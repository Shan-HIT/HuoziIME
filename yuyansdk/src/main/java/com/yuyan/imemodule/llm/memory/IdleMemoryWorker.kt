package com.yuyan.imemodule.llm.memory

import com.yuyan.imemodule.llm.LLMBridge
import com.yuyan.imemodule.llm.scheduler.ModelMutexScheduler
import com.yuyan.imemodule.service.data.StyleConfig
import com.yuyan.imemodule.utils.LogUtil
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.security.MessageDigest
import kotlin.math.max
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class IdleMemoryWorker(
    private val tag: String,
    private val scheduler: ModelMutexScheduler,
    private val userInputHistoryFile: File,
    private val memoryStore: LongTermMemoryStore,
    private val memoryIndexDir: File,
    private val readProgressFile: File,
    private val processMarksFile: File,
    private val preferLatestInputs: Boolean = false,
    private val l1KvDir: File = File((memoryIndexDir.parentFile ?: memoryIndexDir), "imem_l1_kv"),
) {

    private enum class GenKind {
        Ok,
        Failed,
        Preempted,
    }

    private data class GenResult(
        val kind: GenKind,
        val text: String? = null,
    )

    private fun sha256Short(s: String, nBytes: Int = 6): String {
        return try {
            val md = MessageDigest.getInstance("SHA-256")
            val bytes = md.digest(s.toByteArray(Charsets.UTF_8))
            bytes.take(nBytes).joinToString("") { b -> "%02x".format(b) }
        } catch (_: Exception) {
            ""
        }
    }

    private fun logFullTextDebug(kind: String, label: String, text: String, extra: String = "") {
        if (!com.yuyan.imemodule.BuildConfig.DEBUG) return

        val maxChunk = 3400
        val sha = sha256Short(text)
        val total = if (text.isEmpty()) 1 else ((text.length + maxChunk - 1) / maxChunk)
        LogUtil.event(
            LogUtil.Category.MEM,
            tag,
            "prompt_full_begin",
            "kind=$kind label=$label sha=$sha len=${text.length} chunks=$total $extra".trim()
        )
        var part = 1
        var i = 0
        while (i < text.length) {
            val end = minOf(text.length, i + maxChunk)
            val chunk = text.substring(i, end)
            LogUtil.event(
                LogUtil.Category.MEM,
                tag,
                "prompt_full_chunk",
                "kind=$kind label=$label part=$part/$total text=${chunk}"
            )
            i = end
            part++
        }
        if (text.isEmpty()) {
            LogUtil.event(
                LogUtil.Category.MEM,
                tag,
                "prompt_full_chunk",
                "kind=$kind label=$label part=1/1 text="
            )
        }
        LogUtil.event(
            LogUtil.Category.MEM,
            tag,
            "prompt_full_end",
            "kind=$kind label=$label sha=$sha"
        )
    }

    data class ProcessResult(
        val processed: Int,
        val indexed: Int,
    )

    fun tryProcessBatch(
        handle: Long,
        vectorDbHandle: Long = handle,
        maxLines: Int = 12,
        enableL1KvPrecompute: Boolean = true,
        stopNativeIfGenerating: () -> Unit,
        generateMemoryWorker: (prompt: String, maxTokens: Int, callback: LLMBridge.TokenCallback) -> Unit = { prompt, maxTokens, callback ->
            LLMBridge.generateMemoryWorker(handle, prompt, maxTokens, callback)
        },
    ): ProcessResult {
        val token = scheduler.tryEnterMemoryProcessing()
        if (token == null) {
            if (com.yuyan.imemodule.BuildConfig.DEBUG && LogUtil.rateLimit("mem.worker.skip_not_idle", 900)) {
                LogUtil.eventD(
                    LogUtil.Category.MEM,
                    tag,
                    "memory_worker_batch_skip",
                    "reason=scheduler_not_idle state=${scheduler.state()}"
                )
            }
            return ProcessResult(0, 0)
        }

        val tBatchStartMs = System.currentTimeMillis()
        LogUtil.event(
            LogUtil.Category.MEM,
            tag,
            "memory_worker_batch_begin",
            "maxLines=$maxLines historyExists=${userInputHistoryFile.exists()}"
        )

        try {
            if (!userInputHistoryFile.exists()) {
                LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_batch_skip", "reason=no_history_file")
                return ProcessResult(0, 0)
            }
            stopNativeIfGenerating()
            if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
            LLMBridge.vectorDbInit(vectorDbHandle, memoryIndexDir.absolutePath)

            val rawStartLine = readLastProcessedLine()
            val lines = userInputHistoryFile.readLines()
            var startLine = rawStartLine
            if (preferLatestInputs) {
                val tailStart = (lines.size - maxLines).coerceAtLeast(0)
                if (startLine < tailStart) {
                    LogUtil.event(
                        LogUtil.Category.MEM,
                        tag,
                        "memory_worker_batch_seek_tail",
                        "from=$startLine to=$tailStart totalLines=${lines.size} maxLines=$maxLines skipped=${tailStart - startLine}"
                    )
                    startLine = tailStart
                }
            }
            LogUtil.event(
                LogUtil.Category.MEM,
                tag,
                "memory_worker_batch_progress",
                "startLine=$startLine totalLines=${lines.size} rawStartLine=$rawStartLine preferLatest=$preferLatestInputs"
            )
            if (startLine >= lines.size) {
                LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_batch_skip", "reason=already_fully_processed")
                return ProcessResult(0, 0)
            }

            var processed = 0
            var indexed = 0
            var advanced = 0

            processMarksFile.parentFile?.mkdirs()
            FileWriter(processMarksFile, true).use { markW ->
                for (i in startLine until minOf(lines.size, startLine + maxLines)) {
                    if (!scheduler.isMemoryTokenValid(token)) {
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=preempted_unprocessed")
                        break
                    }

                    val raw = lines[i].trim()
                    if (raw.isBlank()) {
                        processed++
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = System.currentTimeMillis(), status = "skipped_blank")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=skipped_blank")
                        continue
                    }

                    val obj = try {
                        JSONObject(raw)
                    } catch (_: Exception) {
                        processed++
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = System.currentTimeMillis(), status = "skipped_invalid_json")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=skipped_invalid_json")
                        continue
                    }

                    val ts = obj.optLong("timestamp", System.currentTimeMillis())
                    val content = obj.optString("content", "").trim()
                    val source = obj.optString("source", "").trim()
                    val pkg = obj.optString("pkg", "").trim()
                    processed++

                    if (content.length < 6) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "skipped_short")
                        val preview = content.take(16).replace("\n", " ")
                        val sha = sha256Short(content)
                        LogUtil.event(
                            LogUtil.Category.MEM,
                            tag,
                            "memory_worker_line",
                            "line=$i status=skipped_short len=${content.length} sha=$sha source=${source.ifEmpty { "<none>" }} pkg=${pkg.ifEmpty { "<none>" }} preview=\"$preview\""
                        )
                        continue
                    }
                    if (MemToolcallParser.isNoMemMarker(content)) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "skipped_no_mem_marker")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=skipped_no_mem_marker")
                        continue
                    }

                    val gen = generateMemoryWorkerOutput(
                        handle = handle,
                        memoryToken = token,
                        userText = content,
                        timestamp = ts,
                        stopNativeIfGenerating = stopNativeIfGenerating,
                        generateMemoryWorker = generateMemoryWorker,
                    )
                    when (gen.kind) {
                        GenKind.Preempted -> {
                            LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=preempted_unprocessed")
                            break
                        }

                        GenKind.Failed -> {
                            advanced++
                            appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "extract_failed")
                            LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=extract_failed")
                            continue
                        }
                        GenKind.Ok -> {                            
                        }
                    }

                    val extracted = gen.text
                        ?.trim()
                        ?.takeIf { it.isNotBlank() }

                    if (extracted == null) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "extract_failed")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=extract_failed")
                        continue
                    }

                    if (MemToolcallParser.isNoMemMarker(extracted)) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "no_mem")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=no_mem")
                        continue
                    }

                    val objOut = try {
                        JSONObject(extracted)
                    } catch (_: Exception) {
                        val start = extracted.indexOf('{')
                        val end = extracted.lastIndexOf('}')
                        if (start != -1 && end != -1 && end > start) {
                            try { JSONObject(extracted.substring(start, end + 1)) } catch (_: Exception) { null }
                        } else null
                    }
                    if (objOut == null) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "extract_invalid_json")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=extract_invalid_json")
                        continue
                    }
                    val summary = objOut.optString("summary", "").trim()
                    if (summary.length < 3) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "extract_empty_summary")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=extract_empty_summary")
                        continue
                    }

                    val datetime = objOut.optString("datetime", "").trim()
                    val location = objOut.optString("location", "").trim()
                    val item = objOut.optString("item", "").trim()
                    val detailField = objOut.optString("detail", "").trim()
                    val participantsArr = objOut.optJSONArray("participants")
                    val participants = buildList {
                        if (participantsArr != null) {
                            for (j in 0 until participantsArr.length()) {
                                val p = participantsArr.optString(j, "").trim()
                                if (p.isNotEmpty()) add(p)
                            }
                        }
                    }
                    val indexText = buildString {
                        append(summary)
                        if (datetime.isNotEmpty()) append(" | 时间: ").append(datetime)
                        if (location.isNotEmpty()) append(" | 地点: ").append(location)
                        if (participants.isNotEmpty()) append(" | 参与者: ").append(participants.joinToString(", "))
                        if (item.isNotEmpty()) append(" | 事项: ").append(item)
                        if (detailField.isNotEmpty() && detailField != summary) append(" | 细节: ").append(detailField)
                    }.trim()
                    if (indexText.length < 6) {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "index_skipped_short")
                        LogUtil.event(LogUtil.Category.MEM, tag, "memory_worker_line", "line=$i status=index_skipped_short")
                        continue
                    }

                    val label = try {
                        val lab = LLMBridge.vectorDbAddText(vectorDbHandle, indexText)
                        if (lab > 0) lab else null
                    } catch (e: Exception) {
                        LogUtil.e(tag, "MemoryWorker", "vectorDbAddText failed: ${e.message}")
                        null
                    }

                    LogUtil.event(
                        LogUtil.Category.MEM,
                        tag,
                        "memory_worker_index",
                        "line=$i indexedOk=${label != null} label=${label ?: -1}"
                    )
                    if (enableL1KvPrecompute && label != null && label > 0) {
                        try {
                            if (!l1KvDir.exists()) l1KvDir.mkdirs()
                            val memLineForPrompt = ("- " + indexText.replace("\n", " ").trim()).trim()
                            val styles = L1KvBlobPaths.defaultStyleKeys()

                            for (styleKey in styles) {
                                val outFile = L1KvBlobPaths.resolveFile(l1KvDir, label, styleKey)
                                val alreadyOk = outFile.exists() && (outFile.length() > 0)
                                if (alreadyOk) {
                                    if (com.yuyan.imemodule.BuildConfig.DEBUG) {
                                        LogUtil.eventD(
                                            LogUtil.Category.MEM,
                                            tag,
                                            "l1_kv_build_skip",
                                            "line=$i label=$label style=$styleKey reason=exists size=${outFile.length()}"
                                        )
                                    }
                                    continue
                                }

                                val t0 = System.currentTimeMillis()
                                if (com.yuyan.imemodule.BuildConfig.DEBUG) {
                                    LogUtil.eventD(
                                        LogUtil.Category.MEM,
                                        tag,
                                        "l1_kv_build_begin",
                                        "line=$i label=$label style=$styleKey out=${outFile.name}"
                                    )
                                }

                                val ok = try {
                                    LLMBridge.buildMemoryKvBlob(handle, memLineForPrompt, outFile.absolutePath)
                                } catch (_: UnsatisfiedLinkError) {
                                    false
                                } catch (_: Exception) {
                                    false
                                }

                                val size = if (outFile.exists()) outFile.length() else -1L
                                val elapsed = max(0L, System.currentTimeMillis() - t0)
                                LogUtil.event(
                                    LogUtil.Category.MEM,
                                    tag,
                                    "l1_kv_build_end",
                                    "line=$i label=$label style=$styleKey ok=$ok size=$size elapsedMs=$elapsed"
                                )
                            }
                        } catch (e: Exception) {
                            LogUtil.eventW(
                                LogUtil.Category.MEM,
                                tag,
                                "l1_kv_error",
                                "line=$i label=$label ex=${e.javaClass.simpleName}:${e.message}"
                            )
                        }
                    }

                    val now = System.currentTimeMillis()
                    val memId = LongTermMemoryStore.stableId(ts, indexText)
                    val indexedOk = label != null

                    val rec = MemoryRecord(
                        id = memId,
                        timestamp = ts,
                        who = "user",
                        what = item.ifEmpty { "memory_worker" },
                        detail = indexText,
                        source = "memory_worker",
                        vectorLabel = label,
                        processedAtMs = now,
                        indexedOk = indexedOk,
                        sourceLineIndex = i,
                    )

                    val storedOk = try {
                        memoryStore.append(rec)
                        true
                    } catch (e: Exception) {
                        LogUtil.e(tag, "MemoryWorker", "memoryStore.append failed: ${e.message}")
                        false
                    }

                    if (storedOk) {
                        if (indexedOk) indexed++
                        advanced++
                        appendProcessMark(
                            markW,
                            lineIndex = i,
                            timestamp = ts,
                            status = if (indexedOk) "indexed" else "stored_no_index",
                            memoryId = memId,
                            indexedOk = indexedOk,
                            vectorLabel = label,
                        )

                        LogUtil.event(
                            LogUtil.Category.MEM,
                            tag,
                            "memory_worker_store",
                            "line=$i memoryId=$memId storedOk=true indexedOk=$indexedOk"
                        )
                    } else {
                        advanced++
                        appendProcessMark(markW, lineIndex = i, timestamp = ts, status = "store_failed", memoryId = memId)
                        LogUtil.event(
                            LogUtil.Category.MEM,
                            tag,
                            "memory_worker_store",
                            "line=$i memoryId=$memId storedOk=false"
                        )
                    }
                }
            }

            writeLastProcessedLine(startLine + advanced)
            LogUtil.event(
                LogUtil.Category.MEM,
                tag,
                "memory_worker_batch_done",
                "processed=$processed advanced=$advanced indexed=$indexed elapsedMs=${System.currentTimeMillis() - tBatchStartMs}"
            )
            return ProcessResult(processed, indexed)
        } finally {
            scheduler.markMemoryFinished(token)
            LogUtil.event(
                LogUtil.Category.MEM,
                tag,
                "memory_worker_batch_end",
                "elapsedMs=${System.currentTimeMillis() - tBatchStartMs}"
            )
        }
    }

    private fun readLastProcessedLine(): Int {
        return try {
            if (!readProgressFile.exists()) return 0
            readProgressFile.readText().trim().toIntOrNull() ?: 0
        } catch (_: Exception) {
            0
        }
    }

    private fun writeLastProcessedLine(line: Int) {
        try {
            readProgressFile.parentFile?.mkdirs()
            readProgressFile.writeText(line.toString())
        } catch (_: Exception) {
        }
    }

    private fun appendProcessMark(
        w: FileWriter,
        lineIndex: Int,
        timestamp: Long,
        status: String,
        memoryId: String? = null,
        indexedOk: Boolean? = null,
        vectorLabel: Long? = null,
    ) {
        try {
            val obj = JSONObject().apply {
                put("lineIndex", lineIndex)
                put("timestamp", timestamp)
                put("status", status)
                if (memoryId != null) put("memoryId", memoryId)
                if (indexedOk != null) put("indexedOk", indexedOk)
                if (vectorLabel != null) put("vectorLabel", vectorLabel)
            }
            w.write(obj.toString())
            w.write("\n")
        } catch (_: Exception) {
        }
    }

    private fun generateMemoryWorkerOutput(
        handle: Long,
        memoryToken: String,
        userText: String,
        timestamp: Long,
        stopNativeIfGenerating: () -> Unit,
        generateMemoryWorker: (prompt: String, maxTokens: Int, callback: LLMBridge.TokenCallback) -> Unit,
    ): GenResult {
        if (!scheduler.isMemoryTokenValid(memoryToken)) return GenResult(GenKind.Preempted)

        
        stopNativeIfGenerating()

        val prompt = buildMemoryWorkerPrompt(userText)
        logFullTextDebug(
            kind = "memory_worker_prompt",
            label = timestamp.toString(),
            text = prompt,
            extra = "userLen=${userText.length}"
        )
        val sb = StringBuilder()
        val latch = CountDownLatch(1)
        val gotError = AtomicBoolean(false)

        val cb = object : LLMBridge.TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                if (!scheduler.isMemoryTokenValid(memoryToken)) {
                    try { LLMBridge.stop(handle) } catch (_: Exception) {}
                    return
                }
                for (t in tokens) {
                    
                    var s = t
                    s = s.replace("<think>", "")
                    val endThink = s.indexOf("</think>")
                    if (endThink != -1) s = s.substring(0, endThink)
                    sb.append(s)
                }
            }

            override fun onFinished() {
                latch.countDown()
            }

            override fun onError(err: String) {
                gotError.set(true)
                LogUtil.e(tag, "MemoryWorker", "native error: $err")
                latch.countDown()
            }
        }

        try {
            generateMemoryWorker(prompt, 192, cb)
        } catch (e: Exception) {
            LogUtil.e(tag, "MemoryWorker", "generateMemoryWorker failed: ${e.message}")
            return GenResult(GenKind.Failed)
        }

        
        val t0 = System.currentTimeMillis()
        val timeoutMs = 8000L
        while (true) {
            if (!scheduler.isMemoryTokenValid(memoryToken)) {
                try { LLMBridge.stop(handle) } catch (_: Exception) {}
                return GenResult(GenKind.Preempted)
            }
            val ok = latch.await(80, TimeUnit.MILLISECONDS)
            if (ok) break
            if (System.currentTimeMillis() - t0 > timeoutMs) {
                try { LLMBridge.stop(handle) } catch (_: Exception) {}
                return GenResult(GenKind.Failed)
            }
        }

        if (gotError.get()) return GenResult(GenKind.Failed)
        if (!scheduler.isMemoryTokenValid(memoryToken)) return GenResult(GenKind.Preempted)

        val out = sb.toString()
        
        val cut = out.indexOf("<|im_end|>")
        val cleaned = if (cut != -1) out.substring(0, cut) else out
        val finalOut = cleaned.trim()
        logFullTextDebug(
            kind = "memory_worker_output",
            label = timestamp.toString(),
            text = finalOut,
            extra = "rawLen=${out.length}"
        )
        return GenResult(GenKind.Ok, finalOut)
    }

    private fun buildMemoryWorkerPrompt(userText: String): String {
        
        val sys = "[MEMORY_WORKER] 结构化提取记忆。\n" +
            "输出格式：JSON。若无有效信息，输出<NO_MEM>。\n" +
            "字段约束：\n" +
            "- 必有：summary\n" +
            "- 可选：datetime, location, participants(数组), item, detail\n" +
            "决策规则：\n" +
            "- 只在出现明确可执行计划/约定时输出 JSON（只包含原文中能直接支持的字段）。\n" +
            "- 其余情况一律输出<NO_MEM>。\n" +
            "严格要求：只输出 JSON 或 <NO_MEM>，不要任何额外解释。"

        return "<|im_start|>system\n" +
            sys +
            "<|im_end|>\n" +
            "<|im_start|>user\n" +
            userText +
            "<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}
