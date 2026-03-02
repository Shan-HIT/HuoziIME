package com.yuyan.imemodule.service.manager

import com.yuyan.imemodule.BuildConfig
import com.yuyan.imemodule.service.ImeService
import com.yuyan.imemodule.llm.LLMBridge
import com.yuyan.imemodule.llm.LLMBridge.TokenCallback
import com.yuyan.imemodule.llm.BaselinePhraseCandidatesEngine
import com.yuyan.imemodule.llm.PhraseCandidatesEngine
import com.yuyan.imemodule.llm.MemoryPromptSegments
import com.yuyan.imemodule.llm.memory.IdleMemoryWorker
import com.yuyan.imemodule.llm.memory.L1KvBlobPaths
import com.yuyan.imemodule.llm.memory.LongTermMemoryStore
import com.yuyan.imemodule.llm.memory.MemoryTestReset
import com.yuyan.imemodule.llm.memory.MemToolcallParser
import com.yuyan.imemodule.llm.memory.ParametricLogExporter
import com.yuyan.imemodule.llm.postprocess.CandidatePostProcessor
import com.yuyan.imemodule.llm.scheduler.ModelMutexScheduler
import com.yuyan.imemodule.service.data.FeedbackConfig
import com.yuyan.imemodule.service.data.GenerationMode
import com.yuyan.imemodule.service.data.StyleConfig
import com.yuyan.imemodule.utils.LogUtil
import com.yuyan.imemodule.utils.toast
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.io.InputStreamReader
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.math.max
import kotlin.math.min

class ImeLLMManager(
    private val service: ImeService
) {
    private val TAG = "IMEM-OS-LLM"

    companion object {
        private const val PREFS_NAME = "yuyan_ime_prefs"
        private const val PREF_ENABLE_RECALL_KV_SPLICE = "enable_kv_splice_recall"
    }

    private val requestIdCounter = AtomicLong(0)
    private val activeRequestId = AtomicLong(0)
    private val activeRequestStartedAtMs = AtomicLong(0)

    private val managerJob = SupervisorJob()
    private val managerScope = CoroutineScope(managerJob + Dispatchers.IO)

    private val sessionSeq = AtomicLong(1)
    @Volatile private var sessionSignature: String = ""

    @Volatile private var isPersistentInferenceMode: Boolean = false

    @Volatile private var enableRecallKvSplice: Boolean = false

    fun isRecallKvSpliceEnabled(): Boolean = enableRecallKvSplice

    fun setRecallKvSpliceEnabled(enabled: Boolean, reason: String = "user"): Boolean {
        val prev = enableRecallKvSplice
        enableRecallKvSplice = enabled
        try {
            service.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE)
                .edit()
                .putBoolean(PREF_ENABLE_RECALL_KV_SPLICE, enabled)
                .apply()
        } catch (_: Exception) {
        }
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "kv_splice_switch",
            "enabled=$enabled prev=$prev reason=$reason"
        )
        return enabled
    }

    private val lastActivityAtMs = AtomicLong(System.currentTimeMillis())
    private var idleClearJob: Job? = null

    private fun previewText(text: String, maxChars: Int = 48): String {
        if (text.length <= maxChars) return text
        return text.substring(0, maxChars) + "…"
    }

    private val hanRegex = Regex("\\p{IsHan}")

    private fun preferCjkOutput(vararg texts: String): Boolean {
        for (t in texts) {
            if (t.isNotEmpty() && hanRegex.containsMatchIn(t)) return true
        }
        return false
    }

    private fun sha256Short(s: String, nBytes: Int = 6): String {
        return try {
            val md = MessageDigest.getInstance("SHA-256")
            val bytes = md.digest(s.toByteArray(Charsets.UTF_8))
            bytes.take(nBytes).joinToString("") { b -> "%02x".format(b) }
        } catch (_: Exception) {
            ""
        }
    }

    private fun sanitizeForSingleLineLog(s: String): String {
        if (s.isEmpty()) return s
        return s
            .replace("\\r", "\\\\r")
            .replace("\\n", "\\\\n")
            .replace("\\t", "\\\\t")
    }

    private fun logFullPromptDebug(kind: String, rid: Long, prompt: String) {
        val hash = sha256Short(prompt)
        val len = prompt.length

        if (!BuildConfig.DEBUG) {
            LogUtil.event(LogUtil.Category.LLM, TAG, "prompt_meta", "kind=$kind rid=$rid sha=$hash len=$len")
            return
        }

        val chunkSize = 1800
        val totalChunks = if (len == 0) 1 else ((len + chunkSize - 1) / chunkSize)
        LogUtil.event(LogUtil.Category.LLM, TAG, "prompt_full_begin", "kind=$kind rid=$rid sha=$hash len=$len chunks=$totalChunks")
        if (len == 0) {
            LogUtil.event(LogUtil.Category.LLM, TAG, "prompt_full_chunk", "kind=$kind rid=$rid part=1/1 text=")
            LogUtil.event(LogUtil.Category.LLM, TAG, "prompt_full_end", "kind=$kind rid=$rid sha=$hash")
            return
        }

        var idx = 0
        var part = 1
        while (idx < len) {
            val end = minOf(len, idx + chunkSize)
            val chunk = sanitizeForSingleLineLog(prompt.substring(idx, end))
            LogUtil.event(
                LogUtil.Category.LLM,
                TAG,
                "prompt_full_chunk",
                "kind=$kind rid=$rid part=$part/$totalChunks text=$chunk"
            )
            idx = end
            part++
        }
        LogUtil.event(LogUtil.Category.LLM, TAG, "prompt_full_end", "kind=$kind rid=$rid sha=$hash")
    }
    
    var generationHandle: Long = 0L

    private var embeddingHandle: Long = 0L

    private val PREF_MEM_INCLUDE_LAST_MSG_IN_QUERY = "mem_include_last_msg_in_query"
    
    private val generationLock = ReentrantLock()
    private val isNativeGenerating = AtomicBoolean(false)
    private val lastGenEndTime = AtomicLong(0L)

    private var lastMemFeedbackAtMs = 0L
    private var lastMemFeedbackMsg: String? = null

    private val lastIdleMemoryRunAtMs = AtomicLong(0L)

    private fun showMemFeedback(msg: String, minIntervalMs: Long = 900L, force: Boolean = false) {
        if (msg.isBlank()) return
        val now = System.currentTimeMillis()
        if (!force) {
            if (lastMemFeedbackMsg == msg && now - lastMemFeedbackAtMs < minIntervalMs) return
            if (now - lastMemFeedbackAtMs < minIntervalMs) return
        }
        lastMemFeedbackAtMs = now
        lastMemFeedbackMsg = msg
        service.showFeedback(msg)
    }

    private fun debugToastOnce(key: String, msg: String, minIntervalMs: Long = 1200L) {
        if (!BuildConfig.DEBUG) return
        if (!LogUtil.rateLimit(key, minIntervalMs)) return
        try {
            service.applicationContext.toast(msg)
        } catch (_: Exception) {
        }
    }

    private fun memToastOnce(key: String, msg: String, minIntervalMs: Long = 1200L) {
        if (!LogUtil.rateLimit(key, minIntervalMs)) return
        try {
            service.applicationContext.toast(msg)
        } catch (_: Exception) {
        }
    }

    private data class RetrievedItem(
        val label: Long,
        val rawScore: Float,
        val vecCos: Float,
        val vecDist: Float,
        val sim01: Float,
        val text: String,
    )

    private enum class VecScoreMode { COSINE, DISTANCE, UNKNOWN }

    private val scheduler = ModelMutexScheduler()

    private val memoryIndexDir: File by lazy { File(service.filesDir, "imem_l2_index") }
    private val memoryStore: LongTermMemoryStore by lazy { LongTermMemoryStore(File(service.filesDir, "imem_l2_store")) }
    private val parametricExporter: ParametricLogExporter by lazy {
        ParametricLogExporter(File(service.filesDir, "imem_l3/parametric_logs.jsonl"))
    }
    private val idleMemoryWorker: IdleMemoryWorker by lazy {
        IdleMemoryWorker(
            tag = TAG,
            scheduler = scheduler,
            userInputHistoryFile = File(service.filesDir, "user_input_history.jsonl"),
            memoryStore = memoryStore,
            memoryIndexDir = memoryIndexDir,
            readProgressFile = File(service.filesDir, "imem_l2_store/user_input_history.progress"),
            processMarksFile = File(service.filesDir, "imem_l2_store/user_input_history.process_marks.jsonl"),
            preferLatestInputs = false,
        )
    }
    
    var completionJob: Job? = null

    private var idleMemoryJob: Job? = null
    
    var currentGenerationMode = GenerationMode.PHRASE
    var currentStyleMode = StyleConfig.STYLE_BUSINESS
    var lastSubmittedPrompt: String = ""
    var isAiCompletionEnabled = true
    
    var syncedHistoryStr: String = "无"
    var syncedLastMsgStr: String = "无"
    var syncedMemoryStr: String = "无"
    private val tmpCounter = AtomicInteger(0)

    @Volatile private var memoryContextKey: String = ""
    @Volatile private var memoryRetrievalDoneForContext: Boolean = false
    @Volatile private var memoryInjectedForContext: String? = null

    @Volatile private var inferenceCompletedForContext: Boolean = false
    @Volatile private var inferenceCompletedPrefix: String = ""

    private fun getVectorDbHandleOrFallback(): Long {
        return embeddingHandle
    }

    private fun buildMemoryContextKey(): String {
        val history = syncedHistoryStr.ifBlank { "无" }
        val lastMsg = syncedLastMsgStr.ifBlank { "无" }
        return "style=$currentStyleMode|h=${sha25612Hex(history)}|l=${sha25612Hex(lastMsg)}"
    }

    private fun refreshContextScopedState(currentKey: String) {
        if (currentKey == memoryContextKey) return
        memoryContextKey = currentKey
        memoryRetrievalDoneForContext = false
        memoryInjectedForContext = null
        syncedMemoryStr = "无"
        inferenceCompletedForContext = false
        inferenceCompletedPrefix = ""
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "mem_context_reset",
            "ctx=$currentKey"
        )
    }

    private fun markMemoryRetrievedForContext(contextKey: String, injectedMemory: String) {
        if (contextKey != memoryContextKey) return
        memoryRetrievalDoneForContext = true
        memoryInjectedForContext = injectedMemory
        syncedMemoryStr = injectedMemory
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "mem_context_cached",
            "ctx=$contextKey injectedLen=${injectedMemory.length}"
        )
    }

    private fun isTerminalCandidate(candidate: String): Boolean {
        val t = candidate.trim()
        if (t.isEmpty()) return false
        if (t.contains("<MEM_RETRIEVAL>") || t.contains("<NO_MEM>")) return false
        val tail = t.last()
        return tail == '。' || tail == '！' || tail == '？' || tail == '!' || tail == '?' || tail == ';' || tail == '；'
    }

    private fun maybeMarkInferenceCompleted(
        contextKey: String,
        requestId: Long,
        source: String,
        currentInput: String,
        candidates: List<String>,
    ) {
        if (contextKey != memoryContextKey) return
        if (inferenceCompletedForContext) return
        val top = candidates.firstOrNull()?.trim().orEmpty()
        if (!isTerminalCandidate(top)) return

        inferenceCompletedForContext = true
        inferenceCompletedPrefix = (currentInput + top).trim()
        LogUtil.event(
            LogUtil.Category.LLM,
            TAG,
            "inference_completed",
            "rid=$requestId source=$source ctx=$contextKey topLen=${top.length} topPreview=\"${previewText(top, 48)}\""
        )
        if (LogUtil.rateLimit("llm.inference_complete.toast", 1200L)) {
            service.showFeedback("✅ 推理已完成")
        }
    }

    private fun shouldTemporarilySkipAfterCompletion(currentInput: String): Boolean {
        if (!inferenceCompletedForContext || inferenceCompletedPrefix.isBlank()) return false
        val prefix = inferenceCompletedPrefix
        if (!currentInput.startsWith(prefix)) return false

        val extra = currentInput.substring(prefix.length)
        if (extra.isEmpty()) return true

        val compact = extra.filterNot { it.isWhitespace() }
        if (compact.isEmpty()) return true

        val onlyPunctuation = compact.all {
            it == '。' || it == '，' || it == '；' || it == '！' || it == '？' ||
                it == '.' || it == ',' || it == ';' || it == '!' || it == '?' || it == '…'
        }
        return onlyPunctuation && compact.length <= 2
    }

    private fun ensureEmbeddingVectorDbReadyOnce() {
        val h = getVectorDbHandleOrFallback()
        if (h == 0L) return

        try {
            if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
            LLMBridge.vectorDbInit(h, memoryIndexDir.absolutePath)

            val metaFile = File(memoryIndexDir, "embedding_meta.json")
            val expectedAsset = "bge-small-zh-v1.5-q8_0.gguf"
            val expectedImpl = "v2_seq_embeddings"
            val currentAsset = try {
                if (!metaFile.exists()) null else JSONObject(metaFile.readText()).optString("asset", "").trim().ifEmpty { null }
            } catch (_: Exception) {
                null
            }
            val currentImpl = try {
                if (!metaFile.exists()) null else JSONObject(metaFile.readText()).optString("impl", "").trim().ifEmpty { null }
            } catch (_: Exception) {
                null
            }

            if (currentAsset != expectedAsset || currentImpl != expectedImpl) {
                LogUtil.event(
                    LogUtil.Category.MEM,
                    TAG,
                    "vector_db_rebuild_needed",
                    "fromAsset=${currentAsset ?: "<none>"} toAsset=$expectedAsset fromImpl=${currentImpl ?: "<none>"} toImpl=$expectedImpl"
                )
                val ok = try {
                    LLMBridge.vectorDbRebuildFromTexts(h, memoryIndexDir.absolutePath)
                } catch (_: UnsatisfiedLinkError) {
                    false
                } catch (_: Exception) {
                    false
                }
                LogUtil.event(LogUtil.Category.MEM, TAG, "vector_db_rebuild_done", "ok=$ok")
                try {
                    metaFile.writeText(
                        JSONObject().apply {
                            put("asset", expectedAsset)
                            put("impl", expectedImpl)
                            put("updatedAtMs", System.currentTimeMillis())
                        }.toString()
                    )
                } catch (_: Exception) {
                }
            }
        } catch (_: Exception) {
        }
    }

    fun markActivity(reason: String = "unspecified") {
        lastActivityAtMs.set(System.currentTimeMillis())
        scheduleIdleKvClear()
        if (LogUtil.rateLimit("llm.activity.$reason", 600)) {
            LogUtil.eventD(LogUtil.Category.CORE, TAG, "activity", "reason=$reason")
        }
    }

    fun onContextSignature(signature: String, reason: String = "context_sync"): Boolean {
        val sig = signature.trim()
        if (sig.isBlank()) return false
        markActivity(reason)
        val changed = sig != sessionSignature
        if (!changed) {
            try { if (generationHandle != 0L) com.yuyan.imemodule.llm.LLMBridge.setSessionSignature(generationHandle, sig) } catch (_: Exception) {}
            return false
        }
        sessionSignature = sig
        sessionSeq.incrementAndGet()
        try { if (generationHandle != 0L) com.yuyan.imemodule.llm.LLMBridge.setSessionSignature(generationHandle, sig) } catch (_: Exception) {}
        LogUtil.event(LogUtil.Category.CORE, TAG, "session_switch", "reason=$reason sig=${previewText(sig, 12)}")
        stopCompletion(reason = "context_updated:$reason")
        return true
    }

    private fun scheduleIdleKvClear() {
        idleClearJob?.cancel()
        idleClearJob = managerScope.launch {
            while (isActive) {
                delay(2000)
                val idleMs = System.currentTimeMillis() - lastActivityAtMs.get()
                if (idleMs < 30_000) continue
                if (isPersistentInferenceMode) continue
                if (generationHandle == 0L) continue
                if (isNativeGenerating.get()) continue
                if (scheduler.state() != ModelMutexScheduler.State.Idle) continue
                clearKvKeepSystem(reason = "idle_30s")
                break
            }
        }
    }

    fun enterPersistentInferenceMode(reason: String = "unspecified") {
        if (generationHandle == 0L) return
        isPersistentInferenceMode = true
        markActivity("enter_persistent")

        try { idleMemoryJob?.cancel() } catch (_: Exception) {}
        idleMemoryJob = null
        scheduler.preemptForInference()

        stopCompletion(reason = "enter_persistent:$reason")
        sessionSeq.incrementAndGet()

        val ok = resetToStyleBaseline(reason = "enter_persistent:$reason")
        LogUtil.event(
            LogUtil.Category.CORE,
            TAG,
            "persistent_enter",
            "reason=$reason ok=$ok style=$currentStyleMode"
        )

        scheduler.markInferenceFinished()
    }

    fun exitPersistentInferenceMode(reason: String = "unspecified") {
        isPersistentInferenceMode = false
        markActivity("exit_persistent")
        LogUtil.event(LogUtil.Category.CORE, TAG, "persistent_exit", "reason=$reason")
    }

    private fun resetToStyleBaseline(reason: String): Boolean {
        if (generationHandle == 0L) return false

        val cacheFileName = StyleConfig.CACHE_FILES[currentStyleMode] ?: "kv_cache_default.bin"
        val cachePath = File(service.filesDir, cacheFileName).absolutePath
        val cacheFile = File(cachePath)
        val targetPrompt = StyleConfig.PROMPTS[currentStyleMode] ?: ""

        generationLock.lock()
        return try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                LLMBridge.stop(generationHandle)
            }

            try { LLMBridge.clearKvKeepSystem(generationHandle) } catch (_: Exception) {}

            var loaded = false
            val cacheExisted = cacheFile.exists()
            if (cacheExisted) {
                loaded = LLMBridge.loadSession(generationHandle, cachePath)
            }
            if (!loaded && !cacheExisted) {
                val formatted = getFormattedSystemPrompt(targetPrompt)
                if (LLMBridge.saveKVCacheSnapshot(generationHandle, formatted, cachePath)) {
                    loaded = LLMBridge.loadSession(generationHandle, cachePath)
                }
            }

            if (loaded) {
                updateReusablePrefixTokenCount()
            }

            LogUtil.event(
                LogUtil.Category.MEM,
                TAG,
                "style_baseline_reset",
                "reason=$reason loaded=$loaded style=$currentStyleMode"
            )
            loaded
        } catch (_: Exception) {
            false
        } finally {
            generationLock.unlock()
        }
    }

    fun clearKvKeepSystem(reason: String = "unspecified") {
        if (generationHandle == 0L) return
        markActivity("kv_clear_request")
        generationLock.lock()
        try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                com.yuyan.imemodule.llm.LLMBridge.stop(generationHandle)
            }
            val ok = try {
                com.yuyan.imemodule.llm.LLMBridge.clearKvKeepSystem(generationHandle)
            } catch (_: Exception) {
                false
            }
            LogUtil.event(
                LogUtil.Category.MEM,
                TAG,
                "kv_clear",
                "reason=$reason ok=$ok"
            )
        } finally {
            generationLock.unlock()
        }
    }

    fun injectTestMemory(text: String): Boolean {
        val indexText = text.trim()
        if (indexText.length < 3) return false
        if (generationHandle == 0L) return false
        markActivity("mem_inject")

        var label: Long? = null
        generationLock.lock()
        try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                com.yuyan.imemodule.llm.LLMBridge.stop(generationHandle)
            }
            try {
                if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
                val vdbHandle = getVectorDbHandleOrFallback()
                com.yuyan.imemodule.llm.LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath)
            } catch (_: Exception) {
            }
            label = try {
                val lab = com.yuyan.imemodule.llm.LLMBridge.vectorDbAddText(getVectorDbHandleOrFallback(), indexText)
                if (lab > 0) lab else null
            } catch (_: Exception) {
                null
            }
        } finally {
            generationLock.unlock()
        }

        val ts = System.currentTimeMillis()
        val rec = com.yuyan.imemodule.llm.memory.MemoryRecord(
            id = com.yuyan.imemodule.llm.memory.LongTermMemoryStore.stableId(ts, indexText),
            timestamp = ts,
            who = "manual",
            what = "manual_inject",
            detail = indexText,
            source = "manual_inject",
            vectorLabel = label,
        )
        return try {
            memoryStore.append(rec)
            true
        } catch (_: Exception) {
            false
        }
    }

    fun listMemories(limit: Int = 200): List<com.yuyan.imemodule.llm.memory.MemoryRecord> {
        return try {
            memoryStore.list(limit = limit, includeDeleted = false)
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun deleteMemory(id: String): Boolean {
        return try {
            memoryStore.softDelete(id)
        } catch (_: Exception) {
            false
        }
    }

    fun debugSearchMemory(query: String, k: Int = 3): List<String> {
        val q = query.trim()
        if (q.isEmpty() || generationHandle == 0L) return emptyList()

        val deletedLabels: Set<Long> = try {
            memoryStore.loadDeletedVectorLabels()
        } catch (_: Exception) {
            emptySet()
        }

        val out = ArrayList<String>()
        generationLock.lock()
        try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                com.yuyan.imemodule.llm.LLMBridge.stop(generationHandle)
            }
            try {
                if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
                val vdbHandle = getVectorDbHandleOrFallback()
                com.yuyan.imemodule.llm.LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath)
            } catch (_: Exception) {
            }
            val labels = try {
                com.yuyan.imemodule.llm.LLMBridge.vectorDbSearch(getVectorDbHandleOrFallback(), q, k)
            } catch (_: Exception) {
                LongArray(0)
            }
            for (lab in labels) {
                if (deletedLabels.contains(lab)) continue
                try {
                    val txt = com.yuyan.imemodule.llm.LLMBridge.vectorDbGetText(getVectorDbHandleOrFallback(), lab).trim()
                    if (txt.isNotEmpty()) out.add(txt)
                } catch (_: Exception) {
                }
            }
        } finally {
            generationLock.unlock()
        }
        return out
    }

    private fun sha256Short(s: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(s.toByteArray(Charsets.UTF_8))
        return bytes.take(8).joinToString("") { "%02x".format(it) }
    }

    private fun sha25612Hex(s: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(s.toByteArray(Charsets.UTF_8))
        return bytes.take(12).joinToString("") { "%02x".format(it) }
    }

    private fun sha256Hex(bytes: ByteArray): String {
        val out = MessageDigest.getInstance("SHA-256").digest(bytes)
        return out.joinToString("") { "%02x".format(it) }
    }

    private fun loadImportedPreMemIds(idsFile: File): MutableSet<String> {
        if (!idsFile.exists()) return HashSet()
        return try {
            idsFile.readLines()
                .asSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .toMutableSet()
        } catch (_: Exception) {
            HashSet()
        }
    }

    private fun importPrebuiltMemoriesIfNeeded() {
        if (generationHandle == 0L) return

        val storeDir = File(service.filesDir, "imem_l2_store")
        storeDir.mkdirs()
        val stateFile = File(storeDir, "pre_mem.state.json")
        val idsFile = File(storeDir, "pre_mem.imported_ids.txt")

        val assetBytes = try {
            service.assets.open("pre_mem.jsonl").use { it.readBytes() }
        } catch (e: Exception) {
            LogUtil.e(TAG, "PreMem", "open pre_mem.jsonl failed: ${e.message}")
            return
        }
        val assetHash = sha256Hex(assetBytes)

        var stateSaysDone = false
        var stateHashMatches = false
        var existingVdbCount = -1

        try {
            if (stateFile.exists()) {
                val s = stateFile.readText().trim()
                if (s.isNotEmpty()) {
                    val obj = JSONObject(s)
                    val prevHash = obj.optString("sha256", "")
                    val done = obj.optBoolean("done", false)
                    stateSaysDone = done
                    stateHashMatches = prevHash == assetHash
                }
            }
        } catch (_: Exception) {
        }

        if (stateSaysDone && stateHashMatches) {
            existingVdbCount = try {
                if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
                val vdbHandle = getVectorDbHandleOrFallback()
                LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath)
                ensureEmbeddingVectorDbReadyOnce()
                LLMBridge.vectorDbCount(vdbHandle)
            } catch (_: Exception) {
                -1
            }
            if (existingVdbCount > 0) {
                LogUtil.i(TAG, "PreMem", "pre_mem.jsonl already imported (hash match), vdbCount=$existingVdbCount")
                return
            }
            LogUtil.event(
                LogUtil.Category.MEM,
                TAG,
                "pre_mem_state_inconsistent",
                "done=true hashMatch=true vdbCount=$existingVdbCount action=reindex_only"
            )
        }

        val reindexOnly = stateSaysDone && stateHashMatches && existingVdbCount == 0

        val importedIds = loadImportedPreMemIds(idsFile)
        var imported = 0
        var skipped = 0
        var indexedOk = 0
        var indexedFail = 0
        var l1KvBuiltOk = 0
        var l1KvBuiltFail = 0
        var l1KvSkipped = 0

        assetBytes.inputStream().bufferedReader(Charsets.UTF_8).use { r ->
            r.forEachLine { rawLine ->
                val line = rawLine.trim()
                if (line.isEmpty()) return@forEachLine

                val id = "pre_mem_" + sha25612Hex(line)
                if (!reindexOnly && importedIds.contains(id)) {
                    skipped++
                    return@forEachLine
                }

                val obj = try {
                    JSONObject(line)
                } catch (_: Exception) {
                    skipped++
                    return@forEachLine
                }

                val summary = obj.optString("summary", "").trim()
                if (summary.length < 3) {
                    skipped++
                    return@forEachLine
                }

                val datetime = obj.optString("datetime", "").trim()
                val location = obj.optString("location", "").trim()
                val item = obj.optString("item", "").trim()
                val detailField = obj.optString("detail", "").trim()
                val participantsArr = obj.optJSONArray("participants")
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
                    skipped++
                    return@forEachLine
                }

                var label: Long? = null
                generationLock.lock()
                try {
                    if (isNativeGenerating.compareAndSet(true, false)) {
                        LLMBridge.stop(generationHandle)
                    }
                    try {
                        if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
                        val vdbHandle = getVectorDbHandleOrFallback()
                        LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath)
                    } catch (_: Exception) {
                    }
                    label = try {
                        val lab = LLMBridge.vectorDbAddText(getVectorDbHandleOrFallback(), indexText)
                        if (lab > 0) lab else null
                    } catch (_: Exception) {
                        null
                    }
                } finally {
                    generationLock.unlock()
                }

                if (label == null) {
                    indexedFail++
                    LogUtil.event(
                        LogUtil.Category.MEM,
                        TAG,
                        "pre_mem_index_failed",
                        "id=$id len=${indexText.length} preview=\"${previewText(indexText, 48)}\""
                    )
                }

                if (enableRecallKvSplice && label != null && label > 0) {
                    val l1Dir = L1KvBlobPaths.resolveDir(service.filesDir)
                    val memLineForPrompt = ("- " + indexText.replace("\n", " ").trim()).trim()
                    val styleKeys = L1KvBlobPaths.defaultStyleKeys()

                    generationLock.lock()
                    try {
                        if (isNativeGenerating.compareAndSet(true, false)) {
                            LLMBridge.stop(generationHandle)
                        }
                        if (!l1Dir.exists()) l1Dir.mkdirs()

                        try { LLMBridge.clearKvKeepSystem(generationHandle) } catch (_: Exception) {}

                        for (styleKey in styleKeys) {
                            val outFile = L1KvBlobPaths.resolveFile(l1Dir, label!!, styleKey)
                            val alreadyOk = outFile.exists() && outFile.length() > 0
                            if (alreadyOk) {
                                l1KvSkipped++
                                if (BuildConfig.DEBUG) {
                                    LogUtil.eventD(
                                        LogUtil.Category.MEM,
                                        TAG,
                                        "pre_mem_l1_kv_skip",
                                        "id=$id label=$label style=$styleKey reason=exists size=${outFile.length()}"
                                    )
                                }
                                continue
                            }

                            val t0 = System.currentTimeMillis()
                            if (BuildConfig.DEBUG) {
                                LogUtil.eventD(
                                    LogUtil.Category.MEM,
                                    TAG,
                                    "pre_mem_l1_kv_build_begin",
                                    "id=$id label=$label style=$styleKey out=${outFile.name} memLen=${memLineForPrompt.length}"
                                )
                            }

                            val ok = try {
                                LLMBridge.buildMemoryKvBlob(generationHandle, memLineForPrompt, outFile.absolutePath)
                            } catch (_: UnsatisfiedLinkError) {
                                false
                            } catch (_: Exception) {
                                false
                            }

                            val size = if (outFile.exists()) outFile.length() else -1L
                            val elapsed = max(0L, System.currentTimeMillis() - t0)

                            if (ok && size > 0) l1KvBuiltOk++ else l1KvBuiltFail++

                            LogUtil.event(
                                LogUtil.Category.MEM,
                                TAG,
                                "pre_mem_l1_kv_build_end",
                                "id=$id label=$label style=$styleKey ok=$ok size=$size elapsedMs=$elapsed"
                            )
                        }

                        try { LLMBridge.clearKvKeepSystem(generationHandle) } catch (_: Exception) {}
                    } finally {
                        generationLock.unlock()
                    }
                }

                val ts = System.currentTimeMillis()
                val rec = com.yuyan.imemodule.llm.memory.MemoryRecord(
                    id = id,
                    timestamp = ts,
                    who = "pre_mem",
                    what = item.ifEmpty { "pre_mem" },
                    detail = indexText,
                    source = "pre_mem",
                    vectorLabel = label,
                    processedAtMs = ts,
                    indexedOk = label != null,
                )

                if (!reindexOnly) {
                    val storedOk = try {
                        memoryStore.append(rec)
                        true
                    } catch (_: Exception) {
                        false
                    }
                    if (!storedOk) {
                        skipped++
                        return@forEachLine
                    }

                    try {
                        FileWriter(idsFile, true).use { w ->
                            w.write(id)
                            w.write("\n")
                        }
                        importedIds.add(id)
                    } catch (_: Exception) {
                    }
                }

                imported++
                if (label != null) indexedOk++
            }
        }

        generationLock.lock()
        try {
            val vdbHandle = getVectorDbHandleOrFallback()
            try { LLMBridge.vectorDbClose(vdbHandle) } catch (_: Exception) {}
            try { LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath) } catch (_: Exception) {}
        } finally {
            generationLock.unlock()
        }

        try {
            stateFile.writeText(
                JSONObject().apply {
                    put("sha256", assetHash)
                    put("done", true)
                    put("imported", imported)
                    put("skipped", skipped)
                    put("indexedOk", indexedOk)
                    put("reindexOnly", reindexOnly)
                    put("updatedAtMs", System.currentTimeMillis())
                }.toString()
            )
        } catch (_: Exception) {
        }

        LogUtil.i(TAG, "PreMem", "pre_mem import done: imported=$imported skipped=$skipped indexedOk=$indexedOk")
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "pre_mem_import_summary",
            "imported=$imported skipped=$skipped indexedOk=$indexedOk indexedFail=$indexedFail l1KvBuiltOk=$l1KvBuiltOk l1KvBuiltFail=$l1KvBuiltFail l1KvSkipped=$l1KvSkipped"
        )
    }

    @Volatile private var phraseCandidatesEngine: PhraseCandidatesEngine = BaselinePhraseCandidatesEngine()

    private val reusablePrefixTokenCountCache = ConcurrentHashMap<String, Int>()

    @Volatile private var enableIncrementalPrefillPipeline: Boolean = true

    fun setPhraseCandidatesEngine(engine: PhraseCandidatesEngine) {
        phraseCandidatesEngine = engine
    }

    suspend fun initIMEMSystem() {
        val startTime = System.currentTimeMillis()
        LogUtil.i(TAG, "Init", ">>>>>>>> 开始初始化 LLM 核心 <<<<<<<<")
        
        val prefs = service.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE)
        currentStyleMode = prefs.getString("persisted_style_mode", StyleConfig.STYLE_BUSINESS) ?: StyleConfig.STYLE_BUSINESS

        enableRecallKvSplice = try {
            prefs.getBoolean(PREF_ENABLE_RECALL_KV_SPLICE, false)
        } catch (_: Exception) {
            false
        }
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "kv_splice_switch",
            "enabled=$enableRecallKvSplice reason=init"
        )
        
        val assetsManager = ImeModelAssetsManager(service)
        val modelPath = assetsManager.ensureModelCopied()
        val embeddingModelPath = assetsManager.ensureEmbeddingModelCopied()
        
        val available = Runtime.getRuntime().availableProcessors()
        val threads = min(8, max(1, available))
        
        generationHandle = LLMBridge.createGenerationInstance(modelPath, "", threads, 999)

        embeddingHandle = try {
            LLMBridge.createEmbeddingInstance(embeddingModelPath, threads, 0)
        } catch (_: UnsatisfiedLinkError) {
            0L
        } catch (_: Exception) {
            0L
        }
        
        if (generationHandle != 0L) {
            LogUtil.i(TAG, "Init", "✅ Generation Instance 已创建，开始 Prefill 检查")

            try {
                LLMBridge.nativeSetDisableKvReuse(generationHandle, false)
                LogUtil.event(LogUtil.Category.MEM, TAG, "kv_reuse_policy", "disableKvReuse=false")
            } catch (_: Exception) {
            }

            checkAndPrefillStyles()

            try {
                if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
                LLMBridge.vectorDbInit(getVectorDbHandleOrFallback(), memoryIndexDir.absolutePath)
                ensureEmbeddingVectorDbReadyOnce()
            } catch (e: Exception) {
                LogUtil.e(TAG, "Init", "vectorDbInit failed: ${e.message}")
            }

            try {
                importPrebuiltMemoriesIfNeeded()
            } catch (e: Exception) {
                LogUtil.e(TAG, "PreMem", "import failed: ${e.message}")
            }
            try {
                val cnt = LLMBridge.vectorDbCount(getVectorDbHandleOrFallback())
                LogUtil.event(LogUtil.Category.MEM, TAG, "vector_db_ready", "count=$cnt indexDir=\"${memoryIndexDir.absolutePath}\"")
            } catch (_: Exception) {
            }
            
            val currentCacheName = StyleConfig.CACHE_FILES[currentStyleMode] ?: "kv_cache_default.bin"
            val currentCachePath = File(service.filesDir, currentCacheName).absolutePath
            val currentCacheFile = File(currentCachePath)
            
            if (currentCacheFile.exists() && currentCacheFile.length() > 0) {
                val loaded = LLMBridge.loadSession(generationHandle, currentCachePath)
                if (loaded) {
                    LogUtil.i(TAG, "Init", "✅ System Prompt 热启动成功")
                    updateReusablePrefixTokenCount()
                }
                else LogUtil.e(TAG, "Init", "❌ Cache 加载失败")
            }
        }
        
        LogUtil.i(TAG, "Init", "🚀 跳过 Warmup，直接进入就绪状态")
        
        val duration = System.currentTimeMillis() - startTime
        LogUtil.i(TAG, "Init", "🚀 初始化完成 | 总耗时: ${duration}ms")
        if (FeedbackConfig.ENABLE_INIT_SUCCESS) {
            service.showFeedback("IMEM-OS 核心初始化完成 (${duration}ms)")
        }
    }

    private fun checkAndPrefillStyles() {
        StyleConfig.PROMPTS.forEach { (styleKey, promptContent) ->
            val fileName = StyleConfig.CACHE_FILES[styleKey]
            if (fileName != null) {
                val cacheFile = File(service.filesDir, fileName)
                if (!cacheFile.exists() || cacheFile.length() <= 0L) {
                    LogUtil.i(TAG, "Prefill", "生成风格缓存: $styleKey")
                    val formatted = getFormattedSystemPrompt(promptContent)
                    LLMBridge.saveKVCacheSnapshot(generationHandle, formatted, cacheFile.absolutePath)
                }
            }
        }
    }

    fun requestCompletion(
        forceEmpty: Boolean = false,
        reason: String = "unspecified",
        explicitInput: String? = null,
    ) {
        if (!isAiCompletionEnabled || generationHandle == 0L) return

        markActivity("request_completion")

        val mySession = sessionSeq.get()

        try { idleMemoryJob?.cancel() } catch (_: Exception) {}
        idleMemoryJob = null
        scheduler.preemptForInference()
        
        val usedPrefix = if (syncedHistoryStr == "无" && syncedLastMsgStr == "无") "我" else ""
        val currentInput = when {
            explicitInput != null -> explicitInput
            forceEmpty -> usedPrefix
            else -> service.getTextBeforeCursor(100)
        }

        val contextKey = buildMemoryContextKey()
        refreshContextScopedState(contextKey)

        if (inferenceCompletedForContext) {
            if (shouldTemporarilySkipAfterCompletion(currentInput)) {
                LogUtil.event(
                    LogUtil.Category.LLM,
                    TAG,
                    "inference_completed_skip",
                    "reason=$reason ctx=$contextKey inputLen=${currentInput.length}"
                )
                if (LogUtil.rateLimit("llm.inference_complete.skip.toast", 1200L)) {
                    service.showFeedback("✅ 本轮补全已结束")
                }
                scheduler.markInferenceFinished()
                return
            } else {
                inferenceCompletedForContext = false
                inferenceCompletedPrefix = ""
                LogUtil.event(
                    LogUtil.Category.LLM,
                    TAG,
                    "inference_completed_resume",
                    "reason=$reason ctx=$contextKey inputLen=${currentInput.length}"
                )
            }
        }

        val preferCjk = preferCjkOutput(currentInput, syncedHistoryStr, syncedLastMsgStr)

        if (currentInput.isBlank() && !forceEmpty && explicitInput == null) {
            if (LogUtil.rateLimit("llm.req.skip_blank", 500)) {
                LogUtil.eventD(
                    LogUtil.Category.LLM,
                    TAG,
                    "request_skip_blank",
                    "reason=$reason forceEmpty=$forceEmpty"
                )
            }
            scheduler.markInferenceFinished()
            return
        }
        
        val systemPromptContent = StyleConfig.PROMPTS[currentStyleMode] ?: ""
        val promptMemoryOverride = if (memoryRetrievalDoneForContext) memoryInjectedForContext else null
        val promptSegments = buildChatMLPromptSegments(systemPromptContent, currentInput, memoryOverride = promptMemoryOverride)
        val finalPrompt = promptSegments.fullPrompt
        
        if (isNativeGenerating.get() && finalPrompt == lastSubmittedPrompt) return
        lastSubmittedPrompt = finalPrompt
        
        completionJob?.cancel()
        service.updateUiState(isGenerating = true)
        
        val thisRequestId = requestIdCounter.incrementAndGet()
        val promptLength = finalPrompt.length

        activeRequestId.set(thisRequestId)
        activeRequestStartedAtMs.set(System.currentTimeMillis())
        LogUtil.event(
            LogUtil.Category.LLM,
            TAG,
            "request_begin",
            "rid=$thisRequestId reason=$reason forceEmpty=$forceEmpty explicit=${explicitInput != null} usedPrefix=\"$usedPrefix\" inputLen=${currentInput.length} inputPreview=\"${previewText(currentInput)}\" mode=${currentGenerationMode.name} style=$currentStyleMode"
        )
        
        logFullPromptDebug(kind = "request", rid = thisRequestId, prompt = finalPrompt)

        completionJob = managerScope.launch {
            try {
                if (enableIncrementalPrefillPipeline) {
                    val prefillOk = prefillPromptBlocking(
                        prompt = promptSegments.prefillBase,
                        rid = thisRequestId,
                        kind = "request_base"
                    )
                    logIncrementalPrefillMetrics(
                        rid = thisRequestId,
                        kind = "request_base",
                        segments = promptSegments,
                        ok = prefillOk,
                    )
                }

                var nativeStartNano = 0L
            var nativeStartElapsedMs = 0L
            var firstTokenElapsedMs = 0L
                var toolcallRerunDone = false
                var toolcallRerunInFlight = false
                val toolcallState = AtomicReference("NONE")
                val reqStartMs = System.currentTimeMillis()
                LogUtil.eventD(
                    LogUtil.Category.LLM,
                    TAG,
                    "request_io_start",
                    "rid=$thisRequestId thread=${Thread.currentThread().name}"
                )
                val cb = object : TokenCallback {
                    override fun onTokenCandidates(tokens: Array<String>) {
                        if (sessionSeq.get() != mySession) return
                        if (firstTokenElapsedMs == 0L) {
                            firstTokenElapsedMs = android.os.SystemClock.elapsedRealtime()
                            val firstTokenDelayMs = if (nativeStartElapsedMs > 0) firstTokenElapsedMs - nativeStartElapsedMs else -1
                            LogUtil.event(
                                LogUtil.Category.LLM,
                                TAG,
                                "first_token",
                                "rid=$thisRequestId delayMs=$firstTokenDelayMs"
                            )
                        }

                        val outcome = CandidatePostProcessor.process(
                            rawCandidates = tokens,
                            ctx = CandidatePostProcessor.CandidateContext(
                                usedPrefix = usedPrefix,
                                forceEmpty = forceEmpty,
                                allowMemToolcall = !memoryRetrievalDoneForContext && !toolcallRerunDone && !toolcallRerunInFlight && toolcallState.get() == "NONE",
                                preferCjk = preferCjk,
                                instructionPrefix = currentInput,
                            )
                        )

                        when (outcome) {
                            is CandidatePostProcessor.CandidateOutcome.MemRetrieval -> {
                                if (memoryRetrievalDoneForContext) {
                                    LogUtil.eventD(
                                        LogUtil.Category.MEM,
                                        TAG,
                                        "mem_retrieval_skip_cached",
                                        "rid=$thisRequestId ctx=$contextKey"
                                    )
                                    return
                                }
                                val q = outcome.query
                                toolcallRerunInFlight = true
                                toolcallState.set("IN_FLIGHT")
                                LogUtil.event(
                                    LogUtil.Category.MEM,
                                    TAG,
                                    "mem_retrieval_requested",
                                    "rid=$thisRequestId queryLen=${q.length} queryPreview=\"${previewText(q)}\""
                                )
                                parametricExporter.logEvent(
                                    "mem_retrieval_requested",
                                    JSONObject().apply {
                                        put("query", q)
                                        put("style", currentStyleMode)
                                    }
                                )

                                managerScope.launch {
                                    handleMemRetrievalAndRerun(
                                        requestId = thisRequestId,
                                        query = q,
                                        originalPrompt = finalPrompt,
                                        nCandidates = 4,
                                        usedPrefix = usedPrefix,
                                        forceEmpty = forceEmpty,
                                        sessionGuard = mySession,
                                        memoryContextKey = contextKey,
                                        baseInput = currentInput,
                                    )
                                    toolcallRerunDone = true
                                    toolcallRerunInFlight = false
                                    toolcallState.set("DONE")
                                }
                                return
                            }

                            is CandidatePostProcessor.CandidateOutcome.Show -> {
                                val finalTokens = outcome.candidates.toTypedArray()
                                LogUtil.eventD(
                                    LogUtil.Category.LLM,
                                    TAG,
                                    "candidates_ready",
                                    "rid=$thisRequestId count=${finalTokens.size} topPreview=\"${finalTokens.firstOrNull()?.let { previewText(it) } ?: ""}\""
                                )
                                maybeMarkInferenceCompleted(
                                    contextKey = contextKey,
                                    requestId = thisRequestId,
                                    source = "request",
                                    currentInput = currentInput,
                                    candidates = finalTokens.toList(),
                                )
                                LogUtil.d(TAG, "Callback", "候选词: ${finalTokens.contentToString()}")
                                service.showAiSuggestion(finalTokens)
                            }

                            is CandidatePostProcessor.CandidateOutcome.DropAll -> {
                                LogUtil.eventD(
                                    LogUtil.Category.LLM,
                                    TAG,
                                    "candidates_dropped",
                                    "rid=$thisRequestId reason=${outcome.reason}"
                                )
                                service.showAiSuggestion(emptyArray())
                            }
                        }
                    }

                    override fun onFinished() {
                        if (sessionSeq.get() != mySession) return
                        val duration = (System.nanoTime() - nativeStartNano) / 1_000_000
                        val latency = if (firstTokenElapsedMs > 0 && nativeStartElapsedMs > 0) (firstTokenElapsedMs - nativeStartElapsedMs).toLong() else duration
                        LogUtil.event(
                            LogUtil.Category.LLM,
                            TAG,
                            "request_finished",
                            "rid=$thisRequestId totalMs=$duration firstTokenMs=$latency elapsedMs=${System.currentTimeMillis() - reqStartMs}"
                        )
                        LogUtil.i(TAG, "GenPerformance", ">>> 本轮推理结束 | 总耗时: ${duration}ms | 首字延迟: ${latency}ms <<<")
                        isNativeGenerating.set(false)
                        lastGenEndTime.set(System.currentTimeMillis())

                        if (activeRequestId.get() == thisRequestId) {
                            activeRequestId.set(0L)
                        }

                        scheduler.markInferenceFinished()
                    }

                    override fun onError(err: String) {
                        if (sessionSeq.get() != mySession) return
                        isNativeGenerating.set(false)
                        LogUtil.e(TAG, "Callback", "Error: $err")
                        LogUtil.eventE(
                            LogUtil.Category.LLM,
                            TAG,
                            "request_error",
                            "rid=$thisRequestId err=\"${previewText(err, 120)}\""
                        )
                        if (activeRequestId.get() == thisRequestId) {
                            activeRequestId.set(0L)
                        }

                        scheduler.markInferenceFinished()
                    }
                }
                
                val lockWaitStartMs = System.currentTimeMillis()
                generationLock.lock()
                try {
                    val lockWaitMs = System.currentTimeMillis() - lockWaitStartMs
                    if (lockWaitMs > 8) {
                        LogUtil.eventW(
                            LogUtil.Category.LLM,
                            TAG,
                            "lock_wait",
                            "rid=$thisRequestId waitMs=$lockWaitMs"
                        )
                    }
                    isNativeGenerating.set(false)
                    LLMBridge.stop(generationHandle)
                    if (!isNativeGenerating.compareAndSet(false, true)) return@launch
                    
                    nativeStartNano = System.nanoTime()
                    nativeStartElapsedMs = android.os.SystemClock.elapsedRealtime()
                    LogUtil.event(
                        LogUtil.Category.LLM,
                        TAG,
                        "native_generate_begin",
                        "rid=$thisRequestId mode=${currentGenerationMode.name} promptLen=$promptLength"
                    )
                    when (currentGenerationMode) {
                        GenerationMode.SENTENCE -> LLMBridge.generateSentenceCandidates(generationHandle, finalPrompt, 4, cb)
                        GenerationMode.PHRASE -> phraseCandidatesEngine.generate(generationHandle, finalPrompt, 4, cb)
                        else -> LLMBridge.generateCandidates(generationHandle, finalPrompt, 4, cb)
                    }
                } finally {
                    generationLock.unlock()
                }
            } catch (e: Exception) {
                LogUtil.e(TAG, "Gen", "异常: ${e.message}")
                LogUtil.eventE(
                    LogUtil.Category.LLM,
                    TAG,
                    "request_exception",
                    "rid=$thisRequestId ex=${e.javaClass.simpleName}:${e.message}"
                )
                if (activeRequestId.get() == thisRequestId) {
                    activeRequestId.set(0L)
                }
            }
        }
    }

    private fun handleMemRetrievalAndRerun(
        requestId: Long,
        query: String,
        originalPrompt: String,
        nCandidates: Int,
        usedPrefix: String,
        forceEmpty: Boolean,
        sessionGuard: Long,
        memoryContextKey: String,
        baseInput: String,
    ) {
        if (generationHandle == 0L) {
            LogUtil.eventW(LogUtil.Category.MEM, TAG, "mem_retrieval_abort", "rid=$requestId reason=handle_zero")
            return
        }
        val nowSession = sessionSeq.get()
        if (nowSession != sessionGuard) {
            LogUtil.eventW(
                LogUtil.Category.MEM,
                TAG,
                "mem_retrieval_abort",
                "rid=$requestId reason=session_mismatch expected=$sessionGuard actual=$nowSession"
            )
            return
        }
        if (memoryContextKey != this.memoryContextKey) {
            LogUtil.eventW(
                LogUtil.Category.MEM,
                TAG,
                "mem_retrieval_abort",
                "rid=$requestId reason=context_key_mismatch expected=${this.memoryContextKey} actual=$memoryContextKey"
            )
            return
        }

        showMemFeedback("🧠 正在检索记忆…", force = true)

        val tStartMs = System.currentTimeMillis()
        val vdbCount = try {
            if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
            val vdbHandle = getVectorDbHandleOrFallback()
            LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath)
            LLMBridge.vectorDbCount(vdbHandle)
        } catch (_: Exception) {
            -1
        }
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "mem_retrieval_begin",
            "rid=$requestId queryLen=${query.length} queryPreview=\"${previewText(query)}\" vdbCount=$vdbCount"
        )

        val vectorTopK = 20
        val vecCosThreshold = 0.4f
        val rerankVecWeight = 0.88f
        val rerankLexWeight = 0.12f

        fun detectVecScoreMode(scores: List<Float>): VecScoreMode {
            val clean = scores.filter { !it.isNaN() }
            if (clean.size < 2) return VecScoreMode.UNKNOWN

            if (clean.any { it > 1.02f || it < -1.02f }) return VecScoreMode.DISTANCE

            val first = clean.first()
            val last = clean.last()
            val eps = 0.0005f
            return when {
                last > first + eps -> VecScoreMode.DISTANCE
                last < first - eps -> VecScoreMode.COSINE
                else -> VecScoreMode.UNKNOWN
            }
        }

        fun toCosine(raw: Float, mode: VecScoreMode): Float {
            if (raw.isNaN()) return Float.NaN
            val cos = when (mode) {
                VecScoreMode.COSINE -> raw
                VecScoreMode.DISTANCE -> 1f - raw
                VecScoreMode.UNKNOWN -> raw
            }
            return cos.coerceIn(-1f, 1f)
        }

        fun extractEnvTag(fullPrompt: String, tag: String): String? {
            val open = "<$tag>\n"
            val close = "\n</$tag>"
            val i0 = fullPrompt.indexOf(open)
            if (i0 < 0) return null
            val i1 = fullPrompt.indexOf(close, startIndex = i0 + open.length)
            if (i1 < 0) return null
            return fullPrompt.substring(i0 + open.length, i1)
        }

        fun normalizeMemQuery(q: String): String {
            return q
                .replace("高层例会", "管理层会议")
                .replace("高层", "管理层")
                .replace("例会", "会议")
                .trim()
        }

        fun cjkBigrams(s: String): Set<String> {
            val clean = s.filter { it.code in 0x4E00..0x9FFF }
            if (clean.length < 2) return emptySet()
            val out = HashSet<String>(clean.length)
            for (i in 0 until clean.length - 1) out.add(clean.substring(i, i + 2))
            return out
        }

        fun wordTokens(s: String): Set<String> {
            return s
                .lowercase()
                .split(Regex("[^a-z0-9]+"))
                .map { it.trim() }
                .filter { it.length >= 2 }
                .toSet()
        }

        fun lexicalScore(queryText: String, candidateText: String): Float {
            val qCjk = cjkBigrams(queryText)
            val cCjk = cjkBigrams(candidateText)
            val cjkScore = if (qCjk.isEmpty() || cCjk.isEmpty()) 0f else {
                val inter = qCjk.intersect(cCjk).size.toFloat()
                val uni = (qCjk.size + cCjk.size - inter).coerceAtLeast(1f)
                (inter / uni).coerceIn(0f, 1f)
            }

            val qW = wordTokens(queryText)
            val cW = wordTokens(candidateText)
            val wScore = if (qW.isEmpty() || cW.isEmpty()) 0f else {
                val inter = qW.intersect(cW).size.toFloat()
                val uni = (qW.size + cW.size - inter).coerceAtLeast(1f)
                (inter / uni).coerceIn(0f, 1f)
            }

            return (0.85f * cjkScore + 0.15f * wScore).coerceIn(0f, 1f)
        }

        val lastMsgFromPrompt = extractEnvTag(originalPrompt, "last_msg")?.trim()?.takeIf { it.isNotBlank() && it != "无" }
        val includeLastMsgInEmbed = try {
            service.getSharedPreferences("yuyan_ime_prefs", android.content.Context.MODE_PRIVATE)
                .getBoolean(PREF_MEM_INCLUDE_LAST_MSG_IN_QUERY, false)
        } catch (_: Exception) {
            false
        }
        val normQuery = normalizeMemQuery(query)
        val queryEmbedText = buildString {
            append(normQuery)
            if (includeLastMsgInEmbed && !lastMsgFromPrompt.isNullOrBlank()) {
                append("\n")
                append(lastMsgFromPrompt)
            }
        }
        LogUtil.eventD(
            LogUtil.Category.MEM,
            TAG,
            "mem_retrieval_query_ctx",
            "rid=$requestId vectorTopK=$vectorTopK qNorm=\"${previewText(normQuery, 80)}\" lastMsgIncluded=$includeLastMsgInEmbed lastMsgLen=${lastMsgFromPrompt?.length ?: 0} lastMsgPreview=\"${previewText(lastMsgFromPrompt ?: "", 80)}\" embedLen=${queryEmbedText.length} embedPreview=\"${previewText(queryEmbedText, 90)}\" rerankWVec=$rerankVecWeight rerankWLex=$rerankLexWeight"
        )

        val seg = MemoryPromptSegments.splitFromFullPrompt(originalPrompt)
        if (seg == null) {
            LogUtil.eventW(LogUtil.Category.MEM, TAG, "mem_prompt_split_failed", "rid=$requestId")
        }
        val existingMemory = seg?.memoryText?.trim()?.takeIf { it.isNotBlank() && it != "无" }

        val retrieved = ArrayList<RetrievedItem>()
        val deletedLabels: Set<Long> = try {
            memoryStore.loadDeletedVectorLabels()
        } catch (_: Exception) {
            emptySet()
        }
        val lockWaitStartMs = System.currentTimeMillis()
        generationLock.lock()
        try {
            val lockWaitMs = System.currentTimeMillis() - lockWaitStartMs
            if (lockWaitMs > 8) {
                LogUtil.eventW(LogUtil.Category.MEM, TAG, "mem_lock_wait", "rid=$requestId waitMs=$lockWaitMs")
            }
            if (isNativeGenerating.compareAndSet(true, false)) {
                LLMBridge.stop(generationHandle)
            }
            try {
                if (!memoryIndexDir.exists()) memoryIndexDir.mkdirs()
                val vdbHandle = getVectorDbHandleOrFallback()
                LLMBridge.vectorDbInit(vdbHandle, memoryIndexDir.absolutePath)
            } catch (_: Exception) {
            }

            val tSearchStartMs = System.currentTimeMillis()
            val packed = try {
                LLMBridge.vectorDbSearchScored(getVectorDbHandleOrFallback(), queryEmbedText, vectorTopK)
            } catch (_: UnsatisfiedLinkError) {
                LongArray(0)
            } catch (_: Exception) {
                LongArray(0)
            }
            val labelsFallback = if (packed.isEmpty()) {
                try {
                    LLMBridge.vectorDbSearch(getVectorDbHandleOrFallback(), queryEmbedText, vectorTopK)
                } catch (_: Exception) {
                    LongArray(0)
                }
            } else LongArray(0)

            val ranked = ArrayList<Pair<Long, Float>>()
            if (packed.isNotEmpty()) {
                for (p in packed) {
                    val lab = (p ushr 32)
                    val bits = (p and 0xFFFF_FFFFL).toInt()
                    val score = Float.fromBits(bits)
                    ranked.add(lab to score)
                }
            } else {
                for (lab in labelsFallback) {
                    ranked.add(lab to Float.NaN)
                }
            }

            val scoreMode = detectVecScoreMode(ranked.map { it.second })
            if (packed.isNotEmpty() && scoreMode != VecScoreMode.COSINE) {
                LogUtil.eventW(
                    LogUtil.Category.MEM,
                    TAG,
                    "mem_vector_score_mode",
                    "rid=$requestId mode=$scoreMode note=expected_cosine"
                )
            }
            val vecCosSamples = if (packed.isNotEmpty()) {
                ranked.mapNotNull { (_, raw) ->
                    if (raw.isNaN()) null else toCosine(raw, scoreMode).takeIf { !it.isNaN() }
                }
            } else emptyList()
            val isVecDegenerate = if (vecCosSamples.size >= 3) {
                val mn = vecCosSamples.minOrNull() ?: 0f
                val mx = vecCosSamples.maxOrNull() ?: 0f
                val range = mx - mn
                range < 0.0010f
            } else false
            if (isVecDegenerate) {
                val mn = vecCosSamples.minOrNull() ?: 0f
                val mx = vecCosSamples.maxOrNull() ?: 0f
                LogUtil.eventW(
                    LogUtil.Category.MEM,
                    TAG,
                    "mem_vector_degenerate",
                    "rid=$requestId note=flat_scores action=no_mem mode=$scoreMode cosMin=${String.format("%.4f", mn)} cosMax=${String.format("%.4f", mx)} n=${vecCosSamples.size}"
                )
            }

            val rankedLog = ranked.joinToString(prefix = "[", postfix = "]") { (lab, score) ->
                if (score.isNaN()) {
                    "{id=$lab}"
                } else {
                    val cos = toCosine(score, scoreMode)
                    val dist = if (cos.isNaN()) Float.NaN else (1f - cos)
                    val sim01 = if (cos.isNaN()) 0f else ((cos + 1f) / 2f).coerceIn(0f, 1f)
                    val rawPart = "raw=${String.format("%.4f", score)}"
                    val cosPart = if (cos.isNaN()) "cos=NaN" else "cos=${String.format("%.4f", cos)}"
                    val distPart = if (dist.isNaN()) "dist=NaN" else "dist=${String.format("%.4f", dist)}"
                    "{id=$lab $rawPart mode=$scoreMode $cosPart $distPart sim01=${String.format("%.4f", sim01)}}"
                }
            }

            LogUtil.eventD(
                LogUtil.Category.MEM,
                TAG,
                "mem_vector_search",
                "rid=$requestId k=$vectorTopK results=${ranked.size} searchMs=${System.currentTimeMillis() - tSearchStartMs} scoreMode=$scoreMode ranked=$rankedLog"
            )

            if (!isVecDegenerate) for ((lab, score) in ranked) {
                if (deletedLabels.contains(lab)) continue
                try {
                    val txt = LLMBridge.vectorDbGetText(getVectorDbHandleOrFallback(), lab)
                    val clean = txt.trim()
                    if (clean.isEmpty()) continue
                    val cos = toCosine(score, scoreMode)
                    val dist = if (cos.isNaN()) Float.NaN else (1f - cos)
                    val sim01 = if (cos.isNaN()) 0f else ((cos + 1f) / 2f).coerceIn(0f, 1f)
                    retrieved.add(
                        RetrievedItem(
                            label = lab,
                            rawScore = score,
                            vecCos = if (cos.isNaN()) 0f else cos,
                            vecDist = if (dist.isNaN()) 0f else dist,
                            sim01 = sim01,
                            text = clean,
                        )
                    )
                } catch (_: Exception) {
                }
            }
        } finally {
            generationLock.unlock()
        }

        data class Reranked(
            val item: RetrievedItem,
            val lex: Float,
            val combined: Float,
        )

        val reranked = retrieved.map { it0 ->
            val lex = lexicalScore(queryEmbedText, it0.text)
            val vec = it0.sim01
            val combined = (rerankVecWeight * vec + rerankLexWeight * lex).coerceIn(0f, 1f)
            Reranked(it0, lex = lex, combined = combined)
        }.sortedWith(compareByDescending<Reranked> { it.combined }.thenByDescending { it.item.sim01 })

        if (com.yuyan.imemodule.BuildConfig.DEBUG) {
            val detailLog = reranked.take(20).joinToString(prefix = "[", postfix = "]") { rr ->
                val rawPart = if (rr.item.rawScore.isNaN()) "raw=NaN" else "raw=${String.format("%.4f", rr.item.rawScore)}"
                "{id=${rr.item.label} $rawPart cos=${String.format("%.4f", rr.item.vecCos)} dist=${String.format("%.4f", rr.item.vecDist)} sim01=${String.format("%.4f", rr.item.sim01)} lex=${String.format("%.4f", rr.lex)} comb=${String.format("%.4f", rr.combined)} len=${rr.item.text.length} preview=\"${previewText(rr.item.text, 22)}\"}"
            }
            LogUtil.eventD(
                LogUtil.Category.MEM,
                TAG,
                "mem_scores",
                "rid=$requestId topN=${minOf(20, reranked.size)} wVec=$rerankVecWeight wLex=$rerankLexWeight scores=$detailLog"
            )
        }

        val rerankLog = reranked.take(12).joinToString(prefix = "[", postfix = "]") { rr ->
            val rawPart = if (rr.item.rawScore.isNaN()) "raw=NaN" else "raw=${String.format("%.4f", rr.item.rawScore)}"
            "{id=${rr.item.label} $rawPart cos=${String.format("%.4f", rr.item.vecCos)} sim01=${String.format("%.4f", rr.item.sim01)} lex=${String.format("%.4f", rr.lex)} comb=${String.format("%.4f", rr.combined)} len=${rr.item.text.length} preview=\"${previewText(rr.item.text, 34)}\"}"
        }
        LogUtil.eventD(
            LogUtil.Category.MEM,
            TAG,
            "mem_rerank",
            "rid=$requestId candidates=${retrieved.size} topPreview=$rerankLog"
        )

        showMemFeedback(
            if (reranked.isEmpty()) "🧠 未召回到记忆" else "🧠 召回 1 条记忆",
            force = true
        )

        val selected: RetrievedItem? = reranked
            .firstOrNull { it.item.vecCos >= vecCosThreshold }?.item
        val injectedMemory = buildString {
            if (selected == null) {
                append("<NO_MEM>")
            } else {
                append("- ")
                append(selected.text.replace("\n", " ").trim())
            }
        }.trim()

        markMemoryRetrievedForContext(memoryContextKey, injectedMemory)

        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "mem_selected",
            if (selected == null) {
                "rid=$requestId selected=none reason=below_threshold_or_empty vecCosThreshold=$vecCosThreshold existingMemLen=${existingMemory?.length ?: 0} injectedLen=${injectedMemory.length}"
            } else {
                val rawPart = if (selected.rawScore.isNaN()) "raw=NaN" else "raw=${String.format("%.4f", selected.rawScore)}"
                "rid=$requestId selectedId=${selected.label} $rawPart cos=${String.format("%.4f", selected.vecCos)} dist=${String.format("%.4f", selected.vecDist)} sim01=${String.format("%.4f", selected.sim01)} vecCosThreshold=$vecCosThreshold existingMemLen=${existingMemory?.length ?: 0} injectedLen=${injectedMemory.length} injectedPreview=\"${previewText(injectedMemory, 90)}\""
            }
        )

        parametricExporter.logEvent(
            "mem_retrieval_injected",
            JSONObject().apply {
                put("query", query)
                put("hit_count", if (selected == null) 0 else 1)
            }
        )

        val retrievedLog = reranked.take(12).joinToString(prefix = "[", postfix = "]") { rr ->
            val rawPart = if (rr.item.rawScore.isNaN()) "raw=NaN" else "raw=${String.format("%.4f", rr.item.rawScore)}"
            "{id=${rr.item.label} $rawPart sim01=${String.format("%.4f", rr.item.sim01)} lex=${String.format("%.4f", rr.lex)} comb=${String.format("%.4f", rr.combined)} len=${rr.item.text.length} preview=\"${previewText(rr.item.text, 28)}\"}"
        }
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "mem_retrieval_done",
            "rid=$requestId hitCount=${if (selected == null) 0 else 1} elapsedMs=${System.currentTimeMillis() - tStartMs} injectedLen=${injectedMemory.length} retrievedTop=$retrievedLog"
        )

        val rerunPrompt = if (seg == null) {
            originalPrompt
        } else {
            seg.prefixBeforeMemory + injectedMemory + seg.suffixAfterMemory
        }
        val rerunSegments = if (seg == null) {
            null
        } else {
            ChatMLPromptSegments(
                prefillBase = seg.prefixBeforeMemory + injectedMemory,
                decodeTail = seg.suffixAfterMemory,
            )
        }

        logFullPromptDebug(kind = "rerun", rid = requestId, prompt = rerunPrompt)

        val cb = object : TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                if (sessionSeq.get() != sessionGuard) return
                if (com.yuyan.imemodule.BuildConfig.DEBUG) {
                    val metric = tokens.firstOrNull { it.startsWith("__METRICS__") }
                    if (metric != null) {
                        LogUtil.eventD(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_metrics",
                            "rid=$requestId ${metric.take(900)}"
                        )
                    }
                }

                val preferCjk = preferCjkOutput(rerunPrompt, usedPrefix, query)

                val outcome = CandidatePostProcessor.process(
                    rawCandidates = tokens,
                    ctx = CandidatePostProcessor.CandidateContext(
                        usedPrefix = usedPrefix,
                        forceEmpty = forceEmpty,
                        allowMemToolcall = false,
                        preferCjk = preferCjk,
                        instructionPrefix = baseInput,
                    )
                )

                when (outcome) {
                    is CandidatePostProcessor.CandidateOutcome.Show -> {
                        val arr = outcome.candidates.toTypedArray()
                        service.showAiSuggestion(arr)
                        maybeMarkInferenceCompleted(
                            contextKey = memoryContextKey,
                            requestId = requestId,
                            source = "rerun",
                            currentInput = baseInput,
                            candidates = arr.toList(),
                        )
                        LogUtil.eventD(
                            LogUtil.Category.LLM,
                            TAG,
                            "rerun_candidates_ready",
                            "rid=$requestId count=${arr.size}"
                        )
                    }

                    else -> {
                        service.showAiSuggestion(emptyArray())
                        LogUtil.eventD(
                            LogUtil.Category.LLM,
                            TAG,
                            "rerun_candidates_dropped",
                            "rid=$requestId"
                        )
                    }
                }
            }

            override fun onFinished() {
                if (sessionSeq.get() != sessionGuard) return
                isNativeGenerating.set(false)
                scheduler.markInferenceFinished()
                LogUtil.event(
                    LogUtil.Category.LLM,
                    TAG,
                    "rerun_finished",
                    "rid=$requestId elapsedMs=${System.currentTimeMillis() - tStartMs}"
                )
            }

            override fun onError(err: String) {
                if (sessionSeq.get() != sessionGuard) return
                isNativeGenerating.set(false)
                LogUtil.e(TAG, "ToolcallRerun", "Error: $err")
                LogUtil.eventE(
                    LogUtil.Category.LLM,
                    TAG,
                    "rerun_error",
                    "rid=$requestId err=\"${previewText(err, 120)}\""
                )

                scheduler.markInferenceFinished()
            }
        }

        generationLock.lock()
        try {
            isNativeGenerating.set(false)
            LLMBridge.stop(generationHandle)
            if (!isNativeGenerating.compareAndSet(false, true)) return
            LogUtil.event(
                LogUtil.Category.LLM,
                TAG,
                "rerun_begin",
                "rid=$requestId promptLen=${rerunPrompt.length} candidates=$nCandidates"
            )

            val usedSpliceFromFile = if (enableRecallKvSplice && seg != null && selected != null && selected.label > 0) {
                val l1Dir = L1KvBlobPaths.resolveDir(service.filesDir)
                val kvFile = L1KvBlobPaths.resolveFile(l1Dir, selected.label, currentStyleMode)
                val kvReady = kvFile.exists() && kvFile.length() > 0

                val ensuredReady = if (kvReady) {
                    true
                } else {
                    if (BuildConfig.DEBUG) {
                        LogUtil.eventD(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_miss",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode exists=${kvFile.exists()} size=${if (kvFile.exists()) kvFile.length() else -1}"
                        )
                    }

                    try {
                        if (!l1Dir.exists()) l1Dir.mkdirs()
                        val t0 = System.currentTimeMillis()
                        LogUtil.eventD(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_repair_begin",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode out=${kvFile.name} memLen=${injectedMemory.length}"
                        )
                        try { LLMBridge.clearKvKeepSystem(generationHandle) } catch (_: Exception) {}
                        val ok = try {
                            LLMBridge.buildMemoryKvBlob(generationHandle, injectedMemory, kvFile.absolutePath)
                        } catch (_: Throwable) {
                            false
                        }
                        try { LLMBridge.clearKvKeepSystem(generationHandle) } catch (_: Exception) {}
                        val elapsed = max(0L, System.currentTimeMillis() - t0)
                        val size = if (kvFile.exists()) kvFile.length() else -1L
                        LogUtil.eventD(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_repair_end",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode ok=$ok size=$size elapsedMs=$elapsed"
                        )
                    } catch (e: Exception) {
                        LogUtil.eventW(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_repair_error",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode ex=${e.javaClass.simpleName}:${e.message}"
                        )
                    }

                    kvFile.exists() && kvFile.length() > 0
                }

                if (ensuredReady) {
                    LogUtil.eventD(
                        LogUtil.Category.MEM,
                        TAG,
                        "rerun_kv_splice_from_file",
                        "rid=$requestId label=${selected.label} style=$currentStyleMode kvSize=${kvFile.length()}"
                    )
                    val rc = try {
                        LLMBridge.generatePhraseCandidatesSpliceMemoryFromKvFile(
                            generationHandle,
                            seg.prefixBeforeMemory,
                            injectedMemory,
                            seg.suffixAfterMemory,
                            kvFile.absolutePath,
                            nCandidates,
                            cb,
                        )
                    } catch (_: UnsatisfiedLinkError) {
                        LogUtil.eventW(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_unavailable",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode"
                        )
                        -999
                    } catch (e: Exception) {
                        LogUtil.eventW(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_exception",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode ex=${e.javaClass.simpleName}:${e.message}"
                        )
                        -998
                    }
                    if (rc != 0) {
                        LogUtil.eventW(
                            LogUtil.Category.MEM,
                            TAG,
                            "rerun_kv_splice_failed",
                            "rid=$requestId label=${selected.label} style=$currentStyleMode rc=$rc"
                        )
                    }
                    rc == 0
                } else {
                    false
                }
            } else {
                if (com.yuyan.imemodule.BuildConfig.DEBUG) {
                    LogUtil.eventD(
                        LogUtil.Category.MEM,
                        TAG,
                        "rerun_kv_splice_skip",
                        "rid=$requestId enabled=$enableRecallKvSplice hasSeg=${seg != null} hasHit=${selected != null} label=${selected?.label ?: -1}"
                    )
                }
                false
            }

            if (!usedSpliceFromFile) {
                if (enableIncrementalPrefillPipeline && rerunSegments != null) {
                    val done = CompletableDeferred<Boolean>()
                    val prefillCb = object : TokenCallback {
                        override fun onTokenCandidates(tokens: Array<String>) {
                        }

                        override fun onFinished() {
                            if (!done.isCompleted) done.complete(true)
                        }

                        override fun onError(err: String) {
                            LogUtil.eventW(
                                LogUtil.Category.MEM,
                                TAG,
                                "prefill_error",
                                "rid=$requestId kind=rerun_base err=\"${previewText(err, 120)}\""
                            )
                            if (!done.isCompleted) done.complete(false)
                        }
                    }

                    var rerunPrefillOk = false
                    try {
                        LLMBridge.prefillPrompt(generationHandle, rerunSegments.prefillBase, prefillCb)
                        rerunPrefillOk = runBlocking { withTimeoutOrNull(12_000L) { done.await() } == true }
                    } catch (_: Exception) {
                        rerunPrefillOk = false
                    }

                    logIncrementalPrefillMetrics(
                        rid = requestId,
                        kind = "rerun_base",
                        segments = rerunSegments,
                        ok = rerunPrefillOk,
                    )

                    if (!rerunPrefillOk) {
                        LogUtil.eventW(
                            LogUtil.Category.MEM,
                            TAG,
                            "incremental_prefill_fallback",
                            "rid=$requestId kind=rerun_base fallback=full_prompt"
                        )
                    }
                }
                phraseCandidatesEngine.generate(generationHandle, rerunPrompt, nCandidates, cb)
            }
        } finally {
            generationLock.unlock()
        }

        LogUtil.event(LogUtil.Category.MEM, TAG, "mem_retrieval_reclaimed", "rid=$requestId")
    }

    private fun tryStartIdleMemoryWork() {
        if (generationHandle == 0L) return
        if (scheduler.state() != ModelMutexScheduler.State.Idle) return

        CoroutineScope(Dispatchers.IO).launch {
            try {
                if (LogUtil.rateLimit("mem.idle.start", 1200)) {
                    LogUtil.eventD(
                        LogUtil.Category.MEM,
                        TAG,
                        "idle_memory_begin",
                        "state=${scheduler.state().name}"
                    )
                }
                showMemFeedback("🧠 正在整理记忆…", minIntervalMs = 1500L)
                val result = idleMemoryWorker.tryProcessBatch(
                    handle = generationHandle,
                    vectorDbHandle = getVectorDbHandleOrFallback(),
                    maxLines = 8,
                    enableL1KvPrecompute = enableRecallKvSplice,
                    stopNativeIfGenerating = {
                        generationLock.lock()
                        try {
                            isNativeGenerating.set(false)
                            LLMBridge.stop(generationHandle)
                        } finally {
                            generationLock.unlock()
                        }
                    },
                    generateMemoryWorker = { prompt, maxTokens, callback ->
                        generationLock.lock()
                        try {
                            isNativeGenerating.set(false)
                            LLMBridge.stop(generationHandle)
                            LLMBridge.generateMemoryWorker(generationHandle, prompt, maxTokens, callback)
                        } finally {
                            generationLock.unlock()
                        }
                    },
                )

                if (result.processed > 0) {
                    LogUtil.i(TAG, "MemoryWorker", "processed=${result.processed}, indexed=${result.indexed}")
                    LogUtil.event(
                        LogUtil.Category.MEM,
                        TAG,
                        "idle_memory_done",
                        "processed=${result.processed} indexed=${result.indexed}"
                    )
                    if (result.indexed > 0) {
                        showMemFeedback("🧠 已记录 ${result.indexed} 条记忆", minIntervalMs = 1200L)
                    }
                }
            } catch (e: Exception) {
                LogUtil.e(TAG, "MemoryWorker", "exception: ${e.message}")
                LogUtil.eventE(
                    LogUtil.Category.MEM,
                    TAG,
                    "idle_memory_exception",
                    "ex=${e.javaClass.simpleName}:${e.message}"
                )
            }
        }
    }

    fun runIdleMemoryWorkOnWindowHidden(minIntervalMs: Long = 120_000L) {
        if (generationHandle == 0L) {
            if (BuildConfig.DEBUG) {
                LogUtil.eventD(LogUtil.Category.MEM, TAG, "idle_memory_skip", "reason=handle_0")
            }
            return
        }
        markActivity("window_hidden")
        val now = System.currentTimeMillis()
        val genDelta = now - lastGenEndTime.get()
        val minGapMs = 800L
        val delayMs = (minGapMs - genDelta).coerceAtLeast(0L)

        idleMemoryJob?.cancel()
        idleMemoryJob = managerScope.launch {
            if (delayMs > 0L) {
                if (BuildConfig.DEBUG) {
                    LogUtil.eventD(
                        LogUtil.Category.MEM,
                        TAG,
                        "idle_memory_delay",
                        "reason=recent_gen_end delayMs=$delayMs genDeltaMs=$genDelta"
                    )
                    memToastOnce("mem.idle.delay", "🧠 稍后整理记忆（${delayMs}ms）")
                }
                delay(delayMs)
            }

            val now2 = System.currentTimeMillis()
            val lastRun = lastIdleMemoryRunAtMs.get()
            val idleDelta = now2 - lastRun
            if (idleDelta < minIntervalMs) {
                if (BuildConfig.DEBUG) {
                    LogUtil.eventD(
                        LogUtil.Category.MEM,
                        TAG,
                        "idle_memory_skip",
                        "reason=throttled deltaMs=$idleDelta minIntervalMs=$minIntervalMs"
                    )
                    memToastOnce("mem.idle.skip.throttle", "🧠 跳过整理：节流中 (${idleDelta}ms)")
                }
                return@launch
            }

            try {
                LogUtil.event(LogUtil.Category.MEM, TAG, "idle_memory_begin", "reason=window_hidden")
                memToastOnce("mem.idle.begin", "🧠 正在后台整理记忆…")
                clearKvKeepSystem(reason = "memory_worker_begin")
                val result = idleMemoryWorker.tryProcessBatch(
                    handle = generationHandle,
                    vectorDbHandle = getVectorDbHandleOrFallback(),
                    maxLines = 8,
                    enableL1KvPrecompute = enableRecallKvSplice,
                    stopNativeIfGenerating = {
                        generationLock.lock()
                        try {
                            isNativeGenerating.set(false)
                            com.yuyan.imemodule.llm.LLMBridge.stop(generationHandle)
                        } finally {
                            generationLock.unlock()
                        }
                    },
                    generateMemoryWorker = { prompt, maxTokens, callback ->
                        generationLock.lock()
                        try {
                            isNativeGenerating.set(false)
                            com.yuyan.imemodule.llm.LLMBridge.stop(generationHandle)
                            com.yuyan.imemodule.llm.LLMBridge.generateMemoryWorker(generationHandle, prompt, maxTokens, callback)
                        } finally {
                            generationLock.unlock()
                        }
                    }
                )

                lastIdleMemoryRunAtMs.set(System.currentTimeMillis())

                LogUtil.event(
                    LogUtil.Category.MEM,
                    TAG,
                    "idle_memory_done",
                    "processed=${result.processed} indexed=${result.indexed}"
                )

                memToastOnce(
                    "mem.idle.done",
                    if (result.indexed > 0) "🧠 后台整理完成：新增 ${result.indexed} 条" else "🧠 后台整理完成：无新增"
                )
            } finally {
                clearKvKeepSystem(reason = "memory_worker_end")
                LogUtil.event(LogUtil.Category.MEM, TAG, "idle_memory_end", "reason=window_hidden")
            }
        }
    }

    fun preemptIdleMemoryProcessing(reason: String = "window_shown") {
        try { idleMemoryJob?.cancel() } catch (_: Exception) {}
        idleMemoryJob = null
        scheduler.preemptForInference()
        scheduler.markInferenceFinished()
        LogUtil.event(LogUtil.Category.MEM, TAG, "idle_memory_preempt", "reason=$reason")
        memToastOnce("mem.idle.preempt", "🧠 已中断后台整理，恢复推理模式")
    }
    
    fun stopCompletion(reason: String = "unspecified") {
        markActivity("stop_completion")
        val rid = activeRequestId.get()
        LogUtil.event(
            LogUtil.Category.LLM,
            TAG,
            "stop_requested",
            "rid=$rid reason=$reason nativeGenerating=${isNativeGenerating.get()} handle=${generationHandle}"
        )
        completionJob?.cancel()
        completionJob = null
        if (generationHandle != 0L) {
            managerScope.launch {
                generationLock.lock()
                try {
                    if (isNativeGenerating.compareAndSet(true, false)) {
                        LLMBridge.stop(generationHandle)
                        lastGenEndTime.set(System.currentTimeMillis())
                    }
                } finally {
                    generationLock.unlock()
                }
            }
        }
    }

    fun triggerDayModeSafe(swapPath: String): Boolean {
        if (generationHandle == 0L) return false
        generationLock.lock()
        return try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                LLMBridge.stop(generationHandle)
            }
            LLMBridge.triggerDayMode(generationHandle, swapPath)
        } finally {
            generationLock.unlock()
        }
    }

    fun triggerNightModeSafe(swapPath: String): Boolean {
        if (generationHandle == 0L) return false
        generationLock.lock()
        return try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                LLMBridge.stop(generationHandle)
            }
            LLMBridge.triggerNightMode(generationHandle, swapPath)
        } finally {
            generationLock.unlock()
        }
    }
    fun executeStyleSwitch(nextStyle: String) {
        val startTime = System.currentTimeMillis()
        val prefs = service.getSharedPreferences("yuyan_ime_prefs", android.content.Context.MODE_PRIVATE)
        prefs.edit().putString("persisted_style_mode", nextStyle).apply()
        currentStyleMode = nextStyle
        
        val cacheFileName = StyleConfig.CACHE_FILES[nextStyle] ?: "kv_cache_default.bin"
        val cachePath = File(service.filesDir, cacheFileName).absolutePath
        val cacheFile = File(cachePath)
        val targetPrompt = StyleConfig.PROMPTS[nextStyle] ?: ""
        
        managerScope.launch {
            if (generationHandle == 0L) return@launch
            var loaded = false
            generationLock.lock()
            try {
                if (isNativeGenerating.compareAndSet(true, false)) {
                    LLMBridge.stop(generationHandle)
                }
                val cacheExisted = cacheFile.exists()
                if (cacheExisted) {
                    loaded = LLMBridge.loadSession(generationHandle, cachePath)
                }
                if (!loaded && !cacheExisted) {
                    val formatted = getFormattedSystemPrompt(targetPrompt)
                    if (LLMBridge.saveKVCacheSnapshot(generationHandle, formatted, cachePath)) {
                        loaded = LLMBridge.loadSession(generationHandle, cachePath)
                    }
                }
                if (loaded) {
                    updateReusablePrefixTokenCount()
                }
            } finally {
                generationLock.unlock()
            }
            val duration = System.currentTimeMillis() - startTime
            LogUtil.i(TAG, "StyleSwitch", "风格切换完成: $nextStyle | 耗时: ${duration}ms")
            service.showFeedback(if (loaded) "✅ 风格已切换" else "❌ 切换失败")
        }
    }

    fun initializeStyleByContactName(rawName: String?, reason: String = "context_sync") : String? {
        val name = rawName?.trim() ?: return null
        if (name.isBlank()) return null

        val inferred = when {
            containsAnyKeyword(name, listOf("老板", "领导", "总监", "经理", "总", "boss", "ceo")) -> StyleConfig.STYLE_BUSINESS
            containsAnyKeyword(name, listOf("妹妹", "小", "弟")) -> StyleConfig.STYLE_WARM
            containsAnyKeyword(name, listOf("死党", "兄弟", "哥们", "闺蜜", "buddy", "bro", "朋友")) -> StyleConfig.STYLE_INRERNET
            else -> null
        }

        if (inferred == null) {
            LogUtil.eventD(
                LogUtil.Category.CORE,
                TAG,
                "style_init_by_contact_skip",
                "reason=$reason name=${previewText(name, 24)}"
            )
            return null
        }

        if (inferred != currentStyleMode) {
            currentStyleMode = inferred
            sessionSeq.incrementAndGet()
            LogUtil.event(
                LogUtil.Category.CORE,
                TAG,
                "style_init_by_contact",
                "reason=$reason name=${previewText(name, 24)} style=$inferred"
            )
        } else {
            LogUtil.eventD(
                LogUtil.Category.CORE,
                TAG,
                "style_init_by_contact_nochange",
                "reason=$reason name=${previewText(name, 24)} style=$inferred"
            )
        }
        return inferred
    }

    fun initializeStyleByConversationHint(rawHint: String?, reason: String = "context_sync_hint"): String? {
        val hint = rawHint?.trim() ?: return null
        if (hint.isBlank()) return null

        val inferred = when {
            containsAnyKeyword(hint, listOf("老板", "领导", "总监", "经理", "预算", "项目", "上线", "kpi", "汇报", "会议")) -> StyleConfig.STYLE_BUSINESS
            containsAnyKeyword(hint, listOf("死党", "兄弟", "哥们", "闺蜜", "网吧", "solo", "打野", "偷袭", "开黑", "buddy", "bro")) -> StyleConfig.STYLE_INRERNET
            containsAnyKeyword(hint, listOf("妹妹", "小妹", "表妹", "妈妈", "安慰", "抱抱", "别难过", "姐姐")) -> StyleConfig.STYLE_WARM
            else -> null
        }

        if (inferred == null) {
            LogUtil.eventD(
                LogUtil.Category.CORE,
                TAG,
                "style_init_by_hint_skip",
                "reason=$reason hintLen=${hint.length}"
            )
            return null
        }

        if (inferred != currentStyleMode) {
            currentStyleMode = inferred
            sessionSeq.incrementAndGet()
            LogUtil.event(
                LogUtil.Category.CORE,
                TAG,
                "style_init_by_hint",
                "reason=$reason hintLen=${hint.length} style=$inferred"
            )
        } else {
            LogUtil.eventD(
                LogUtil.Category.CORE,
                TAG,
                "style_init_by_hint_nochange",
                "reason=$reason hintLen=${hint.length} style=$inferred"
            )
        }
        return inferred
    }

    private fun containsAnyKeyword(text: String, keywords: List<String>): Boolean {
        val lower = text.lowercase()
        return keywords.any { kw -> lower.contains(kw.lowercase()) }
    }
    
    fun triggerSilentPrefill() {
        if (generationHandle == 0L || !isAiCompletionEnabled) return
        markActivity("silent_prefill")
        val startTime = System.currentTimeMillis()
        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "silent_prefill_begin",
            "style=$currentStyleMode"
        )
        managerScope.launch {
            try {
                if (isPersistentInferenceMode) {
                    resetToStyleBaseline(reason = "silent_prefill")
                }
                val prefixFromEditor = service.getAllTextBeforeCursor()
                val fallbackPrefix = if (syncedHistoryStr == "无" && syncedLastMsgStr == "无") "我" else ""
                val prefix = if (prefixFromEditor.isNotBlank()) prefixFromEditor else fallbackPrefix
                val systemPrompt = StyleConfig.PROMPTS[currentStyleMode] ?: ""
                val prefillSegments = buildChatMLPromptSegments(systemPrompt, prefix)
                val prefillPrompt = prefillSegments.prefillBase

                logFullPromptDebug(kind = "silent_prefill", rid = -1L, prompt = prefillPrompt)

                updateReusablePrefixTokenCount()

                val ok = prefillPromptBlocking(
                    prompt = prefillPrompt,
                    rid = -1L,
                    kind = "silent_base"
                )
                val duration = System.currentTimeMillis() - startTime
                if (ok) {
                    service.showFeedback("⚡️ 聊天环境已同步")
                    LogUtil.i(TAG, "Sync", "环境同步成功 | 耗时: ${duration}ms")
                    LogUtil.event(
                        LogUtil.Category.MEM,
                        TAG,
                        "silent_prefill_ok",
                        "elapsedMs=$duration prefillBaseLen=${prefillPrompt.length} tailLen=${prefillSegments.decodeTail.length} prefixLen=${prefix.length} promptPreview=\"${previewText(prefillPrompt)}\""
                    )
                    requestCompletion(forceEmpty = false, reason = "silent_prefill", explicitInput = prefix)
                } else {
                    LogUtil.e(TAG, "Sync", "环境同步失败/超时 | 耗时: ${duration}ms")
                    LogUtil.eventW(LogUtil.Category.MEM, TAG, "silent_prefill_timeout", "elapsedMs=$duration")
                    generationLock.lock()
                    try {
                        if (isNativeGenerating.compareAndSet(true, false)) {
                            LLMBridge.stop(generationHandle)
                        }
                    } finally {
                        generationLock.unlock()
                    }
                }
            } catch (e: Exception) {
                LogUtil.e(TAG, "Prefill", "异常: ${e.message}")
                LogUtil.eventE(
                    LogUtil.Category.MEM,
                    TAG,
                    "silent_prefill_exception",
                    "ex=${e.javaClass.simpleName}:${e.message}"
                )
            }
        }
    }

    fun loadSessionSafe(path: String): Boolean {
        if (generationHandle == 0L) return false
        generationLock.lock()
        return try {
            if (isNativeGenerating.compareAndSet(true, false)) {
                LLMBridge.stop(generationHandle)
            }
            LLMBridge.loadSession(generationHandle, path)
        } finally {
            generationLock.unlock()
        }
    }

    private fun getFormattedSystemPrompt(content: String): String {
        return "<|im_start|>system\n$content<|im_end|>\n"
    }

    private data class ChatMLPromptSegments(
        val prefillBase: String,
        val decodeTail: String,
    ) {
        val fullPrompt: String
            get() = prefillBase + decodeTail
    }
    private fun estimateTokenCount(text: String): Int {
        if (generationHandle == 0L || text.isBlank()) return 0
        return try {
            LLMBridge.tokenize(generationHandle, text)?.size ?: 0
        } catch (_: Exception) {
            0
        }
    }

    private fun logIncrementalPrefillMetrics(
        rid: Long,
        kind: String,
        segments: ChatMLPromptSegments,
        ok: Boolean,
    ) {
        val baseLen = segments.prefillBase.length
        val tailLen = segments.decodeTail.length
        val fullLen = segments.fullPrompt.length
        val includeTokenMetrics = BuildConfig.DEBUG
        val baseTok = if (includeTokenMetrics) estimateTokenCount(segments.prefillBase) else -1
        val tailTok = if (includeTokenMetrics) estimateTokenCount(segments.decodeTail) else -1
        val fullTok = if (includeTokenMetrics) estimateTokenCount(segments.fullPrompt) else -1

        LogUtil.event(
            LogUtil.Category.MEM,
            TAG,
            "incremental_prefill",
            "rid=$rid kind=$kind ok=$ok prefillBaseLen=$baseLen tailLen=$tailLen fullLen=$fullLen prefillBaseTok=$baseTok tailTok=$tailTok fullTok=$fullTok"
        )
    }

    private fun getReusablePrefixText(systemPrompt: String): String {
        val systemPart = getFormattedSystemPrompt(systemPrompt)
        return "$systemPart<|im_start|>user\n"
    }

    fun updateReusablePrefixTokenCount() {
        if (generationHandle == 0L) return
        val systemPromptContent = StyleConfig.PROMPTS[currentStyleMode] ?: ""
        val reusablePrefix = getReusablePrefixText(systemPromptContent)

        val cached = reusablePrefixTokenCountCache[currentStyleMode]
        val tokenCount = if (cached != null && cached > 0) {
            cached
        } else {
            val tokens = LLMBridge.tokenize(generationHandle, reusablePrefix)
            val computed = tokens?.size ?: 0
            if (computed > 0) reusablePrefixTokenCountCache[currentStyleMode] = computed
            computed
        }

        if (tokenCount > 0) {
            LLMBridge.setReusablePrefixTokenCount(generationHandle, tokenCount)
        }
    }

    private fun buildUserContentTemplate(history: String, lastMsg: String, memory: String?, prefix: String): String {
        val base = buildUserContentTemplateBase(history, lastMsg, memory)
        val tail = buildUserContentTemplateTail(prefix)
        return base + tail
    }

    private fun buildUserContentTemplateBase(history: String, lastMsg: String, memory: String?): String {
        return buildString {
            append("<env>\n")
            append("<history>\n")
            append(history)
            append("\n</history>\n")
            append("<last_msg>\n")
            append(lastMsg)
            append("\n</last_msg>\n")
            append("<memory>\n")
            append(if (memory.isNullOrEmpty()) "无" else memory)
            append("\n</memory></env>\n")
            append("<instruction>\n")
            append("请根据记忆和LastMsg，补全我的回复：\n")
        }
    }

    private fun buildUserContentTemplateTail(prefix: String): String {
        return buildString {
            append(prefix)
            append("\n</instruction>")
        }
    }

    private fun buildChatMLPromptSegments(systemPrompt: String, prefix: String, memoryOverride: String? = null): ChatMLPromptSegments {
        val historyStr = if (syncedHistoryStr.isNotBlank()) syncedHistoryStr else "无"
        val lastMsgContent = if (syncedLastMsgStr != "无" && syncedLastMsgStr.isNotBlank()) "[对方]: $syncedLastMsgStr" else "无"
        val memoryStr = memoryOverride ?: if (syncedMemoryStr.isNotBlank()) syncedMemoryStr else "无"

        val reusablePrefix = getReusablePrefixText(systemPrompt)
        val userBase = buildUserContentTemplateBase(
            history = historyStr,
            lastMsg = lastMsgContent,
            memory = memoryStr
        )
        val userTail = buildUserContentTemplateTail(prefix)

        val assistantStart = if (prefix.isNotBlank()) {
            "<think>\n\n</think>\n\n$prefix"
        } else {
            "<think>\n\n</think>\n\n"
        }
        return ChatMLPromptSegments(
            prefillBase = reusablePrefix + userBase,
            decodeTail = "$userTail<|im_end|>\n<|im_start|>assistant\n$assistantStart"
        )
    }

    private fun buildChatMLPrompt(systemPrompt: String, prefix: String): String {
        return buildChatMLPromptSegments(systemPrompt, prefix).fullPrompt
    }

    private suspend fun prefillPromptBlocking(
        prompt: String,
        rid: Long,
        kind: String,
        timeoutMs: Long = 12_000L,
    ): Boolean {
        if (generationHandle == 0L) return false
        if (prompt.isBlank()) return true

        val done = CompletableDeferred<Boolean>()
        val cb = object : TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
            }

            override fun onFinished() {
                isNativeGenerating.set(false)
                if (!done.isCompleted) done.complete(true)
            }

            override fun onError(err: String) {
                LogUtil.eventW(
                    LogUtil.Category.MEM,
                    TAG,
                    "prefill_error",
                    "rid=$rid kind=$kind err=\"${previewText(err, 120)}\""
                )
                isNativeGenerating.set(false)
                if (!done.isCompleted) done.complete(false)
            }
        }

        val lockWaitStartMs = System.currentTimeMillis()
        generationLock.lock()
        try {
            val lockWaitMs = System.currentTimeMillis() - lockWaitStartMs
            if (lockWaitMs > 8) {
                LogUtil.eventW(LogUtil.Category.MEM, TAG, "prefill_lock_wait", "rid=$rid kind=$kind waitMs=$lockWaitMs")
            }
            if (isNativeGenerating.compareAndSet(true, false)) {
                LLMBridge.stop(generationHandle)
            }
            if (!isNativeGenerating.compareAndSet(false, true)) return false
            LLMBridge.prefillPrompt(generationHandle, prompt, cb)
        } finally {
            generationLock.unlock()
        }

        val ok = withTimeoutOrNull(timeoutMs) { done.await() } == true
        if (!ok) {
            generationLock.lock()
            try {
                if (isNativeGenerating.compareAndSet(true, false)) {
                    LLMBridge.stop(generationHandle)
                }
            } finally {
                generationLock.unlock()
            }
        }
        return ok
    }

    fun release() {
        try { idleClearJob?.cancel() } catch (_: Exception) {}
        try { managerJob.cancel() } catch (_: Exception) {}
        if (embeddingHandle != 0L) {
            try {
                LLMBridge.vectorDbClose(embeddingHandle)
            } catch (_: Exception) {
            }
            try {
                LLMBridge.freeModel(embeddingHandle)
            } catch (_: Exception) {
            }
            embeddingHandle = 0L
        }
        if (generationHandle != 0L) {
            LLMBridge.freeModel(generationHandle)
            generationHandle = 0L
        }
    }

    fun clearAllMemoryForTest(): Boolean {
        return try {
            val res = MemoryTestReset.clearAllForTest(service, handle = getVectorDbHandleOrFallback())
            res.ok
        } catch (_: Exception) {
            false
        }
    }
}