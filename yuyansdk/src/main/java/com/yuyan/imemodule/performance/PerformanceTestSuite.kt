package com.yuyan.imemodule.performance

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.Process
import com.yuyan.imemodule.llm.LLMBridge
import com.yuyan.imemodule.llm.MemoryPromptSegments
import com.yuyan.imemodule.llm.postprocess.CandidatePostProcessor
import com.yuyan.imemodule.service.data.StyleConfig
import com.yuyan.imemodule.service.manager.ImeModelAssetsManager
import com.yuyan.imemodule.utils.LogUtil
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.LinkedHashMap
import java.util.LinkedHashSet
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class PerformanceTestSuite(private val context: Context) {
    private val tag = "PerformanceTestSuite"
    internal val javaTracker = PerformanceTracker(context)

    private val running = AtomicBoolean(false)
    private val currentTest = AtomicReference("")
    private val flowSeq = AtomicLong(0)
    private val lifecycleToleranceMs = 50L
    private val flowTimeoutMs = 120_000L

    data class TestResult(
        val testName: String,
        val success: Boolean,
        val iterations: Int,
        val totalMetrics: Int,
        val durationMs: Long,
        val statistics: TestStatistics,
        val exportPath: String? = null,
        val warning: String? = null,
        val error: String? = null
    )

    data class TestStatistics(
        val avgTimeMs: Double,
        val minTimeMs: Long,
        val maxTimeMs: Long,
        val p50: Long,
        val p90: Long,
        val p95: Long,
        val p99: Long,
        val throughput: Double
    )

    data class TestStatus(
        val isRunning: Boolean,
        val currentTest: String,
        val statistics: PerformanceTracker.Statistics,
        val nativeSessionCount: Int
    )

    private data class SuiteConfig(
        val name: String,
        val styles: List<String>,
        val promptBuckets: List<Int>,
        val decodeBuckets: List<Int>,
        val decodeRepeats: Int,
        val historyBuckets: List<Int>,
        val includePrefill: Boolean = true,
        val includeDecode: Boolean = true,
        val includeImeFirst: Boolean = true,
        val repeatFactor: Int = 1,
        val contextRepeatFactor: Int = repeatFactor,
        val warmupRounds: Int = 0,
        val decodeControlMode: DecodeControlMode = DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS,
        val decodePromptChars: Int = 256,
        val decodeNCandidates: Int = 4,
        val decodeInvalidTokenRatio: Double = 0.60,
        val decodeInvalidMinBudget: Int = 12
    )

    enum class DecodeControlMode(val wireValue: String) {
        BENCHMARK_DECODE_MAX_STEPS("benchmark_decode_max_steps"),
        RUNTIME_GENERATE_CANDIDATES("runtime_generate_candidates"),
        RUNTIME_GENERATE_PHRASE_CANDIDATES("runtime_generate_phrase_candidates");

        companion object {
            fun fromWireValue(raw: String?): DecodeControlMode {
                val norm = raw?.trim()?.lowercase(Locale.US)
                return when (norm) {
                    "runtime_generate_candidates", "generate_candidates", "generatecandidates", "candidates" -> RUNTIME_GENERATE_CANDIDATES
                    "runtime_generate_phrase_candidates", "generate_phrase_candidates", "phrase_candidates", "phrase" -> RUNTIME_GENERATE_PHRASE_CANDIDATES
                    else -> BENCHMARK_DECODE_MAX_STEPS
                }
            }
        }
    }

    private data class DecodeSampleValidity(
        val isInvalid: Boolean,
        val reason: String?,
        val budget: Int?,
        val tokens: Double?,
        val threshold: Double?,
    )

    private data class FlowLifecycle(
        val flowId: String,
        val styleMode: String,
        val tag: String,
        val bucket: String,
        val iteration: Int,
        val promptBuildMs: Long,
        val preNativeOverheadMs: Long,
        val nativeWaitMs: Long,
        val postprocessMs: Long,
        val nativePrefillMs: Long,
        val nativeDecodeMs: Long,
        val nativeGapMs: Long,
        val totalFlowMs: Long,
        val eqTotalDiffMs: Long,
        val eqNativeDiffMs: Long,
        val passed: Boolean,
        val error: String?
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("flow_id", flowId)
            put("style_mode", styleMode)
            put("tag", tag)
            put("bucket", bucket)
            put("iteration", iteration)
            put("prompt_build_ms", promptBuildMs)
            put("pre_native_overhead_ms", preNativeOverheadMs)
            put("native_wait_ms", nativeWaitMs)
            put("postprocess_ms", postprocessMs)
            put("native_prefill_ms", nativePrefillMs)
            put("native_decode_ms", nativeDecodeMs)
            put("native_gap_ms", nativeGapMs)
            put("total_flow_ms", totalFlowMs)
            put("eq_total_diff_ms", eqTotalDiffMs)
            put("eq_native_diff_ms", eqNativeDiffMs)
            put("passed", passed)
            if (!error.isNullOrBlank()) put("error", error)
        }
    }

    private data class FlowRun(
        val lifecycle: FlowLifecycle,
        val metrics: JSONObject?,
        val candidates: List<String>,
        val postprocessOutcome: String,
        val error: String?
    )

    private data class NativeResult(
        val rawCandidates: List<String>,
        val metrics: JSONObject
    )

    private data class KvSpliceScenario(
        val id: String,
        val style: String,
        val history: String,
        val lastMsg: String,
        val prefix: String,
        val memories: List<String>
    )

    suspend fun runCpuExtraRamTest(progress: (String) -> Unit): TestResult {
        val testName = "cpu_extra_ram_test"
        if (!running.compareAndSet(false, true)) throw IllegalStateException("A test is already running")
        currentTest.set(testName)
        val start = System.currentTimeMillis()
        val perfToken = LogUtil.enterPerfMode()
        runCatching { LLMBridge.nativeEnterPerfMode() }

        val runDir = File(
            javaTracker.getExportDirectory(),
            "${testName}_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}"
        ).apply { mkdirs() }

        val summaryFile = File(runDir, "summary.json")
        val rawFile = File(runDir, "raw_cpu_extra_ram_metrics.jsonl")
        val progressFile = File(runDir, "progress_log.txt")
        val deviceFile = File(runDir, "device_snapshot.json")

        val warnings = mutableListOf<String>()
        val errors = mutableListOf<String>()
        val latencies = mutableListOf<Long>()
        val cpuSamples = mutableListOf<Double>()
        val sampleRows = mutableListOf<JSONObject>()
        val cpuSampler = CpuUsageSampler(Process.myPid())
        val idlePss = readPssKb()
        var peakPss = idlePss

        val style = StyleConfig.STYLE_INRERNET
        val historyBuckets = (16..512 step 16).toList()
        val repeatFactor = 3
        val totalSteps = max(1, historyBuckets.size * repeatFactor + 2)
        var done = 0
        var handle = 0L

        BufferedWriter(FileWriter(progressFile, true)).use { progressWriter ->
            BufferedWriter(FileWriter(rawFile, true)).use { rawWriter ->
                fun log(line: String) {
                    progress(line)
                    progressWriter.write("${System.currentTimeMillis()}\t$line\n")
                    progressWriter.flush()
                }

                fun step(label: String) {
                    done = (done + 1).coerceAtMost(totalSteps)
                    progress("__PROGRESS__${JSONObject().put("done", done).put("total", totalSteps).put("label", label)}")
                    cpuSampler.sample()?.let { cpuSamples.add(it) }
                }

                fun writeRaw(obj: JSONObject) {
                    rawWriter.write(obj.toString())
                    rawWriter.write("\n")
                    rawWriter.flush()
                }

                try {
                    step("setup model")
                    val modelPath = ImeModelAssetsManager(context).ensureModelCopied()
                    val modelVariant = detectModelVariant(modelPath)
                    val modelSizeBytes = File(modelPath).length()

                    handle = createHandle(modelPath)
                    if (handle == 0L) throw RuntimeException("cpu_extra_ram_create_handle_failed")

                    // Prime CPU sampler baseline.
                    cpuSampler.sample()

                    repeat(repeatFactor) { round ->
                        for (hist in historyBuckets) {
                            val sampleStartNs = System.nanoTime()
                            step("$style cpu+ram hist=$hist iter=${round + 1}")
                            ensureRunning()

                            val pssBefore = readPssKb()
                            peakPss = max(peakPss, pssBefore)
                            cpuSampler.sample()?.let { cpuSamples.add(it) }

                            val promptSegments = buildContextSyncPromptSegments(
                                historyChars = hist,
                                lastMsg = "let us continue and decide the next step",
                                prefix = "",
                                style = style
                            )
                            val prompt = promptSegments.fullPrompt

                            val kvLoadStart = System.nanoTime()
                            val baselineLoaded = ensureImeFirstBaselineLoaded(handle, style)
                            val kvLoadMs = elapsedMs(kvLoadStart)
                            if (!baselineLoaded) throw RuntimeException("cpu_extra_ram_baseline_load_failed")
                            val pssAfterKvLoad = readPssKb()
                            peakPss = max(peakPss, pssAfterKvLoad)

                            val prefillStart = System.nanoTime()
                            val prefillRes = prefillOnly(handle, promptSegments.prefillBase)
                            val prefillExternalMs = elapsedMs(prefillStart)
                            val prefillMs = metricLong(prefillRes.metrics, "prefill_ms")
                                ?: metricLong(prefillRes.metrics, "e2e_ms")
                                ?: prefillExternalMs
                            val pssAfterPrefill = readPssKb()
                            peakPss = max(peakPss, pssAfterPrefill)

                            val decodeStart = System.nanoTime()
                            val decodeRes = benchmarkDecode(handle, prompt, 1)
                            val decodeExternalMs = elapsedMs(decodeStart)
                            val inferTtftMs = extractTtft(decodeRes.metrics) ?: -1L
                            val inferPrefillMs = decodeRes.metrics.optLong("prefill_ms", -1L)
                            val inferDecodeMs = decodeRes.metrics.optLong("decode_ms", -1L)
                            val inferE2eMs = decodeRes.metrics.optLong("e2e_ms", -1L)
                            val pssAfterDecode = readPssKb()
                            peakPss = max(peakPss, pssAfterDecode)
                            cpuSampler.sample()?.let { cpuSamples.add(it) }

                            val sampleLatencyMs = elapsedMs(sampleStartNs)
                            latencies.add(sampleLatencyMs)

                            val samplePeakPss = max(max(pssBefore, pssAfterKvLoad), max(pssAfterPrefill, pssAfterDecode))
                            val extraPeakKb = max(0, samplePeakPss - idlePss)
                            val extraKvLoadKb = max(0, pssAfterKvLoad - pssBefore)
                            val extraPrefillKb = max(0, pssAfterPrefill - pssAfterKvLoad)
                            val extraDecodeKb = max(0, pssAfterDecode - pssAfterPrefill)

                            val row = JSONObject()
                                .put("tag", "cpu_extra_ram")
                                .put("style_mode", style)
                                .put("bucket", "hist_$hist")
                                .put("iteration", round)
                                .put("sample_latency_ms", sampleLatencyMs)
                                .put("kv_load_ms", kvLoadMs)
                                .put("context_prefill_ms", prefillMs)
                                .put("context_prefill_external_ms", prefillExternalMs)
                                .put("ttft_1token_infer_ms", inferTtftMs)
                                .put("ttft_1token_prefill_ms", inferPrefillMs)
                                .put("ttft_1token_decode_ms", inferDecodeMs)
                                .put("ttft_1token_e2e_ms", inferE2eMs)
                                .put("ttft_1token_decode_external_ms", decodeExternalMs)
                                .put("pss_before_kb", pssBefore)
                                .put("pss_after_kv_load_kb", pssAfterKvLoad)
                                .put("pss_after_prefill_kb", pssAfterPrefill)
                                .put("pss_after_decode_kb", pssAfterDecode)
                                .put("extra_peak_kb", extraPeakKb)
                                .put("extra_kv_load_kb", extraKvLoadKb)
                                .put("extra_prefill_kb", extraPrefillKb)
                                .put("extra_decode_kb", extraDecodeKb)
                            sampleRows.add(row)
                            writeRaw(row)
                        }
                    }

                    step("export reports")
                    val cpuUsage = buildCpuUsageSummary(cpuSamples)
                    val duration = max(0L, System.currentTimeMillis() - start)

                    val byBucket = LinkedHashMap<String, MutableList<JSONObject>>()
                    for (row in sampleRows) {
                        val bucket = row.optString("bucket", "unknown")
                        byBucket.getOrPut(bucket) { mutableListOf() }.add(row)
                    }
                    val bucketSummary = JSONArray()
                    for ((bucket, rows) in byBucket) {
                        val sampleLatencyMs = rows.mapNotNull { metricLong(it, "sample_latency_ms") }
                        val inferTtftMs = rows.mapNotNull { metricLong(it, "ttft_1token_infer_ms") }
                        val extraPeakKb = rows.mapNotNull { metricLong(it, "extra_peak_kb") }
                        val extraKvLoadKb = rows.mapNotNull { metricLong(it, "extra_kv_load_kb") }
                        val extraPrefillKb = rows.mapNotNull { metricLong(it, "extra_prefill_kb") }
                        val extraDecodeKb = rows.mapNotNull { metricLong(it, "extra_decode_kb") }
                        bucketSummary.put(
                            JSONObject()
                                .put("bucket", bucket)
                                .put("samples", rows.size)
                                .put("sample_latency_ms", buildLatencySummary(sampleLatencyMs))
                                .put("ttft_1token_infer_ms", buildLatencySummary(inferTtftMs))
                                .put("extra_peak_kb", buildLatencySummary(extraPeakKb))
                                .put("extra_kv_load_kb", buildLatencySummary(extraKvLoadKb))
                                .put("extra_prefill_kb", buildLatencySummary(extraPrefillKb))
                                .put("extra_decode_kb", buildLatencySummary(extraDecodeKb))
                        )
                    }

                    val extraPeakOverallKb = max(0, peakPss - idlePss)
                    val systemSummary = JSONObject()
                        .put("style", style)
                        .put("model_variant", modelVariant)
                        .put("model_file_size_bytes", modelSizeBytes)
                        .put("idle_pss_kb", idlePss)
                        .put("peak_pss_kb", peakPss)
                        .put("extra_peak_kb", extraPeakOverallKb)
                        .put("cpu_usage_percent", cpuUsage)

                    deviceFile.writeText(
                        JSONObject()
                            .put("timestamp", start)
                            .put("style", style)
                            .put("idle_pss_kb", idlePss)
                            .put("peak_pss_kb", peakPss)
                            .put("extra_peak_kb", extraPeakOverallKb)
                            .put("cpu_usage_percent", cpuUsage)
                            .toString(2),
                        Charsets.UTF_8
                    )

                    val success = errors.isEmpty()
                    summaryFile.writeText(
                        JSONObject()
                            .put("test_name", testName)
                            .put("success", success)
                            .put("start_time_ms", start)
                            .put("duration_ms", duration)
                            .put("style", style)
                            .put("history_buckets", JSONArray(historyBuckets))
                            .put("repeat_factor", repeatFactor)
                            .put("bucket_summary", bucketSummary)
                            .put("system", systemSummary)
                            .put("warnings", JSONArray(warnings))
                            .put("errors", JSONArray(errors))
                            .toString(2),
                        Charsets.UTF_8
                    )

                    return TestResult(
                        testName = testName,
                        success = success,
                        iterations = latencies.size,
                        totalMetrics = sampleRows.size,
                        durationMs = duration,
                        statistics = buildStatistics(latencies, duration),
                        exportPath = runDir.absolutePath,
                        warning = if (warnings.isEmpty()) null else warnings.joinToString("\n"),
                        error = if (errors.isEmpty()) null else errors.joinToString("\n")
                    )
                } catch (t: Throwable) {
                    errors.add("fatal_error: ${t.message ?: t::class.java.simpleName}")
                    val duration = max(0L, System.currentTimeMillis() - start)
                    return TestResult(
                        testName = testName,
                        success = false,
                        iterations = latencies.size,
                        totalMetrics = sampleRows.size,
                        durationMs = duration,
                        statistics = buildStatistics(latencies, duration),
                        exportPath = runDir.absolutePath,
                        warning = if (warnings.isEmpty()) null else warnings.joinToString("\n"),
                        error = errors.joinToString("\n")
                    )
                } finally {
                    freeHandle(handle)
                    runCatching { LLMBridge.nativeExitPerfMode() }
                    runCatching { LogUtil.exitPerfMode(perfToken) }
                    running.set(false)
                    currentTest.set("")
                }
            }
        }
    }

    suspend fun runMassivePrefillTest(progress: (String) -> Unit): TestResult {
        return runSuite(
            buildMassiveConfig(
                name = "massive_prefill_test",
                decodeControlMode = DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS,
                includePrefill = true,
                includeDecode = false,
                includeImeFirst = false
            ),
            progress
        )
    }

    suspend fun runMassiveDecodeTest(
        decodeMode: String,
        progress: (String) -> Unit,
    ): TestResult {
        val decodeControlMode = DecodeControlMode.fromWireValue(decodeMode)
        return runSuite(
            buildMassiveConfig(
                name = "massive_decode_test",
                decodeControlMode = decodeControlMode,
                includePrefill = false,
                includeDecode = true,
                includeImeFirst = false
            ),
            progress
        )
    }

    suspend fun runMassiveImeFirstTest(progress: (String) -> Unit): TestResult {
        return runSuite(
            buildMassiveConfig(
                name = "massive_ime_first_test",
                decodeControlMode = DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS,
                includePrefill = false,
                includeDecode = false,
                includeImeFirst = true
            ),
            progress
        )
    }

    private fun buildMassiveConfig(
        name: String,
        decodeControlMode: DecodeControlMode,
        includePrefill: Boolean = true,
        includeDecode: Boolean = true,
        includeImeFirst: Boolean = true
    ): SuiteConfig {
        return SuiteConfig(
            name = name,
            styles = listOf(StyleConfig.STYLE_INRERNET),
            promptBuckets = (0..512 step 16).toList(),
            decodeBuckets = (0..64 step 4).toList(),
            decodeRepeats = 1,
            historyBuckets = (16..512 step 16).toList(),
            includePrefill = includePrefill,
            includeDecode = includeDecode,
            includeImeFirst = includeImeFirst,
            repeatFactor = 30,
            contextRepeatFactor = 10,
            warmupRounds = 1,
            decodeControlMode = decodeControlMode,
            decodePromptChars = 256
        )
    }

    private suspend fun runKvSpliceVsReuseExperiment(progress: (String) -> Unit): TestResult {
        val testName = "kv_splice_vs_reuse_test"
        if (!running.compareAndSet(false, true)) throw IllegalStateException("A test is already running")
        currentTest.set(testName)
        val start = System.currentTimeMillis()
        val perfToken = LogUtil.enterPerfMode()
        runCatching { LLMBridge.nativeEnterPerfMode() }

        val runDir = File(
            javaTracker.getExportDirectory(),
            "${testName}_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}"
        ).apply { mkdirs() }

        val summaryFile = File(runDir, "summary.json")
        val rawFile = File(runDir, "kv_splice_vs_reuse_raw.jsonl")
        val reportFile = File(runDir, "kv_splice_vs_reuse_report.md")
        val progressFile = File(runDir, "progress_log.txt")
        val deviceFile = File(runDir, "device_snapshot.json")

        val warnings = mutableListOf<String>()
        val errors = mutableListOf<String>()
        val cpuSamples = mutableListOf<Double>()
        val cpuSampler = CpuUsageSampler(Process.myPid())
        val idlePss = readPssKb()
        var peakPss = idlePss

        val scenarios = buildKvSpliceVsReuseScenarios()
        val repeatFactor = 3
        val warmupRounds = 1
        val nCandidates = 5

        val baselineExternalMs = mutableListOf<Long>()
        val baselineE2eMs = mutableListOf<Long>()
        val baselinePrefillMs = mutableListOf<Long>()
        val baselineDecodeMs = mutableListOf<Long>()
        val baselineTtftMs = mutableListOf<Long>()
        val baselineDecodeTps = mutableListOf<Double>()

        val spliceExternalMs = mutableListOf<Long>()
        val spliceExternalAmortizedMs = mutableListOf<Double>()
        val spliceE2eMs = mutableListOf<Long>()
        val splicePrefillOnlyMs = mutableListOf<Long>()
        val splicePrefillMs = mutableListOf<Long>()
        val spliceDecodeMs = mutableListOf<Long>()
        val spliceTtftMs = mutableListOf<Long>()
        val spliceDecodeTps = mutableListOf<Double>()
        val prefillExternalMs = mutableListOf<Long>()

        val overlapScores = mutableListOf<Double>()
        val top1SimilarityScores = mutableListOf<Double>()
        var top1ExactMatchCount = 0
        var comparedCount = 0
        var baselineMetricMissingCount = 0
        var spliceMetricMissingCount = 0
        var prefillMetricMissingCount = 0
        val statsLatencies = mutableListOf<Long>()

        var baselineHandle = 0L
        var spliceHandle = 0L

        val measuredPairsPerRound = scenarios.sumOf { it.memories.size }
        val measuredPairs = measuredPairsPerRound * repeatFactor
        val totalSteps = max(1, measuredPairs + scenarios.size * repeatFactor + warmupRounds + 4)
        var done = 0

        BufferedWriter(FileWriter(progressFile, true)).use { progressWriter ->
            BufferedWriter(FileWriter(rawFile, true)).use { rawWriter ->
                fun log(line: String) {
                    progress(line)
                    progressWriter.write("${System.currentTimeMillis()}\t$line\n")
                    progressWriter.flush()
                }

                fun step(label: String) {
                    done = (done + 1).coerceAtMost(totalSteps)
                    progress("__PROGRESS__${JSONObject().put("done", done).put("total", totalSteps).put("label", label)}")
                    cpuSampler.sample()?.let { cpuSamples.add(it) }
                }

                fun writeRaw(obj: JSONObject) {
                    rawWriter.write(obj.toString())
                    rawWriter.write("\n")
                    rawWriter.flush()
                }

                try {
                    log("[Phase] kv_splice_vs_reuse setup")
                    javaTracker.clearCache()
                    runCatching { NativePerformanceTracker.nativeClearAll() }
                    val modelPath = ImeModelAssetsManager(context).ensureModelCopied()
                    val modelFile = File(modelPath)
                    val modelSizeBytes = modelFile.length()
                    val modelVariant = detectModelVariant(modelPath)
                    baselineHandle = createHandle(modelPath)
                    if (baselineHandle == 0L) throw RuntimeException("kv_compare_create_baseline_handle_failed")
                    spliceHandle = createHandle(modelPath)
                    if (spliceHandle == 0L) throw RuntimeException("kv_compare_create_splice_handle_failed")
                    step("model initialized")

                    repeat(warmupRounds) { round ->
                        if (scenarios.isEmpty()) return@repeat
                        val scenario = scenarios[round % scenarios.size]
                        val warmMemory = scenario.memories.firstOrNull().orEmpty()
                        val warmPrompt = buildChatMlPrompt(
                            getSystemPrompt(scenario.style),
                            scenario.history,
                            scenario.lastMsg,
                            warmMemory,
                            scenario.prefix
                        )
                        val warmSeg = MemoryPromptSegments.splitFromFullPrompt(warmPrompt)
                            ?: throw RuntimeException("kv_compare_prompt_split_failed warmup scenario=${scenario.id}")
                        runCatching { generate(baselineHandle, warmPrompt, nCandidates) }
                            .onFailure { throw RuntimeException("kv_compare_warmup_baseline_failed:${it.message}") }
                        runCatching {
                            prefillOnly(spliceHandle, warmSeg.toBasePromptWithoutMemory())
                            generateSpliceMemory(
                                spliceHandle,
                                warmSeg.prefixBeforeMemory,
                                warmSeg.memoryText,
                                warmSeg.suffixAfterMemory,
                                nCandidates
                            )
                        }.onFailure { throw RuntimeException("kv_compare_warmup_splice_failed:${it.message}") }
                        step("warmup round=${round + 1}/$warmupRounds")
                    }

                    log("[Phase] kv_splice_vs_reuse measured runs")
                    var pairIndex = 0
                    for (scenario in scenarios) {
                        currentCoroutineContext().ensureActive()
                        ensureRunning()
                        for (round in 0 until repeatFactor) {
                            currentCoroutineContext().ensureActive()
                            ensureRunning()
                            runCatching { LLMBridge.clearKvKeepSystem(baselineHandle) }
                            runCatching { LLMBridge.clearKvKeepSystem(spliceHandle) }
                            val basePrompt = buildChatMlPrompt(
                                getSystemPrompt(scenario.style),
                                scenario.history,
                                scenario.lastMsg,
                                "none",
                                scenario.prefix
                            )
                            val baseSeg = MemoryPromptSegments.splitFromFullPrompt(basePrompt)
                                ?: throw RuntimeException("kv_compare_prompt_split_failed scenario=${scenario.id}")

                            val prefillStart = System.nanoTime()
                            val prefillResult = prefillOnly(spliceHandle, baseSeg.toBasePromptWithoutMemory())
                            val prefillElapsed = elapsedMs(prefillStart)
                            prefillExternalMs.add(prefillElapsed)
                            metricLong(prefillResult.metrics, "prefill_ms")?.let { splicePrefillOnlyMs.add(it) } ?: run {
                                prefillMetricMissingCount++
                            }
                            step("${scenario.id} prefill round=${round + 1}/$repeatFactor")
                            val prefillAmortized = prefillElapsed.toDouble() / scenario.memories.size.coerceAtLeast(1).toDouble()

                            for ((memoryIndex, memory) in scenario.memories.withIndex()) {
                                currentCoroutineContext().ensureActive()
                                ensureRunning()

                                val fullPrompt = buildChatMlPrompt(
                                    getSystemPrompt(scenario.style),
                                    scenario.history,
                                    scenario.lastMsg,
                                    memory,
                                    scenario.prefix
                                )
                                val seg = MemoryPromptSegments.splitFromFullPrompt(fullPrompt)
                                    ?: throw RuntimeException("kv_compare_prompt_split_failed scenario=${scenario.id} memory=${memoryIndex + 1}")

                                try {
                                    val baselineStart = System.nanoTime()
                                    val baseline = generate(baselineHandle, fullPrompt, nCandidates)
                                    val baselineElapsed = elapsedMs(baselineStart)

                                    val spliceStart = System.nanoTime()
                                    val splice = generateSpliceMemory(
                                        spliceHandle,
                                        seg.prefixBeforeMemory,
                                        seg.memoryText,
                                        seg.suffixAfterMemory,
                                        nCandidates
                                    )
                                    val spliceElapsed = elapsedMs(spliceStart)
                                    val spliceAmortized = spliceElapsed.toDouble() + prefillAmortized

                                    baselineExternalMs.add(baselineElapsed)
                                    spliceExternalMs.add(spliceElapsed)
                                    spliceExternalAmortizedMs.add(spliceAmortized)

                                    val baselineE2e = metricLong(baseline.metrics, "e2e_ms") ?: run {
                                        baselineMetricMissingCount++
                                        baselineElapsed
                                    }
                                    val spliceE2e = metricLong(splice.metrics, "e2e_ms") ?: run {
                                        spliceMetricMissingCount++
                                        spliceElapsed
                                    }
                                    baselineE2eMs.add(baselineE2e)
                                    spliceE2eMs.add(spliceE2e)
                                    statsLatencies.add(baselineE2e)
                                    statsLatencies.add(spliceE2e)

                                    metricLong(baseline.metrics, "prefill_ms")?.let { baselinePrefillMs.add(it) }
                                    metricLong(baseline.metrics, "decode_ms")?.let { baselineDecodeMs.add(it) }
                                    metricLong(baseline.metrics, "ttft_ms")?.let { baselineTtftMs.add(it) }
                                    metricDouble(baseline.metrics, "decode_tps")?.let { baselineDecodeTps.add(it) }

                                    metricLong(splice.metrics, "prefill_ms")?.let { splicePrefillMs.add(it) }
                                    metricLong(splice.metrics, "decode_ms")?.let { spliceDecodeMs.add(it) }
                                    metricLong(splice.metrics, "ttft_ms")?.let { spliceTtftMs.add(it) }
                                    metricDouble(splice.metrics, "decode_tps")?.let { spliceDecodeTps.add(it) }

                                    val overlapTopK = topKOverlap(baseline.rawCandidates, splice.rawCandidates, nCandidates)
                                    val baselineTop1 = baseline.rawCandidates.firstOrNull().orEmpty()
                                    val spliceTop1 = splice.rawCandidates.firstOrNull().orEmpty()
                                    val top1Similarity = candidateSimilarity(baselineTop1, spliceTop1)
                                    val top1ExactMatch = baselineTop1.isNotBlank() && baselineTop1.trim() == spliceTop1.trim()

                                    overlapScores.add(overlapTopK)
                                    top1SimilarityScores.add(top1Similarity)
                                    comparedCount += 1
                                    if (top1ExactMatch) top1ExactMatchCount += 1

                                    pairIndex += 1
                                    writeRaw(
                                        JSONObject()
                                            .put("test_name", testName)
                                            .put("scenario_id", scenario.id)
                                            .put("style_mode", scenario.style)
                                            .put("round", round + 1)
                                            .put("memory_index", memoryIndex + 1)
                                            .put("memory_chars", memory.length)
                                            .put("full_prompt_length_chars", fullPrompt.length)
                                            .put("baseline_external_ms", baselineElapsed)
                                            .put("splice_external_ms", spliceElapsed)
                                            .put("splice_prefill_external_ms_once", prefillElapsed)
                                            .put("splice_external_amortized_ms", spliceAmortized)
                                            .put("baseline_metrics", JSONObject(baseline.metrics.toString()))
                                            .put("splice_metrics", JSONObject(splice.metrics.toString()))
                                            .put("baseline_candidates", JSONArray(baseline.rawCandidates))
                                            .put("splice_candidates", JSONArray(splice.rawCandidates))
                                            .put("topk_overlap", overlapTopK)
                                            .put("top1_similarity", top1Similarity)
                                            .put("top1_exact_match", top1ExactMatch)
                                    )
                                } catch (t: Throwable) {
                                    val err = "kv_compare_pair_failed scenario=${scenario.id} round=${round + 1} memory=${memoryIndex + 1}: ${t.message}"
                                    errors.add(err)
                                    writeRaw(
                                        JSONObject()
                                            .put("test_name", testName)
                                            .put("scenario_id", scenario.id)
                                            .put("round", round + 1)
                                            .put("memory_index", memoryIndex + 1)
                                            .put("error", err)
                                    )
                                }

                                peakPss = max(peakPss, readPssKb())
                                step("kv compare $pairIndex/$measuredPairs")
                            }
                        }
                    }

                    if (baselineMetricMissingCount > 0) {
                        warnings.add("baseline_e2e_metric_missing_count=$baselineMetricMissingCount")
                    }
                    if (spliceMetricMissingCount > 0) {
                        warnings.add("splice_e2e_metric_missing_count=$spliceMetricMissingCount")
                    }
                    if (prefillMetricMissingCount > 0) {
                        warnings.add("splice_prefill_metric_missing_count=$prefillMetricMissingCount")
                    }
                    if (comparedCount == 0) {
                        warnings.add("no_valid_kv_compare_pairs_recorded")
                    }

                    step("export summary")
                    val duration = max(0L, System.currentTimeMillis() - start)
                    cpuSampler.sample()?.let { cpuSamples.add(it) }
                    val cpuUsage = buildCpuUsageSummary(cpuSamples)
                    val systemSummary = JSONObject()
                        .put("idle_pss_kb", idlePss)
                        .put("peak_pss_kb", peakPss)
                        .put("model_path", modelPath)
                        .put("model_variant", modelVariant)
                        .put("model_file_size_bytes", modelSizeBytes)
                        .put("baseline_handle_type", "generation")
                        .put("splice_handle_type", "generation")

                    val summary = JSONObject()
                        .put("test_name", testName)
                        .put("success", errors.isEmpty())
                        .put("start_time_ms", start)
                        .put("duration_ms", duration)
                        .put(
                            "config",
                            JSONObject()
                                .put("repeat_factor", repeatFactor)
                                .put("warmup_rounds", warmupRounds)
                                .put("n_candidates", nCandidates)
                                .put("scenario_count", scenarios.size)
                                .put("memories_per_round", measuredPairsPerRound)
                        )
                        .put("pair_count", comparedCount)
                        .put(
                            "efficiency",
                            JSONObject()
                                .put(
                                    "baseline",
                                    JSONObject()
                                        .put("external_latency_ms", buildLatencySummary(baselineExternalMs))
                                        .put("e2e_ms", buildLatencySummary(baselineE2eMs))
                                        .put("prefill_ms", buildLatencySummary(baselinePrefillMs))
                                        .put("decode_ms", buildLatencySummary(baselineDecodeMs))
                                        .put("ttft_ms", buildLatencySummary(baselineTtftMs))
                                        .put("decode_tps", buildDoubleSummary(baselineDecodeTps))
                                )
                                .put(
                                    "splice",
                                    JSONObject()
                                        .put("external_latency_ms", buildLatencySummary(spliceExternalMs))
                                        .put("external_latency_amortized_ms", buildDoubleSummary(spliceExternalAmortizedMs))
                                        .put("prefill_external_ms_once", buildLatencySummary(prefillExternalMs))
                                        .put("prefill_only_ms", buildLatencySummary(splicePrefillOnlyMs))
                                        .put("e2e_ms", buildLatencySummary(spliceE2eMs))
                                        .put("prefill_ms", buildLatencySummary(splicePrefillMs))
                                        .put("decode_ms", buildLatencySummary(spliceDecodeMs))
                                        .put("ttft_ms", buildLatencySummary(spliceTtftMs))
                                        .put("decode_tps", buildDoubleSummary(spliceDecodeTps))
                                )
                                .put(
                                    "delta",
                                    JSONObject()
                                        .put("speedup_e2e_p50_baseline_over_splice", speedupMedianRatioLong(baselineE2eMs, spliceE2eMs) ?: JSONObject.NULL)
                                        .put("speedup_external_p50_baseline_over_splice", speedupMedianRatioLong(baselineExternalMs, spliceExternalMs) ?: JSONObject.NULL)
                                        .put(
                                            "speedup_external_p50_baseline_over_splice_amortized",
                                            speedupMedianRatioDouble(
                                                baselineExternalMs.map { it.toDouble() },
                                                spliceExternalAmortizedMs
                                            ) ?: JSONObject.NULL
                                        )
                                )
                        )
                        .put(
                            "effect",
                            JSONObject()
                                .put("topk_overlap", buildDoubleSummary(overlapScores))
                                .put("top1_similarity", buildDoubleSummary(top1SimilarityScores))
                                .put("top1_exact_match_count", top1ExactMatchCount)
                                .put("top1_exact_match_rate", safeRatio(top1ExactMatchCount, comparedCount))
                        )
                        .put("system", systemSummary)
                        .put("cpu_usage_percent", cpuUsage)
                        .put("warnings", JSONArray(warnings))
                        .put("errors", JSONArray(errors))
                        .put(
                            "files",
                            JSONObject()
                                .put("raw_pairs", rawFile.name)
                                .put("report", reportFile.name)
                        )

                    summaryFile.writeText(summary.toString(2), Charsets.UTF_8)
                    reportFile.writeText(buildKvSpliceVsReuseReport(summary), Charsets.UTF_8)

                    val deviceSnapshot = JSONObject()
                        .put("timestamp", start)
                        .put("device", Build.MODEL ?: "unknown")
                        .put("brand", Build.BRAND ?: "unknown")
                        .put("manufacturer", Build.MANUFACTURER ?: "unknown")
                        .put("sdk", Build.VERSION.SDK_INT)
                        .put("idle_pss_kb", idlePss)
                        .put("peak_pss_kb", peakPss)
                        .put("cpu_usage_percent", cpuUsage)
                        .put("model_path", modelPath)
                        .put("model_variant", modelVariant)
                        .put("model_file_size_bytes", modelSizeBytes)
                    deviceFile.writeText(deviceSnapshot.toString(2), Charsets.UTF_8)

                    val success = errors.isEmpty()
                    return TestResult(
                        testName = testName,
                        success = success,
                        iterations = comparedCount,
                        totalMetrics = comparedCount,
                        durationMs = duration,
                        statistics = buildStatistics(statsLatencies, duration),
                        exportPath = runDir.absolutePath,
                        warning = if (warnings.isEmpty()) null else warnings.joinToString("\n"),
                        error = if (errors.isEmpty()) null else errors.joinToString("\n")
                    )
                } catch (t: Throwable) {
                    errors.add("fatal_error: ${t.message ?: t::class.java.simpleName}")
                    val duration = max(0L, System.currentTimeMillis() - start)
                    return TestResult(
                        testName = testName,
                        success = false,
                        iterations = comparedCount,
                        totalMetrics = comparedCount,
                        durationMs = duration,
                        statistics = buildStatistics(statsLatencies, duration),
                        exportPath = runDir.absolutePath,
                        warning = if (warnings.isEmpty()) null else warnings.joinToString("\n"),
                        error = errors.joinToString("\n")
                    )
                } finally {
                    freeHandle(spliceHandle)
                    freeHandle(baselineHandle)
                    runCatching { LLMBridge.nativeExitPerfMode() }
                    runCatching { LogUtil.exitPerfMode(perfToken) }
                    running.set(false)
                    currentTest.set("")
                }
            }
        }
    }

    private suspend fun runSuite(config: SuiteConfig, progress: (String) -> Unit): TestResult {
        if (!running.compareAndSet(false, true)) throw IllegalStateException("A test is already running")
        currentTest.set(config.name)
        val start = System.currentTimeMillis()
        val perfToken = LogUtil.enterPerfMode()
        runCatching { LLMBridge.nativeEnterPerfMode() }

        val runDir = File(
            javaTracker.getExportDirectory(),
            "${config.name}_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}"
        ).apply { mkdirs() }

        val summaryFile = File(runDir, "summary.json")
        val rawFile = File(runDir, "raw_llm_metrics.jsonl")
        val lifecycleFile = File(runDir, "lifecycle_flow_metrics.jsonl")
        val progressFile = File(runDir, "progress_log.txt")
        val nativeExportFile = File(runDir, "native_perf_export.json")
        val deviceFile = File(runDir, "device_snapshot.json")
        val plotScriptFile = File(runDir, "plot_metrics.py")

        val allFlows = mutableListOf<FlowLifecycle>()
        val latencies = mutableListOf<Long>()
        val warnings = mutableListOf<String>()
        val errors = mutableListOf<String>()
        val cpuSamples = mutableListOf<Double>()
        val resultsByStyle = JSONObject()
        val idlePss = readPssKb()
        var peakPss = idlePss
        val cpuSampler = CpuUsageSampler(Process.myPid())
        val repeatFactor = config.repeatFactor.coerceAtLeast(1)
        val contextRepeatFactor = config.contextRepeatFactor.coerceAtLeast(1)
        val warmupRounds = config.warmupRounds.coerceAtLeast(0)

        val measuredStepsPerStyle =
            (if (config.includePrefill) config.promptBuckets.size * repeatFactor else 0) +
                (if (config.includeDecode) config.decodeBuckets.size * config.decodeRepeats * repeatFactor else 0) +
                (if (config.includeImeFirst) config.historyBuckets.size * contextRepeatFactor else 0)
        val warmupStepsPerStyle = warmupRounds * (
            (if (config.includePrefill) config.promptBuckets.size else 0) +
                (if (config.includeDecode) config.decodeBuckets.size else 0) +
                (if (config.includeImeFirst) config.historyBuckets.size else 0)
            )
        val stepsPerStyle = warmupStepsPerStyle + measuredStepsPerStyle
        val totalSteps = max(1, config.styles.size * stepsPerStyle + 1)
        var done = 0

        BufferedWriter(FileWriter(progressFile, true)).use { progressWriter ->
            BufferedWriter(FileWriter(rawFile, true)).use { rawWriter ->
                BufferedWriter(FileWriter(lifecycleFile, true)).use { lifecycleWriter ->
                    fun log(line: String) {
                        progress(line)
                        progressWriter.write("${System.currentTimeMillis()}\t$line\n")
                        progressWriter.flush()
                    }

                    fun step(label: String) {
                        done = (done + 1).coerceAtMost(totalSteps)
                        progress("__PROGRESS__${JSONObject().put("done", done).put("total", totalSteps).put("label", label)}")
                        cpuSampler.sample()?.let { cpuSamples.add(it) }
                    }

                    fun writeJsonl(writer: BufferedWriter, obj: JSONObject) {
                        writer.write(obj.toString())
                        writer.write("\n")
                        writer.flush()
                    }

                    fun registerFlow(flow: FlowRun) {
                        allFlows.add(flow.lifecycle)
                        writeJsonl(lifecycleWriter, flow.lifecycle.toJson())
                        if (flow.error == null) {
                            latencies.add(flow.lifecycle.totalFlowMs)
                        } else when {
                            isLifecycleWarning(flow.error) -> {
                                warnings.add(flow.error)
                                latencies.add(flow.lifecycle.totalFlowMs)
                            }
                            isNonFatalGenerationWarning(flow.error) -> {
                                warnings.add(flow.error)
                            }
                            else -> {
                                errors.add(flow.error)
                            }
                        }
                        peakPss = max(peakPss, readPssKb())
                        cpuSampler.sample()?.let { cpuSamples.add(it) }
                    }

                    fun normalizedWarnings(): List<String> {
                        if (warnings.isEmpty()) return emptyList()
                        val counts = LinkedHashMap<String, Int>()
                        for (w in warnings) {
                            counts[w] = (counts[w] ?: 0) + 1
                        }
                        return counts.map { (msg, count) ->
                            if (count > 1) "$msg (x$count)" else msg
                        }
                    }

                    try {
                        javaTracker.clearCache()
                        runCatching { NativePerformanceTracker.nativeClearAll() }
                        val modelPath = ImeModelAssetsManager(context).ensureModelCopied()
                        val modelFile = File(modelPath)
                        val modelSizeBytes = modelFile.length()
                        val modelVariant = detectModelVariant(modelPath)

                        val deviceSnapshot = JSONObject()
                            .put("timestamp", start)
                            .put("device", Build.MODEL ?: "unknown")
                            .put("brand", Build.BRAND ?: "unknown")
                            .put("manufacturer", Build.MANUFACTURER ?: "unknown")
                            .put("sdk", Build.VERSION.SDK_INT)
                            .put("model_path", modelPath)
                            .put("model_file_size_bytes", modelSizeBytes)
                            .put("model_variant", modelVariant)
                            .put("idle_pss_kb", idlePss)

                        for (style in config.styles) {
                            currentCoroutineContext().ensureActive()
                            ensureRunning()
                            log("[Style] $style")
                            val styleObj = runStyle(config, style, modelPath, rawWriter, ::step, ::registerFlow, errors)
                            resultsByStyle.put(style, styleObj)
                        }

                        step("export reports")
                        val exportRes = runCatching {
                            javaTracker.exportToDirectory(runDir, "perf_report_${config.name}")
                        }.getOrElse { PerformanceTracker.ExportResult(false, 0, it.message ?: "export_failed") }
                        if (!exportRes.success) errors.add("export_failed: ${exportRes.message}")
                        runCatching { nativeExportFile.writeText(NativePerformanceTracker.nativeExportJson(), Charsets.UTF_8) }

                        val lifecycleBreakdown = buildLifecycleBreakdown(allFlows)
                        val lifecycleMs = buildLifecycleFallback(allFlows)
                        val lifecycleInvariant = JSONObject()
                            .put("checked", lifecycleBreakdown.optInt("strict_checked_flows", 0))
                            .put("passed", lifecycleBreakdown.optInt("passed_flows", 0))
                            .put("failed", lifecycleBreakdown.optInt("failed_flows", 0))
                            .put("tolerance_ms", lifecycleToleranceMs)
                            .put("samples_file", lifecycleFile.name)
                        if (lifecycleInvariant.optInt("failed", 0) > 0) {
                            warnings.add(
                                "lifecycle_strict_validation_failed(failed=${lifecycleInvariant.optInt("failed", 0)}," +
                                    "tolerance=${lifecycleToleranceMs}ms)"
                            )
                        }

                        val duration = max(0L, System.currentTimeMillis() - start)
                        val compactWarnings = normalizedWarnings()
                        val success = errors.isEmpty()
                        deviceSnapshot.put("peak_pss_kb", peakPss)
                        cpuSampler.sample()?.let { cpuSamples.add(it) }
                        val cpuUsage = buildCpuUsageSummary(cpuSamples)
                        val kvCacheSummary = buildKvCacheSummary(emptyList())
                        val systemSummary = JSONObject()
                            .put("model_path", modelPath)
                            .put("model_variant", modelVariant)
                            .put("model_file_size_bytes", modelSizeBytes)
                            .put("kv_cache_size_bytes", kvCacheSummary)
                            .put("idle_pss_kb", idlePss)
                            .put("peak_pss_kb", peakPss)
                        deviceSnapshot.put("cpu_usage_percent", cpuUsage)
                        deviceSnapshot.put("kv_cache_size_bytes", kvCacheSummary)
                        deviceFile.writeText(deviceSnapshot.toString(2), Charsets.UTF_8)
                        runCatching { writePlotScript(plotScriptFile) }
                            .onFailure { warnings.add("plot_script_write_failed: ${it.message}") }

                        summaryFile.writeText(
                            JSONObject()
                                .put("test_name", config.name)
                                .put("success", success)
                                .put("start_time_ms", start)
                                .put("duration_ms", duration)
                                .put("styles_tested", JSONArray(config.styles))
                                .put("results_by_style", resultsByStyle)
                                .put("lifecycle_breakdown", lifecycleBreakdown)
                                .put("lifecycle_ms", lifecycleMs)
                                .put("lifecycle_invariant", lifecycleInvariant)
                                .put("lifecycle_phase_order", JSONArray(listOf("prompt_build_ms", "pre_native_overhead_ms", "native_wait_ms", "postprocess_ms")))
                                .put("dialog_tests", JSONObject().put("enabled", false))
                                .put(
                                    "decode",
                                    JSONObject()
                                        .put("control_mode", config.decodeControlMode.wireValue)
                                        .put("fixed_prompt_chars", config.decodePromptChars)
                                        .put("n_candidates", config.decodeNCandidates)
                                        .put("invalid_token_ratio", config.decodeInvalidTokenRatio)
                                        .put("invalid_min_budget", config.decodeInvalidMinBudget)
                                )
                                .put("system", systemSummary)
                                .put("cpu_usage_percent", cpuUsage)
                                .put("plot_script", plotScriptFile.name)
                                .put("plot_command", "python ${plotScriptFile.name} --run-dir . --out-dir plots --summary-md research_summary.md")
                                .put("warnings", JSONArray(compactWarnings))
                                .put("errors", JSONArray(errors))
                                .toString(2),
                            Charsets.UTF_8
                        )

                        return TestResult(
                            testName = config.name,
                            success = success,
                            iterations = latencies.size,
                            totalMetrics = allFlows.size,
                            durationMs = duration,
                            statistics = buildStatistics(latencies, duration),
                            exportPath = runDir.absolutePath,
                            warning = if (compactWarnings.isEmpty()) null else compactWarnings.joinToString("\n"),
                            error = if (errors.isEmpty()) null else errors.joinToString("\n")
                        )
                    } catch (t: Throwable) {
                        errors.add("fatal_error: ${t.message ?: t::class.java.simpleName}")
                        val duration = max(0L, System.currentTimeMillis() - start)
                        return TestResult(
                            testName = config.name,
                            success = false,
                            iterations = latencies.size,
                            totalMetrics = allFlows.size,
                            durationMs = duration,
                            statistics = buildStatistics(latencies, duration),
                            exportPath = runDir.absolutePath,
                            warning = if (warnings.isEmpty()) null else warnings.joinToString("\n"),
                            error = errors.joinToString("\n")
                        )
                    } finally {
                        runCatching { LLMBridge.nativeExitPerfMode() }
                        runCatching { LogUtil.exitPerfMode(perfToken) }
                        running.set(false)
                        currentTest.set("")
                    }
                }
            }
        }
    }

    private fun buildKvSpliceVsReuseScenarios(): List<KvSpliceScenario> {
        return listOf(
            KvSpliceScenario(
                id = "business_revision",
                style = StyleConfig.STYLE_INRERNET,
                history = "[me]: Q3 report draft is out.\n[peer]: The structure is weak and lacks a clear action plan.\n[me]: I will revise it in 30 minutes.",
                lastMsg = "Please give me a concise response that confirms ownership and timeline.",
                prefix = "Sure, ",
                memories = listOf(
                    "none",
                    "Remember: the stakeholder cares most about concise structure and quantified decisions.",
                    "Remember: deadline is tonight 10 PM and risk-control points must be explicit."
                )
            ),
            KvSpliceScenario(
                id = "project_blocker",
                style = StyleConfig.STYLE_INRERNET,
                history = "[me]: We need API readiness this week.\n[peer]: Scope keeps changing and we are short on people.\n[me]: Let's lock a minimal deliverable.",
                lastMsg = "Can you propose a practical reply with owner and milestone?",
                prefix = "We can ",
                memories = listOf(
                    "none",
                    "Remember: prioritize minimum deliverable and explicit owner assignment.",
                    "Remember: include one escalation path if dependency is still blocked."
                )
            ),
            KvSpliceScenario(
                id = "incident_response",
                style = StyleConfig.STYLE_INRERNET,
                history = "[me]: P99 latency spiked at 10:05.\n[peer]: Business is pushing for a quick answer.\n[me]: We need stop-loss then root-cause isolation.",
                lastMsg = "Draft a short incident-response message for the team.",
                prefix = "First, ",
                memories = listOf(
                    "none",
                    "Remember: stop-loss first, then isolate CPU/IO/downstream bottleneck.",
                    "Remember: include rollback trigger and next checkpoint time."
                )
            )
        )
    }

    private fun buildKvSpliceVsReuseReport(summary: JSONObject): String {
        fun metricP50(parent: JSONObject?, key: String): String {
            val obj = parent?.optJSONObject(key) ?: return "-"
            val v = obj.opt("p50")
            return when (v) {
                null, JSONObject.NULL -> "-"
                is Number -> String.format(Locale.US, "%.2f", v.toDouble())
                else -> v.toString()
            }
        }
        fun metricN(parent: JSONObject?, key: String): String {
            val obj = parent?.optJSONObject(key) ?: return "0"
            return obj.optInt("n", 0).toString()
        }
        fun valueOrDash(v: Any?): String {
            return when (v) {
                null, JSONObject.NULL -> "-"
                is Number -> String.format(Locale.US, "%.4f", v.toDouble())
                else -> v.toString()
            }
        }

        val config = summary.optJSONObject("config")
        val efficiency = summary.optJSONObject("efficiency")
        val baseline = efficiency?.optJSONObject("baseline")
        val splice = efficiency?.optJSONObject("splice")
        val delta = efficiency?.optJSONObject("delta")
        val effect = summary.optJSONObject("effect")
        val warnings = summary.optJSONArray("warnings")
        val errors = summary.optJSONArray("errors")

        return buildString {
            append("# KV-Splice vs KV Reuse Test Report\n\n")
            append("## 1. Test Config\n")
            append("- test_name: ${summary.optString("test_name", "-")}\n")
            append("- duration_ms: ${summary.optLong("duration_ms", -1)}\n")
            append("- scenario_count: ${config?.optInt("scenario_count", 0) ?: 0}\n")
            append("- repeat_factor: ${config?.optInt("repeat_factor", 0) ?: 0}\n")
            append("- n_candidates: ${config?.optInt("n_candidates", 0) ?: 0}\n")
            append("- pair_count: ${summary.optInt("pair_count", 0)}\n\n")

            append("## 2. Efficiency (P50)\n")
            append("| Metric | Baseline P50 | Splice P50 | Samples |\n")
            append("|---|---:|---:|---:|\n")
            append("| e2e_ms | ${metricP50(baseline, "e2e_ms")} | ${metricP50(splice, "e2e_ms")} | ${metricN(baseline, "e2e_ms")} |\n")
            append("| external_ms | ${metricP50(baseline, "external_latency_ms")} | ${metricP50(splice, "external_latency_ms")} | ${metricN(baseline, "external_latency_ms")} |\n")
            append("| prefill_ms | ${metricP50(baseline, "prefill_ms")} | ${metricP50(splice, "prefill_ms")} | ${metricN(baseline, "prefill_ms")} |\n")
            append("| decode_ms | ${metricP50(baseline, "decode_ms")} | ${metricP50(splice, "decode_ms")} | ${metricN(baseline, "decode_ms")} |\n")
            append("| ttft_ms | ${metricP50(baseline, "ttft_ms")} | ${metricP50(splice, "ttft_ms")} | ${metricN(baseline, "ttft_ms")} |\n\n")

            append("## 3. Speedup (Baseline/Splice)\n")
            append("- e2e_p50_speedup: ${valueOrDash(delta?.opt("speedup_e2e_p50_baseline_over_splice"))}\n")
            append("- external_p50_speedup: ${valueOrDash(delta?.opt("speedup_external_p50_baseline_over_splice"))}\n")
            append("- external_p50_speedup_amortized: ${valueOrDash(delta?.opt("speedup_external_p50_baseline_over_splice_amortized"))}\n\n")

            append("## 4. Effect Consistency\n")
            append("- topk_overlap_p50: ${metricP50(effect, "topk_overlap")}\n")
            append("- top1_similarity_p50: ${metricP50(effect, "top1_similarity")}\n")
            append("- top1_exact_match_rate: ${valueOrDash(effect?.opt("top1_exact_match_rate"))}\n")
            append("- top1_exact_match_count: ${effect?.optInt("top1_exact_match_count", 0) ?: 0}\n\n")

            append("## 5. Warnings / Errors\n")
            append("- warnings: ${warnings?.length() ?: 0}\n")
            if (warnings != null) {
                for (i in 0 until warnings.length()) {
                    append("  - ${warnings.optString(i)}\n")
                }
            }
            append("- errors: ${errors?.length() ?: 0}\n")
            if (errors != null) {
                for (i in 0 until errors.length()) {
                    append("  - ${errors.optString(i)}\n")
                }
            }
        }
    }

    private data class SampledJsonLine(
        val lineNumber: Int,
        val payload: JSONObject
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("line_number", lineNumber)
            .put("payload", payload)
    }

    private fun buildLatencySummary(values: List<Long>): JSONObject {
        val valid = values.filter { it >= 0L }
        if (valid.isEmpty()) {
            return JSONObject()
                .put("n", 0)
                .put("avg", JSONObject.NULL)
                .put("std", JSONObject.NULL)
                .put("min", JSONObject.NULL)
                .put("max", JSONObject.NULL)
                .put("p50", JSONObject.NULL)
                .put("p90", JSONObject.NULL)
                .put("p99", JSONObject.NULL)
        }
        val doubles = valid.map { it.toDouble() }
        val mean = doubles.average()
        val std = if (doubles.size > 1) {
            val varSample = doubles.sumOf { (it - mean) * (it - mean) } / (doubles.size - 1).toDouble()
            sqrt(max(0.0, varSample))
        } else 0.0
        return JSONObject()
            .put("n", valid.size)
            .put("avg", mean)
            .put("std", std)
            .put("min", valid.minOrNull() ?: JSONObject.NULL)
            .put("max", valid.maxOrNull() ?: JSONObject.NULL)
            .put("p50", percentile(valid, 0.50) ?: JSONObject.NULL)
            .put("p90", percentile(valid, 0.90) ?: JSONObject.NULL)
            .put("p99", percentile(valid, 0.99) ?: JSONObject.NULL)
    }

    private fun buildDoubleSummary(values: List<Double>): JSONObject {
        val valid = values.filter { it.isFinite() && it >= 0.0 }
        if (valid.isEmpty()) {
            return JSONObject()
                .put("n", 0)
                .put("avg", JSONObject.NULL)
                .put("std", JSONObject.NULL)
                .put("min", JSONObject.NULL)
                .put("max", JSONObject.NULL)
                .put("p50", JSONObject.NULL)
                .put("p90", JSONObject.NULL)
                .put("p99", JSONObject.NULL)
        }
        val mean = valid.average()
        val std = if (valid.size > 1) {
            val varSample = valid.sumOf { (it - mean) * (it - mean) } / (valid.size - 1).toDouble()
            sqrt(max(0.0, varSample))
        } else 0.0
        return JSONObject()
            .put("n", valid.size)
            .put("avg", mean)
            .put("std", std)
            .put("min", valid.minOrNull() ?: JSONObject.NULL)
            .put("max", valid.maxOrNull() ?: JSONObject.NULL)
            .put("p50", percentileDouble(valid, 0.50) ?: JSONObject.NULL)
            .put("p90", percentileDouble(valid, 0.90) ?: JSONObject.NULL)
            .put("p99", percentileDouble(valid, 0.99) ?: JSONObject.NULL)
    }

    private fun metricLong(metrics: JSONObject?, key: String): Long? {
        if (metrics == null || !metrics.has(key)) return null
        val raw = metrics.opt(key)
        val value = when (raw) {
            is Number -> raw.toLong()
            is String -> raw.toLongOrNull()
            else -> null
        } ?: return null
        return value.takeIf { it >= 0L }
    }

    private fun metricDouble(metrics: JSONObject?, key: String): Double? {
        if (metrics == null || !metrics.has(key)) return null
        val raw = metrics.opt(key)
        val value = when (raw) {
            is Number -> raw.toDouble()
            is String -> raw.toDoubleOrNull()
            else -> null
        } ?: return null
        return value.takeIf { it.isFinite() && it >= 0.0 }
    }

    private fun topKOverlap(a: List<String>, b: List<String>, k: Int): Double {
        val limit = k.coerceAtLeast(1)
        val aa = a.asSequence()
            .filter { it.isNotBlank() && !it.startsWith("__METRICS__") }
            .take(limit)
            .toList()
        val bb = b.asSequence()
            .filter { it.isNotBlank() && !it.startsWith("__METRICS__") }
            .take(limit)
            .toList()
        if (aa.isEmpty() || bb.isEmpty()) return 0.0
        val inter = aa.toSet().intersect(bb.toSet()).size
        return inter.toDouble() / min(aa.size, bb.size).coerceAtLeast(1).toDouble()
    }

    private fun speedupMedianRatioLong(baseline: List<Long>, target: List<Long>): Double? {
        val b = percentile(baseline.filter { it >= 0L }, 0.50)?.toDouble() ?: return null
        val t = percentile(target.filter { it >= 0L }, 0.50)?.toDouble() ?: return null
        if (b <= 0.0 || t <= 0.0) return null
        return b / t
    }

    private fun speedupMedianRatioDouble(baseline: List<Double>, target: List<Double>): Double? {
        val b = percentileDouble(baseline.filter { it.isFinite() && it >= 0.0 }, 0.50) ?: return null
        val t = percentileDouble(target.filter { it.isFinite() && it >= 0.0 }, 0.50) ?: return null
        if (b <= 0.0 || t <= 0.0) return null
        return b / t
    }

    private fun safeRatio(numerator: Int, denominator: Int): Double {
        if (denominator <= 0) return 0.0
        return numerator.toDouble() / denominator.toDouble()
    }

    private fun loadJsonlObjects(file: File, limit: Int = 5000): List<JSONObject> {
        if (!file.exists() || limit <= 0) return emptyList()
        val out = mutableListOf<JSONObject>()
        runCatching {
            file.forEachLine(Charsets.UTF_8) { raw ->
                if (out.size >= limit) return@forEachLine
                val t = raw.trim()
                if (t.isEmpty()) return@forEachLine
                runCatching { JSONObject(t) }.getOrNull()?.let { out.add(it) }
            }
        }
        return out
    }

    private fun loadProcessStatusCounts(file: File): Map<String, Int> {
        if (!file.exists()) return emptyMap()
        val out = LinkedHashMap<String, Int>()
        runCatching {
            file.forEachLine(Charsets.UTF_8) { raw ->
                val t = raw.trim()
                if (t.isEmpty()) return@forEachLine
                val status = runCatching { JSONObject(t).optString("status", "").trim() }.getOrDefault("")
                if (status.isEmpty()) return@forEachLine
                out[status] = (out[status] ?: 0) + 1
            }
        }
        return out
    }

    private fun candidateSimilarity(a: String, b: String): Double {
        if (a.isBlank() || b.isBlank()) return 0.0
        val cjk = jaccard(cjkBigrams(a), cjkBigrams(b))
        val word = jaccard(wordTokens(a), wordTokens(b))
        return (0.85 * cjk + 0.15 * word).coerceIn(0.0, 1.0)
    }

    private fun cjkBigrams(text: String): Set<String> {
        val clean = text.filter { it.code in 0x4E00..0x9FFF }
        if (clean.length < 2) return emptySet()
        val out = LinkedHashSet<String>()
        for (i in 0 until clean.length - 1) {
            out.add(clean.substring(i, i + 2))
        }
        return out
    }

    private fun wordTokens(text: String): Set<String> {
        return text.lowercase(Locale.US)
            .split(Regex("[^a-z0-9]+"))
            .map { it.trim() }
            .filter { it.length >= 2 }
            .toSet()
    }

    private fun jaccard(a: Set<String>, b: Set<String>): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val inter = a.intersect(b).size.toDouble()
        val union = (a.size + b.size - inter).coerceAtLeast(1.0)
        return inter / union
    }

    private suspend fun runStyle(
        config: SuiteConfig,
        style: String,
        modelPath: String,
        rawWriter: BufferedWriter,
        step: (String) -> Unit,
        registerFlow: (FlowRun) -> Unit,
        errors: MutableList<String>
    ): JSONObject {
        val prefill = mutableListOf<JSONObject>()
        val decode = mutableListOf<JSONObject>()
        val imeFirst = mutableListOf<JSONObject>()
        val repeatFactor = config.repeatFactor.coerceAtLeast(1)
        val contextRepeatFactor = config.contextRepeatFactor.coerceAtLeast(1)
        val warmupRounds = config.warmupRounds.coerceAtLeast(0)

        fun writeRaw(obj: JSONObject) {
            rawWriter.write(obj.toString())
            rawWriter.write("\n")
            rawWriter.flush()
        }

        val prefillPromptCache = mutableMapOf<Int, Pair<String, Int>>()

        suspend fun runPrefillBucket(bucket: Int, round: Int, measured: Boolean) {
            val start = System.nanoTime()
            val createStart = System.nanoTime()
            val handle = createBenchmarkHandle(modelPath)
            val createMs = elapsedMs(createStart)
            val promptInfo = prefillPromptCache[bucket] ?: buildPrefillPromptByTokenBudget(handle, bucket).also {
                prefillPromptCache[bucket] = it
            }
            val prompt = promptInfo.first
            val promptTokens = promptInfo.second
            val buildMs = elapsedMs(start)
            try {
                val flow = runFlow(
                    handle = handle,
                    prompt = prompt,
                    style = style,
                    tag = "prefill_scaling",
                    bucket = "tokens_$bucket",
                    iteration = round,
                    promptBuildMs = buildMs,
                    flowStartNs = start,
                    instructionPrefix = "",
                    preNativeOverheadMs = createMs,
                    prefillOnlyMode = true,
                    skipPostProcess = true
                )
                if (!measured) return
                registerFlow(flow)
                prefill.add(
                    JSONObject()
                        .put("bucket", "tokens_$bucket")
                        .put("iteration", round)
                        .put("prefill_target_tokens", bucket)
                        .put("prefill_prompt_tokens", promptTokens)
                        .put("prefill_ms", flow.metrics?.optLong("prefill_ms", -1) ?: -1)
                        .put("prefill_tokens", flow.metrics?.optLong("prefill_tokens", -1) ?: -1)
                        .put("prefill_tps", flow.metrics?.optDouble("prefill_tps", -1.0) ?: -1.0)
                        .put("error", flow.error ?: JSONObject.NULL)
                )
                writeRaw(
                    rawFlowObject(style, "prefill_scaling", "tokens_$bucket", round, prompt, flow)
                        .put("prefill_target_tokens", bucket)
                        .put("prefill_prompt_tokens", promptTokens)
                        .put("prefill_token_mode", true)
                )
            } finally {
                freeHandle(handle)
            }
        }

        if (config.includePrefill) {
            repeat(warmupRounds) { round ->
                for (bucket in config.promptBuckets) {
                    step("$style warmup prefill tokens=$bucket round=${round + 1}")
                    runPrefillBucket(bucket = bucket, round = round, measured = false)
                }
            }
            repeat(repeatFactor) { round ->
                for (bucket in config.promptBuckets) {
                    step("$style prefill tokens=$bucket iter=${round + 1}")
                    runPrefillBucket(bucket = bucket, round = round, measured = true)
                }
            }
        }

        val decodePrompt = buildPrompt(max(64, config.decodePromptChars), style)
        val decodePromptFingerprint = Integer.toHexString(decodePrompt.hashCode())

        suspend fun runDecodeBucketSample(bucket: Int, iteration: Int, measured: Boolean, bucketRecords: MutableList<JSONObject>) {
            val start = System.nanoTime()
            val prompt = decodePrompt
            val buildMs = elapsedMs(start)
            val createStart = System.nanoTime()
            val handle = if (config.decodeControlMode == DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS) {
                createBenchmarkHandle(modelPath)
            } else {
                createHandle(modelPath)
            }
            val createMs = elapsedMs(createStart)
            try {
                val flow = runFlow(
                    handle = handle,
                    prompt = prompt,
                    style = style,
                    tag = "decode",
                    bucket = "steps_$bucket",
                    iteration = iteration,
                    promptBuildMs = buildMs,
                    flowStartNs = start,
                    instructionPrefix = "",
                    preNativeOverheadMs = createMs,
                    decodeStepBudget = bucket,
                    decodeControlMode = config.decodeControlMode,
                    decodeNCandidates = config.decodeNCandidates,
                    skipPostProcess = true
                )
                if (!measured) return
                registerFlow(flow)
                val validity = assessDecodeSampleValidity(
                    metrics = flow.metrics,
                    stepBudget = bucket,
                    tokenRatio = config.decodeInvalidTokenRatio,
                    minBudget = config.decodeInvalidMinBudget
                )
                flow.metrics?.let {
                    bucketRecords.add(
                        JSONObject(it.toString())
                            .put("decode_control_mode", config.decodeControlMode.wireValue)
                            .put("decode_step_budget", bucket)
                            .put("decode_prompt_fingerprint", decodePromptFingerprint)
                            .put("decode_prompt_chars", config.decodePromptChars)
                            .put("decode_same_sample_repeated", true)
                            .put("decode_n_candidates", config.decodeNCandidates)
                            .put("decode_sample_invalid", validity.isInvalid)
                            .put("decode_invalid_reason", validity.reason ?: JSONObject.NULL)
                            .put("decode_invalid_threshold_tokens", validity.threshold ?: JSONObject.NULL)
                    )
                }
                writeRaw(
                    rawFlowObject(style, "decode", "steps_$bucket", iteration, prompt, flow)
                        .put("decode_control_mode", config.decodeControlMode.wireValue)
                        .put("decode_step_budget", bucket)
                        .put("decode_step_budget_applied", config.decodeControlMode == DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS)
                        .put("decode_handle_isolated", true)
                        .put("decode_prompt_fingerprint", decodePromptFingerprint)
                        .put("decode_prompt_chars", config.decodePromptChars)
                        .put("decode_same_sample_repeated", true)
                        .put("decode_n_candidates", config.decodeNCandidates)
                        .put("decode_sample_invalid", validity.isInvalid)
                        .put("decode_invalid_reason", validity.reason ?: JSONObject.NULL)
                        .put("decode_invalid_threshold_tokens", validity.threshold ?: JSONObject.NULL)
                )
            } finally {
                freeHandle(handle)
            }
        }

        if (config.includeDecode) {
            repeat(warmupRounds) { round ->
                for (bucket in config.decodeBuckets) {
                    step("$style warmup decode steps=$bucket round=${round + 1}")
                    runDecodeBucketSample(bucket = bucket, iteration = round, measured = false, bucketRecords = mutableListOf())
                }
            }
            for (bucket in config.decodeBuckets) {
                val bucketRecords = mutableListOf<JSONObject>()
                repeat(repeatFactor) { round ->
                    repeat(config.decodeRepeats) { idx ->
                        val iteration = round * config.decodeRepeats + idx
                        step("$style decode steps=$bucket iter=${iteration + 1}")
                        runDecodeBucketSample(bucket = bucket, iteration = iteration, measured = true, bucketRecords = bucketRecords)
                    }
                }
                decode.add(
                    summarizeDecodeBucket(
                        records = bucketRecords,
                        steps = bucket,
                        tokenRatio = config.decodeInvalidTokenRatio,
                        minBudget = config.decodeInvalidMinBudget
                    )
                        .put("decode_control_mode", config.decodeControlMode.wireValue)
                        .put("decode_step_budget", bucket)
                        .put("decode_step_budget_applied", config.decodeControlMode == DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS)
                        .put("decode_handle_isolated", true)
                        .put("decode_prompt_fingerprint", decodePromptFingerprint)
                        .put("decode_prompt_chars", config.decodePromptChars)
                        .put("decode_same_sample_repeated", true)
                        .put("decode_n_candidates", config.decodeNCandidates)
                )
            }
        }

        suspend fun runImeFirst(hist: Int, round: Int, measured: Boolean, handle: Long) {
            val start = System.nanoTime()
            val promptSegments = buildContextSyncPromptSegments(hist, "let us continue and decide the next step", "", style)
            val prompt = promptSegments.fullPrompt
            val buildMs = elapsedMs(start)
            try {
                val kvLoadStart = System.nanoTime()
                val baselineLoaded = ensureImeFirstBaselineLoaded(handle, style)
                val kvLoadMs = elapsedMs(kvLoadStart)
                if (!baselineLoaded) throw RuntimeException("ime_first_style_baseline_load_failed")

                val contextPrefillStart = System.nanoTime()
                val contextPrefill = prefillOnly(handle, promptSegments.prefillBase)
                val contextPrefillExternalMs = elapsedMs(contextPrefillStart)
                val contextPrefillMs = metricLong(contextPrefill.metrics, "prefill_ms")
                    ?: metricLong(contextPrefill.metrics, "e2e_ms")
                    ?: contextPrefillExternalMs

                val flow = runFlow(
                    handle = handle,
                    prompt = prompt,
                    style = style,
                    tag = "ime_first",
                    bucket = "hist_$hist",
                    iteration = round,
                    promptBuildMs = buildMs,
                    flowStartNs = start,
                    instructionPrefix = "",
                    preNativeOverheadMs = kvLoadMs + contextPrefillExternalMs
                )
                if (!measured) return
                registerFlow(flow)
                val generateTtft = extractTtft(flow.metrics) ?: -1L
                val imeFirstCandidate = if (generateTtft >= 0L) kvLoadMs + contextPrefillMs + generateTtft else -1L

                var ttftOneTokenMs = -1L
                var ttftOneTokenInferMs = -1L
                var ttftOneTokenKvLoadMs = -1L
                var ttftOneTokenContextPrefillMs = -1L
                var ttftOneTokenContextPrefillExternalMs = -1L
                var ttftOneTokenPrefillMs = -1L
                var ttftOneTokenDecodeMs = -1L
                var ttftOneTokenE2eMs = -1L
                var ttftOneTokenError: String? = null

                runCatching {
                    val kvLoadStartForTtft = System.nanoTime()
                    val baselineLoadedForTtft = ensureImeFirstBaselineLoaded(handle, style)
                    ttftOneTokenKvLoadMs = elapsedMs(kvLoadStartForTtft)
                    if (!baselineLoadedForTtft) throw RuntimeException("ime_ttft1_style_baseline_load_failed")

                    val contextPrefillStartForTtft = System.nanoTime()
                    val contextPrefillForTtft = prefillOnly(handle, promptSegments.prefillBase)
                    ttftOneTokenContextPrefillExternalMs = elapsedMs(contextPrefillStartForTtft)
                    ttftOneTokenContextPrefillMs = metricLong(contextPrefillForTtft.metrics, "prefill_ms")
                        ?: metricLong(contextPrefillForTtft.metrics, "e2e_ms")
                        ?: ttftOneTokenContextPrefillExternalMs

                    val oneTokenDecode = benchmarkDecode(handle, prompt, 1)
                    ttftOneTokenInferMs = extractTtft(oneTokenDecode.metrics) ?: -1L
                    ttftOneTokenPrefillMs = oneTokenDecode.metrics.optLong("prefill_ms", -1L)
                    ttftOneTokenDecodeMs = oneTokenDecode.metrics.optLong("decode_ms", -1L)
                    ttftOneTokenE2eMs = oneTokenDecode.metrics.optLong("e2e_ms", -1L)
                    ttftOneTokenMs = if (ttftOneTokenInferMs >= 0L) {
                        ttftOneTokenKvLoadMs + ttftOneTokenContextPrefillMs + ttftOneTokenInferMs
                    } else {
                        -1L
                    }
                }.onFailure { err ->
                    ttftOneTokenError = err.message ?: err::class.java.simpleName
                }

                val ttftForReport = if (ttftOneTokenMs >= 0L) ttftOneTokenMs else generateTtft
                val record = JSONObject()
                    .put("bucket", "hist_$hist")
                    .put("iteration", round)
                    .put("ime_first_candidate_ms", imeFirstCandidate)
                    .put("ttft_ms", ttftForReport)
                    .put("ttft_generate_ms", generateTtft)
                    .put("ttft_1token_ms", ttftOneTokenMs)
                    .put("ttft_1token_infer_ms", ttftOneTokenInferMs)
                    .put("ttft_1token_kv_load_ms", ttftOneTokenKvLoadMs)
                    .put("ttft_1token_context_prefill_ms", ttftOneTokenContextPrefillMs)
                    .put("ttft_1token_context_prefill_external_ms", ttftOneTokenContextPrefillExternalMs)
                    .put("ttft_1token_prefill_ms", ttftOneTokenPrefillMs)
                    .put("ttft_1token_decode_ms", ttftOneTokenDecodeMs)
                    .put("ttft_1token_e2e_ms", ttftOneTokenE2eMs)
                    .put("kv_load_ms", kvLoadMs)
                    .put("context_prefill_ms", contextPrefillMs)
                    .put("context_prefill_external_ms", contextPrefillExternalMs)
                    .put("prefill_ms", flow.metrics?.optLong("prefill_ms", -1) ?: -1)
                    .put("decode_ms", flow.metrics?.optLong("decode_ms", -1) ?: -1)
                    .put("e2e_ms", flow.metrics?.optLong("e2e_ms", -1) ?: -1)
                    .put("decode_tps", flow.metrics?.optDouble("decode_tps", -1.0) ?: -1.0)
                if (!flow.error.isNullOrBlank()) record.put("error", flow.error)
                if (!ttftOneTokenError.isNullOrBlank()) record.put("ttft_1token_error", ttftOneTokenError)
                imeFirst.add(record)
                writeRaw(
                    rawFlowObject(style, "ime_first", "hist_$hist", round, prompt, flow)
                        .put("ime_first_candidate_ms", imeFirstCandidate)
                        .put("ttft_ms", ttftForReport)
                        .put("ttft_generate_ms", generateTtft)
                        .put("ttft_1token_ms", ttftOneTokenMs)
                        .put("ttft_1token_infer_ms", ttftOneTokenInferMs)
                        .put("ttft_1token_kv_load_ms", ttftOneTokenKvLoadMs)
                        .put("ttft_1token_context_prefill_ms", ttftOneTokenContextPrefillMs)
                        .put("ttft_1token_context_prefill_external_ms", ttftOneTokenContextPrefillExternalMs)
                        .put("ttft_1token_prefill_ms", ttftOneTokenPrefillMs)
                        .put("ttft_1token_decode_ms", ttftOneTokenDecodeMs)
                        .put("ttft_1token_e2e_ms", ttftOneTokenE2eMs)
                        .put("ttft_1token_error", ttftOneTokenError ?: JSONObject.NULL)
                        .put("kv_load_ms", kvLoadMs)
                        .put("context_prefill_ms", contextPrefillMs)
                        .put("context_prefill_external_ms", contextPrefillExternalMs)
                )
            } catch (t: Throwable) {
                if (measured) {
                    val err = "ime_first_failed style=$style hist=$hist iter=${round + 1} err=${t.message}"
                    errors.add(err)
                    val record = JSONObject()
                        .put("bucket", "hist_$hist")
                        .put("iteration", round)
                        .put("error", t.message ?: "ime_first_failed")
                    imeFirst.add(record)
                    writeRaw(JSONObject(record.toString()).put("tag", "ime_first").put("style_mode", style))
                }
            }
        }

        if (config.includeImeFirst) {
            val imeFirstHandle = createHandle(modelPath)
            if (imeFirstHandle == 0L) {
                errors.add("ime_first_create_handle_failed style=$style")
            } else {
                try {
                    repeat(warmupRounds) { round ->
                        for (hist in config.historyBuckets) {
                            step("$style warmup ime_first hist=$hist round=${round + 1}")
                            runImeFirst(hist = hist, round = round, measured = false, handle = imeFirstHandle)
                        }
                    }
                    repeat(contextRepeatFactor) { round ->
                        for (hist in config.historyBuckets) {
                            step("$style ime_first hist=$hist iter=${round + 1}")
                            runImeFirst(hist = hist, round = round, measured = true, handle = imeFirstHandle)
                        }
                    }
                } finally {
                    freeHandle(imeFirstHandle)
                }
            }
        }

        return JSONObject()
            .put("prefill_buckets", JSONArray(prefill))
            .put("prefill_scaling", JSONArray(prefill))
            .put("decode_buckets", JSONArray(decode))
            .put("ime_first", JSONArray(imeFirst))
    }

    private suspend fun runFlow(
        handle: Long,
        prompt: String,
        style: String,
        tag: String,
        bucket: String,
        iteration: Int,
        promptBuildMs: Long,
        flowStartNs: Long,
        instructionPrefix: String,
        preNativeOverheadMs: Long = 0,
        decodeStepBudget: Int? = null,
        decodeControlMode: DecodeControlMode = DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS,
        decodeNCandidates: Int = 4,
        prefillOnlyMode: Boolean = false,
        skipPostProcess: Boolean = false
    ): FlowRun {
        currentCoroutineContext().ensureActive()
        ensureRunning()
        val flowId = "flow_${flowSeq.incrementAndGet()}"

        val nativeStart = System.nanoTime()
        val native = runCatching {
            if (prefillOnlyMode) {
                prefillOnly(handle, prompt)
            } else if (decodeStepBudget != null) {
                when (decodeControlMode) {
                    DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS ->
                        benchmarkDecode(handle, prompt, decodeStepBudget)

                    DecodeControlMode.RUNTIME_GENERATE_CANDIDATES ->
                        generateForDecodeRuntime(handle, prompt, decodeNCandidates, preferGenerateCandidates = true)

                    DecodeControlMode.RUNTIME_GENERATE_PHRASE_CANDIDATES ->
                        generateForDecodeRuntime(handle, prompt, decodeNCandidates, preferGenerateCandidates = false)
                }
            } else {
                generate(handle, prompt, 5)
            }
        }.getOrElse { err ->
            val nativeWait = elapsedMs(nativeStart)
            val total = elapsedMs(flowStartNs)
            val lifecycle = FlowLifecycle(
                flowId,
                style,
                tag,
                bucket,
                iteration,
                promptBuildMs,
                preNativeOverheadMs,
                nativeWait,
                0,
                0,
                0,
                nativeWait,
                total,
                abs(total - (promptBuildMs + preNativeOverheadMs + nativeWait)),
                0,
                false,
                "generation_failed: ${err.message}"
            )
            javaTracker.recordGenerationSessionFromBenchmark("$tag:$bucket", prompt.take(256), style, -1, -1, -1, -1, -1, false, lifecycle.error)
            return FlowRun(lifecycle, null, emptyList(), "native_error", lifecycle.error)
        }
        val nativeWait = elapsedMs(nativeStart)
        val candidates: List<String>
        val postOutcome: String
        val postMs: Long
        if (skipPostProcess) {
            candidates = native.rawCandidates
            postOutcome = "benchmark_no_postprocess"
            postMs = 0L
        } else {
            val ppStart = System.nanoTime()
            val post = CandidatePostProcessor.process(
                native.rawCandidates.toTypedArray(),
                CandidatePostProcessor.CandidateContext(allowMemToolcall = true, preferCjk = true, instructionPrefix = instructionPrefix)
            )
            when (post) {
                is CandidatePostProcessor.CandidateOutcome.Show -> {
                    candidates = post.candidates
                    postOutcome = "show"
                }
                is CandidatePostProcessor.CandidateOutcome.MemRetrieval -> {
                    candidates = emptyList()
                    postOutcome = "mem_retrieval"
                }
                is CandidatePostProcessor.CandidateOutcome.DropAll -> {
                    candidates = emptyList()
                    postOutcome = "drop_all:${post.reason}"
                }
            }
            postMs = elapsedMs(ppStart)
        }
        val total = elapsedMs(flowStartNs)

        val prefill = native.metrics.optLong("prefill_ms", -1)
        val decode = native.metrics.optLong("decode_ms", -1)
        val p = if (prefill >= 0) prefill else 0L
        val d = if (decode >= 0) decode else 0L
        val gap = max(0L, nativeWait - p - d)
        val diffTotal = abs(total - (promptBuildMs + preNativeOverheadMs + nativeWait + postMs))
        val diffNative = abs(nativeWait - (p + d + gap))

        var err: String? = null
        if (prefill < 0 || decode < 0) err = "missing_native_metrics(prefill_ms/decode_ms)"
        if (err == null && diffTotal > lifecycleToleranceMs) err = "lifecycle_total_mismatch(diff=${diffTotal}ms,tolerance=${lifecycleToleranceMs}ms)"
        if (err == null && diffNative > lifecycleToleranceMs) err = "lifecycle_native_mismatch(diff=${diffNative}ms,tolerance=${lifecycleToleranceMs}ms)"

        val lifecycle = FlowLifecycle(flowId, style, tag, bucket, iteration, promptBuildMs, preNativeOverheadMs, nativeWait, postMs, p, d, gap, total, diffTotal, diffNative, err == null, err)
        javaTracker.recordGenerationSessionFromBenchmark(
            "$tag:$bucket",
            prompt.take(256),
            style,
            native.metrics.optLong("e2e_ms", -1),
            extractTtft(native.metrics) ?: -1,
            p,
            d,
            native.metrics.optLong("decode_tokens", -1).toInt(),
            err == null,
            err
        )
        return FlowRun(lifecycle, native.metrics, candidates, postOutcome, err)
    }

    private suspend fun generateWithBridge(
        handle: Long,
        prompt: String,
        nCandidates: Int,
        apiName: String,
        invoke: (Long, String, Int, LLMBridge.TokenCallback) -> Int,
    ): NativeResult = withContext(Dispatchers.IO) {
        val deferred = CompletableDeferred<NativeResult>()
        val lock = Any()
        val candidates = mutableListOf<String>()
        var metrics: JSONObject? = null

        fun fail(msg: String) {
            if (!deferred.isCompleted) deferred.completeExceptionally(RuntimeException(msg))
        }

        fun completeIfReady() {
            if (deferred.isCompleted) return
            val ready = synchronized(lock) {
                val m = metrics ?: return@synchronized null
                NativeResult(candidates.toList(), JSONObject(m.toString()))
            }
            if (ready != null) deferred.complete(ready)
        }

        val cb = object : LLMBridge.TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                synchronized(lock) {
                    for (t in tokens) {
                        if (t.startsWith("__METRICS__")) {
                            runCatching { JSONObject(t.removePrefix("__METRICS__")) }.getOrNull()?.let { metrics = it }
                        } else if (t.isNotBlank()) {
                            candidates.add(t)
                        }
                    }
                }
                completeIfReady()
            }

            override fun onFinished() {
                completeIfReady()
                if (!deferred.isCompleted) fail("${apiName}_finished_without_metrics")
            }

            override fun onError(err: String) {
                fail("${apiName}_native_error:$err")
            }
        }

        val rc = invoke(handle, prompt, nCandidates.coerceIn(1, 20), cb)
        if (rc != 0) throw RuntimeException("${apiName}_failed:$rc")
        try {
            withTimeout(flowTimeoutMs) { deferred.await() }
        } catch (_: TimeoutCancellationException) {
            runCatching { LLMBridge.stop(handle) }
            throw RuntimeException("${apiName}_timeout_${flowTimeoutMs}ms")
        }
    }

    private suspend fun generate(handle: Long, prompt: String, nCandidates: Int): NativeResult =
        generateWithBridge(
            handle = handle,
            prompt = prompt,
            nCandidates = nCandidates,
            apiName = "generatePhraseCandidates",
            invoke = { h, p, n, cb -> LLMBridge.generatePhraseCandidates(h, p, n, cb) }
        )

    private suspend fun generateCandidates(handle: Long, prompt: String, nCandidates: Int): NativeResult =
        generateWithBridge(
            handle = handle,
            prompt = prompt,
            nCandidates = nCandidates,
            apiName = "generateCandidates",
            invoke = { h, p, n, cb -> LLMBridge.generateCandidates(h, p, n, cb) }
        )

    private suspend fun generateForDecodeRuntime(
        handle: Long,
        prompt: String,
        nCandidates: Int,
        preferGenerateCandidates: Boolean,
    ): NativeResult {
        if (!preferGenerateCandidates) {
            val phrase = generate(handle, prompt, nCandidates)
            phrase.metrics.put("decode_api", DecodeControlMode.RUNTIME_GENERATE_PHRASE_CANDIDATES.wireValue)
            return phrase
        }

        return try {
            val direct = generateCandidates(handle, prompt, nCandidates)
            direct.metrics.put("decode_api", DecodeControlMode.RUNTIME_GENERATE_CANDIDATES.wireValue)
            direct
        } catch (firstErr: Throwable) {
            if (firstErr is CancellationException) throw firstErr
            val fallback = generate(handle, prompt, nCandidates)
            fallback.metrics.put("decode_api", "${DecodeControlMode.RUNTIME_GENERATE_CANDIDATES.wireValue}_fallback_to_phrase")
            fallback.metrics.put("decode_api_fallback_reason", firstErr.message ?: firstErr::class.java.simpleName)
            fallback
        }
    }

    private suspend fun prefillOnly(handle: Long, prompt: String): NativeResult = withContext(Dispatchers.IO) {
        val deferred = CompletableDeferred<NativeResult>()
        val lock = Any()
        val candidates = mutableListOf<String>()
        var metrics: JSONObject? = null

        fun fail(msg: String) {
            if (!deferred.isCompleted) deferred.completeExceptionally(RuntimeException(msg))
        }

        fun completeIfReady() {
            if (deferred.isCompleted) return
            val ready = synchronized(lock) {
                val m = metrics ?: return@synchronized null
                NativeResult(candidates.toList(), JSONObject(m.toString()))
            }
            if (ready != null) deferred.complete(ready)
        }

        val cb = object : LLMBridge.TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                synchronized(lock) {
                    for (t in tokens) {
                        if (t.startsWith("__METRICS__")) {
                            runCatching { JSONObject(t.removePrefix("__METRICS__")) }.getOrNull()?.let { metrics = it }
                        } else if (t.isNotBlank()) {
                            candidates.add(t)
                        }
                    }
                }
                completeIfReady()
            }

            override fun onFinished() {
                completeIfReady()
                if (!deferred.isCompleted) fail("prefill_finished_without_metrics")
            }

            override fun onError(err: String) {
                fail("native_error:$err")
            }
        }

        val rc = LLMBridge.prefillPrompt(handle, prompt, cb)
        if (rc != 0) throw RuntimeException("prefillPrompt_failed:$rc")
        try {
            withTimeout(flowTimeoutMs) { deferred.await() }
        } catch (_: TimeoutCancellationException) {
            runCatching { LLMBridge.stop(handle) }
            throw RuntimeException("prefill_timeout_${flowTimeoutMs}ms")
        }
    }

    private suspend fun generateSpliceMemory(
        handle: Long,
        prefixBeforeMemory: String,
        memory: String,
        suffixAfterMemory: String,
        nCandidates: Int
    ): NativeResult = withContext(Dispatchers.IO) {
        val deferred = CompletableDeferred<NativeResult>()
        val lock = Any()
        val candidates = mutableListOf<String>()
        var metrics: JSONObject? = null

        fun fail(msg: String) {
            if (!deferred.isCompleted) deferred.completeExceptionally(RuntimeException(msg))
        }

        fun completeIfReady() {
            if (deferred.isCompleted) return
            val ready = synchronized(lock) {
                val m = metrics ?: return@synchronized null
                NativeResult(candidates.toList(), JSONObject(m.toString()))
            }
            if (ready != null) deferred.complete(ready)
        }

        val cb = object : LLMBridge.TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                synchronized(lock) {
                    for (t in tokens) {
                        if (t.startsWith("__METRICS__")) {
                            runCatching { JSONObject(t.removePrefix("__METRICS__")) }.getOrNull()?.let { metrics = it }
                        } else if (t.isNotBlank()) {
                            candidates.add(t)
                        }
                    }
                }
                completeIfReady()
            }

            override fun onFinished() {
                completeIfReady()
                if (!deferred.isCompleted) fail("splice_generation_finished_without_metrics")
            }

            override fun onError(err: String) {
                fail("native_error:$err")
            }
        }

        val rc = LLMBridge.generatePhraseCandidatesSpliceMemory(
            handle,
            prefixBeforeMemory,
            memory,
            suffixAfterMemory,
            nCandidates.coerceIn(1, 20),
            cb
        )
        if (rc != 0) throw RuntimeException("generatePhraseCandidatesSpliceMemory_failed:$rc")
        try {
            withTimeout(flowTimeoutMs) { deferred.await() }
        } catch (_: TimeoutCancellationException) {
            runCatching { LLMBridge.stop(handle) }
            throw RuntimeException("splice_generation_timeout_${flowTimeoutMs}ms")
        }
    }

    private suspend fun benchmarkDecode(handle: Long, prompt: String, maxDecodeSteps: Int): NativeResult = withContext(Dispatchers.IO) {
        val deferred = CompletableDeferred<NativeResult>()
        val lock = Any()
        val candidates = mutableListOf<String>()
        var metrics: JSONObject? = null

        fun fail(msg: String) {
            if (!deferred.isCompleted) deferred.completeExceptionally(RuntimeException(msg))
        }

        fun completeIfReady() {
            if (deferred.isCompleted) return
            val ready = synchronized(lock) {
                val m = metrics ?: return@synchronized null
                NativeResult(candidates.toList(), JSONObject(m.toString()))
            }
            if (ready != null) deferred.complete(ready)
        }

        val cb = object : LLMBridge.TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                synchronized(lock) {
                    for (t in tokens) {
                        if (t.startsWith("__METRICS__")) {
                            runCatching { JSONObject(t.removePrefix("__METRICS__")) }.getOrNull()?.let { metrics = it }
                        } else if (t.isNotBlank()) {
                            candidates.add(t)
                        }
                    }
                }
                completeIfReady()
            }

            override fun onFinished() {
                completeIfReady()
                if (!deferred.isCompleted) fail("benchmark_finished_without_metrics")
            }

            override fun onError(err: String) {
                fail("native_error:$err")
            }
        }

        val rc = LLMBridge.benchmarkDecode(handle, prompt, maxDecodeSteps.coerceAtLeast(0), cb)
        if (rc != 0) throw RuntimeException("benchmarkDecode_failed:$rc")
        try {
            withTimeout(flowTimeoutMs) { deferred.await() }
        } catch (_: TimeoutCancellationException) {
            runCatching { LLMBridge.stop(handle) }
            throw RuntimeException("benchmark_timeout_${flowTimeoutMs}ms")
        }
    }

    private fun rawFlowObject(style: String, tag: String, bucket: String, iteration: Int, prompt: String, flow: FlowRun): JSONObject {
        val obj = JSONObject()
        obj.put("tag", tag)
        obj.put("bucket", bucket)
        obj.put("style_mode", style)
        obj.put("iteration", iteration)
        obj.put("prompt_length_chars", prompt.length)
        obj.put("prompt_preview", prompt.take(256))
        obj.put("postprocess_outcome", flow.postprocessOutcome)
        obj.put("candidates", JSONArray(flow.candidates))
        obj.put("lifecycle", flow.lifecycle.toJson())
        flow.metrics?.keys()?.forEach { k -> obj.put(k, flow.metrics.get(k)) }
        if (!flow.error.isNullOrBlank()) obj.put("error", flow.error)
        return obj
    }

    private fun assessDecodeSampleValidity(
        metrics: JSONObject?,
        stepBudget: Int,
        tokenRatio: Double,
        minBudget: Int,
    ): DecodeSampleValidity {
        if (metrics == null) {
            return DecodeSampleValidity(
                isInvalid = false,
                reason = null,
                budget = stepBudget.takeIf { it > 0 },
                tokens = null,
                threshold = null
            )
        }
        val budget = stepBudget.takeIf { it > 0 }
        val tokensRaw = metrics.optDouble("decode_tokens", Double.NaN)
        val tokens = tokensRaw.takeIf { it.isFinite() && it >= 0.0 }
        if (budget == null || budget < max(1, minBudget) || tokens == null) {
            return DecodeSampleValidity(
                isInvalid = false,
                reason = null,
                budget = budget,
                tokens = tokens,
                threshold = if (budget != null) budget.toDouble() * tokenRatio else null
            )
        }
        val threshold = budget.toDouble() * tokenRatio
        val invalid = tokens < threshold
        val reason = if (invalid) {
            "decode_tokens(${String.format(Locale.US, "%.1f", tokens)}) < " +
                "${String.format(Locale.US, "%.2f", tokenRatio)}*step_budget(${budget})"
        } else {
            null
        }
        return DecodeSampleValidity(
            isInvalid = invalid,
            reason = reason,
            budget = budget,
            tokens = tokens,
            threshold = threshold
        )
    }

    private fun summarizeDecodeBucket(
        records: List<JSONObject>,
        steps: Int,
        tokenRatio: Double,
        minBudget: Int,
    ): JSONObject {
        val decodeMs = records.mapNotNull { it.optLong("decode_ms", -1).takeIf { v -> v >= 0 } }
        val decodeTokens = records.mapNotNull { it.optLong("decode_tokens", -1).takeIf { v -> v >= 0 } }
        val decodeTps = records.mapNotNull {
            val v = it.optDouble("decode_tps", Double.NaN)
            if (v.isFinite() && v >= 0.0) v else null
        }
        val perTokenLatencyMs = records.mapNotNull {
            val ms = it.optDouble("decode_ms", Double.NaN)
            val tokens = it.optDouble("decode_tokens", Double.NaN)
            if (ms.isFinite() && tokens.isFinite() && ms >= 0.0 && tokens > 0.0) ms / tokens else null
        }
        val invalidCount = records.count { rec ->
            if (rec.has("decode_sample_invalid")) rec.optBoolean("decode_sample_invalid", false)
            else assessDecodeSampleValidity(rec, steps, tokenRatio, minBudget).isInvalid
        }
        val fallbackCount = records.count { rec ->
            rec.optString("decode_api", "").contains("fallback", ignoreCase = true)
        }
        val validCount = max(0, records.size - invalidCount)
        return JSONObject()
            .put("bucket", "steps_$steps")
            .put("decode_ms_p50", percentile(decodeMs, 0.50) ?: JSONObject.NULL)
            .put("decode_tokens_p50", percentile(decodeTokens, 0.50) ?: JSONObject.NULL)
            .put("decode_tps_p50", percentileDouble(decodeTps, 0.50) ?: JSONObject.NULL)
            .put("decode_ms_per_token_p50", percentileDouble(perTokenLatencyMs, 0.50) ?: JSONObject.NULL)
            .put("samples", records.size)
            .put("valid_samples", validCount)
            .put("invalid_samples", invalidCount)
            .put("invalid_sample_rate", if (records.isNotEmpty()) invalidCount.toDouble() / records.size else 0.0)
            .put("decode_api_fallback_samples", fallbackCount)
            .put("invalid_rule", "decode_tokens < ${String.format(Locale.US, "%.2f", tokenRatio)} * step_budget (step_budget >= ${max(1, minBudget)})")
    }

    private fun buildLifecycleBreakdown(flows: List<FlowLifecycle>): JSONObject {
        val nonfatalSkipped = flows.filter { isNonFatalGenerationWarning(it.error) }
        val strictChecked = flows.filterNot { isNonFatalGenerationWarning(it.error) }
        val failed = strictChecked.filter { !it.passed }
        return JSONObject()
            .put("tolerance_ms", lifecycleToleranceMs)
            .put("total_flows", flows.size)
            .put("strict_checked_flows", strictChecked.size)
            .put("nonfatal_skipped_flows", nonfatalSkipped.size)
            .put("passed_flows", strictChecked.size - failed.size)
            .put("failed_flows", failed.size)
            .put("strict_validation_passed", failed.isEmpty())
            .put("max_total_equation_diff_ms", flows.maxOfOrNull { it.eqTotalDiffMs } ?: 0)
            .put("max_native_equation_diff_ms", flows.maxOfOrNull { it.eqNativeDiffMs } ?: 0)
            .put("flow_metrics_file", "lifecycle_flow_metrics.jsonl")
            .put("failed_flow_ids", JSONArray(failed.map { it.flowId }))
    }

    private fun buildLifecycleFallback(flows: List<FlowLifecycle>): JSONObject {
        fun avg(values: List<Long>): Double = if (values.isEmpty()) 0.0 else values.sum().toDouble() / values.size
        return JSONObject()
            .put("prompt_build_ms", avg(flows.map { it.promptBuildMs }))
            .put("pre_native_overhead_ms", avg(flows.map { it.preNativeOverheadMs }))
            .put("native_wait_ms", avg(flows.map { it.nativeWaitMs }))
            .put("postprocess_ms", avg(flows.map { it.postprocessMs }))
            .put("native_prefill_ms", avg(flows.map { it.nativePrefillMs }))
            .put("native_decode_ms", avg(flows.map { it.nativeDecodeMs }))
            .put("native_gap_ms", avg(flows.map { it.nativeGapMs }))
            .put("total_flow_ms", avg(flows.map { it.totalFlowMs }))
    }

    private fun buildStatistics(values: List<Long>, durationMs: Long): TestStatistics {
        if (values.isEmpty()) return TestStatistics(0.0, 0, 0, 0, 0, 0, 0, 0.0)
        val sorted = values.sorted()
        val avg = sorted.sum().toDouble() / sorted.size
        val throughput = if (durationMs > 0) sorted.size * 1000.0 / durationMs else 0.0
        return TestStatistics(
            avgTimeMs = avg,
            minTimeMs = sorted.first(),
            maxTimeMs = sorted.last(),
            p50 = percentile(sorted, 0.50) ?: 0,
            p90 = percentile(sorted, 0.90) ?: 0,
            p95 = percentile(sorted, 0.95) ?: 0,
            p99 = percentile(sorted, 0.99) ?: 0,
            throughput = throughput
        )
    }

    private fun percentile(values: List<Long>, q: Double): Long? {
        if (values.isEmpty()) return null
        val sorted = values.sorted()
        val idx = ((sorted.size - 1) * q).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    private fun percentileDouble(values: List<Double>, q: Double): Double? {
        if (values.isEmpty()) return null
        val sorted = values.sorted()
        val idx = ((sorted.size - 1) * q).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    private fun extractTtft(metrics: JSONObject?): Long? {
        if (metrics == null) return null
        val ttft = metrics.optLong("ttft_ms", -1)
        if (ttft >= 0) return ttft
        val prefill = metrics.optLong("prefill_ms", -1)
        val lat = metrics.optJSONArray("decode_latencies")
        if (prefill >= 0 && lat != null && lat.length() > 0) {
            val d0 = lat.optDouble(0, -1.0)
            if (d0 >= 0.0) return prefill + d0.toLong()
        }
        return null
    }

    private fun writePlotScript(file: File) {
        val content = runCatching {
            context.assets.open("performance/plot_metrics_template.py").bufferedReader(Charsets.UTF_8).use { it.readText() }
        }.getOrElse {
            """
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            raise SystemExit("plot_metrics template missing from assets/performance/plot_metrics_template.py")
            """.trimIndent()
        }
        file.writeText(content.trimEnd() + "\n", Charsets.UTF_8)
        runCatching { file.setExecutable(true, false) }
    }

    private fun isLifecycleWarning(error: String?): Boolean {
        if (error.isNullOrBlank()) return false
        return error.startsWith("lifecycle_total_mismatch(") || error.startsWith("lifecycle_native_mismatch(")
    }

    private fun isNonFatalGenerationWarning(error: String?): Boolean {
        if (error.isNullOrBlank()) return false
        return error == "generation_failed: generation_finished_without_metrics"
    }

    private fun tokenizeCountForPrefillTest(handle: Long, text: String): Int {
        if (handle == 0L) return -1
        return runCatching { LLMBridge.tokenize(handle, text)?.size ?: -1 }.getOrDefault(-1)
    }

    private fun buildPrefillPromptByTokenBudget(handle: Long, targetTokens: Int): Pair<String, Int> {
        if (targetTokens <= 0) {
            val emptyCount = tokenizeCountForPrefillTest(handle, "")
            if (emptyCount > 0) return "" to emptyCount
            val fallbackPrompt = " 0"
            return fallbackPrompt to max(0, tokenizeCountForPrefillTest(handle, fallbackPrompt))
        }

        val prompt = StringBuilder()
        var bestCount = 0
        val maxPromptChars = max(1024, targetTokens * 10 + 256)
        var i = 0
        var guard = 0
        while (bestCount < targetTokens && prompt.length < maxPromptChars && guard < targetTokens * 64) {
            val prevLen = prompt.length
            val prevCount = bestCount

            prompt.append(" t").append(i)
            if (i % 8 == 0) prompt.append('\n')

            val curCount = tokenizeCountForPrefillTest(handle, prompt.toString())
            if (curCount < 0) {
                prompt.setLength(prevLen)
                break
            }
            if (curCount <= targetTokens) {
                bestCount = curCount
            } else {
                var lo = prevLen
                var hi = prompt.length
                var bestLen = prevLen
                var localBest = prevCount
                while (lo <= hi) {
                    val mid = (lo + hi) ushr 1
                    val candidate = prompt.substring(0, mid)
                    val c = tokenizeCountForPrefillTest(handle, candidate)
                    if (c >= 0 && c <= targetTokens) {
                        bestLen = mid
                        localBest = c
                        lo = mid + 1
                    } else {
                        hi = mid - 1
                    }
                }
                prompt.setLength(bestLen)
                bestCount = localBest
                break
            }
            i++
            guard++
        }
        if (bestCount >= targetTokens) return prompt.toString() to bestCount

        val suffixPool = listOf(
            " a", " b", " c", " d", " e", " f", " g", " h", " i", " j", " k", " l",
            " m", " n", " o", " p", " q", " r", " s", " t",
            " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "\n"
        )
        var count = bestCount
        var fineTuneGuard = 0
        while (count < targetTokens && prompt.length < maxPromptChars && fineTuneGuard < 1024) {
            var improved = false
            for (suffix in suffixPool) {
                val candidate = prompt.toString() + suffix
                val c = tokenizeCountForPrefillTest(handle, candidate)
                if (c in (count + 1)..targetTokens) {
                    prompt.append(suffix)
                    count = c
                    improved = true
                    if (count >= targetTokens) break
                }
            }
            if (!improved) break
            fineTuneGuard++
        }
        return prompt.toString() to count
    }

    private fun getSystemPrompt(style: String): String = StyleConfig.PROMPTS[style] ?: ""

    private data class ImeContextPromptSegments(
        val prefillBase: String,
        val decodeTail: String,
    ) {
        val fullPrompt: String
            get() = prefillBase + decodeTail
    }

    private fun buildPrompt(chars: Int, style: String): String {
        val seed = "[me]: keep tests close to real ime flow\n[peer]: lifecycle checks must be strict\n"
        val history = buildString { while (length < chars) append(seed) }.take(chars)
        val last = when (style) {
            StyleConfig.STYLE_BUSINESS -> "please start with a clear conclusion"
            StyleConfig.STYLE_WARM -> "please answer softly and briefly"
            else -> "be short and practical"
        }
        return buildChatMlPrompt(getSystemPrompt(style), history, last, "none", "")
    }

    private fun formatSystemPromptForSession(systemPrompt: String): String {
        return "<|im_start|>system\n$systemPrompt<|im_end|>\n"
    }

    private fun getStyleBaselineCachePath(style: String): String {
        val cacheFileName = StyleConfig.CACHE_FILES[style] ?: "kv_cache_default.bin"
        return File(context.filesDir, cacheFileName).absolutePath
    }

    private fun ensureImeFirstBaselineLoaded(handle: Long, style: String): Boolean {
        if (handle == 0L) return false
        val systemPrompt = getSystemPrompt(style)
        val formattedSystemPrompt = formatSystemPromptForSession(systemPrompt)
        val cachePath = getStyleBaselineCachePath(style)
        val cacheFile = File(cachePath)

        runCatching { LLMBridge.clearKvKeepSystem(handle) }

        var loaded = false
        if (cacheFile.exists() && cacheFile.length() > 0L) {
            loaded = runCatching { LLMBridge.loadSession(handle, cachePath) }.getOrDefault(false)
        }
        if (!loaded) {
            val saved = runCatching {
                LLMBridge.saveKVCacheSnapshot(handle, formattedSystemPrompt, cachePath)
            }.getOrDefault(false)
            if (saved) {
                loaded = runCatching { LLMBridge.loadSession(handle, cachePath) }.getOrDefault(false)
            }
        }
        if (loaded) {
            val reusablePrefix = "${formattedSystemPrompt}<|im_start|>user\n"
            val reusablePrefixTokens = runCatching { LLMBridge.tokenize(handle, reusablePrefix)?.size ?: 0 }.getOrDefault(0)
            if (reusablePrefixTokens > 0) {
                runCatching { LLMBridge.setReusablePrefixTokenCount(handle, reusablePrefixTokens) }
            }
        }
        return loaded
    }

    private fun buildContextSyncPromptSegments(historyChars: Int, lastMsg: String, prefix: String, style: String): ImeContextPromptSegments {
        val history = buildString { while (length < historyChars) append("[me]: history sample\n[peer]: acknowledged\n") }.take(historyChars)
        val historyStr = if (history.isNotBlank()) history else "none"
        val lastMsgStr = if (lastMsg.startsWith("[peer]:")) lastMsg else "[peer]: $lastMsg"
        val reusablePrefix = "${formatSystemPromptForSession(getSystemPrompt(style))}<|im_start|>user\n"
        val userBase = "<env>\n<history>\n$historyStr\n</history>\n<last_msg>\n$lastMsgStr\n</last_msg>\n<memory>\nnone\n</memory></env>\n<instruction>\nComplete my reply based on memory and last message:\n"
        val userTail = "$prefix\n</instruction>"
        val assistantStart = if (prefix.isNotBlank()) "<think>\n\n</think>\n\n$prefix" else "<think>\n\n</think>\n\n"
        return ImeContextPromptSegments(
            prefillBase = reusablePrefix + userBase,
            decodeTail = "$userTail<|im_end|>\n<|im_start|>assistant\n$assistantStart"
        )
    }

    private fun buildContextSyncPrompt(historyChars: Int, lastMsg: String, prefix: String, style: String): String {
        return buildContextSyncPromptSegments(historyChars, lastMsg, prefix, style).fullPrompt
    }

    private fun buildChatMlPrompt(systemPrompt: String, history: String, lastMsg: String, memory: String, prefix: String): String {
        val historyStr = if (history.isNotBlank()) history else "none"
        val lastMsgStr = if (lastMsg.startsWith("[peer]:")) lastMsg else "[peer]: $lastMsg"
        val memoryStr = if (memory.isNotBlank()) memory else "none"
        val userBase = "<env>\n<history>\n$historyStr\n</history>\n<last_msg>\n$lastMsgStr\n</last_msg>\n<memory>\n$memoryStr\n</memory></env>\n<instruction>\nComplete my reply based on memory and last message:\n"
        val userTail = "$prefix\n</instruction>"
        val assistantStart = if (prefix.isNotBlank()) "<think>\n\n</think>\n\n$prefix" else "<think>\n\n</think>\n\n"
        return "<|im_start|>system\n$systemPrompt<|im_end|>\n<|im_start|>user\n$userBase$userTail<|im_end|>\n<|im_start|>assistant\n$assistantStart"
    }

    private fun createHandle(modelPath: String): Long {
        val threads = min(8, max(1, Runtime.getRuntime().availableProcessors()))
        return LLMBridge.createGenerationInstance(modelPath, "", threads, 999)
    }

    private fun createBenchmarkHandle(modelPath: String): Long {
        val threads = min(8, max(1, Runtime.getRuntime().availableProcessors()))
        val benchmark = LLMBridge.createBenchmarkInstance(modelPath, "", threads, 999)
        return if (benchmark != 0L) benchmark else LLMBridge.createGenerationInstance(modelPath, "", threads, 999)
    }

    private fun freeHandle(handle: Long) {
        runCatching { if (handle != 0L) LLMBridge.freeModel(handle) }
    }

    private fun ensureRunning() {
        if (!running.get()) throw CancellationException("test_stopped")
    }

    private fun elapsedMs(startNs: Long): Long = ((System.nanoTime() - startNs) / 1_000_000L).coerceAtLeast(0L)

    private fun readPssKb(): Int = try {
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        mi.totalPss
    } catch (_: Exception) {
        0
    }

    private fun detectModelVariant(modelPath: String): String {
        val name = File(modelPath).name.lowercase(Locale.US)
        return when {
            "q4_0" in name -> "Q4_0"
            "q4_1" in name -> "Q4_1"
            "q5_0" in name -> "Q5_0"
            "q5_1" in name -> "Q5_1"
            "q8_0" in name -> "Q8_0"
            "bf16" in name -> "BF16"
            "f16" in name -> "F16"
            "f32" in name -> "F32"
            else -> "UNKNOWN"
        }
    }

    private fun buildKvCacheSummary(values: List<Long>): JSONObject {
        val valid = values.filter { it > 0L }
        if (valid.isEmpty()) {
            return JSONObject()
                .put("samples", 0)
                .put("min_bytes", JSONObject.NULL)
                .put("avg_bytes", JSONObject.NULL)
                .put("max_bytes", JSONObject.NULL)
        }
        return JSONObject()
            .put("samples", valid.size)
            .put("min_bytes", valid.minOrNull() ?: JSONObject.NULL)
            .put("avg_bytes", valid.average())
            .put("max_bytes", valid.maxOrNull() ?: JSONObject.NULL)
    }

    private fun buildCpuUsageSummary(samples: List<Double>): JSONObject {
        val valid = samples.filter { it.isFinite() && it >= 0.0 }
        if (valid.isEmpty()) {
            return JSONObject()
                .put("sample_count", 0)
                .put("avg_percent", JSONObject.NULL)
                .put("peak_percent", JSONObject.NULL)
        }
        return JSONObject()
            .put("sample_count", valid.size)
            .put("avg_percent", valid.average())
            .put("peak_percent", valid.maxOrNull() ?: JSONObject.NULL)
    }

    private data class CpuTicks(val totalJiffies: Long, val processJiffies: Long)

    private class CpuUsageSampler(private val pid: Int) {
        private var last: CpuTicks? = null

        fun sample(): Double? {
            val now = readCpuTicks(pid) ?: return null
            val prev = last
            last = now
            if (prev == null) return null
            val totalDelta = now.totalJiffies - prev.totalJiffies
            val processDelta = now.processJiffies - prev.processJiffies
            if (totalDelta <= 0L || processDelta < 0L) return null
            return (processDelta.toDouble() * 100.0 / totalDelta.toDouble()).coerceIn(0.0, 100.0)
        }

        private fun readCpuTicks(pid: Int): CpuTicks? {
            val total = readTotalJiffies() ?: return null
            val process = readProcessJiffies(pid) ?: return null
            return CpuTicks(total, process)
        }

        private fun readTotalJiffies(): Long? {
            val cpuLine = runCatching {
                File("/proc/stat").useLines { seq -> seq.firstOrNull { it.startsWith("cpu ") } }
            }.getOrNull() ?: return null
            val parts = cpuLine.trim().split(Regex("\\s+"))
            if (parts.size <= 1) return null
            var total = 0L
            for (i in 1 until parts.size) {
                total += parts[i].toLongOrNull() ?: return null
            }
            return total
        }

        private fun readProcessJiffies(pid: Int): Long? {
            val line = runCatching { File("/proc/$pid/stat").readText(Charsets.UTF_8) }.getOrNull()?.trim() ?: return null
            val rParen = line.lastIndexOf(')')
            if (rParen <= 0 || rParen + 2 >= line.length) return null
            val tail = line.substring(rParen + 2).trim()
            val fields = tail.split(Regex("\\s+"))
            if (fields.size <= 12) return null
            val utime = fields[11].toLongOrNull() ?: return null
            val stime = fields[12].toLongOrNull() ?: return null
            return utime + stime
        }
    }

    fun getTestStatus(): TestStatus {
        return TestStatus(
            isRunning = running.get(),
            currentTest = currentTest.get(),
            statistics = javaTracker.getStatistics(),
            nativeSessionCount = runCatching { NativePerformanceTracker.nativeGetSessionCount() }.getOrDefault(0)
        )
    }

    fun stopCurrentTest() {
        if (running.get()) {
            running.set(false)
            LogUtil.w(tag, "Test", "Current test stopped by user")
        }
    }
}
