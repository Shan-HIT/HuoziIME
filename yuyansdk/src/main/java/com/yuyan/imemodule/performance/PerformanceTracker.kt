package com.yuyan.imemodule.performance

import android.content.Context
import android.os.Environment
import com.yuyan.imemodule.utils.LogUtil
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class PerformanceTracker(private val context: Context) {
    private val TAG = "PerformanceTracker"

    private val currentSessionId = AtomicLong(0)
    private val sessionIdMap = ConcurrentHashMap<Long, SessionInfo>()

    private val metricsQueue = ConcurrentLinkedQueue<PerformanceMetric>()
    private val activeSessions = ConcurrentHashMap<Long, ActiveSessionState>()

    private val isExporting = AtomicBoolean(false)

    private val totalMetricsRecorded = AtomicLong(0)
    private val metricsByType = ConcurrentHashMap<String, AtomicLong>()

    var maxCacheSize = 10000
    var autoExportEnabled = false
    var autoExportThreshold = 1000

    data class SessionInfo(
        val sessionId: Long,
        val startTime: Long,
        val mode: String,
        val prompt: String,
        val metadata: Map<String, Any> = emptyMap()
    )

    data class ActiveSessionState(
        val sessionId: Long,
        val startTime: Long,
        val mode: String,
        val prompt: String,
        var prefillTime: Long = 0,
        var decodeStartTime: Long = 0,
        var decodeSteps: Int = 0,
        var totalDecodeTime: Long = 0,
        var candidates: MutableList<String> = mutableListOf(),
        var success: Boolean = true,
        var errorMessage: String? = null,
        val additionalMetrics: MutableMap<String, Any> = mutableMapOf()
    )

    fun beginSession(mode: String, prompt: String, metadata: Map<String, Any> = emptyMap()): Long {
        val sessionId = currentSessionId.incrementAndGet()
        val now = System.currentTimeMillis()

        sessionIdMap[sessionId] = SessionInfo(sessionId, now, mode, prompt, metadata)
        activeSessions[sessionId] = ActiveSessionState(sessionId, now, mode, prompt)

        LogUtil.d(TAG, "Session", "开始新会话: $sessionId | 模式: $mode")
        return sessionId
    }

    fun recordPrefill(
        sessionId: Long,
        promptLength: Int,
        tokenCount: Int,
        reuseTokenCount: Int,
        timeMs: Long,
        kvCacheSizeBefore: Long = 0,
        kvCacheSizeAfter: Long = 0,
        modelName: String = "unknown"
    ) {
        val rate = if (timeMs > 0) (tokenCount - reuseTokenCount).toDouble() * 1000.0 / timeMs else 0.0

        val metric = PrefillMetric(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId.toString(),
            promptLength = promptLength,
            tokenCount = tokenCount,
            reuseTokenCount = reuseTokenCount,
            timeMs = timeMs,
            rateTokensPerSec = rate,
            kvCacheSizeBefore = kvCacheSizeBefore,
            kvCacheSizeAfter = kvCacheSizeAfter,
            modelName = modelName
        )

        addMetric(metric)

        activeSessions[sessionId]?.apply {
            prefillTime = timeMs
            additionalMetrics["prompt_length"] = promptLength
            additionalMetrics["token_count"] = tokenCount
            additionalMetrics["reuse_token_count"] = reuseTokenCount
        }

        LogUtil.d(TAG, "Prefill", "会话 $sessionId | 时间: ${timeMs}ms | 速率: ${"%.2f".format(rate)} T/s")
    }

    fun recordDecodeStep(
        sessionId: Long,
        stepNumber: Int,
        tokenGenerated: String,
        tokenId: Int,
        timeMs: Long,
        kvCacheHit: Boolean = false,
        candidateBranchCount: Int = 1
    ) {
        val session = activeSessions[sessionId] ?: return

        if (session.decodeSteps == 0) {
            session.decodeStartTime = System.currentTimeMillis()
        }

        session.decodeSteps++
        session.totalDecodeTime += timeMs

        val cumulativeTime = session.totalDecodeTime
        val tokensPerSec = if (cumulativeTime > 0) session.decodeSteps.toDouble() * 1000.0 / cumulativeTime else 0.0

        val metric = DecodeMetric(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId.toString(),
            stepNumber = stepNumber,
            tokenGenerated = tokenGenerated,
            tokenId = tokenId,
            timeMs = timeMs,
            cumulativeTimeMs = cumulativeTime,
            tokensPerSec = tokensPerSec,
            kvCacheHit = kvCacheHit,
            candidateBranchCount = candidateBranchCount
        )

        addMetric(metric)

        LogUtil.d(TAG, "Decode", "会话 $sessionId | 步骤 $stepNumber | Token: $tokenGenerated | 时间: ${timeMs}ms")
    }

    fun completeSession(
        sessionId: Long,
        candidates: List<String>,
        success: Boolean = true,
        errorMessage: String? = null
    ) {
        val session = activeSessions.remove(sessionId) ?: return

        session.success = success
        session.errorMessage = errorMessage
        session.candidates.addAll(candidates)

        val totalTime = System.currentTimeMillis() - session.startTime
        val firstTokenLatency = if (session.decodeStartTime > 0) session.decodeStartTime - session.startTime else totalTime

        val metric = GenerationSessionMetric(
            timestamp = session.startTime,
            sessionId = sessionId.toString(),
            mode = session.mode,
            prompt = session.prompt,
            generatedCandidates = session.candidates,
            totalTimeMs = totalTime,
            firstTokenLatencyMs = firstTokenLatency,
            prefillTimeMs = session.prefillTime,
            decodeTimeMs = session.totalDecodeTime,
            totalTokensGenerated = session.decodeSteps,
            avgTokensPerSec = if (session.totalDecodeTime > 0) session.decodeSteps.toDouble() * 1000.0 / session.totalDecodeTime else 0.0,
            styleMode = session.additionalMetrics["style_mode"] as? String ?: "unknown",
            success = success,
            errorMessage = errorMessage
        )

        addMetric(metric)

        val resultLabel = if (success) "success" else "failed"
        LogUtil.i(
            TAG,
            "SessionComplete",
            "Session=$sessionId total=${totalTime}ms ttft=${firstTokenLatency}ms " +
                "prefill=${session.prefillTime}ms decode=${session.totalDecodeTime}ms " +
                "steps=${session.decodeSteps} candidates=${candidates.size} result=$resultLabel"
        )

        if (autoExportEnabled && totalMetricsRecorded.get() % autoExportThreshold == 0L) {
            CoroutineScope(Dispatchers.IO).launch {
                exportToSdCard()
            }
        }
    }

    fun recordGenerationSessionFromBenchmark(
        mode: String,
        prompt: String,
        styleMode: String,
        e2eTimeMs: Long,
        ttftMs: Long,
        prefillTimeMs: Long,
        decodeTimeMs: Long,
        decodeTokens: Int,
        success: Boolean,
        errorMessage: String? = null
    ) {
        val sessionId = currentSessionId.incrementAndGet()
        val avgTps = if (decodeTimeMs > 0) decodeTokens.toDouble() * 1000.0 / decodeTimeMs.toDouble() else 0.0

        val metric = GenerationSessionMetric(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId.toString(),
            mode = mode,
            prompt = prompt,
            generatedCandidates = emptyList(),
            totalTimeMs = e2eTimeMs.coerceAtLeast(0),
            firstTokenLatencyMs = ttftMs.coerceAtLeast(0),
            prefillTimeMs = prefillTimeMs.coerceAtLeast(0),
            decodeTimeMs = decodeTimeMs.coerceAtLeast(0),
            totalTokensGenerated = decodeTokens.coerceAtLeast(0),
            avgTokensPerSec = avgTps,
            styleMode = styleMode,
            success = success,
            errorMessage = errorMessage
        )

        addMetric(metric)
    }

    fun recordContextSync(
        syncType: String,
        historyLength: Int,
        lastMsgLength: Int,
        prefillTimeMs: Long,
        sessionLoadTimeMs: Long = 0,
        success: Boolean = true
    ) {
        val sessionId = currentSessionId.incrementAndGet()
        val hasInvalidTiming = prefillTimeMs < 0 || sessionLoadTimeMs < 0
        val safePrefillMs = prefillTimeMs.coerceAtLeast(0)
        val safeLoadMs = sessionLoadTimeMs.coerceAtLeast(0)
        val totalTime = safePrefillMs + safeLoadMs
        val safeSuccess = success && !hasInvalidTiming

        val metric = ContextSyncMetric(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId.toString(),
            syncType = syncType,
            historyLength = historyLength,
            lastMsgLength = lastMsgLength,
            prefillTimeMs = safePrefillMs,
            sessionLoadTimeMs = safeLoadMs,
            totalTimeMs = totalTime,
            success = safeSuccess
        )

        addMetric(metric)

        if (hasInvalidTiming) {
            LogUtil.w(TAG, "ContextSync", "上下文同步(无效计时) | 类型: $syncType | prefill=${prefillTimeMs}ms | load=${sessionLoadTimeMs}ms")
        }
        LogUtil.i(TAG, "ContextSync", "上下文同步 | 类型: $syncType | 耗时: ${totalTime}ms")
    }

    private fun addMetric(metric: PerformanceMetric) {
        metricsQueue.offer(metric)
        val totalCount = totalMetricsRecorded.incrementAndGet()

        metricsByType.getOrPut(metric.operationType) { AtomicLong(0) }.incrementAndGet()

        if (metricsQueue.size > maxCacheSize) {
            LogUtil.w(TAG, "Cache", "Metric cache is near capacity (${metricsQueue.size}/$maxCacheSize)")
        }
    }

    suspend fun exportToJson(): ExportResult = withContext(Dispatchers.IO) {
        if (!isExporting.compareAndSet(false, true)) {
            return@withContext ExportResult(false, 0, "Export already in progress")
        }

        try {
            val startTime = System.currentTimeMillis()
            val exportDir = getExportDirectory()
            if (!exportDir.exists()) exportDir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "performance_metrics_$timestamp.json"
            val file = File(exportDir, fileName)

            val jsonArray = JSONArray()
            var count = 0

            while (metricsQueue.isNotEmpty()) {
                val metric = metricsQueue.poll()
                if (metric != null) {
                    jsonArray.put(metric.toJson())
                    count++
                }
            }

            file.writeText(jsonArray.toString(2))

            val duration = System.currentTimeMillis() - startTime
            LogUtil.i(TAG, "Export", "导出JSON完成 | 文件: $fileName | 记录数: $count | 耗时: ${duration}ms")

            ExportResult(true, count, file.absolutePath)
        } catch (e: Exception) {
            LogUtil.e(TAG, "Export", "导出失败: ${e.message}")
            ExportResult(false, 0, e.message ?: "Export failed")
        } finally {
            isExporting.set(false)
        }
    }

    suspend fun exportToCsv(): ExportResult = withContext(Dispatchers.IO) {
        if (!isExporting.compareAndSet(false, true)) {
            return@withContext ExportResult(false, 0, "Export already in progress")
        }

        try {
            val startTime = System.currentTimeMillis()
            val exportDir = getExportDirectory()
            if (!exportDir.exists()) exportDir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "performance_metrics_$timestamp.csv"
            val file = File(exportDir, fileName)

            val metricsByType = linkedMapOf<String, MutableList<PerformanceMetric>>()
            while (metricsQueue.isNotEmpty()) {
                val metric = metricsQueue.poll()
                if (metric != null) {
                    metricsByType.getOrPut(metric.operationType) { mutableListOf() }.add(metric)
                }
            }

            file.bufferedWriter().use { writer ->
                metricsByType.forEach { (type, metrics) ->
                    if (metrics.isNotEmpty()) {
                        writer.write("\n=== $type ===\n")
                        writer.write(metrics.first().getCsvHeader())
                        writer.write("\n")
                        metrics.forEach { metric ->
                            writer.write(metric.toCsvRow())
                            writer.write("\n")
                        }
                    }
                }

                writer.write("\n=== Summary ===\n")
                writer.write("total_metrics,${totalMetricsRecorded.get()}\n")
                writer.write("export_timestamp,${System.currentTimeMillis()}\n")
                writer.write("export_time_ms,${System.currentTimeMillis() - startTime}\n")
            }

            val count = metricsByType.values.sumOf { it.size }
            val duration = System.currentTimeMillis() - startTime

            LogUtil.i(TAG, "Export", "导出CSV完成 | 文件: $fileName | 记录数: $count | 耗时: ${duration}ms")

            ExportResult(true, count, file.absolutePath)
        } catch (e: Exception) {
            LogUtil.e(TAG, "Export", "导出失败: ${e.message}")
            ExportResult(false, 0, e.message ?: "Export failed")
        } finally {
            isExporting.set(false)
        }
    }

    suspend fun exportToSdCard(): ExportResult = withContext(Dispatchers.IO) {
        val exportDir = getExportDirectory()
        if (!exportDir.exists()) exportDir.mkdirs()
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        exportToDirectory(exportDir, "perf_report_$timestamp")
    }

    suspend fun exportToDirectory(exportDir: File, baseFileName: String): ExportResult = withContext(Dispatchers.IO) {
        if (!isExporting.compareAndSet(false, true)) {
            return@withContext ExportResult(false, 0, "Export already in progress")
        }

        try {
            val startTime = System.currentTimeMillis()
            if (!exportDir.exists()) exportDir.mkdirs()

            val jsonFile = File(exportDir, "$baseFileName.json")
            val jsonArray = JSONArray()
            val metricsList = mutableListOf<PerformanceMetric>()

            while (metricsQueue.isNotEmpty()) {
                val metric = metricsQueue.poll()
                if (metric != null) {
                    jsonArray.put(metric.toJson())
                    metricsList.add(metric)
                }
            }
            jsonFile.writeText(jsonArray.toString(2))

            val csvFile = File(exportDir, "$baseFileName.csv")
            val groupedMetrics = metricsList.groupBy { it.operationType }
            csvFile.bufferedWriter().use { writer ->
                groupedMetrics.forEach { (type, metrics) ->
                    if (metrics.isNotEmpty()) {
                        writer.write("\n### $type ###\n")
                        writer.write(metrics.first().getCsvHeader())
                        writer.write("\n")
                        metrics.forEach { m ->
                            writer.write(m.toCsvRow())
                            writer.write("\n")
                        }
                    }
                }
            }

            val reportFile = File(exportDir, "$baseFileName.txt")
            generateAnalysisReport(metricsList, reportFile)

            val duration = System.currentTimeMillis() - startTime
            LogUtil.i(TAG, "Export", "导出完成 | 记录数: ${metricsList.size} | 耗时: ${duration}ms | 目录: ${exportDir.absolutePath}")
            ExportResult(true, metricsList.size, exportDir.absolutePath)
        } catch (e: Exception) {
            LogUtil.e(TAG, "Export", "导出失败: ${e.message}")
            ExportResult(false, 0, e.message ?: "Export failed")
        } finally {
            isExporting.set(false)
        }
    }

    private fun generateAnalysisReport(metrics: List<PerformanceMetric>, file: File) {
        file.bufferedWriter().use { writer ->
            writer.write("=" .repeat(80))
            writer.write("\n")
            writer.write("Performance Analysis Report")
            writer.write("=" .repeat(80))
            writer.write("\n\n")

            writer.write("导出时间: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())}\n")
            writer.write("总记录数: ${metrics.size}\n\n")

            val grouped = metrics.groupBy { it.operationType }

            writer.write("-".repeat(80))
            writer.write("\n")
            writer.write("Metrics by type:\n")
            writer.write("-".repeat(80))
            writer.write("\n")

            grouped.forEach { (type, typeMetrics) ->
                writer.write("\n【$type】\n")
                writer.write("  记录数: ${typeMetrics.size}\n")

                when (type) {
                    "prefill" -> {
                        val prefills = typeMetrics.filterIsInstance<PrefillMetric>()
                        val avgTime = prefills.map { it.timeMs }.average()
                        val avgRate = prefills.map { it.rateTokensPerSec }.average()
                        writer.write("  平均耗时: ${"%.2f".format(avgTime)}ms\n")
                        writer.write("  平均速率: ${"%.2f".format(avgRate)} T/s\n")
                        writer.write("  最长耗时: ${prefills.maxOfOrNull { it.timeMs }}ms\n")
                        writer.write("  最短耗时: ${prefills.minOfOrNull { it.timeMs }}ms\n")
                    }
                    "decode" -> {
                        val decodes = typeMetrics.filterIsInstance<DecodeMetric>()
                        val avgTime = decodes.map { it.timeMs }.average()
                        val avgRate = decodes.map { it.tokensPerSec }.average()
                        writer.write("  平均单步耗时: ${"%.2f".format(avgTime)}ms\n")
                        writer.write("  平均速率: ${"%.2f".format(avgRate)} T/s\n")
                        writer.write("  KV缓存命中: ${decodes.count { it.kvCacheHit }}/${decodes.size}\n")
                    }
                    "generation_session" -> {
                        val sessions = typeMetrics.filterIsInstance<GenerationSessionMetric>()
                        val success = sessions.filter { it.success }
                        val successRate = sessions.count { it.success }.toDouble() / sessions.size * 100

                        if (success.isNotEmpty()) {
                            val avgTotalTime = success.map { it.totalTimeMs }.average()
                            val avgFirstToken = success.map { it.firstTokenLatencyMs }.average()
                            writer.write("  平均总耗时(成功): ${"%.2f".format(avgTotalTime)}ms\n")
                            writer.write("  平均首字延迟(成功): ${"%.2f".format(avgFirstToken)}ms\n")
                        } else {
                            writer.write("  平均总耗时(成功): N/A\n")
                            writer.write("  平均首字延迟(成功): N/A\n")
                        }
                        writer.write("  成功率: ${"%.2f".format(successRate)}%\n")
                    }
                    else -> {
                        writer.write("  (详细数据请查看CSV/JSON文件)\n")
                    }
                }
            }

            writer.write("\n")
            writer.write("-".repeat(80))
            writer.write("\n")
            writer.write("Session stability and latency distribution:\n")
            writer.write("-".repeat(80))
            writer.write("\n")

            val sessions = grouped["generation_session"]?.filterIsInstance<GenerationSessionMetric>()
            if (sessions != null && sessions.isNotEmpty()) {
                val success = sessions.filter { it.success }
                if (success.isNotEmpty()) {
                    val sortedLatency = success.map { it.firstTokenLatencyMs }.sorted()
                    writer.write("\n首字延迟分布(成功):\n")
                    writer.write("  P50: ${sortedLatency[sortedLatency.size * 50 / 100]}ms\n")
                    writer.write("  P90: ${sortedLatency[sortedLatency.size * 90 / 100]}ms\n")
                    writer.write("  P95: ${sortedLatency[sortedLatency.size * 95 / 100]}ms\n")
                    writer.write("  P99: ${sortedLatency[sortedLatency.size * 99 / 100]}ms\n")
                } else {
                    writer.write("\n首字延迟分布(成功): N/A\n")
                }
            }

            writer.write("\n")
            writer.write("=" .repeat(80))
            writer.write("\n")
        }
    }

    internal fun getExportDirectory(): File {
        val externalDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
        if (externalDir != null) {
            return File(externalDir, "performance_reports")
        }

        return File(context.filesDir, "performance_reports")
    }

    fun getStatistics(): Statistics {
        return Statistics(
            totalMetrics = totalMetricsRecorded.get(),
            cachedMetrics = metricsQueue.size,
            activeSessions = activeSessions.size,
            metricsByType = metricsByType.mapValues { it.value.get() }
        )
    }

    fun clearCache() {
        metricsQueue.clear()
        LogUtil.i(TAG, "Cache", "Metric cache cleared")
    }

    data class ExportResult(
        val success: Boolean,
        val recordCount: Int,
        val message: String
    )

    data class Statistics(
        val totalMetrics: Long,
        val cachedMetrics: Int,
        val activeSessions: Int,
        val metricsByType: Map<String, Long>
    )
}
