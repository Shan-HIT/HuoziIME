package com.yuyan.imemodule.performance

import android.app.ProgressDialog
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.yuyan.imemodule.R
import com.yuyan.imemodule.utils.LogUtil
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class PerformanceTestActivity : AppCompatActivity() {
    private lateinit var testSuite: PerformanceTestSuite
    private var currentJob: Job? = null
    private var progressDialog: ProgressDialog? = null
    private var massiveDecodeMode: String = PerformanceTestSuite.DecodeControlMode.RUNTIME_GENERATE_PHRASE_CANDIDATES.wireValue
    private val massiveDecodeModes = listOf(
        PerformanceTestSuite.DecodeControlMode.RUNTIME_GENERATE_PHRASE_CANDIDATES.wireValue,
        PerformanceTestSuite.DecodeControlMode.BENCHMARK_DECODE_MAX_STEPS.wireValue,
        PerformanceTestSuite.DecodeControlMode.RUNTIME_GENERATE_CANDIDATES.wireValue,
    )

    private enum class SuiteMode {
        CPU_EXTRA_RAM,
        MASSIVE_PREFILL,
        MASSIVE_DECODE,
        MASSIVE_IME_FIRST,
    }

    private lateinit var runMassivePrefillTestButton: Button
    private lateinit var runMassiveFullTestButton: Button
    private lateinit var runMassiveDecodeTestButton: Button
    private lateinit var runMassiveImeFirstTestButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var currentPhaseTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_performance_test)

        testSuite = PerformanceTestSuite(this)
        initViews()
    }

    private fun initViews() {
        runMassivePrefillTestButton = findViewById(R.id.run_micro_test_button)
        runMassiveFullTestButton = findViewById(R.id.run_massive_test_button)
        runMassiveDecodeTestButton = findViewById(R.id.run_cherry_pick_memory_test_button)
        runMassiveImeFirstTestButton = findViewById(R.id.run_kv_splice_vs_reuse_test_button)
        resultTextView = findViewById(R.id.result_text)
        statusTextView = findViewById(R.id.status_text)
        currentPhaseTextView = findViewById(R.id.current_phase_text)

        runMassivePrefillTestButton.text = "Massive Prefill"
        runMassiveFullTestButton.text = "CPU/Extra RAM"
        runMassiveDecodeTestButton.text = "Massive Decode"
        runMassiveImeFirstTestButton.text = "Massive ImeFirst"

        runMassivePrefillTestButton.setOnClickListener { runSuite(SuiteMode.MASSIVE_PREFILL) }
        runMassiveFullTestButton.setOnClickListener { runSuite(SuiteMode.CPU_EXTRA_RAM) }
        runMassiveDecodeTestButton.setOnClickListener { runSuite(SuiteMode.MASSIVE_DECODE) }
        runMassiveDecodeTestButton.setOnLongClickListener {
            cycleMassiveDecodeMode()
            true
        }
        runMassiveImeFirstTestButton.setOnClickListener { runSuite(SuiteMode.MASSIVE_IME_FIRST) }
        updateStatus()
    }

    private fun cycleMassiveDecodeMode() {
        val currentIdx = massiveDecodeModes.indexOf(massiveDecodeMode).let { if (it >= 0) it else 0 }
        val nextIdx = (currentIdx + 1) % massiveDecodeModes.size
        massiveDecodeMode = massiveDecodeModes[nextIdx]
        val msg = "Massive decode mode -> $massiveDecodeMode"
        appendStatus("[Massive] $msg")
        Toast.makeText(this@PerformanceTestActivity, msg, Toast.LENGTH_SHORT).show()
    }

    private fun runSuite(mode: SuiteMode) {
        currentJob?.cancel()
        currentJob = CoroutineScope(Dispatchers.Main).launch {
            val title = when (mode) {
                SuiteMode.CPU_EXTRA_RAM -> "Running CPU/Extra RAM test..."
                SuiteMode.MASSIVE_PREFILL -> "Running massive prefill test..."
                SuiteMode.MASSIVE_DECODE -> "Running massive decode test... [$massiveDecodeMode]"
                SuiteMode.MASSIVE_IME_FIRST -> "Running massive ime_first test..."
            }
            showProgressDialog(title, true)
            currentPhaseTextView.text = "Current phase: preparing"
            resultTextView.text = ""

            try {
                val result = withContext(Dispatchers.IO) {
                    when (mode) {
                        SuiteMode.CPU_EXTRA_RAM -> testSuite.runCpuExtraRamTest { line -> launch(Dispatchers.Main) { handleProgressLine(line) } }
                        SuiteMode.MASSIVE_PREFILL -> testSuite.runMassivePrefillTest { line -> launch(Dispatchers.Main) { handleProgressLine(line) } }
                        SuiteMode.MASSIVE_DECODE -> testSuite.runMassiveDecodeTest(decodeMode = massiveDecodeMode) { line -> launch(Dispatchers.Main) { handleProgressLine(line) } }
                        SuiteMode.MASSIVE_IME_FIRST -> testSuite.runMassiveImeFirstTest { line -> launch(Dispatchers.Main) { handleProgressLine(line) } }
                    }
                }

                hideProgressDialog()
                displayTestResult(result)
                updateStatus()

                result.exportPath?.let {
                    Toast.makeText(this@PerformanceTestActivity, "Test finished. Export: $it", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                hideProgressDialog()
                LogUtil.e("PerformanceTest", "Suite failed", e.message ?: "unknown")
                Toast.makeText(this@PerformanceTestActivity, "Test failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun appendStatus(line: String) {
        val old = statusTextView.text?.toString().orEmpty()
        val next = if (old.isEmpty()) line else "$old\n$line"
        val lines = next.split('\n')
        statusTextView.text = if (lines.size > 200) lines.takeLast(200).joinToString("\n") else next
    }

    private fun handleProgressLine(line: String) {
        if (line.startsWith("__PROGRESS__")) {
            val payload = line.removePrefix("__PROGRESS__")
            runCatching {
                val obj = JSONObject(payload)
                val done = obj.optInt("done", 0).coerceAtLeast(0)
                val total = obj.optInt("total", 1).coerceAtLeast(1)
                val label = obj.optString("label", "running")
                val percent = ((done.toDouble() / total.toDouble()) * 100.0).toInt().coerceIn(0, 100)

                progressDialog?.let { dlg ->
                    if (dlg.isIndeterminate) {
                        dlg.isIndeterminate = false
                        dlg.max = 100
                    }
                }
                updateProgressDialog("$label ($done/$total)", percent)
                currentPhaseTextView.text = "Current phase: $label"
            }
            return
        }

        appendStatus(line)
        when {
            line.startsWith("[Phase]") -> currentPhaseTextView.text = "Current phase: ${line.removePrefix("[Phase] ")}"
            line.startsWith("[Style]") -> currentPhaseTextView.text = "Current style: ${line.removePrefix("[Style] ")}"
        }
    }

    private fun displayTestResult(result: PerformanceTestSuite.TestResult) {
        val sb = StringBuilder()
        sb.append("Test result: ${result.testName}\n")
        sb.append("=".repeat(48)).append("\n\n")
        sb.append("Status: ${if (result.success) "SUCCESS" else "FAILED"}\n")
        sb.append("Iterations: ${result.iterations}\n")
        sb.append("Duration: ${result.durationMs} ms\n")
        sb.append("Avg/Min/Max: ${"%.2f".format(result.statistics.avgTimeMs)} / ${result.statistics.minTimeMs} / ${result.statistics.maxTimeMs} ms\n")
        sb.append("P50/P90/P95/P99: ${result.statistics.p50}/${result.statistics.p90}/${result.statistics.p95}/${result.statistics.p99} ms\n")
        sb.append("Throughput: ${"%.2f".format(result.statistics.throughput)} ops/s\n")
        sb.append("Metrics: ${result.totalMetrics}\n")

        result.warning?.let {
            sb.append("\nWarnings:\n").append(it).append("\n")
        }

        result.error?.let {
            sb.append("\nErrors:\n").append(it).append("\n")
        }

        result.exportPath?.let {
            sb.append("\nExport path:\n$it\n")
            val plotScript = File(it, "plot_metrics.py")
            if (plotScript.exists()) {
                sb.append("\nPlot Script:\n")
                sb.append(plotScript.absolutePath).append("\n")
                sb.append("Run command:\n")
                sb.append("python \"").append(plotScript.absolutePath).append("\" --run-dir \"")
                    .append(it).append("\" --out-dir \"plots\" --summary-md \"research_summary.md\"\n")
            }
            val dashboard = runCatching { formatDashboardFromRunDir(it) }.getOrNull()
            if (!dashboard.isNullOrBlank()) {
                sb.append("\n").append(dashboard).append("\n")
            }
            val allMetrics = runCatching { formatAllMetricsFromRunDir(it) }.getOrNull()
            if (!allMetrics.isNullOrBlank()) {
                sb.append("\n").append(allMetrics).append("\n")
            }
        }

        val text = sb.toString()
        resultTextView.text = text

        result.exportPath?.let { path ->
            runCatching { File(path, "ui_test_result.txt").writeText(text, Charsets.UTF_8) }
        }
    }

    private fun formatDashboardFromRunDir(exportPath: String): String? {
        val runDir = File(exportPath)
        val summaryFile = File(runDir, "summary.json")
        if (!summaryFile.exists()) return null

        val summary = JSONObject(summaryFile.readText())
        val lifecycleInvariant = summary.optJSONObject("lifecycle_invariant")
        val lifecycleMs = summary.optJSONObject("lifecycle_ms")

        val out = StringBuilder()
        out.append("Dashboard\n")
        out.append("-".repeat(48)).append("\n")

        out.append("Styles: ")
        val styles = summary.optJSONArray("styles_tested")
        if (styles == null || styles.length() == 0) {
            out.append("(missing)\n")
        } else {
            val values = mutableListOf<String>()
            for (i in 0 until styles.length()) values.add(styles.optString(i))
            out.append(values.joinToString(", ")).append("\n")
        }

        out.append("Duration: ${summary.optLong("duration_ms", -1)} ms\n")

        if (lifecycleInvariant != null) {
            out.append("\n[Lifecycle Invariant]\n")
            out.append("Checked: ${lifecycleInvariant.optInt("checked", 0)}\n")
            out.append("Passed: ${lifecycleInvariant.optInt("passed", 0)}\n")
            out.append("Failed: ${lifecycleInvariant.optInt("failed", 0)}\n")
            out.append("Tolerance: ${lifecycleInvariant.optLong("tolerance_ms", 0)} ms\n")
        }

        if (lifecycleMs != null) {
            out.append("\n[Lifecycle Average Ms]\n")
            val keys = lifecycleMs.keys().asSequence().toList().sorted()
            for (k in keys) {
                out.append("- $k: ${lifecycleMs.opt(k)}\n")
            }
        }

        return out.toString().trimEnd()
    }

    private fun formatAllMetricsFromRunDir(exportPath: String): String {
        val runDir = File(exportPath)
        val out = StringBuilder()
        out.append("All Metrics Dump\n")
        out.append("-".repeat(48)).append("\n")

        appendFileSection(out, runDir, "summary.json", parseJson = true)
        appendFileSection(out, runDir, "device_snapshot.json", parseJson = true)
        appendFileSection(out, runDir, "lifecycle_flow_metrics.jsonl")
        appendFileSection(out, runDir, "raw_llm_metrics.jsonl")
        appendFileSection(out, runDir, "native_perf_export.json", parseJson = true)
        appendFileSection(out, runDir, "memory_sample_records.jsonl")
        appendFileSection(out, runDir, "memory_processing_marks.jsonl")
        appendFileSection(out, runDir, "memory_eval_records.jsonl")
        appendFileSection(out, runDir, "memory_triplets.jsonl")
        appendFileSection(out, runDir, "memory_human_report.md")
        appendFileSection(out, runDir, "kv_splice_vs_reuse_raw.jsonl")
        appendFileSection(out, runDir, "kv_splice_vs_reuse_report.md")

        return out.toString().trimEnd()
    }

    private fun appendFileSection(out: StringBuilder, runDir: File, fileName: String, parseJson: Boolean = false) {
        val file = File(runDir, fileName)
        out.append("\n[").append(fileName).append("]\n")
        if (!file.exists()) {
            out.append("(missing)\n")
            return
        }

        val raw = file.readText(Charsets.UTF_8)
        if (raw.isBlank()) {
            out.append("(empty)\n")
            return
        }

        val content = if (parseJson) prettyJson(raw) else raw
        out.append(content.trimEnd()).append("\n")
    }

    private fun prettyJson(raw: String): String {
        val t = raw.trim()
        return when {
            t.startsWith("{") -> runCatching { JSONObject(t).toString(2) }.getOrDefault(raw)
            t.startsWith("[") -> runCatching { org.json.JSONArray(t).toString(2) }.getOrDefault(raw)
            else -> raw
        }
    }

    private fun updateStatus() {
        val status = testSuite.getTestStatus()
        statusTextView.text = buildString {
            append("Current status\n")
            append("-".repeat(30)).append("\n")
            append("Running: ${if (status.isRunning) "YES (${status.currentTest})" else "NO"}\n")
            append("Massive decode mode: $massiveDecodeMode\n")
            append("Cached metrics: ${status.statistics.cachedMetrics}\n")
            append("Native sessions: ${status.nativeSessionCount}\n")
            append("Active sessions: ${status.statistics.activeSessions}\n")
            if (status.statistics.metricsByType.isNotEmpty()) {
                append("\nMetrics by type:\n")
                status.statistics.metricsByType.forEach { (type, count) -> append("$type: $count\n") }
            }
        }
    }

    private fun showProgressDialog(message: String, indeterminate: Boolean) {
        hideProgressDialog()
        progressDialog = ProgressDialog(this).apply {
            setTitle("Performance Test")
            setMessage(message)
            setProgressStyle(ProgressDialog.STYLE_HORIZONTAL)
            setIndeterminate(indeterminate)
            setCancelable(false)
            setCanceledOnTouchOutside(false)
            max = 100
            progress = 0
            show()
        }
    }

    private fun updateProgressDialog(message: String, progress: Int) {
        progressDialog?.apply {
            setMessage(message)
            if (!isIndeterminate) setProgress(progress)
        }
    }

    private fun hideProgressDialog() {
        progressDialog?.dismiss()
        progressDialog = null
    }

    override fun onDestroy() {
        super.onDestroy()
        currentJob?.cancel()
        hideProgressDialog()
    }
}
