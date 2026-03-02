package com.yuyan.imemodule.performance

object NativePerformanceTracker {

    init {
        try {
            System.loadLibrary("llama_jni")
            initNative()
        } catch (e: UnsatisfiedLinkError) {
            e.printStackTrace()
        }
    }

    enum class MetricType(val nativeValue: Int) {
        PREFILL(0),
        DECODE_STEP(1),
        GENERATION_SESSION(2),
        CONTEXT_SYNC(5)
    }

    @JvmStatic
    external fun initNative()

    @JvmStatic
    external fun nativeCreateSession(type: Int): String

    @JvmStatic
    external fun nativeRecordPrefill(
        sessionId: String,
        promptLength: Int,
        tokenCount: Int,
        reuseCount: Int,
        timeMs: Long,
        rate: Double,
        kvCacheBefore: Long,
        kvCacheAfter: Long
    )

    @JvmStatic
    external fun nativeRecordDecodeStep(
        sessionId: String,
        step: Int,
        token: String,
        tokenId: Int,
        timeMs: Long,
        cumulativeMs: Long,
        tps: Double,
        kvHit: Boolean,
        branchCount: Int
    )

    @JvmStatic
    external fun nativeCompleteSession(
        sessionId: String,
        mode: String,
        prompt: String,
        candidates: Array<String>,
        firstTokenLatency: Long,
        prefillMs: Long,
        decodeMs: Long,
        totalTokens: Int,
        success: Boolean
    )

    @JvmStatic
    external fun nativeRecordContextSync(
        sessionId: String,
        syncType: String,
        historyLen: Int,
        lastMsgLen: Int,
        prefillMs: Long,
        sessionLoadMs: Long,
        success: Boolean
    )

    @JvmStatic
    external fun nativeExportJson(): String

    @JvmStatic
    external fun nativeClearAll()

    @JvmStatic
    external fun nativeGetSessionCount(): Int

    @JvmStatic
    external fun nativeGetTimestampNs(): Long

    @JvmStatic
    external fun nativeRecordCustomMetric(
        sessionId: String,
        metricName: String,
        metricValue: String
    )

    @JvmStatic
    external fun nativeStartTimer(): Long

    @JvmStatic
    external fun nativeStopTimer(timerPtr: Long): Long

    @JvmStatic
    external fun nativeGetElapsedMs(timerPtr: Long): Long

    fun createSession(type: MetricType): String {
        return nativeCreateSession(type.nativeValue)
    }

    fun getTimestampNs(): Long = nativeGetTimestampNs()

    fun getTimestampMs(): Long = nativeGetTimestampNs() / 1_000_000

    class Timer {
        private var timerPtr: Long = 0
        private var isRunning = false

        fun start(): Timer {
            if (!isRunning) {
                timerPtr = nativeStartTimer()
                isRunning = true
            }
            return this
        }

        fun stop(): Long {
            return if (isRunning) {
                isRunning = false
                nativeStopTimer(timerPtr)
            } else 0L
        }

        fun elapsedMs(): Long {
            return if (isRunning) {
                nativeGetElapsedMs(timerPtr)
            } else 0L
        }

        fun reset() {
            if (isRunning) {
                nativeStopTimer(timerPtr)
                isRunning = false
            }
            timerPtr = 0
        }

        companion object {
            fun start(): Timer {
                return Timer().start()
            }
        }
    }

    class ScopedTimer(private val name: String, private val onEnd: ((String, Long) -> Unit)?) {
        private var startTime: Long = 0
        private var timer: Timer? = null

        init {
            timer = Timer.start()
            startTime = System.currentTimeMillis()
        }

        fun end(): Long {
            val elapsed = timer?.stop() ?: 0L
            onEnd?.invoke(name, elapsed)
            return elapsed
        }

        companion object {
            fun create(name: String, onEnd: ((String, Long) -> Unit)? = null): ScopedTimer {
                return ScopedTimer(name, onEnd)
            }
        }
    }
}
