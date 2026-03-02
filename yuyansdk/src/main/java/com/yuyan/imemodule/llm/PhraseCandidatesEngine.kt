package com.yuyan.imemodule.llm

import android.os.SystemClock
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

interface PhraseCandidatesEngine {
    fun generate(handle: Long, fullPrompt: String, nCandidates: Int, callback: LLMBridge.TokenCallback): Int
}

class BaselinePhraseCandidatesEngine : PhraseCandidatesEngine {
    override fun generate(handle: Long, fullPrompt: String, nCandidates: Int, callback: LLMBridge.TokenCallback): Int {
        return LLMBridge.generatePhraseCandidates(handle, fullPrompt, nCandidates, callback)
    }
}
class KvSplicePhraseCandidatesEngine(
    private val prefillTimeoutMs: Long = 15_000,
) : PhraseCandidatesEngine {

    @Volatile private var lastBasePrompt: String? = null

    override fun generate(handle: Long, fullPrompt: String, nCandidates: Int, callback: LLMBridge.TokenCallback): Int {
        val seg = MemoryPromptSegments.splitFromFullPrompt(fullPrompt)
            ?: return LLMBridge.generatePhraseCandidates(handle, fullPrompt, nCandidates, callback)

        val basePrompt = seg.toBasePromptWithoutMemory()
        if (lastBasePrompt != basePrompt) {
            val ok = blockingPrefill(handle, basePrompt)
            if (!ok) {
                return LLMBridge.generatePhraseCandidates(handle, fullPrompt, nCandidates, callback)
            }
            lastBasePrompt = basePrompt
        }

        return LLMBridge.generatePhraseCandidatesSpliceMemory(
            handle,
            seg.prefixBeforeMemory,
            seg.memoryText,
            seg.suffixAfterMemory,
            nCandidates,
            callback,
        )
    }

    private fun blockingPrefill(handle: Long, basePrompt: String): Boolean {
        val done = CountDownLatch(1)
        var success = true

        val cb = object : LLMBridge.TokenCallback {
            override fun onTokenCandidates(tokens: Array<String>) {
                
            }

            override fun onFinished() {
                done.countDown()
            }

            override fun onError(err: String) {
                success = false
                done.countDown()
            }
        }

        val rc = LLMBridge.prefillPrompt(handle, basePrompt, cb)
        if (rc != 0) return false

        val t0 = SystemClock.uptimeMillis()
        val completed = done.await(prefillTimeoutMs, TimeUnit.MILLISECONDS)
        val elapsed = SystemClock.uptimeMillis() - t0
        if (!completed) return false
        if (elapsed > prefillTimeoutMs) return false
        return success
    }
}
