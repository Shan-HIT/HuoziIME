package com.yuyan.imemodule.llm

import com.yuyan.imemodule.BuildConfig

object LLMBridge {
    init {
        System.loadLibrary("llama_jni")

        nativeSetLogMinPriority(if (BuildConfig.DEBUG) android.util.Log.DEBUG else android.util.Log.ERROR)
    }

    @JvmStatic external fun nativeSetLogMinPriority(minPriority: Int)
    @JvmStatic external fun nativeEnterPerfMode(): Int
    @JvmStatic external fun nativeExitPerfMode(): Int

    @JvmStatic external fun nativeSetDisableKvReuse(handle: Long, disable: Boolean)

    @JvmStatic external fun createGenerationInstance(modelPath: String, loraPath: String, nThreads: Int, nGpuLayers: Int): Long
    @JvmStatic external fun createEmbeddingInstance(modelPath: String, nThreads: Int, nGpuLayers: Int): Long
    @JvmStatic external fun createBenchmarkInstance(modelPath: String, loraPath: String, nThreads: Int, nGpuLayers: Int): Long
    @JvmStatic external fun loadLora(handle: Long, loraPath: String?)
    @JvmStatic external fun unloadLora(handle: Long)
    @JvmStatic external fun freeModel(handle: Long)
    @JvmStatic external fun tokenize(handle: Long, text: String): IntArray?
    @JvmStatic external fun generateCandidates(handle: Long, prompt: String, n_candidates: Int, callback: TokenCallback): Int
    @JvmStatic external fun generateSentenceCandidates(handle: Long, prompt: String, n_candidates: Int, callback: TokenCallback): Int
    @JvmStatic external fun stop(handle: Long)
    @JvmStatic external fun saveKVCacheSnapshot(handle: Long, text: String, savePath: String): Boolean
    @JvmStatic external fun loadSession(handle: Long, path: String): Boolean
    @JvmStatic external fun triggerNightMode(handle: Long, swapPath: String): Boolean
    @JvmStatic external fun triggerDayMode(handle: Long, swapPath: String): Boolean
    /** Updates the native handle's current session signature (used for Day/Night restore isolation). */
    @JvmStatic external fun setSessionSignature(handle: Long, signature: String)
    /** Clears KV cache for user turns while keeping system prompt tokens and resets radix tree. */
    @JvmStatic external fun clearKvKeepSystem(handle: Long): Boolean
    @JvmStatic external fun generatePhraseCandidates(handle: Long, prompt: String, n_candidates: Int, callback: TokenCallback): Int

    @JvmStatic external fun generateMemoryWorker(handle: Long, prompt: String, maxTokens: Int, callback: TokenCallback): Int

    @JvmStatic external fun prefillPrompt(handle: Long, prompt: String, callback: TokenCallback): Int

    @JvmStatic external fun generatePhraseCandidatesSpliceMemory(
        handle: Long,
        prefixBeforeMemory: String,
        memory: String,
        suffixAfterMemory: String,
        n_candidates: Int,
        callback: TokenCallback
    ): Int

    @JvmStatic external fun buildMemoryKvBlob(handle: Long, memoryLine: String, savePath: String): Boolean

    @JvmStatic external fun generatePhraseCandidatesSpliceMemoryFromKvFile(
        handle: Long,
        prefixBeforeMemory: String,
        memory: String,
        suffixAfterMemory: String,
        memoryKvPath: String,
        n_candidates: Int,
        callback: TokenCallback,
    ): Int

    @JvmStatic external fun saveSeqStateFile(handle: Long, seqId: Int, tokens: IntArray, savePath: String): Long
    @JvmStatic external fun loadSeqStateFile(handle: Long, seqId: Int, loadPath: String, maxTokens: Int = 8192): IntArray?
    @JvmStatic external fun warmup(handle: Long)
    @JvmStatic external fun setReusablePrefixTokenCount(handle: Long, tokenCount: Int)
    @JvmStatic external fun getReusablePrefixTokenCount(handle: Long): Int

    @JvmStatic external fun benchmarkDecode(handle: Long, prompt: String, maxDecodeSteps: Int, callback: TokenCallback): Int

    @JvmStatic external fun getModelInfoJson(handle: Long): String

    @JvmStatic external fun vectorDbInit(handle: Long, indexDir: String, maxElements: Int = 2048, m: Int = 16, efConstruction: Int = 200): Boolean
    @JvmStatic external fun vectorDbClose(handle: Long)
    @JvmStatic external fun vectorDbAddText(handle: Long, text: String): Long
    @JvmStatic external fun vectorDbSearch(handle: Long, queryText: String, k: Int = 3): LongArray

    @JvmStatic external fun vectorDbSearchScored(handle: Long, queryText: String, k: Int = 3): LongArray
    @JvmStatic external fun vectorDbGetText(handle: Long, label: Long): String
    @JvmStatic external fun vectorDbCount(handle: Long): Int

    @JvmStatic external fun vectorDbRebuildFromTexts(handle: Long, indexDir: String, maxElements: Int = 2048, m: Int = 16, efConstruction: Int = 200): Boolean
    interface TokenCallback {
        fun onTokenCandidates(tokens: Array<String>)
        fun onFinished()
        fun onError(err: String)
    }
}