package com.yuyan.imemodule.performance

import org.json.JSONObject

sealed class PerformanceMetric {
    abstract val timestamp: Long
    abstract val operationType: String

    abstract fun toJson(): JSONObject

    abstract fun toCsvRow(): String

    abstract fun getCsvHeader(): String
}

data class PrefillMetric(
    override val timestamp: Long,
    val sessionId: String,
    val promptLength: Int,
    val tokenCount: Int,
    val reuseTokenCount: Int,
    val timeMs: Long,
    val rateTokensPerSec: Double,
    val kvCacheSizeBefore: Long,
    val kvCacheSizeAfter: Long,
    val modelName: String = "unknown"
) : PerformanceMetric() {
    override val operationType: String = "prefill"

    override fun toJson(): JSONObject {
        return JSONObject().apply {
            put("timestamp", timestamp)
            put("operation_type", operationType)
            put("session_id", sessionId)
            put("prompt_length", promptLength)
            put("token_count", tokenCount)
            put("reuse_token_count", reuseTokenCount)
            put("time_ms", timeMs)
            put("rate_tokens_per_sec", rateTokensPerSec)
            put("kv_cache_size_before", kvCacheSizeBefore)
            put("kv_cache_size_after", kvCacheSizeAfter)
            put("model_name", modelName)
            put("new_tokens", tokenCount - reuseTokenCount)
        }
    }

    override fun toCsvRow(): String {
        return "$timestamp,$operationType,$sessionId,$promptLength,$tokenCount,$reuseTokenCount,$timeMs,$rateTokensPerSec,$kvCacheSizeBefore,$kvCacheSizeAfter,$modelName"
    }

    override fun getCsvHeader(): String {
        return "timestamp,operation_type,session_id,prompt_length,token_count,reuse_token_count,time_ms,rate_tokens_per_sec,kv_cache_size_before,kv_cache_size_after,model_name"
    }
}

data class DecodeMetric(
    override val timestamp: Long,
    val sessionId: String,
    val stepNumber: Int,
    val tokenGenerated: String,
    val tokenId: Int,
    val timeMs: Long,
    val cumulativeTimeMs: Long,
    val tokensPerSec: Double,
    val kvCacheHit: Boolean,
    val candidateBranchCount: Int
) : PerformanceMetric() {
    override val operationType: String = "decode"

    override fun toJson(): JSONObject {
        return JSONObject().apply {
            put("timestamp", timestamp)
            put("operation_type", operationType)
            put("session_id", sessionId)
            put("step_number", stepNumber)
            put("token_generated", tokenGenerated)
            put("token_id", tokenId)
            put("time_ms", timeMs)
            put("cumulative_time_ms", cumulativeTimeMs)
            put("tokens_per_sec", tokensPerSec)
            put("kv_cache_hit", kvCacheHit)
            put("candidate_branch_count", candidateBranchCount)
        }
    }

    override fun toCsvRow(): String {
        return "$timestamp,$operationType,$sessionId,$stepNumber,$tokenGenerated,$tokenId,$timeMs,$cumulativeTimeMs,$tokensPerSec,$kvCacheHit,$candidateBranchCount"
    }

    override fun getCsvHeader(): String {
        return "timestamp,operation_type,session_id,step_number,token_generated,token_id,time_ms,cumulative_time_ms,tokens_per_sec,kv_cache_hit,candidate_branch_count"
    }
}

data class GenerationSessionMetric(
    override val timestamp: Long,
    val sessionId: String,
    val mode: String,
    val prompt: String,
    val generatedCandidates: List<String>,
    val totalTimeMs: Long,
    val firstTokenLatencyMs: Long,
    val prefillTimeMs: Long,
    val decodeTimeMs: Long,
    val totalTokensGenerated: Int,
    val avgTokensPerSec: Double,
    val styleMode: String,
    val success: Boolean,
    val errorMessage: String? = null
) : PerformanceMetric() {
    override val operationType: String = "generation_session"

    override fun toJson(): JSONObject {
        return JSONObject().apply {
            put("timestamp", timestamp)
            put("operation_type", operationType)
            put("session_id", sessionId)
            put("mode", mode)
            put("prompt", prompt.take(100))
            put("generated_candidates", org.json.JSONArray(generatedCandidates))
            put("total_time_ms", totalTimeMs)
            put("first_token_latency_ms", firstTokenLatencyMs)
            put("prefill_time_ms", prefillTimeMs)
            put("decode_time_ms", decodeTimeMs)
            put("total_tokens_generated", totalTokensGenerated)
            put("avg_tokens_per_sec", avgTokensPerSec)
            put("style_mode", styleMode)
            put("success", success)
            errorMessage?.let { put("error_message", it) }
        }
    }

    override fun toCsvRow(): String {
        val candidatesStr = generatedCandidates.joinToString("|") { it.replace(",", "\\,") }
        val promptEscaped = prompt.replace(",", "\\,").take(50)
        return "$timestamp,$operationType,$sessionId,$mode,$promptEscaped,$candidatesStr,$totalTimeMs,$firstTokenLatencyMs,$prefillTimeMs,$decodeTimeMs,$totalTokensGenerated,$avgTokensPerSec,$styleMode,$success,${errorMessage ?: ""}"
    }

    override fun getCsvHeader(): String {
        return "timestamp,operation_type,session_id,mode,prompt,generated_candidates,total_time_ms,first_token_latency_ms,prefill_time_ms,decode_time_ms,total_tokens_generated,avg_tokens_per_sec,style_mode,success,error_message"
    }
}

data class ContextSyncMetric(
    override val timestamp: Long,
    val sessionId: String,
    val syncType: String, 
    val historyLength: Int,
    val lastMsgLength: Int,
    val prefillTimeMs: Long,
    val sessionLoadTimeMs: Long,
    val totalTimeMs: Long,
    val success: Boolean
) : PerformanceMetric() {
    override val operationType: String = "context_sync"

    override fun toJson(): JSONObject {
        return JSONObject().apply {
            put("timestamp", timestamp)
            put("operation_type", operationType)
            put("session_id", sessionId)
            put("sync_type", syncType)
            put("history_length", historyLength)
            put("last_msg_length", lastMsgLength)
            put("prefill_time_ms", prefillTimeMs)
            put("session_load_time_ms", sessionLoadTimeMs)
            put("total_time_ms", totalTimeMs)
            put("success", success)
        }
    }

    override fun toCsvRow(): String {
        return "$timestamp,$operationType,$sessionId,$syncType,$historyLength,$lastMsgLength,$prefillTimeMs,$sessionLoadTimeMs,$totalTimeMs,$success"
    }

    override fun getCsvHeader(): String {
        return "timestamp,operation_type,session_id,sync_type,history_length,last_msg_length,prefill_time_ms,session_load_time_ms,total_time_ms,success"
    }
}
