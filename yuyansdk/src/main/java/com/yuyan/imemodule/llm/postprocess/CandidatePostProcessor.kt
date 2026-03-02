package com.yuyan.imemodule.llm.postprocess

import com.yuyan.imemodule.llm.memory.MemToolcallParser

object CandidatePostProcessor {

    data class CandidateContext(
        val usedPrefix: String = "",
        val forceEmpty: Boolean = false,
        val allowMemToolcall: Boolean = true,
        val preferCjk: Boolean = false,
        val instructionPrefix: String = "",
    )

    sealed interface CandidateOutcome {
        data class Show(val candidates: List<String>) : CandidateOutcome
        data class MemRetrieval(val query: String) : CandidateOutcome
        data class DropAll(val reason: String) : CandidateOutcome
    }

    interface CandidateReranker {
        fun rerank(candidates: List<String>, ctx: CandidateContext): List<String>
    }

    @Volatile
    var reranker: CandidateReranker? = null

    private val noiseExact = setOf("ensions", "neider")

    private val hanRegex = Regex("\\p{IsHan}")
    private val latinLetterRegex = Regex("[A-Za-z]")

    private val queryWideRegex = Regex(
        """(?i)\bquery\s*[:=]\s*(?:\"([^\"]{1,512})\"|'([^']{1,512})'|“([^”]{1,512})”|「([^」]{1,512})」|『([^』]{1,512})』|([^\s,;]{1,128}))"""
    )

    fun process(rawCandidates: Array<String>, ctx: CandidateContext): CandidateOutcome {
        if (rawCandidates.isEmpty()) return CandidateOutcome.DropAll("empty")

        if (ctx.allowMemToolcall) {
            val q = tryParseMemQueryWide(rawCandidates)
            if (!q.isNullOrBlank()) {
                return CandidateOutcome.MemRetrieval(q)
            }
        }

        val out = ArrayList<String>(rawCandidates.size)
        var filteredByLang = 0
        for (raw in rawCandidates) {
            if (raw.isBlank()) continue
            if (raw.startsWith("__METRICS__")) continue

            var clean = stripInvisibleAndControl(raw)

            clean = ThinkTagStripper.stripOrNull(clean) ?: continue

            clean = clean
                .replace("<|im_end|>", "")
                .replace("<|endoftext|>", "")
                .trim()

            if (clean.isEmpty()) continue

            if (ctx.preferCjk && shouldSuppressMostlyLatin(clean)) {
                filteredByLang++
                continue
            }

            if (looksLikeProtocolOrControl(clean)) continue

            val isBad = clean.any { it in '\u0400'..'\u04FF' } ||
                clean.startsWith("_") ||
                clean in noiseExact

            if (isBad) continue
            if (clean.contains("[对方]") || clean.contains("[我]")) continue

            if (ctx.instructionPrefix.isNotEmpty() && clean.startsWith(ctx.instructionPrefix)) {
                clean = clean.removePrefix(ctx.instructionPrefix).trimStart()
                if (clean.isEmpty()) continue
            }

            if (ctx.forceEmpty && clean.isNotEmpty()) {
                clean = ctx.usedPrefix + clean
            }

            if (clean.isNotEmpty()) out.add(clean)
        }

        val deduped = out.distinct()
            .toMutableList()

        if (deduped.size > 1) {
            deduped.removeAll { it.equals("nothing", ignoreCase = true) }
        }

        if (deduped.isEmpty()) {
            return if (filteredByLang > 0) CandidateOutcome.DropAll("lang_filtered") else CandidateOutcome.DropAll("all_filtered")
        }

        val reranked = reranker?.rerank(deduped, ctx) ?: deduped
        return CandidateOutcome.Show(reranked)
    }

    private fun shouldSuppressMostlyLatin(text: String): Boolean {
        if (text.isBlank()) return true
        if (hanRegex.containsMatchIn(text)) return false
        val letters = latinLetterRegex.findAll(text).count()
        if (letters < 6) return false
        val ratio = letters.toFloat() / text.length.coerceAtLeast(1)
        return ratio >= 0.45f
    }

    private fun looksLikeProtocolOrControl(text: String): Boolean {
        val s = stripInvisibleAndControl(text).trim()
        if (s.isEmpty()) return true

        if (MemToolcallParser.looksLikeMemToolcall(s)) return true

        if (queryWideRegex.containsMatchIn(s)) return true

        if (s.contains("<NO_MEM>", ignoreCase = true)) return true
        if (s.contains("<MEM_RETRIEVAL>", ignoreCase = true) || s.contains("</MEM_RETRIEVAL>", ignoreCase = true)) return true
        if (s.contains("search query", ignoreCase = true)) return true
        if (s.contains("no longer", ignoreCase = true)) return true

        return false
    }

    private fun tryParseMemQueryWide(candidates: Array<String>): String? {
        val strict = MemToolcallParser.tryParseMemRetrievalQuery(candidates)
        if (!strict.isNullOrBlank()) return strict

        for (raw in candidates) {
            val s = stripInvisibleAndControl(raw).trim()
            if (s.isEmpty()) continue
            val m = queryWideRegex.find(s) ?: continue
            val q = m.groupValues.drop(1).firstOrNull { it.isNotBlank() }?.trim() ?: continue
            if (q.length !in 1..512) continue
            return q
        }
        return null
    }

    fun stripInvisibleAndControl(input: String): String {
        if (input.isEmpty()) return input
        val sb = StringBuilder(input.length)
        for (ch in input) {
            if (ch.isISOControl()) continue
            when (ch) {
                '\u200B', '\u200C', '\u200D', '\u200E', '\u200F',
                '\u202A', '\u202B', '\u202C', '\u202D', '\u202E',
                '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
                '\u2066', '\u2067', '\u2068', '\u2069',
                '\uFEFF' -> continue
            }
            sb.append(ch)
        }
        return sb.toString()
    }
}
