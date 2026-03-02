package com.yuyan.imemodule.llm.memory

object MemToolcallParser {

    private val queryRegex = Regex("""query\s*=\s*\"([^\"]{1,512})\"""")

    fun looksLikeMemToolcall(text: String): Boolean {
        val s = text.trim()
        if (s.isEmpty()) return false
        if (s.contains("<MEM_RETRIEVAL>", ignoreCase = true) || s.contains("</MEM_RETRIEVAL>", ignoreCase = true)) return true
        return queryRegex.containsMatchIn(s)
    }

    fun tryParseMemRetrievalQuery(candidates: Array<String>): String? {
        for (raw in candidates) {
            val s = raw.trim()
            if (s.isEmpty()) continue

            if (s.contains("<MEM_RETRIEVAL>", ignoreCase = true) || s.contains("</MEM_RETRIEVAL>", ignoreCase = true)) {
                val m = queryRegex.find(s)
                if (m != null) return m.groupValues[1].trim()
            }

            if (s.startsWith("query=\"") || s.contains("query=\"")) {
                val m = queryRegex.find(s)
                if (m != null) return m.groupValues[1].trim()
            }
        }
        return null
    }

    fun isNoMemMarker(text: String): Boolean {
        return text.trim().equals("<NO_MEM>", ignoreCase = true)
    }
}
