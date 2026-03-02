package com.yuyan.imemodule.llm.postprocess

object ThinkTagStripper {

    private const val START = "<think>"
    private const val END = "</think>"

    fun stripOrNull(text: String): String? {
        if (text.isEmpty()) return text

        var out = text
        while (true) {
            val startIdx = out.indexOf(START)
            if (startIdx == -1) break
            val endIdx = out.indexOf(END, startIdx + START.length)
            if (endIdx == -1) return null
            out = out.removeRange(startIdx, endIdx + END.length)
        }

        if (out.contains(END)) {
            out = out.replace(END, "")
        }
        return out
    }
}
