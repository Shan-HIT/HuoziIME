package com.yuyan.imemodule.llm

data class MemoryPromptSegments(
    val prefixBeforeMemory: String,
    val memoryText: String,
    val suffixAfterMemory: String,
) {
    fun toFullPrompt(): String = prefixBeforeMemory + memoryText + suffixAfterMemory

    /** Base prompt used for splice: remove the memory content but keep tags and newlines. */
    fun toBasePromptWithoutMemory(): String = prefixBeforeMemory + "" + suffixAfterMemory

    companion object {
        private const val MARKER_OPEN = "<memory>\n"
        private const val MARKER_CLOSE = "\n</memory>"

        @JvmStatic
        fun splitFromFullPrompt(fullPrompt: String): MemoryPromptSegments? {
            val i0 = fullPrompt.indexOf(MARKER_OPEN)
            if (i0 < 0) return null
            val i1 = fullPrompt.indexOf(MARKER_CLOSE, startIndex = i0)
            if (i1 < 0) return null

            val prefixBefore = fullPrompt.substring(0, i0 + MARKER_OPEN.length)
            val memory = fullPrompt.substring(i0 + MARKER_OPEN.length, i1)
            val suffixAfter = fullPrompt.substring(i1)

            if (prefixBefore + memory + suffixAfter != fullPrompt) return null
            return MemoryPromptSegments(prefixBefore, memory, suffixAfter)
        }
    }
}
