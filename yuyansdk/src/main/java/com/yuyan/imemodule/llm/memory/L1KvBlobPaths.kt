package com.yuyan.imemodule.llm.memory

import com.yuyan.imemodule.service.data.StyleConfig
import java.io.File


object L1KvBlobPaths {
    const val DIR_NAME: String = "imem_l1_kv"

    fun resolveDir(filesDir: File): File = File(filesDir, DIR_NAME)

    fun resolveFile(dir: File, label: Long, styleKey: String): File {
        return File(dir, "mem_${label}_${styleKey}.bin")
    }

    fun defaultStyleKeys(): List<String> = listOf(
        StyleConfig.STYLE_BUSINESS,
        StyleConfig.STYLE_WARM,
        StyleConfig.STYLE_INRERNET,
    )
}
