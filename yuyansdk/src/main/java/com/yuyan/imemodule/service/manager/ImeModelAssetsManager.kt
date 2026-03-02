package com.yuyan.imemodule.service.manager

import android.content.Context
import com.yuyan.imemodule.utils.LogUtil
import java.io.File
import java.io.FileOutputStream

class ImeModelAssetsManager(private val context: Context) {
    private val TAG = "IMEM-OS-Assets"

    fun ensureEmbeddingModelCopied(): String {
        val startTime = System.currentTimeMillis()
        val assetName = "bge-small-zh-v1.5-q8_0.gguf"
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

        LogUtil.i(TAG, "Assets", "正在解压 Embedding 模型 (首次运行)...")
        try {
            context.assets.open(assetName).use { ins ->
                FileOutputStream(outFile).use { os ->
                    val buffer = ByteArray(8 * 1024)
                    var read: Int
                    var totalRead = 0L
                    while (ins.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                        totalRead += read
                    }
                    os.flush()

                    val duration = System.currentTimeMillis() - startTime
                    val speed = if (duration > 0) (totalRead / 1024.0 / 1024.0) / (duration / 1000.0) else 0.0
                    LogUtil.i(TAG, "Assets", "✅ Embedding 模型解压完成 | 耗时: ${duration}ms | 速率: %.2f MB/s".format(speed))
                }
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Assets", "Embedding 模型解压失败: $e")
        }
        return outFile.absolutePath
    }

    fun ensureMagicLoraCopied(): String {
        val startTime = System.currentTimeMillis()
        val assetName = "magicdata_ime_lora_v6.gguf"
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

        LogUtil.i(TAG, "Assets", "正在解压 Lora 权重 (首次运行)...")
        try {
            context.assets.open(assetName).use { ins ->
                FileOutputStream(outFile).use { os ->
                    val buffer = ByteArray(8 * 1024)
                    var read: Int
                    var totalRead = 0L
                    while (ins.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                        totalRead += read
                    }
                    os.flush()

                    val duration = System.currentTimeMillis() - startTime
                    val speed = if (duration > 0) (totalRead / 1024.0 / 1024.0) / (duration / 1000.0) else 0.0
                    LogUtil.i(TAG, "Assets", "✅ Lora 解压完成 | 耗时: ${duration}ms | 速率: %.2f MB/s".format(speed))
                }
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Assets", "Lora 解压失败: $e")
        }
        return outFile.absolutePath
    }

    fun ensureModelCopied(): String {
        val startTime = System.currentTimeMillis()
        val assetName = "scirime_grpo_v2_744-q4_0.gguf"
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

        LogUtil.i(TAG, "Assets", "正在解压模型文件 (首次运行)...")
        try {
            context.assets.open(assetName).use { ins ->
                FileOutputStream(outFile).use { os ->
                    val buffer = ByteArray(8 * 1024)
                    var read: Int
                    var totalRead = 0L
                    while (ins.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                        totalRead += read
                    }
                    os.flush()

                    val duration = System.currentTimeMillis() - startTime
                    val speed = if (duration > 0) (totalRead / 1024.0 / 1024.0) / (duration / 1000.0) else 0.0
                    LogUtil.i(TAG, "Assets", "✅ 模型解压完成 | 耗时: ${duration}ms | 速率: %.2f MB/s".format(speed))
                }
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Assets", "模型解压失败: $e")
        }
        return outFile.absolutePath
    }
}
