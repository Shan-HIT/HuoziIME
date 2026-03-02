package com.yuyan.imemodule.llm.memory

import android.content.Context
import com.yuyan.imemodule.llm.LLMBridge
import java.io.File

object MemoryTestReset {

    data class ResetResult(
        val ok: Boolean,
        val deletedPaths: List<String>,
        val errors: List<String>,
    )

    fun clearAllForTest(context: Context, handle: Long? = null): ResetResult {
        val errors = ArrayList<String>()
        val deleted = ArrayList<String>()

        if (handle != null && handle != 0L) {
            try {
                LLMBridge.stop(handle)
            } catch (e: Exception) {
                errors.add("stop: ${e.message}")
            }
            try {
                LLMBridge.vectorDbClose(handle)
            } catch (e: Exception) {
                errors.add("vectorDbClose: ${e.message}")
            }
        }

        fun deletePath(f: File) {
            try {
                if (!f.exists()) return
                val ok = if (f.isDirectory) f.deleteRecursively() else f.delete()
                if (ok) deleted.add(f.absolutePath) else errors.add("delete failed: ${f.absolutePath}")
            } catch (e: Exception) {
                errors.add("delete exception: ${f.absolutePath} ${e.message}")
            }
        }

        val filesDir = context.filesDir
        val l2StoreDir = File(filesDir, "imem_l2_store")
        val l2IndexDir = File(filesDir, "imem_l2_index")
        val inputHistory = File(filesDir, "user_input_history.jsonl")
        val inputFragments = File(filesDir, "user_input_fragments.jsonl")
        val l2Progress = File(l2StoreDir, "user_input_history.progress")
        val l2Marks = File(l2StoreDir, "user_input_history.process_marks.jsonl")
        val l3Parametric = File(filesDir, "imem_l3/parametric_logs.jsonl")

        deletePath(l2Progress)
        deletePath(l2Marks)
        deletePath(inputHistory)
        deletePath(inputFragments)
        deletePath(l3Parametric)
        deletePath(l2StoreDir)
        deletePath(l2IndexDir)

        try { l2StoreDir.mkdirs() } catch (_: Exception) {}
        try { l2IndexDir.mkdirs() } catch (_: Exception) {}

        return ResetResult(
            ok = errors.isEmpty(),
            deletedPaths = deleted,
            errors = errors,
        )
    }
}
