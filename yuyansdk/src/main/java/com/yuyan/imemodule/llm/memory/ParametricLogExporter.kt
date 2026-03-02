package com.yuyan.imemodule.llm.memory

import org.json.JSONObject
import java.io.File
import java.io.FileWriter

class ParametricLogExporter(private val file: File) {

    init {
        file.parentFile?.mkdirs()
    }

    fun logEvent(type: String, payload: JSONObject) {
        val obj = JSONObject()
        obj.put("timestamp", System.currentTimeMillis())
        obj.put("type", type)
        obj.put("payload", payload)
        FileWriter(file, true).use { w ->
            w.write(obj.toString())
            w.write("\n")
        }
    }
}
