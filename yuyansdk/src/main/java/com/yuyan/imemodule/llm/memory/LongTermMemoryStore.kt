package com.yuyan.imemodule.llm.memory

import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.security.MessageDigest

data class MemoryRecord(
    val id: String,
    val timestamp: Long,
    val who: String,
    val what: String,
    val detail: String,
    val source: String,
    val vectorLabel: Long? = null,
    val processedAtMs: Long? = null,
    val indexedOk: Boolean? = null,
    val sourceLineIndex: Int? = null,
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("timestamp", timestamp)
            put("who", who)
            put("what", what)
            put("detail", detail)
            put("source", source)
            if (vectorLabel != null) put("vectorLabel", vectorLabel)
            if (processedAtMs != null) put("processedAtMs", processedAtMs)
            if (indexedOk != null) put("indexedOk", indexedOk)
            if (sourceLineIndex != null) put("sourceLineIndex", sourceLineIndex)
        }
    }
}

class LongTermMemoryStore(private val rootDir: File) {

    private val memoriesFile: File = File(rootDir, "memories.jsonl")
    private val deletedIdsFile: File = File(rootDir, "deleted_ids.txt")

    init {
        if (!rootDir.exists()) rootDir.mkdirs()
    }

    /**
     * Test/debug-only hard reset.
     * Physically removes on-disk memory records and tombstones.
     */
    fun clearAll(): Boolean {
        return try {
            if (rootDir.exists()) {
                rootDir.deleteRecursively()
            }
            rootDir.mkdirs()
            true
        } catch (_: Exception) {
            false
        }
    }

    fun append(record: MemoryRecord) {
        FileWriter(memoriesFile, true).use { w ->
            w.write(record.toJson().toString())
            w.write("\n")
        }
    }

    fun list(limit: Int = 200, includeDeleted: Boolean = false): List<MemoryRecord> {
        val deleted = if (includeDeleted) emptySet() else loadDeletedIds()
        val out = ArrayList<MemoryRecord>()
        if (!memoriesFile.exists()) return out
        try {
            memoriesFile.forEachLine { line ->
                val s = line.trim()
                if (s.isEmpty()) return@forEachLine
                try {
                    val obj = JSONObject(s)
                    val rec = MemoryRecord(
                        id = obj.optString("id", "").trim(),
                        timestamp = obj.optLong("timestamp", 0L),
                        who = obj.optString("who", "").trim(),
                        what = obj.optString("what", "").trim(),
                        detail = obj.optString("detail", "").trim(),
                        source = obj.optString("source", "").trim(),
                        vectorLabel = if (obj.has("vectorLabel")) obj.optLong("vectorLabel") else null,
                        processedAtMs = if (obj.has("processedAtMs")) obj.optLong("processedAtMs") else null,
                        indexedOk = if (obj.has("indexedOk")) obj.optBoolean("indexedOk") else null,
                        sourceLineIndex = if (obj.has("sourceLineIndex")) obj.optInt("sourceLineIndex") else null,
                    )
                    if (rec.id.isNotEmpty() && (includeDeleted || !deleted.contains(rec.id))) {
                        out.add(rec)
                    }
                } catch (_: Exception) {
                }
            }
        } catch (_: Exception) {
            return emptyList()
        }

        out.sortByDescending { it.timestamp }
        if (out.size > limit) return out.subList(0, limit)
        return out
    }

    fun softDelete(id: String): Boolean {
        val clean = id.trim()
        if (clean.isEmpty()) return false
        return try {
            rootDir.mkdirs()
            FileWriter(deletedIdsFile, true).use { w ->
                w.write(clean)
                w.write("\n")
            }
            true
        } catch (_: Exception) {
            false
        }
    }

    fun loadDeletedIds(): Set<String> {
        if (!deletedIdsFile.exists()) return emptySet()
        return try {
            deletedIdsFile.readLines()
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .toSet()
        } catch (_: Exception) {
            emptySet()
        }
    }

    /** Returns vector labels that correspond to soft-deleted records. */
    fun loadDeletedVectorLabels(): Set<Long> {
        val deleted = loadDeletedIds()
        if (deleted.isEmpty()) return emptySet()
        val labels = HashSet<Long>()
        if (!memoriesFile.exists()) return labels
        try {
            memoriesFile.forEachLine { line ->
                val s = line.trim()
                if (s.isEmpty()) return@forEachLine
                try {
                    val obj = JSONObject(s)
                    val id = obj.optString("id", "").trim()
                    if (id.isEmpty() || !deleted.contains(id)) return@forEachLine
                    if (obj.has("vectorLabel")) {
                        val lab = obj.optLong("vectorLabel")
                        labels.add(lab)
                    }
                } catch (_: Exception) {
                }
            }
        } catch (_: Exception) {
        }
        return labels
    }

    companion object {
        fun stableId(timestamp: Long, content: String): String {
            val md = MessageDigest.getInstance("SHA-256")
            val bytes = md.digest((timestamp.toString() + "\n" + content).toByteArray(Charsets.UTF_8))
            return bytes.take(12).joinToString("") { b -> "%02x".format(b) }
        }
    }
}
