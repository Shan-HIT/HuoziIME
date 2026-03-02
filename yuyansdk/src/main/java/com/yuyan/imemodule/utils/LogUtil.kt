package com.yuyan.imemodule.utils

import android.util.Log
import android.view.KeyEvent
import com.yuyan.imemodule.BuildConfig
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * 日志管理工具
 */
object LogUtil {
    enum class Level { VERBOSE, DEBUG, INFO, WARN, ERROR }

    enum class Category {
        CORE,
        INPUT,
        GHOST,
        UI,
        LLM,
        MEM,
        PERF,
    }

    private val BASE_TAG = "IMEM"

    // ===== Policy =====
    // BuildConfig.DEBUG is the default baseline.
    // - Debug build: everything is enabled by default.
    // - Release build: only ERROR is enabled by default.
    // Runtime overrides exist but are evaluated at call-time (no cached vals).
    private val runtimeMasterEnabled = AtomicBoolean(BuildConfig.DEBUG)
    private val runtimeAllowInfoInRelease = AtomicBoolean(false)
    private val runtimeAllowWarnInRelease = AtomicBoolean(false)

    // Category toggles (evaluated at call-time).
    private val inputEnabled = AtomicBoolean(BuildConfig.DEBUG)
    private val ghostEnabled = AtomicBoolean(BuildConfig.DEBUG)
    private val uiEnabled = AtomicBoolean(BuildConfig.DEBUG)
    private val llmEnabled = AtomicBoolean(BuildConfig.DEBUG)
    private val memEnabled = AtomicBoolean(BuildConfig.DEBUG)

    // Perf mode: silence verbose logs to avoid skewing benchmarks.
    private val perfModeDepth = AtomicInteger(0)

    // Basic rate limiter for high-frequency callbacks.
    private val lastLogAtMs = ConcurrentHashMap<String, AtomicLong>()

    /**
     * Global log switch.
     * NOTE: Unlike the old implementation, this actually takes effect immediately.
     */
    @JvmStatic
    fun setOpenLog(isOpen: Boolean) {
        runtimeMasterEnabled.set(isOpen)
    }

    /** Enable/disable input (key/commit/composing/cursor) logs. */
    @JvmStatic
    fun setInputLogEnabled(enabled: Boolean) {
        inputEnabled.set(enabled)
    }

    @JvmStatic
    fun setGhostLogEnabled(enabled: Boolean) {
        ghostEnabled.set(enabled)
    }

    @JvmStatic
    fun setUiLogEnabled(enabled: Boolean) {
        uiEnabled.set(enabled)
    }

    @JvmStatic
    fun setLlmLogEnabled(enabled: Boolean) {
        llmEnabled.set(enabled)
    }

    @JvmStatic
    fun setMemLogEnabled(enabled: Boolean) {
        memEnabled.set(enabled)
    }

    /**
     * Optional: allow INFO/WARN logs in release at runtime.
     * Default is false to keep release noise low.
     */
    @JvmStatic
    fun setReleaseVerbosity(allowInfo: Boolean = false, allowWarn: Boolean = false) {
        runtimeAllowInfoInRelease.set(allowInfo)
        runtimeAllowWarnInRelease.set(allowWarn)
    }

    @JvmStatic
    fun enterPerfMode(): Int {
        return perfModeDepth.incrementAndGet()
    }

    @JvmStatic
    fun exitPerfMode(token: Int) {
        // Best-effort decrement: do not go negative.
        while (true) {
            val cur = perfModeDepth.get()
            if (cur <= 0) return
            if (perfModeDepth.compareAndSet(cur, cur - 1)) return
        }
    }

    inline fun <T> withPerfMode(block: () -> T): T {
        val t = enterPerfMode()
        return try {
            block()
        } finally {
            exitPerfMode(t)
        }
    }

    @JvmStatic
    fun shouldLog(level: Level, category: Category): Boolean {
        if (!runtimeMasterEnabled.get()) return false

        // Perf mode: keep only WARN/ERROR unless category is PERF.
        val perfDepth = perfModeDepth.get()
        if (perfDepth > 0 && category != Category.PERF) {
            if (level != Level.ERROR && level != Level.WARN) return false
        }

        // Release default policy: ERROR only.
        if (!BuildConfig.DEBUG) {
            if (level == Level.ERROR) return true
            if (level == Level.WARN) return runtimeAllowWarnInRelease.get()
            if (level == Level.INFO) return runtimeAllowInfoInRelease.get()
            return false
        }

        // Debug build policy: category toggles.
        return when (category) {
            Category.INPUT -> inputEnabled.get()
            Category.GHOST -> ghostEnabled.get()
            Category.UI -> uiEnabled.get()
            Category.LLM -> llmEnabled.get()
            Category.MEM -> memEnabled.get()
            else -> true
        }
    }

    /**
     * Rate-limit helper to reduce log spam on hot callbacks.
     * Returns true if allowed to log now.
     */
    @JvmStatic
    fun rateLimit(key: String, intervalMs: Long): Boolean {
        if (intervalMs <= 0) return true
        val now = System.currentTimeMillis()
        val last = lastLogAtMs.getOrPut(key) { AtomicLong(0L) }
        val prev = last.get()
        if (now - prev < intervalMs) return false
        return last.compareAndSet(prev, now)
    }

    private fun emit(level: Level, tag: String, msg: String) {
        when (level) {
            Level.VERBOSE -> Log.v(BASE_TAG, "$tag|$msg")
            Level.DEBUG -> Log.d(BASE_TAG, "$tag|$msg")
            Level.INFO -> Log.i(BASE_TAG, "$tag|$msg")
            Level.WARN -> Log.w(BASE_TAG, "$tag|$msg")
            Level.ERROR -> Log.e(BASE_TAG, "$tag|$msg")
        }
    }

    private inline fun emitLazy(level: Level, category: Category, tag: String, crossinline msg: () -> String) {
        if (!shouldLog(level, category)) return
        emit(level, tag, msg())
    }

    fun v(TAG: String, method: String, msg: String) = emitLazy(Level.VERBOSE, Category.CORE, TAG) { "$method()-->$msg" }

    @JvmStatic
    fun d(TAG: String, msg: String) = emitLazy(Level.DEBUG, Category.CORE, TAG) { msg }

    @JvmStatic
    fun d(TAG: String, method: String, msg: String) = emitLazy(Level.DEBUG, Category.CORE, TAG) { "$method()-->$msg" }

    fun i(TAG: String, method: String, msg: String) = emitLazy(Level.INFO, Category.CORE, TAG) { "$method()-->$msg" }

    fun w(TAG: String, method: String, msg: String) = emitLazy(Level.WARN, Category.CORE, TAG) { "$method()-->$msg" }

    @JvmStatic
    fun e(TAG: String, method: String, msg: String) = emitLazy(Level.ERROR, Category.CORE, TAG) { "$method()-->$msg" }

    // ==================== Structured events ====================
    @JvmStatic
    fun event(category: Category, tag: String, name: String, details: String = "") {
        emitLazy(Level.INFO, category, tag) { if (details.isBlank()) name else "$name | $details" }
    }

    @JvmStatic
    fun eventD(category: Category, tag: String, name: String, details: String = "") {
        emitLazy(Level.DEBUG, category, tag) { if (details.isBlank()) name else "$name | $details" }
    }

    @JvmStatic
    fun eventW(category: Category, tag: String, name: String, details: String = "") {
        emitLazy(Level.WARN, category, tag) { if (details.isBlank()) name else "$name | $details" }
    }

    @JvmStatic
    fun eventE(category: Category, tag: String, name: String, details: String = "") {
        emitLazy(Level.ERROR, category, tag) { if (details.isBlank()) name else "$name | $details" }
    }

    // ==================== 输入日志专用方法 ====================

    /**
     * 记录按键事件
     * @param keyCode 按键代码
     * @param keyChar 按键字符（如果有）
     * @param action 按键动作（DOWN/UP）
     */
    @JvmStatic
    fun logKeyEvent(TAG: String, keyCode: Int, keyChar: Int = 0, action: String = "DOWN") {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val keyCodeName = KeyEvent.keyCodeToString(keyCode)
        val charInfo = if (keyChar > 0) " | char: '${keyChar.toChar()}' (0x${keyChar.toString(16)})" else ""
        emit(Level.INFO, TAG, "[INPUT] KEY_EVENT | keyCode: $keyCode ($keyCodeName) | action: $action$charInfo")
    }

    /**
     * 记录输入框文本提交
     * @param text 提交的文本
     * @param cursorPos 光标位置（可选）
     * @param source 提交来源（commitText/commitGhostText/commitDecInfoText等）
     */
    @JvmStatic
    fun logCommitText(TAG: String, text: String, cursorPos: Int = -1, source: String = "commitText") {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val cursorInfo = if (cursorPos >= 0) " | cursorPos: $cursorPos" else ""
        val displayText = if (text.length > 50) "\"${text.take(50)}...\"(len=${text.length})" else "\"$text\""
        emit(Level.INFO, TAG, "[INPUT] COMMIT_TEXT | source: $source | text: $displayText$cursorInfo")
    }

    /**
     * 记录组合文本设置
     * @param text 组合文本
     * @param newCursorPosition 新光标位置
     */
    @JvmStatic
    fun logSetComposingText(TAG: String, text: String, newCursorPosition: Int = 1) {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val displayText = if (text.length > 50) "\"${text.take(50)}...\"(len=${text.length})" else "\"$text\""
        emit(Level.INFO, TAG, "[INPUT] SET_COMPOSING | text: $displayText | newCursorPos: $newCursorPosition")
    }

    /**
     * 记录文本删除
     * @param beforeText 删除前的文本
     * @param afterText 删除后的文本
     * @param length 删除的长度
     */
    @JvmStatic
    fun logDeleteText(TAG: String, beforeText: String, afterText: String, length: Int = 1) {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val beforeDisplay = if (beforeText.length > 30) "\"${beforeText.take(30)}...\"" else "\"$beforeText\""
        val afterDisplay = if (afterText.length > 30) "\"${afterText.take(30)}...\"" else "\"$afterText\""
        emit(Level.INFO, TAG, "[INPUT] DELETE_TEXT | before: $beforeDisplay | after: $afterDisplay | deletedLength: $length")
    }

    /**
     * 记录光标位置变化
     * @param oldSelStart 旧选择起始位置
     * @param oldSelEnd 旧选择结束位置
     * @param newSelStart 新选择起始位置
     * @param newSelEnd 新选择结束位置
     */
    @JvmStatic
    fun logCursorChange(TAG: String, oldSelStart: Int, oldSelEnd: Int, newSelStart: Int, newSelEnd: Int) {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val oldRange = if (oldSelStart == oldSelEnd) "pos=$oldSelStart" else "range=[$oldSelStart,$oldSelEnd]"
        val newRange = if (newSelStart == newSelEnd) "pos=$newSelStart" else "range=[$newSelStart,$newSelEnd]"
        emit(Level.INFO, TAG, "[INPUT] CURSOR_CHANGE | old: $oldRange | new: $newRange")
    }

    /**
     * 记录IME状态变化
     * @param oldState 旧状态
     * @param newState 新状态
     * @param reason 变化原因
     */
    @JvmStatic
    fun logStateChange(TAG: String, oldState: String, newState: String, reason: String = "") {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val reasonInfo = if (reason.isNotEmpty()) " | reason: $reason" else ""
        emit(Level.INFO, TAG, "[INPUT] STATE_CHANGE | $oldState -> $newState$reasonInfo")
    }

    /**
     * 记录候选词列表更新
     * @param candidates 候选词列表
     * @param activeIndex 当前激活的候选词索引
     */
    @JvmStatic
    fun logCandidatesUpdate(TAG: String, candidates: List<Any>, activeIndex: Int = 0) {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val displayCandidates = candidates.take(5).joinToString(", ") { "\"$it\"" }
        val moreInfo = if (candidates.size > 5) "... (${candidates.size} total)" else ""
        emit(Level.INFO, TAG, "[INPUT] CANDIDATES_UPDATE | count: ${candidates.size} | active: $activeIndex | [$displayCandidates$moreInfo]")
    }

    /**
     * 记录候选词选择
     * @param candidateIndex 选择的候选词索引
     * @param candidateText 选择的候选词文本
     * @param comment 候选词注释（如果有）
     */
    @JvmStatic
    fun logCandidateSelected(TAG: String, candidateIndex: Int, candidateText: String, comment: String = "") {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val commentInfo = if (comment.isNotEmpty()) " | comment: $comment" else ""
        emit(Level.INFO, TAG, "[INPUT] CANDIDATE_SELECTED | index: $candidateIndex | text: \"$candidateText\"$commentInfo")
    }

    /**
     * 记录输入框内容快照（调试用）
     * @param textBeforeCursor 光标前的文本
     * @param textAfterCursor 光标后的文本
     * @param selectedText 选中的文本（如果有）
     */
    @JvmStatic
    fun logInputSnapshot(TAG: String, textBeforeCursor: String, textAfterCursor: String = "", selectedText: String = "") {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        val beforeDisplay = if (textBeforeCursor.length > 50) "\"${textBeforeCursor.take(50)}...\"(len=${textBeforeCursor.length})" else "\"$textBeforeCursor\""
        val afterDisplay = if (textAfterCursor.length > 30) "\"${textAfterCursor.take(30)}...\"" else "\"$textAfterCursor\""
        val selectedDisplay = if (selectedText.isNotEmpty()) " | selected: \"$selectedText\"" else ""
        emit(Level.INFO, TAG, "[INPUT] INPUT_SNAPSHOT | beforeCursor: $beforeDisplay | afterCursor: $afterDisplay$selectedDisplay")
    }

    /**
     * 记录特殊操作（如Ghost Text、联想、分词等）
     * @param operation 操作名称
     * @param details 操作详情
     */
    @JvmStatic
    fun logSpecialOperation(TAG: String, operation: String, details: String) {
        if (!shouldLog(Level.INFO, Category.INPUT)) return
        emit(Level.INFO, TAG, "[INPUT] SPECIAL_OP | $operation | $details")
    }
}
