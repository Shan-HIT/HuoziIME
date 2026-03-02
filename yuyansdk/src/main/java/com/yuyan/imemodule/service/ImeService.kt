package com.yuyan.imemodule.service

import android.content.res.Configuration
import android.inputmethodservice.InputMethodService
import android.os.SystemClock
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.CursorAnchorInfo
import android.view.inputmethod.ExtractedTextRequest
import android.view.inputmethod.InputConnection
import android.widget.Toast
import com.yuyan.imemodule.data.emojicon.YuyanEmojiCompat
import com.yuyan.imemodule.data.theme.Theme
import com.yuyan.imemodule.data.theme.ThemeManager.OnThemeChangeListener
import com.yuyan.imemodule.data.theme.ThemeManager.addOnChangedListener
import com.yuyan.imemodule.data.theme.ThemeManager.onSystemDarkModeChange
import com.yuyan.imemodule.data.theme.ThemeManager.removeOnChangedListener
import com.yuyan.imemodule.data.theme.ThemeManager.setNormalModeTheme
import com.yuyan.imemodule.data.theme.ThemePreset
import com.yuyan.imemodule.keyboard.InputView
import com.yuyan.imemodule.keyboard.KeyboardManager
import com.yuyan.imemodule.keyboard.container.ClipBoardContainer
import com.yuyan.imemodule.prefs.AppPrefs.Companion.getInstance
import com.yuyan.imemodule.prefs.behavior.SkbMenuMode
import com.yuyan.imemodule.service.data.*
import com.yuyan.imemodule.service.manager.ImeLLMManager
import com.yuyan.imemodule.singleton.EnvironmentSingleton
import com.yuyan.imemodule.utils.KeyboardLoaderUtil
import com.yuyan.imemodule.utils.LogUtil
import com.yuyan.imemodule.utils.StringUtils
import com.yuyan.imemodule.utils.isDarkMode
import com.yuyan.imemodule.utils.toast
import com.yuyan.imemodule.view.preference.ManagedPreference
import kotlinx.coroutines.*
import org.json.JSONObject
import splitties.bitflags.hasFlag
import java.io.File
import java.io.FileWriter
import java.security.MessageDigest
import java.util.Locale

class ImeService : InputMethodService() {
    private val TAG = "IMEM-OS-Service"
    private val instanceId: String by lazy { Integer.toHexString(System.identityHashCode(this)) }
    
    internal lateinit var llmManager: ImeLLMManager

    private var mRootContainer: android.widget.FrameLayout? = null
    internal lateinit var mInputView: InputView
    private var controlPanel: LLMControlPanel? = null
    
    private var mFeedbackView: android.widget.TextView? = null
    private val mFeedbackHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var mFeedbackRunnable: Runnable? = null

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(serviceJob + Dispatchers.Main.immediate)
    private val serviceIoScope = CoroutineScope(serviceJob + Dispatchers.IO)
    
    private var isWindowShown = false
    @Volatile
    var lastCursorAnchorInfo: CursorAnchorInfo? = null

    private val trajectoryLock = Any()
    private val inputTrajectoryBuffer = StringBuilder()
    private var trajectoryStartedAtMs: Long = 0L
    private var lastFlushedTrajectorySha: String? = null
    private var lastEditorPackageName: String = "unknown"
    
    private val onThemeChangeListener = OnThemeChangeListener { _: Theme? ->
        if (::mInputView.isInitialized) mInputView.updateTheme()
    }
    
    private val clipboardUpdateContent = getInstance().internal.clipboardUpdateContent
    private val clipboardUpdateContentListener =
        ManagedPreference.OnChangeListener<String> { _, value ->
            if (getInstance().clipboard.clipboardSuggestion.getValue()) {
                if (value.isNotBlank()) {
                    if (::mInputView.isInitialized && mInputView.isShown) {
                        if (KeyboardManager.instance.currentContainer is ClipBoardContainer
                            && (KeyboardManager.instance.currentContainer as ClipBoardContainer).getMenuMode() == SkbMenuMode.ClipBoard
                        ) {
                            (KeyboardManager.instance.currentContainer as ClipBoardContainer).showClipBoardView(SkbMenuMode.ClipBoard)
                        } else {
                            mInputView.showSymbols(arrayOf(value))
                        }
                    }
                }
            }
        }

    private var contextSyncReceiver: android.content.BroadcastReceiver? = null

    companion object {
        private const val ACTION_CONTEXT_SYNC = "com.huoziime.chat.CONTEXT_SYNC"
        private const val ACTION_CONTEXT_SYNC_REQUEST = "com.huoziime.chat.CONTEXT_SYNC_REQUEST"
        private const val LEGACY_ACTION_CONTEXT_SYNC = "com.scirime.chat.CONTEXT_SYNC"

        private const val EXTRA_MCP_MESSAGE = "mcp_message"
        private const val EXTRA_MCP_REQUEST = "mcp_request"
        private const val EXTRA_CONTEXT_DATA = "context_data"

        private const val MCP_JSONRPC_VERSION = "2.0"
        private const val MCP_METHOD_CONTEXT_GET = "huozi/context/get"
        private const val CONTEXT_SCHEMA_HUOZI_V1 = "huozi.context.v1"
        private const val DEFAULT_CONTEXT_LIMIT = 5
    }

    override fun onCreate() {
        super.onCreate()
        val startTime = System.currentTimeMillis()
        LogUtil.event(
            LogUtil.Category.CORE,
            TAG,
            "service_create_begin",
            "instance=$instanceId thread=${Thread.currentThread().name}"
        )
        LogUtil.i(TAG, "Lifecycle", ">>> Service onCreate 开始 <<<")
        
        llmManager = ImeLLMManager(this)
        
        addOnChangedListener(onThemeChangeListener)
        clipboardUpdateContent.registerOnChangeListener(clipboardUpdateContentListener)
        
        serviceIoScope.launch {
            llmManager.initIMEMSystem()
            serviceScope.launch {
                applyThemeForStyle(llmManager.currentStyleMode, reason = "init_restored_style")
            }
            val duration = System.currentTimeMillis() - startTime
            LogUtil.event(
                LogUtil.Category.CORE,
                TAG,
                "service_ready",
                "instance=$instanceId initMs=${duration}"
            )
            LogUtil.i(TAG, "Lifecycle", ">>> Service 核心服务就绪 | 累计耗时: ${duration}ms <<<")
        }
    }
    
    override fun onCreateInputView(): View {
        mInputView = InputView(baseContext, this)
        LogUtil.event(
            LogUtil.Category.UI,
            TAG,
            "create_input_view",
            "instance=$instanceId inputView=${Integer.toHexString(System.identityHashCode(mInputView))}"
        )
        val root = android.widget.FrameLayout(this)
        root.layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        mInputView.layoutParams = android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        root.addView(mInputView)
        mRootContainer = root
        return root
    }

    override fun setInputView(view: View) {
        super.setInputView(view)
        LogUtil.eventD(
            LogUtil.Category.UI,
            TAG,
            "set_input_view",
            "instance=$instanceId view=${view.javaClass.simpleName}"
        )
        val layoutParams = view.layoutParams
        if (layoutParams != null && layoutParams.height != ViewGroup.LayoutParams.MATCH_PARENT) {
            layoutParams.height = ViewGroup.LayoutParams.MATCH_PARENT
            view.layoutParams = layoutParams
        }
    }

    fun getTextBeforeCursor(length: Int): String {
        val text = currentInputConnection?.getTextBeforeCursor(length, 0).toString()
        if (LogUtil.rateLimit("ic.beforeCursor.$length", 250)) {
            LogUtil.logInputSnapshot(TAG, text)
        }
        return text
    }

    fun getAllTextBeforeCursor(maxFallbackChars: Int = 4000): String {
        try {
            val ic = currentInputConnection
            if (ic != null) {
                val req = ExtractedTextRequest()
                val extracted = ic.getExtractedText(req, 0)
                val fullText = extracted?.text?.toString()
                val selStart = extracted?.selectionStart ?: -1
                if (!fullText.isNullOrEmpty() && selStart >= 0) {
                    val safeStart = selStart.coerceIn(0, fullText.length)
                    val before = fullText.substring(0, safeStart)
                    if (LogUtil.rateLimit("ic.beforeCursor.full", 250)) {
                        LogUtil.event(
                            LogUtil.Category.INPUT,
                            TAG,
                            "input_snapshot_full",
                            "beforeCursorLen=${before.length} extractedLen=${fullText.length} method=extractedText"
                        )
                    }
                    return before
                }
            }
        } catch (e: Exception) {
            if (LogUtil.rateLimit("ic.beforeCursor.full.err", 1000)) {
                LogUtil.eventW(
                    LogUtil.Category.INPUT,
                    TAG,
                    "input_snapshot_full_failed",
                    "ex=${e.javaClass.simpleName}:${e.message}"
                )
            }
        }

        val fallback = getTextBeforeCursor(maxFallbackChars)
        if (LogUtil.rateLimit("ic.beforeCursor.full.fallback", 250)) {
            LogUtil.event(
                LogUtil.Category.INPUT,
                TAG,
                "input_snapshot_full",
                "beforeCursorLen=${fallback.length} method=fallback maxChars=$maxFallbackChars"
            )
        }
        return fallback
    }

    fun showAiSuggestion(tokens: Array<String>) {
        LogUtil.event(
            LogUtil.Category.GHOST,
            TAG,
            "suggestions_to_view",
            "count=${tokens.size}"
        )
        serviceScope.launch {
            if (::mInputView.isInitialized) {
                mInputView.showAiSuggestion(tokens)
            }
        }
    }
    
    fun updateUiState(isGenerating: Boolean) {
        LogUtil.eventD(
            LogUtil.Category.UI,
            TAG,
            "ui_state_update",
            "isGenerating=$isGenerating"
        )
        serviceScope.launch {
            if (::mInputView.isInitialized && isGenerating) {
                mInputView.showGeneratingStatus()
            }
        }
    }

    private fun resolveThemeForStyle(style: String): Theme {
        return when (style) {
            StyleConfig.STYLE_BUSINESS -> ThemePreset.TechAI_BUSINESS
            StyleConfig.STYLE_WARM -> ThemePreset.TechAI_WARM
            StyleConfig.STYLE_INRERNET -> ThemePreset.TechAI_INRERNET
            else -> ThemePreset.TechAI_BUSINESS
        }
    }

    private fun applyThemeForStyle(style: String, reason: String) {
        val targetTheme = resolveThemeForStyle(style)
        setNormalModeTheme(targetTheme)
        LogUtil.eventD(
            LogUtil.Category.UI,
            TAG,
            "style_theme_sync",
            "reason=$reason style=$style theme=${targetTheme.name}"
        )
        if (::mInputView.isInitialized) {
            KeyboardLoaderUtil.instance.clearKeyboardMap()
            KeyboardManager.instance.clearKeyboard()
            mInputView.updateTheme()
            KeyboardManager.instance.switchKeyboard()
        }
    }

    fun toggleAiCompletion() {
        val before = llmManager.isAiCompletionEnabled
        llmManager.isAiCompletionEnabled = !llmManager.isAiCompletionEnabled
        LogUtil.event(
            LogUtil.Category.LLM,
            TAG,
            "ai_completion_toggle",
            "before=$before after=${llmManager.isAiCompletionEnabled}"
        )
        if (llmManager.isAiCompletionEnabled) {
            showFeedback("🟢 AI 推理已开启")
            LogUtil.i(TAG, "Control", "AI功能已启用")
            llmManager.requestCompletion(reason = "ai_enabled_toggle")
        } else {
            showFeedback("🚫 AI 推理已关闭")
            llmManager.stopCompletion(reason = "ai_disabled_toggle")
            if (::mInputView.isInitialized) mInputView.hideAiSuggestion("ai_disabled")
        }
    }

    fun switchStyle() {
        openControlPanel(LLMControlPanel.MODE_STYLE)
    }
    
    fun executeStyleSwitch(nextStyle: String) {
        serviceScope.launch {
            applyThemeForStyle(nextStyle, reason = "manual_style_switch")
        }
        llmManager.executeStyleSwitch(nextStyle)
    }

    fun showFeedback(message: String) {
        if (message.isBlank()) return
        LogUtil.event(
            LogUtil.Category.UI,
            TAG,
            "feedback_show",
            "len=${message.length}"
        )
        mFeedbackHandler.post {
            try {
                mFeedbackRunnable?.let { mFeedbackHandler.removeCallbacks(it) }
                if (mFeedbackView != null && mFeedbackView!!.parent != null) {
                    mRootContainer?.removeView(mFeedbackView)
                }
                val density = resources.displayMetrics.density
                val paddingH = (16 * density).toInt()
                val paddingV = (10 * density).toInt()
                val radius = (20 * density).toFloat()
                val marginBottom = (80 * density).toInt()
                val textView = android.widget.TextView(this).apply {
                    text = message
                    setTextColor(android.graphics.Color.WHITE)
                    textSize = 15f
                    setPadding(paddingH, paddingV, paddingH, paddingV)
                    gravity = android.view.Gravity.CENTER
                    background = android.graphics.drawable.GradientDrawable().apply {
                        setColor(android.graphics.Color.parseColor("#D9333333"))
                        cornerRadius = radius
                    }
                    maxWidth = (resources.displayMetrics.widthPixels * 0.8).toInt()
                }
                val params = android.widget.FrameLayout.LayoutParams(
                    android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                    android.view.ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    gravity = android.view.Gravity.BOTTOM or android.view.Gravity.CENTER_HORIZONTAL
                    bottomMargin = marginBottom
                }
                mRootContainer?.addView(textView, params)
                mFeedbackView = textView
                mFeedbackRunnable = Runnable {
                    if (mFeedbackView == textView) {
                        mRootContainer?.removeView(textView)
                        mFeedbackView = null
                        LogUtil.eventD(LogUtil.Category.UI, TAG, "feedback_hide", "reason=timeout")
                    }
                }
                mFeedbackHandler.postDelayed(mFeedbackRunnable!!, 2000)
            } catch (e: Exception) {
                LogUtil.e(TAG, "Feedback", "Hint显示失败: ${e.message}")
            }
        }
    }

    override fun onStartInputView(editorInfo: EditorInfo?, restarting: Boolean) {
        val t = SystemClock.elapsedRealtime()
        LogUtil.event(
            LogUtil.Category.CORE,
            TAG,
            "start_input_view",
            "restarting=$restarting pkg=${editorInfo?.packageName ?: "unknown"} inputType=${editorInfo?.inputType ?: -1} imeOptions=${editorInfo?.imeOptions ?: -1}"
        )
        LogUtil.i(TAG, "Lifecycle", ">>> onStartInputView | Restarting: $restarting | Package: ${editorInfo?.packageName ?: "Unknown"}")
        lastEditorPackageName = editorInfo?.packageName ?: "unknown"
        editorInfo?.let { YuyanEmojiCompat.setEditorInfo(it) }
        currentInputConnection?.requestCursorUpdates(
            InputConnection.CURSOR_UPDATE_IMMEDIATE or InputConnection.CURSOR_UPDATE_MONITOR
        )
        applyThemeForStyle(llmManager.currentStyleMode, reason = "start_input_view")
        if (::mInputView.isInitialized && editorInfo != null) mInputView.onStartInputView(editorInfo, restarting)
        LogUtil.eventD(
            LogUtil.Category.CORE,
            TAG,
            "start_input_view_done",
            "elapsedMs=${SystemClock.elapsedRealtime() - t}"
        )
    }

    override fun onUpdateCursorAnchorInfo(cursorAnchorInfo: CursorAnchorInfo?) {
        super.onUpdateCursorAnchorInfo(cursorAnchorInfo)
        if (cursorAnchorInfo != null) {
            lastCursorAnchorInfo = cursorAnchorInfo
            if (LogUtil.rateLimit("cursorAnchor", 250)) {
                LogUtil.eventD(
                    LogUtil.Category.INPUT,
                    TAG,
                    "cursor_anchor_update",
                    "hasCursorAnchor=true ghostActive=${::mInputView.isInitialized && mInputView.isGhostTextActive()}"
                )
            }
            if (::mInputView.isInitialized && mInputView.isGhostTextActive()) {
                mInputView.repositionGhostText(cursorAnchorInfo)
            }
        }
    }

    override fun onDestroy() {
        LogUtil.event(LogUtil.Category.CORE, TAG, "service_destroy", "instance=$instanceId")
        LogUtil.i(TAG, "Lifecycle", ">>> Service onDestroy <<<")
        super.onDestroy()
        try {
            mFeedbackRunnable?.let { mFeedbackHandler.removeCallbacks(it) }
            mFeedbackRunnable = null
            mFeedbackHandler.removeCallbacksAndMessages(null)
        } catch (_: Exception) {
        }
        try { serviceJob.cancel() } catch (_: Exception) {}
        if (::mInputView.isInitialized) mInputView.resetToIdleState()
        removeOnChangedListener(onThemeChangeListener)
        clipboardUpdateContent.unregisterOnChangeListener(clipboardUpdateContentListener)
        
        if (contextSyncReceiver != null) {
            unregisterReceiver(contextSyncReceiver)
            contextSyncReceiver = null
        }
        
        llmManager.release()
    }
    
    override fun onWindowShown() {
        if (isWindowShown) return
        val tStart = SystemClock.elapsedRealtime()
        LogUtil.event(LogUtil.Category.CORE, TAG, "window_shown", "instance=$instanceId")
        LogUtil.i(TAG, "Lifecycle", ">>> Window Shown (Keyboard UI Pop-up) <<<")
        isWindowShown = true

        llmManager.preemptIdleMemoryProcessing(reason = "window_shown")

        synchronized(trajectoryLock) {
            inputTrajectoryBuffer.setLength(0)
            trajectoryStartedAtMs = System.currentTimeMillis()
        }
        if (::mInputView.isInitialized) {
            mInputView.onWindowShown()
            if (llmManager.isAiCompletionEnabled && !mInputView.isGhostTextActive()) {
                val currentInput = getTextBeforeCursor(100)
                if (currentInput.isBlank()) {
                    LogUtil.eventD(
                        LogUtil.Category.GHOST,
                        TAG,
                        "ghost_placeholder_show",
                        "reason=window_shown_empty_cursor"
                    )
                    mInputView.showAiSuggestion(arrayOf("我", "在", "是", "有", "不"))
                }
            }
        }
        super.onWindowShown()

        llmManager.markActivity("window_shown")

        startContextSyncReceiver()

        if (llmManager.generationHandle != 0L) {
            serviceIoScope.launch {
                llmManager.enterPersistentInferenceMode(reason = "window_shown")
                requestContextSync()
                LogUtil.event(
                    LogUtil.Category.CORE,
                    TAG,
                    "window_shown_ready",
                    "elapsedMs=${SystemClock.elapsedRealtime() - tStart}"
                )
            }
        }
    }

    override fun onWindowHidden() {
        val tStart = SystemClock.elapsedRealtime()
        LogUtil.event(LogUtil.Category.CORE, TAG, "window_hidden", "instance=$instanceId")
        LogUtil.i(TAG, "Lifecycle", ">>> Window Hidden (Keyboard UI Close) <<<")
        isWindowShown = false

        stopContextSyncReceiver()

        if (::mInputView.isInitialized && mInputView.isGhostTextActive()) {
            LogUtil.eventD(LogUtil.Category.GHOST, TAG, "ghost_clear", "reason=window_hidden")
            mInputView.hideAiSuggestion("window_hidden")
        }

        if (::mInputView.isInitialized) mInputView.onWindowHidden()
        super.onWindowHidden()
        llmManager.stopCompletion(reason = "window_hidden")
        llmManager.markActivity("window_hidden")

        if (llmManager.generationHandle != 0L) {
            serviceIoScope.launch {
                val flushedLen = flushTrajectoryToAggregatedHistory(reason = "window_hidden")
                if (flushedLen > 0) {
                    LogUtil.event(
                        LogUtil.Category.MEM,
                        TAG,
                        "input_trajectory_flushed",
                        "reason=window_hidden len=$flushedLen pkg=$lastEditorPackageName"
                    )
                    if (LogUtil.rateLimit("mem.traj.toast", 1200)) {
                        applicationContext.toast("🧾 已记录输入轨迹（$flushedLen 字），后台整理记忆中…")
                    }
                } else {
                    LogUtil.event(
                        LogUtil.Category.MEM,
                        TAG,
                        "input_trajectory_flushed",
                        "reason=window_hidden len=0 pkg=$lastEditorPackageName"
                    )
                }

                llmManager.exitPersistentInferenceMode(reason = "window_hidden")
                llmManager.clearKvKeepSystem(reason = "window_hidden_clear")
                LogUtil.event(
                    LogUtil.Category.MEM,
                    TAG,
                    "idle_memory_trigger",
                    "reason=window_hidden handle=${llmManager.generationHandle}"
                )
                llmManager.runIdleMemoryWorkOnWindowHidden(minIntervalMs = 5_000L)
                LogUtil.event(
                    LogUtil.Category.CORE,
                    TAG,
                    "window_hidden_bg",
                    "elapsedMs=${SystemClock.elapsedRealtime() - tStart}"
                )
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (0 != event.repeatCount) return super.onKeyDown(keyCode, event)
        LogUtil.eventD(
            LogUtil.Category.INPUT,
            TAG,
            "hw_key_down",
            "keyCode=$keyCode shown=$isInputViewShown"
        )
        if (isInputViewShown) {
            if (keyCode == KeyEvent.KEYCODE_BACK) {
                if (controlPanel != null) {
                    closeControlPanel()
                    return true
                }
                return super.onKeyDown(keyCode, event)
            }
            if (keyCode == KeyEvent.KEYCODE_ENTER) {
                sendEnterKeyEvent()
                return true
            }
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        LogUtil.eventD(
            LogUtil.Category.INPUT,
            TAG,
            "hw_key_up",
            "keyCode=$keyCode shown=$isInputViewShown"
        )
        return if (isInputViewShown) mInputView.processKey(event) || super.onKeyUp(keyCode, event)
        else super.onKeyUp(keyCode, event)
    }

    fun sendEnterKeyEvent() {
        LogUtil.i(TAG, "KeyHandler", ">>> 回车键按下")
        LogUtil.event(LogUtil.Category.INPUT, TAG, "enter_key", "action=send")

        clearGhostTextBeforeCommit()

        val ic = currentInputConnection ?: return
        llmManager.stopCompletion(reason = "enter_key")

        YuyanEmojiCompat.mEditorInfo?.run {
            if (inputType and EditorInfo.TYPE_MASK_CLASS == EditorInfo.TYPE_NULL || imeOptions.hasFlag(EditorInfo.IME_FLAG_NO_ENTER_ACTION)) {
                sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
            } else if (!actionLabel.isNullOrEmpty() && actionId != EditorInfo.IME_ACTION_UNSPECIFIED) {
                ic.performEditorAction(actionId)
            } else when (val action = imeOptions and EditorInfo.IME_MASK_ACTION) {
                EditorInfo.IME_ACTION_UNSPECIFIED, EditorInfo.IME_ACTION_NONE -> sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
                else -> ic.performEditorAction(action)
            }
        }
    }

    private fun clearGhostTextBeforeCommit() {
        if (mInputView.isGhostTextActive()) {
            LogUtil.eventD(LogUtil.Category.GHOST, TAG, "ghost_clear", "reason=before_commit")
            mInputView.hideAiSuggestion("before_commit")
        }
    }

    fun commitText(text: String) {
        clearGhostTextBeforeCommit()

        LogUtil.eventD(
            LogUtil.Category.INPUT,
            TAG,
            "commit_text",
            "len=${text.length}"
        )
        LogUtil.logCommitText(TAG, text, source = "commitText")
        currentInputConnection?.commitText(StringUtils.converted2FlowerTypeface(text), 1)
        appendTrajectory(text)
        saveInputFragmentRecord(text)
    }

    private fun startContextSyncReceiver() {
        if (contextSyncReceiver != null) return
        val receiver = object : android.content.BroadcastReceiver() {
            override fun onReceive(context: android.content.Context?, intent: android.content.Intent?) {
                val action = intent?.action ?: return
                if (action != ACTION_CONTEXT_SYNC && action != LEGACY_ACTION_CONTEXT_SYNC) return

                val jsonStr = intent.getStringExtra(EXTRA_MCP_MESSAGE)
                    ?: intent.getStringExtra(EXTRA_CONTEXT_DATA)
                val sourceKey = when {
                    intent.hasExtra(EXTRA_MCP_MESSAGE) -> EXTRA_MCP_MESSAGE
                    intent.hasExtra(EXTRA_CONTEXT_DATA) -> EXTRA_CONTEXT_DATA
                    else -> "none"
                }
                LogUtil.event(
                    LogUtil.Category.MEM,
                    TAG,
                    "context_sync_receive",
                    "action=$action sourceKey=$sourceKey hasData=${!jsonStr.isNullOrBlank()}"
                )
                if (!jsonStr.isNullOrBlank()) handleContextSync(jsonStr)
            }
        }
        contextSyncReceiver = receiver
        val filter = android.content.IntentFilter().apply {
            addAction(ACTION_CONTEXT_SYNC)
            addAction(LEGACY_ACTION_CONTEXT_SYNC)
        }
        try {
            if (android.os.Build.VERSION.SDK_INT >= 33) registerReceiver(receiver, filter, 0x2)
            else registerReceiver(receiver, filter)
        } catch (e: Exception) {
            LogUtil.e(TAG, "Sync", "Register receiver failed: ${e.message}")
            contextSyncReceiver = null
        }
    }

    private fun stopContextSyncReceiver() {
        val r = contextSyncReceiver ?: return
        try {
            unregisterReceiver(r)
        } catch (_: Exception) {
        } finally {
            contextSyncReceiver = null
        }
    }

    private fun requestContextSync() {
        try {
            val requestId = "ime-${System.currentTimeMillis()}"
            val requestPayload = buildContextSyncRequestPayload(requestId)
            val intent = android.content.Intent(ACTION_CONTEXT_SYNC_REQUEST).apply {
                putExtra(EXTRA_MCP_REQUEST, requestPayload)
                putExtra(EXTRA_CONTEXT_DATA, requestPayload)
            }
            sendBroadcast(intent)
            LogUtil.event(
                LogUtil.Category.MEM,
                TAG,
                "context_sync_request",
                "sent=true action=$ACTION_CONTEXT_SYNC_REQUEST requestId=$requestId limit=$DEFAULT_CONTEXT_LIMIT"
            )
        } catch (e: Exception) {
            LogUtil.eventE(LogUtil.Category.MEM, TAG, "context_sync_request_fail", "ex=${e.javaClass.simpleName}:${e.message}")
        }
    }
    
    private fun handleContextSync(jsonStr: String) {
        try {
            val json = JSONObject(jsonStr)
            val context = extractContextObject(json)
            if (context == null) {
                LogUtil.eventW(
                    LogUtil.Category.MEM,
                    TAG,
                    "context_sync_parse_skip",
                    "reason=no_context_payload"
                )
                return
            }
            val isHuoziContextV1 = context.optString("schema") == CONTEXT_SCHEMA_HUOZI_V1
            val contactName = extractConversationName(context)
            var inferredStyle = llmManager.initializeStyleByContactName(contactName, reason = "context_sync_contact")
            if (inferredStyle != null) {
                val styleToApply = inferredStyle
                serviceScope.launch {
                    applyThemeForStyle(styleToApply!!, reason = "context_sync_contact_init")
                }
            }

            val historyArr = context.optJSONArray("history")
            val sb = StringBuilder()
            var foundLast = false
            if (historyArr != null) {
                for (i in 0 until historyArr.length()) {
                    val item = historyArr.optJSONObject(i) ?: continue
                    val roleRaw = item.optString("role").trim().lowercase(Locale.ROOT)
                    val role = when {
                        isHuoziContextV1 && roleRaw == "me" -> "user"
                        isHuoziContextV1 && roleRaw == "user" -> "partner"
                        roleRaw == "me" || roleRaw == "self" || roleRaw == "owner" || roleRaw == "assistant" -> "user"
                        else -> roleRaw
                    }
                    val content = item.optString("content")
                    val displayRole = if (role == "user") "[我]" else "[对方]"
                    sb.append("$displayRole: $content\n")
                }
                for (i in historyArr.length() - 1 downTo 0) {
                    val item = historyArr.optJSONObject(i) ?: continue
                    val role = item.optString("role").trim().lowercase(Locale.ROOT)
                    if (isPeerRole(role, isHuoziContextV1)) {
                        llmManager.syncedLastMsgStr = item.optString("content")
                        foundLast = true
                        break
                    }
                }
            }
            llmManager.syncedHistoryStr = if (sb.isNotEmpty()) sb.toString().trim() else "无"
            if (!foundLast) llmManager.syncedLastMsgStr = "无"

            if (inferredStyle == null) {
                val hint = buildString {
                    append(llmManager.syncedHistoryStr)
                    append("\n")
                    append(llmManager.syncedLastMsgStr)
                }
                inferredStyle = llmManager.initializeStyleByConversationHint(hint, reason = "context_sync_hint")
                if (inferredStyle != null) {
                    val styleToApply = inferredStyle
                    serviceScope.launch {
                        applyThemeForStyle(styleToApply!!, reason = "context_sync_hint_init")
                    }
                }
            }

            val signatureSource = buildString {
                append(llmManager.syncedHistoryStr)
                append("\n--last--\n")
                append(llmManager.syncedLastMsgStr)
            }
                .replace("\r\n", "\n")
                .trim()
            val signature = sha256Short(signatureSource)
            val sessionChanged = llmManager.onContextSignature(signature, reason = "context_sync")
            if (sessionChanged && ::mInputView.isInitialized) {
                mInputView.hideAiSuggestionListOnly("session_changed")
            }

            LogUtil.event(
                LogUtil.Category.MEM,
                TAG,
                "context_sync_parsed",
                "historyCount=${historyArr?.length() ?: 0} foundLast=$foundLast hasContact=${!contactName.isNullOrBlank()} inferredStyle=${inferredStyle ?: "none"} schema=${context.optString("schema", "<none>")}"
            )

            val hasHistory = historyArr != null && historyArr.length() > 0
            if (hasHistory) {
                llmManager.triggerSilentPrefill()
            } else {
                LogUtil.eventD(
                    LogUtil.Category.MEM,
                    TAG,
                    "context_sync_prefill_skip",
                    "reason=no_history_only_contact_init"
                )
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Sync", "解析失败: ${e.message}")
            LogUtil.eventE(LogUtil.Category.MEM, TAG, "context_sync_parse_fail", "ex=${e.javaClass.simpleName}:${e.message}")
        }
    }

    private fun buildContextSyncRequestPayload(requestId: String): String {
        return JSONObject().apply {
            put("jsonrpc", MCP_JSONRPC_VERSION)
            put("id", requestId)
            put("method", MCP_METHOD_CONTEXT_GET)
            put(
                "params",
                JSONObject().apply {
                    put("limit", DEFAULT_CONTEXT_LIMIT)
                }
            )
        }.toString()
    }

    private fun extractContextObject(messageJson: JSONObject): JSONObject? {
        if (looksLikeContextObject(messageJson)) return messageJson

        val params = messageJson.optJSONObject("params")
        val paramsContext = params?.optJSONObject("context")
        if (paramsContext != null && looksLikeContextObject(paramsContext)) return paramsContext
        if (params != null && looksLikeContextObject(params)) return params

        val result = messageJson.optJSONObject("result")
        val resultContext = result?.optJSONObject("context")
        if (resultContext != null && looksLikeContextObject(resultContext)) return resultContext
        if (result != null && looksLikeContextObject(result)) return result

        val directContext = messageJson.optJSONObject("context")
        if (directContext != null && looksLikeContextObject(directContext)) return directContext

        return null
    }

    private fun looksLikeContextObject(json: JSONObject): Boolean {
        return json.has("history")
            || json.has("partner")
            || json.has("environment")
            || json.optString("schema").isNotBlank()
            || json.optString("app_pkg").isNotBlank()
    }

    private fun isSelfRole(role: String, isHuoziContextV1: Boolean): Boolean {
        return when (role) {
            "me", "self", "owner", "assistant" -> true
            "user" -> !isHuoziContextV1
            else -> false
        }
    }

    private fun isPeerRole(role: String, isHuoziContextV1: Boolean): Boolean {
        return when (role) {
            "peer", "partner", "other", "target" -> true
            "user" -> isHuoziContextV1
            else -> !isSelfRole(role, isHuoziContextV1)
        }
    }

    private fun extractConversationName(json: JSONObject): String? {
        val directKeys = arrayOf(
            "person_name",
            "peer_name",
            "target_name",
            "chat_name",
            "display_name",
            "name"
        )
        for (key in directKeys) {
            val value = json.optString(key)
            if (value.isNotBlank()) return value.trim()
        }
        val partner = json.optJSONObject("partner")
        if (partner != null) {
            val partnerName = partner.optString("name")
            if (partnerName.isNotBlank()) return partnerName.trim()
        }

        val profile = json.optJSONObject("profile")
        if (profile != null) {
            for (key in directKeys) {
                val value = profile.optString(key)
                if (value.isNotBlank()) return value.trim()
            }
        }

        val historyArr = json.optJSONArray("history") ?: return null
        for (i in historyArr.length() - 1 downTo 0) {
            val item = historyArr.optJSONObject(i) ?: continue
            val role = item.optString("role").trim().lowercase(Locale.ROOT)
            if (role == "peer" || role == "partner" || role == "user") {
                val n1 = item.optString("name")
                if (n1.isNotBlank()) return n1.trim()
                val n2 = item.optString("display_name")
                if (n2.isNotBlank()) return n2.trim()
            }
        }
        return null
    }

    private fun sha256Short(s: String): String {
        val bytes = MessageDigest.getInstance("SHA-256").digest(s.toByteArray(Charsets.UTF_8))
        return bytes.take(8).joinToString("") { "%02x".format(it) }
    }

    private fun appendTrajectory(text: String) {
        if (text.isBlank()) return
        val trimmed = text.replace("\n", " ")
        synchronized(trajectoryLock) {
            if (trajectoryStartedAtMs == 0L) trajectoryStartedAtMs = System.currentTimeMillis()
            val maxLen = 4000
            val incomingLen = trimmed.length
            val overflow = (inputTrajectoryBuffer.length + incomingLen) - maxLen
            if (overflow > 0 && inputTrajectoryBuffer.isNotEmpty()) {
                val toDelete = overflow.coerceAtMost(inputTrajectoryBuffer.length)
                inputTrajectoryBuffer.delete(0, toDelete)
            }
            inputTrajectoryBuffer.append(trimmed)
        }
    }

    private fun flushTrajectoryToAggregatedHistory(reason: String): Int {
        val content: String
        val startedAt: Long
        val endedAt = System.currentTimeMillis()
        synchronized(trajectoryLock) {
            content = inputTrajectoryBuffer.toString().trim()
            startedAt = trajectoryStartedAtMs
            inputTrajectoryBuffer.setLength(0)
            trajectoryStartedAtMs = 0L
        }

        if (content.length < 6) return 0
        val sha = sha256Short(content)
        if (!sha.isNullOrBlank() && sha == lastFlushedTrajectorySha) {
            LogUtil.event(LogUtil.Category.MEM, TAG, "input_trajectory_dedup", "reason=$reason sha=$sha")
            return 0
        }
        lastFlushedTrajectorySha = sha

        try {
            val file = File(filesDir, "user_input_history.jsonl")
            val json = JSONObject().apply {
                put("timestamp", endedAt)
                put("content", content)
                put("source", "window_hidden_trajectory")
                put("reason", reason)
                put("pkg", lastEditorPackageName)
                if (startedAt > 0L) put("startedAtMs", startedAt)
                put("endedAtMs", endedAt)
                put("len", content.length)
                put("sha", sha)
            }
            FileWriter(file, true).use { writer ->
                writer.write(json.toString() + "\n")
            }
            LogUtil.event(
                LogUtil.Category.MEM,
                TAG,
                "input_trajectory_write",
                "reason=$reason len=${content.length} sha=${sha} pkg=$lastEditorPackageName preview=\"${content.take(24).replace("\n", " ")}\""
            )
        } catch (e: Exception) {
            LogUtil.e(TAG, "DataAnalysis", "保存聚合轨迹失败: ${e.message}")
            LogUtil.eventE(LogUtil.Category.MEM, TAG, "input_trajectory_write_fail", "ex=${e.javaClass.simpleName}:${e.message}")
            return 0
        }
        return content.length
    }

    private fun saveInputFragmentRecord(text: String) {
        if (text.isBlank()) return
        serviceIoScope.launch {
            try {
                val file = File(filesDir, "user_input_fragments.jsonl")
                val now = System.currentTimeMillis()
                val json = JSONObject().apply {
                    put("timestamp", now)
                    put("content", text)
                    put("source", "commitText")
                    put("pkg", lastEditorPackageName)
                    put("len", text.length)
                }
                FileWriter(file, true).use { writer ->
                    writer.write(json.toString() + "\n")
                }
                if (LogUtil.rateLimit("input.frag", 250)) {
                    LogUtil.eventD(
                        LogUtil.Category.INPUT,
                        TAG,
                        "input_fragment_write",
                        "len=${text.length} pkg=$lastEditorPackageName preview=\"${text.take(12).replace("\n", " ")}\""
                    )
                }
            } catch (e: Exception) {
                LogUtil.e(TAG, "DataAnalysis", "保存片段失败: ${e.message}")
            }
        }
    }

    fun toggleControlPanel() {
        if (controlPanel != null && controlPanel!!.parent != null) closeControlPanel()
        else openControlPanel(LLMControlPanel.MODE_STYLE)
    }

    fun closeControlPanel() {
        if (controlPanel != null) {
            mRootContainer?.removeView(controlPanel)
            controlPanel = null
            mInputView.requestLayout()
            requestHideSelf(0)
            mInputView.visibility = View.VISIBLE
        }
    }

    fun openControlPanel(mode: Int) {
        if (mRootContainer == null || !::mInputView.isInitialized) return
        closeControlPanel()
        controlPanel = LLMControlPanel(this, this@ImeService, mode).apply {
            layoutParams = android.widget.FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(android.graphics.Color.WHITE)
            isClickable = true
            isFocusable = true
        }
        mRootContainer?.addView(controlPanel)
        controlPanel?.bringToFront()
        controlPanel?.requestLayout()
        mInputView.requestLayout()
    }

    fun isSentenceMode(): Boolean = (llmManager.currentGenerationMode == GenerationMode.SENTENCE)
    fun setSentenceMode(isSentence: Boolean) {
        llmManager.currentGenerationMode = if (isSentence) GenerationMode.SENTENCE else GenerationMode.TOKEN
    }
    fun getCurrentStyle(): String = llmManager.currentStyleMode
    fun getGenerationModeOrdinal(): Int = llmManager.currentGenerationMode.ordinal
    fun setGenerationModeOrdinal(ordinal: Int) {
        val modes = GenerationMode.values()
        if (ordinal in modes.indices) {
            llmManager.currentGenerationMode = modes[ordinal]
            val modeStr = when (llmManager.currentGenerationMode) {
                GenerationMode.TOKEN -> "单词补全模式"
                GenerationMode.PHRASE -> "短语联想模式"
                GenerationMode.SENTENCE -> "整句生成模式"
            }
            showFeedback("⚡️ 已切换至: $modeStr")
        }
    }
    fun switchGenerationMode() = openControlPanel(LLMControlPanel.MODE_GEN)

    fun openMemoryPanel() = openControlPanel(LLMControlPanel.MODE_MEMORY)

    fun injectTestMemory(text: String): Boolean {
        val ok = llmManager.injectTestMemory(text)
        showFeedback(if (ok) "🧠 已注入测试记忆" else "🧠 注入失败/内容太短")
        return ok
    }

    fun listMemories(limit: Int = 200): List<com.yuyan.imemodule.llm.memory.MemoryRecord> {
        return llmManager.listMemories(limit)
    }

    fun deleteMemory(id: String): Boolean {
        val ok = llmManager.deleteMemory(id)
        showFeedback(if (ok) "🗑️ 记忆已删除" else "🗑️ 删除失败")
        return ok
    }

    fun debugSearchMemory(query: String, k: Int = 3): List<String> {
        return llmManager.debugSearchMemory(query, k)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        serviceScope.launch {
            delay(200)
            EnvironmentSingleton.instance.initData()
            KeyboardLoaderUtil.instance.clearKeyboardMap()
            KeyboardManager.instance.clearKeyboard()
            if (::mInputView.isInitialized) KeyboardManager.instance.switchKeyboard()
        }
        onSystemDarkModeChange(newConfig.isDarkMode())
    }
    override fun onEvaluateFullscreenMode(): Boolean = false
    override fun onComputeInsets(outInsets: Insets) {
        if (!::mInputView.isInitialized) return
        if (controlPanel != null && controlPanel!!.visibility == View.VISIBLE) {
            outInsets.contentTopInsets = 0
            outInsets.visibleTopInsets = 0
            outInsets.touchableInsets = Insets.TOUCHABLE_INSETS_FRAME
            outInsets.touchableRegion.setEmpty()
            return
        }
        val (x, y) = intArrayOf(0, 0).also {
            if (mInputView.isAddPhrases) mInputView.mAddPhrasesLayout.getLocationInWindow(it)
            else mInputView.mSkbRoot.getLocationInWindow(it)
        }
        outInsets.apply {
            if (EnvironmentSingleton.instance.keyboardModeFloat) {
                contentTopInsets = EnvironmentSingleton.instance.mScreenHeight
                visibleTopInsets = EnvironmentSingleton.instance.mScreenHeight
                touchableInsets = Insets.TOUCHABLE_INSETS_REGION
                touchableRegion.set(x, y, x + mInputView.mSkbRoot.width, y + mInputView.mSkbRoot.height)
            } else {
                contentTopInsets = y
                touchableInsets = Insets.TOUCHABLE_INSETS_CONTENT
                touchableRegion.setEmpty()
                visibleTopInsets = y
            }
        }
    }
    override fun onUpdateSelection(oldSelStart: Int, oldSelEnd: Int, newSelStart: Int, newSelEnd: Int, candidatesStart: Int, candidatesEnd: Int) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        if (LogUtil.rateLimit("selectionUpdate", 200)) {
            LogUtil.eventD(
                LogUtil.Category.INPUT,
                TAG,
                "selection_update",
                "old=($oldSelStart,$oldSelEnd) new=($newSelStart,$newSelEnd) candEnd=$candidatesEnd"
            )
        }
        if (::mInputView.isInitialized && mInputView.isShown) {
            mInputView.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesEnd)
        }
    }
    fun sendCombinationKeyEvents(keyEventCode: Int, alt: Boolean = false, ctrl: Boolean = false, shift: Boolean = false) {
        var metaState = 0
        if (alt) metaState = KeyEvent.META_ALT_ON or KeyEvent.META_ALT_LEFT_ON
        if (ctrl) metaState = metaState or KeyEvent.META_CTRL_ON or KeyEvent.META_CTRL_LEFT_ON
        if (shift) metaState = metaState or KeyEvent.META_SHIFT_ON or KeyEvent.META_SHIFT_LEFT_ON
        val eventTime = SystemClock.uptimeMillis()
        sendDownKeyEvent(eventTime, keyEventCode, metaState)
        sendUpKeyEvent(eventTime, keyEventCode, metaState)
    }
    fun sendDownKeyEvent(eventTime: Long, keyEventCode: Int, metaState: Int = 0) {
        currentInputConnection?.sendKeyEvent(KeyEvent(eventTime, eventTime, KeyEvent.ACTION_DOWN, keyEventCode, 0, metaState))
    }
    fun sendUpKeyEvent(eventTime: Long, keyEventCode: Int, metaState: Int = 0) {
        currentInputConnection?.sendKeyEvent(KeyEvent(eventTime, SystemClock.uptimeMillis(), KeyEvent.ACTION_UP, keyEventCode, 0, metaState))
    }
    fun setComposingText(text: CharSequence) {
        LogUtil.logSetComposingText(TAG, text.toString())
        currentInputConnection?.setComposingText(text, 1)
    }
    fun finishComposingText() {
        LogUtil.logSpecialOperation(TAG, "FINISH_COMPOSING", "finish composing text")
        currentInputConnection?.finishComposingText()
    }
    fun commitTextEditMenu(id: Int) { currentInputConnection?.performContextMenuAction(id) }
    fun performEditorAction(editorAction: Int) { currentInputConnection?.performEditorAction(editorAction) }
    fun deleteSurroundingText(length: Int) {
        val beforeText = getTextBeforeCursor(1000)
        currentInputConnection?.deleteSurroundingText(length, 0)
        val afterText = getTextBeforeCursor(1000)
        LogUtil.logDeleteText(TAG, beforeText, afterText, length)
    }
    fun setSelection(start: Int, end: Int) {
        LogUtil.logSpecialOperation(TAG, "SET_SELECTION", "range=[$start,$end]")
        currentInputConnection?.setSelection(start, end)
    }
}
