package com.yuyan.imemodule.keyboard
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.BitmapDrawable
import android.os.Build
import android.view.KeyEvent
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.inputmethod.EditorInfo
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RelativeLayout
import android.widget.FrameLayout
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.get
import com.yuyan.imemodule.R
import com.yuyan.imemodule.callback.CandidateViewListener
import com.yuyan.imemodule.callback.IResponseKeyEvent
import com.yuyan.imemodule.data.emojicon.EmojiconData.SymbolPreset
import com.yuyan.imemodule.data.theme.ThemeManager
import com.yuyan.imemodule.database.DataBaseKT
import com.yuyan.imemodule.entity.keyboard.SoftKey
import com.yuyan.imemodule.manager.InputModeSwitcherManager
import com.yuyan.imemodule.prefs.AppPrefs.Companion.getInstance
import com.yuyan.imemodule.prefs.behavior.KeyboardOneHandedMod
import com.yuyan.imemodule.prefs.behavior.PopupMenuMode
import com.yuyan.imemodule.prefs.behavior.SkbMenuMode
import com.yuyan.imemodule.service.DecodingInfo
import com.yuyan.imemodule.service.ImeService
import com.yuyan.imemodule.singleton.EnvironmentSingleton
import com.yuyan.imemodule.utils.InputMethodUtil
import com.yuyan.imemodule.utils.DevicesUtils
import com.yuyan.imemodule.utils.KeyboardLoaderUtil
import com.yuyan.imemodule.utils.LogUtil
import com.yuyan.imemodule.utils.StringUtils
import com.yuyan.imemodule.view.CandidatesBar
import com.yuyan.imemodule.view.EditPhrasesView
import com.yuyan.imemodule.view.FullDisplayKeyboardBar
import com.yuyan.imemodule.keyboard.container.CandidatesContainer
import com.yuyan.imemodule.keyboard.container.ClipBoardContainer
import com.yuyan.imemodule.keyboard.container.SymbolContainer
import com.yuyan.imemodule.keyboard.container.T9TextContainer
import com.yuyan.imemodule.view.popup.PopupComponent
import com.yuyan.imemodule.view.preference.ManagedPreference
import com.yuyan.imemodule.view.widget.LifecycleRelativeLayout
import com.yuyan.inputmethod.CustomEngine
import com.yuyan.inputmethod.core.CandidateListItem
import splitties.views.bottomPadding
import splitties.views.rightPadding
import kotlin.math.absoluteValue
import androidx.core.graphics.drawable.toDrawable
import androidx.core.graphics.scale
import androidx.core.view.postDelayed
import android.widget.TextView
import android.graphics.Typeface

/**
 * 输入法主界面
 */
@SuppressLint("ViewConstructor")
class InputView(context: Context, service: ImeService) : LifecycleRelativeLayout(context), IResponseKeyEvent {
    private val logTag = "InputView"
    private fun previewText(text: String, maxChars: Int = 32): String {
        if (text.length <= maxChars) return text
        return text.substring(0, maxChars) + "…"
    }
    private fun ghostStateSummary(): String {
        return "ghostActive=$isGhostTextActive ghostLen=${currentGhostText.length} suggestions=${currentSuggestions.size} expanded=$isAiSuggestionExpanded"
    }
    private val clipboardItemTimeout = getInstance().clipboard.clipboardItemTimeout.getValue()
    private var chinesePrediction = true
    var isAddPhrases = false

    internal var service: ImeService
    private var mImeState = ImeState.STATE_IDLE
    private var mChoiceNotifier = ChoiceNotifier()
    var mSkbRoot: RelativeLayout
    var mSkbCandidatesBarView: CandidatesBar
    private var mHoderLayoutLeft: LinearLayout
    private var mHoderLayoutRight: LinearLayout
    private lateinit var mOnehandHoderLayout: LinearLayout
    var mAddPhrasesLayout: EditPhrasesView
    private var mLlKeyboardBottomHolder: LinearLayout
    private var mInputKeyboardContainer: RelativeLayout
    private lateinit var mRightPaddingKey: ManagedPreference.PInt
    private lateinit var mBottomPaddingKey: ManagedPreference.PInt
    private var mFullDisplayKeyboardBar: FullDisplayKeyboardBar? = null
    var hasSelection = false
    var hasSelectionAll = false
    private lateinit var aiSuggestionContainer: LinearLayout
    private lateinit var ghostOverlayContainer: FrameLayout
    private var ghostOverlayTextView: TextView? = null
    // 新增：GhostText 浮动窗口，用于在光标位置显示
    private var ghostTextPopup: GhostTextPopupWindow? = null
    private var isAiSuggestionExpanded = false
    private var currentSuggestions: Array<String> = emptyArray()
    init {
        initNavbarBackground(service)
        this.service = service
        mSkbRoot = LayoutInflater.from(context)
            .inflate(R.layout.sdk_skb_container, this, false) as RelativeLayout
        addView(mSkbRoot)
        mSkbCandidatesBarView = mSkbRoot.findViewById(R.id.candidates_bar)
        mHoderLayoutLeft = mSkbRoot.findViewById(R.id.ll_skb_holder_layout_left)
        mHoderLayoutRight = mSkbRoot.findViewById(R.id.ll_skb_holder_layout_right)
        mInputKeyboardContainer = mSkbRoot.findViewById(R.id.ll_input_keyboard_container)
        mAddPhrasesLayout = EditPhrasesView(context)
        KeyboardManager.instance.setData(mSkbRoot.findViewById(R.id.skb_input_keyboard_view), this)
        mLlKeyboardBottomHolder = mSkbRoot.findViewById(R.id.iv_keyboard_holder)
        aiSuggestionContainer = mSkbRoot.findViewById(R.id.ai_suggestion_container)
        ghostOverlayContainer = mSkbRoot.findViewById(R.id.ghost_overlay_container)
        val root = PopupComponent.get().root
        val viewParent = root.parent
        if (viewParent != null) {
            (viewParent as ViewGroup).removeView(root)
        }
        addView(root, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
            addRule(ALIGN_BOTTOM, mSkbRoot.id)
            addRule(ALIGN_LEFT, mSkbRoot.id)
        })
        initView(context)
    }

    var enableStreamingAnimation = true
    var streamCharDelay = 50L
    private val animHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val activeStreamRunnables = mutableListOf<Runnable>()

    private var isGhostTextActive = false
    private var currentGhostText = ""

    // 用于去重：避免新 top1 与刚刚上屏的文本重复
    private var lastCommittedAiText: String = ""

    private fun commitAiTextDirect(text: String, hideReason: String, nextReason: String, source: String) {
        if (text.isBlank()) return

        lastCommittedAiText = text
        LogUtil.logCommitText("InputView", text, source = source)

        // 先清空本地 ghost 状态，避免 Service 的 pre-commit safety 再次触发 hide
        isGhostTextActive = false
        currentGhostText = ""

        LogUtil.eventD(LogUtil.Category.INPUT, "commit_text_begin", "source=$source")
        service.commitText(text)

        // 清理 AI 建议 UI（列表 + overlay）
        hideAiSuggestion(hideReason)

        // 请求下一轮补全
        LogUtil.event(LogUtil.Category.GHOST, "ai_commit_done", "action=request_next_completion source=$source")
        service.llmManager.requestCompletion(reason = nextReason)
    }
    private fun applyGhostText(text: String) {
        currentGhostText = text
        isGhostTextActive = true
        showGhostOverlay(text)
    }

    private fun showGhostOverlay(text: String) {
        // 使用 PopupWindow 在光标位置显示 GhostText
        val popup = ghostTextPopup ?: GhostTextPopupWindow(context) { tappedText ->
            // 点击气泡直接上屏（无需 AI 键）
            if (tappedText.isNotBlank()) {
                LogUtil.eventD(LogUtil.Category.GHOST, "ghost_bubble_tap", "len=${tappedText.length} preview=\"${previewText(tappedText)}\"")
                commitAiTextDirect(
                    text = tappedText,
                    hideReason = "ghost_bubble_tap",
                    nextReason = "ghost_bubble_tap_next",
                    source = "ghostBubbleTap",
                )
            }
        }.also { ghostTextPopup = it }
        
        // 获取 anchor view（使用 ImeService 的 window decorView）
        val anchor = try {
            service.window?.window?.decorView ?: this
        } catch (e: Exception) {
            LogUtil.eventW(
                LogUtil.Category.GHOST,
                "overlay_anchor_fallback",
                "reason=decorView_exception ex=${e.javaClass.simpleName}:${e.message}"
            )
            this
        }
        
        popup.show(text, service.lastCursorAnchorInfo, anchor)
        LogUtil.eventD(
            LogUtil.Category.GHOST,
            "overlay_show",
            "textLen=${text.length} textPreview=\"${previewText(text)}\" hasCursorAnchor=${service.lastCursorAnchorInfo != null} anchor=${anchor.javaClass.simpleName}"
        )
        
        // 保留旧的 FrameLayout 作为 fallback（当 PopupWindow 不可用时）
        // 如果 PopupWindow 成功显示，隐藏 FrameLayout 容器
        ghostOverlayContainer.visibility = View.GONE
    }

    /**
     * 重新定位 GhostText PopupWindow（当光标位置更新时调用）
     * @param cursorAnchorInfo 新的光标位置信息
     */
    fun repositionGhostText(cursorAnchorInfo: android.view.inputmethod.CursorAnchorInfo?) {
        if (!isGhostTextActive || ghostTextPopup == null) {
            return
        }
        if (LogUtil.rateLimit("ghost.reposition", 250)) {
            LogUtil.eventD(
                LogUtil.Category.GHOST,
                "overlay_reposition",
                "hasCursorAnchor=${cursorAnchorInfo != null} ${ghostStateSummary()}"
            )
        }
        ghostTextPopup?.updatePosition(cursorAnchorInfo)
    }

    private fun hideGhostOverlay() {
        // 隐藏 PopupWindow
        ghostTextPopup?.dismiss()
        LogUtil.eventD(LogUtil.Category.GHOST, "overlay_hide", ghostStateSummary())
        
        // 同时清理旧的 FrameLayout（fallback）
        ghostOverlayContainer.removeAllViews()
        ghostOverlayContainer.visibility = View.GONE
        ghostOverlayTextView = null
    }
    /**
     * 停止所有当前的动画任务（包括生成中和文字流式生成）
     */
    private fun stopAnimations() {
        activeStreamRunnables.forEach { animHandler.removeCallbacks(it) }
        activeStreamRunnables.clear()
    }
    /**
     * 显示“生成中...”的动画状态
     */
    fun showGeneratingStatus() {
        LogUtil.eventD(LogUtil.Category.UI, "gen_status_show", "reason=new_generation")
        stopAnimations()

        // 生成中：隐藏 top2-4 列表，但保留当前 GhostText，避免闪断
        aiSuggestionContainer.visibility = View.GONE
        aiSuggestionContainer.removeAllViews()
        currentSuggestions = emptyArray()
        isAiSuggestionExpanded = false
    }
    fun showAiSuggestion(suggestions: Array<String>) {
        LogUtil.logSpecialOperation("InputView", "AI_SUGGESTION_SHOW", "suggestions count=${suggestions.size}")
        stopAnimations()
        if (suggestions.isEmpty()) {
            LogUtil.logSpecialOperation("InputView", "AI_SUGGESTION_EMPTY", "suggestions empty, hiding")
            LogUtil.event(LogUtil.Category.GHOST, "suggestions_empty", ghostStateSummary())
            hideAiSuggestion("empty_suggestions")
            return
        }

        // 新结果到达：清空旧列表，但不隐藏 GhostText（无缝替换）
        if (currentSuggestions.isNotEmpty()) {
            LogUtil.eventD(
                LogUtil.Category.GHOST,
                "suggestions_replace_previous",
                "incoming=${suggestions.size} ${ghostStateSummary()}"
            )
            hideAiSuggestionListOnly("new_suggestions_arrived")
        }

        // 去重：保持顺序，去掉重复项，并丢弃与刚刚上屏相同的文本
        val distinct = suggestions.asList().filter { it.isNotBlank() }.distinct()
        val filtered = if (lastCommittedAiText.isNotBlank()) {
            distinct.filterNot { it == lastCommittedAiText }
        } else {
            distinct
        }

        if (filtered.isEmpty()) {
            LogUtil.eventW(LogUtil.Category.GHOST, "suggestions_all_deduped", "lastCommittedLen=${lastCommittedAiText.length}")
            hideAiSuggestionListOnly("all_deduped")
            return
        }

        // Failsafe: never allow protocol/control outputs to reach UI/ghost.
        fun stripInvisibleAndControl(s: String): String {
            if (s.isEmpty()) return s
            val sb = StringBuilder(s.length)
            for (ch in s) {
                if (ch.isISOControl()) continue
                when (ch) {
                    '\u200B', '\u200C', '\u200D', '\u200E', '\u200F',
                    '\u202A', '\u202B', '\u202C', '\u202D', '\u202E',
                    '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
                    '\u2066', '\u2067', '\u2068', '\u2069',
                    '\uFEFF' -> continue
                }
                sb.append(ch)
            }
            return sb.toString()
        }

        fun looksLikeProtocol(s: String): Boolean {
            val t = stripInvisibleAndControl(s).trim()
            if (t.isEmpty()) return true
            if (t.startsWith("__METRICS__")) return true
            if (t.contains("<MEM_RETRIEVAL>", ignoreCase = true) || t.contains("</MEM_RETRIEVAL>", ignoreCase = true)) return true
            if (t.contains("<NO_MEM>", ignoreCase = true)) return true
            if (t.contains("search query", ignoreCase = true)) return true
            if (t.contains("no longer", ignoreCase = true)) return true
            // wide query= variants
            if (Regex("""(?i)\\bquery\\s*[:=]""").containsMatchIn(t)) return true
            return false
        }

        val safeFiltered = filtered.filterNot { looksLikeProtocol(it) }

        if (safeFiltered.isEmpty()) {
            LogUtil.eventW(LogUtil.Category.GHOST, "suggestions_all_protocol", "incoming=${suggestions.size}")
            hideAiSuggestionListOnly("all_protocol_filtered")
            return
        }

        val top1 = safeFiltered[0]
        LogUtil.logSpecialOperation("InputView", "GHOST_ACTIVATE", "ghost text activated: \"$top1\"")
        LogUtil.event(
            LogUtil.Category.GHOST,
            "ghost_activate",
            "top1Len=${top1.length} top1Preview=\"${previewText(top1)}\" total=${safeFiltered.size}"
        )
        LogUtil.eventD(LogUtil.Category.GHOST, "ghost_apply_ui_only", "note=no_input_connection_touch")
        applyGhostText(top1)

        if (safeFiltered.size > 1) {
            currentSuggestions = safeFiltered.subList(1, safeFiltered.size).toTypedArray()
            isAiSuggestionExpanded = false
            LogUtil.event(
                LogUtil.Category.GHOST,
                "suggestions_render_remaining",
                "remaining=${currentSuggestions.size}"
            )
            updateAiSuggestionView()
            aiSuggestionContainer.visibility = View.VISIBLE
        } else {

            LogUtil.eventD(LogUtil.Category.GHOST, "suggestions_single_only", "hide_candidate_container=true")
            currentSuggestions = emptyArray()
            aiSuggestionContainer.removeAllViews()
            aiSuggestionContainer.visibility = View.GONE
        }
    }
    private fun updateAiSuggestionView() {
        LogUtil.eventD(
            LogUtil.Category.UI,
            "suggestions_view_update",
            "expanded=$isAiSuggestionExpanded suggestions=${currentSuggestions.size} anim=$enableStreamingAnimation"
        )
        aiSuggestionContainer.removeAllViews()
        val paddingVertical = dpToPx(12f)
        val minHeight = dpToPx(48f)
        val dividerMargin = dpToPx(24f)
        val spaceMargin = dpToPx(8f)
        val primaryColor = Color.parseColor("#2EC4B6")
        val secondaryColor = Color.parseColor("#4A4A4A")
        val pressColor = Color.parseColor("#F0F0F0")
        val backgroundColor = Color.WHITE
        val dividerColor = Color.parseColor("#EAEAEA")
        val primaryTypeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        val secondaryTypeface = Typeface.DEFAULT
        val suggestionClickListener = OnClickListener { view ->
            val suggestion = (view as TextView).tag as String
            LogUtil.eventD(
                LogUtil.Category.GHOST,
                "suggestion_click_commit",
                "len=${suggestion.length} preview=\"${previewText(suggestion)}\""
            )
            stopAnimations()
            commitAiTextDirect(
                text = suggestion,
                hideReason = "ai_suggestion_tap_committed",
                nextReason = "ai_suggestion_tap_next",
                source = "aiSuggestionTap",
            )
        }
        val touchListener = OnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> v.setBackgroundColor(pressColor)
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> v.setBackgroundColor(
                    backgroundColor
                )
            }
            false
        }
        fun animateText(textView: TextView, fullText: String) {
            if (!enableStreamingAnimation) {
                textView.text = fullText
                return
            }
            textView.text = ""
            val length = fullText.length
            var currentIndex = 0
            val streamRunnable = object : Runnable {
                override fun run() {
                    if (textView.parent != null && currentIndex <= length) {
                        textView.text = fullText.substring(0, currentIndex)
                        if (currentIndex < length) {
                            currentIndex++
                            animHandler.postDelayed(this, streamCharDelay)
                        }
                    }
                }
            }
            activeStreamRunnables.add(streamRunnable)
            animHandler.post(streamRunnable)
        }
        if (isAiSuggestionExpanded) {
            currentSuggestions.forEachIndexed { index, suggestion ->
                if (index < currentSuggestions.size - 1) {
                    val textView = TextView(context).apply {
                        tag = suggestion
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).also { it.setMargins(0, 0, 0, 0) }
                        minimumHeight = minHeight
                        gravity = android.view.Gravity.CENTER_VERTICAL
                        setPadding(0, paddingVertical, 0, paddingVertical)
                        setTextColor(if (index == 0) primaryColor else secondaryColor)
                        textSize = if (index == 0) 17f else 16f
                        typeface = if (index == 0) primaryTypeface else secondaryTypeface
                        setOnClickListener(suggestionClickListener)
                        setOnTouchListener(touchListener)
                    }
                    aiSuggestionContainer.addView(textView)

                    if (index == 0) {
                        textView.text = suggestion
                    } else {
                        animateText(textView, suggestion)
                    }
                    val divider = View(context).apply {
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            dpToPx(1f)
                        ).apply {
                            leftMargin = dividerMargin
                            rightMargin = dividerMargin
                        }
                        setBackgroundColor(dividerColor)
                    }
                    aiSuggestionContainer.addView(divider)
                } else {
                    val lastRowLayout = RelativeLayout(context).apply {
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).also { it.setMargins(0, 0, 0, 0) }
                        setBackgroundColor(backgroundColor)
                    }
                    val collapseButton = TextView(context).apply {
                        id = View.generateViewId()
                        text = "▴"
                        setPadding(dpToPx(16f), paddingVertical, dpToPx(16f), paddingVertical)
                        setTextColor(secondaryColor)
                        textSize = 16f
                        setOnClickListener {
                            LogUtil.eventD(LogUtil.Category.UI, "suggestions_collapse", "")
                            isAiSuggestionExpanded = false
                            updateAiSuggestionView()
                        }
                        setOnTouchListener(touchListener)
                    }
                    val collapseParams = RelativeLayout.LayoutParams(
                        RelativeLayout.LayoutParams.WRAP_CONTENT,
                        RelativeLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        addRule(RelativeLayout.ALIGN_PARENT_END)
                        addRule(RelativeLayout.CENTER_VERTICAL)
                    }
                    lastRowLayout.addView(collapseButton, collapseParams)
                    val textView = TextView(context).apply {
                        tag = suggestion
                        val params = RelativeLayout.LayoutParams(
                            RelativeLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply {
                            addRule(RelativeLayout.START_OF, collapseButton.id)
                            addRule(RelativeLayout.CENTER_VERTICAL)
                            setMarginEnd(spaceMargin)
                        }
                        layoutParams = params
                        minimumHeight = minHeight
                        gravity = android.view.Gravity.CENTER_VERTICAL
                        setPadding(0, paddingVertical, 0, paddingVertical)
                        setTextColor(if (currentSuggestions.size == 1) primaryColor else secondaryColor)
                        textSize = if (currentSuggestions.size == 1) 17f else 16f
                        typeface =
                            if (currentSuggestions.size == 1) primaryTypeface else secondaryTypeface
                        setOnClickListener(suggestionClickListener)
                        setOnTouchListener(touchListener)
                    }
                    lastRowLayout.addView(textView)
                    aiSuggestionContainer.addView(lastRowLayout)


                    if (index == 0) {
                        textView.text = suggestion
                    } else {
                        animateText(textView, suggestion)
                    }
                }
            }
        } else {
            val collapsedContainer = RelativeLayout(context).apply {
                layoutParams = RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.MATCH_PARENT,
                    RelativeLayout.LayoutParams.WRAP_CONTENT
                )
                setBackgroundColor(backgroundColor)
            }
            val expandButton = TextView(context).apply {
                id = View.generateViewId()
                text = "▾"
                setPadding(dpToPx(16f), paddingVertical, dpToPx(16f), paddingVertical)
                setTextColor(secondaryColor)
                textSize = 16f
                setOnClickListener {
                    LogUtil.eventD(LogUtil.Category.UI, "suggestions_expand", "")
                    isAiSuggestionExpanded = true
                    updateAiSuggestionView()
                }
                setOnTouchListener(touchListener)
            }
            val expandParams = RelativeLayout.LayoutParams(
                RelativeLayout.LayoutParams.WRAP_CONTENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                addRule(RelativeLayout.ALIGN_PARENT_END)
                addRule(RelativeLayout.CENTER_VERTICAL)
            }
            collapsedContainer.addView(expandButton, expandParams)
            val content = currentSuggestions.firstOrNull() ?: ""
            val suggestionTextView = TextView(context).apply {
                tag = content
                val params = RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.MATCH_PARENT,
                    RelativeLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    addRule(RelativeLayout.START_OF, expandButton.id)
                    addRule(RelativeLayout.CENTER_VERTICAL)
                    setMarginEnd(spaceMargin)
                }
                layoutParams = params
                minimumHeight = minHeight
                gravity = android.view.Gravity.CENTER_VERTICAL
                maxLines = 1
                ellipsize = android.text.TextUtils.TruncateAt.END
                setPadding(0, paddingVertical, 0, paddingVertical)
                setTextColor(primaryColor)
                textSize = 17f
                typeface = primaryTypeface
                setOnClickListener(suggestionClickListener)
                setOnTouchListener(touchListener)
            }
            collapsedContainer.addView(suggestionTextView)
            aiSuggestionContainer.addView(collapsedContainer)

            animateText(suggestionTextView, content)
        }
    }
    private var lastKey75Timestamp: Long = 0

    // 辅助函数：检测AI键的快速连击
    private fun checkToggleAiKey(keyCode: Int): Boolean {
        if (keyCode == 75) {
            val now = System.currentTimeMillis()
            // 300ms内的连击判定为双击
            if (now - lastKey75Timestamp < 300) {
                LogUtil.event(
                    LogUtil.Category.INPUT,
                    "ai_key_double_tap_toggle",
                    "keyCode=75 deltaMs=${now - lastKey75Timestamp}"
                )
                service.toggleAiCompletion()
                lastKey75Timestamp = 0 // 重置以防连续误触
                return true // 消费掉这次事件
            }
            lastKey75Timestamp = now
        }
        return false
    }

    // 修改为Public，以便Service调用来清除GhostText
    fun hideAiSuggestion(reason: String = "unspecified") {
        LogUtil.logSpecialOperation(
            "InputView",
            "AI_SUGGESTION_HIDE",
            "reason=$reason ${ghostStateSummary()}"
        )
        LogUtil.eventD(LogUtil.Category.GHOST, "ghost_hide", "reason=$reason ${ghostStateSummary()}")
        stopAnimations()
        aiSuggestionContainer.visibility = View.GONE
        aiSuggestionContainer.removeAllViews()
        currentSuggestions = emptyArray()
        isAiSuggestionExpanded = false
        hideGhostOverlay()
        if (isGhostTextActive) {
            LogUtil.eventD(
                LogUtil.Category.GHOST,
                "ghost_deactivate",
                "reason=$reason ghostLen=${currentGhostText.length} ghostPreview=\"${previewText(currentGhostText)}\""
            )
            LogUtil.logSpecialOperation(
                "InputView",
                "GHOST_DEACTIVATE",
                "reason=$reason ghost text cleared: was \"${previewText(currentGhostText, 64)}\""
            )

            isGhostTextActive = false
            currentGhostText = ""
        }
    }

    // 仅隐藏 top2-4 列表（用于生成中 / session_changed 等场景，保持 GhostText 连续显示）
    fun hideAiSuggestionListOnly(reason: String = "unspecified") {
        LogUtil.eventD(LogUtil.Category.GHOST, "suggestions_list_hide", "reason=$reason ${ghostStateSummary()}")
        stopAnimations()
        aiSuggestionContainer.visibility = View.GONE
        aiSuggestionContainer.removeAllViews()
        currentSuggestions = emptyArray()
        isAiSuggestionExpanded = false
    }
    private fun dpToPx(dp: Float): Int = (dp * resources.displayMetrics.density).toInt()
    @SuppressLint("ClickableViewAccessibility")
    fun initView(context: Context) {
        if (isAddPhrases) {
            if (mAddPhrasesLayout.parent == null) {
                addView(
                    mAddPhrasesLayout,
                    LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                        addRule(ABOVE, mSkbRoot.id)
                        addRule(ALIGN_LEFT, mSkbRoot.id)
                    })
                mAddPhrasesLayout.handleAddPhrasesView()
            }
        } else {
            removeView(mAddPhrasesLayout)
        }
        mSkbCandidatesBarView.initialize(mChoiceNotifier)
        val oneHandedModSwitch = getInstance().keyboardSetting.oneHandedModSwitch.getValue()
        val oneHandedMod = getInstance().keyboardSetting.oneHandedMod.getValue()
        if (::mOnehandHoderLayout.isInitialized) mOnehandHoderLayout.visibility = GONE
        if (oneHandedModSwitch) {
            mOnehandHoderLayout = when (oneHandedMod) {
                KeyboardOneHandedMod.LEFT -> mHoderLayoutRight
                else -> mHoderLayoutLeft
            }
            mOnehandHoderLayout.visibility = VISIBLE
            mOnehandHoderLayout[0].setOnClickListener { view: View -> onClick(view) }
            mOnehandHoderLayout[1].setOnClickListener { view: View -> onClick(view) }
            (mOnehandHoderLayout[1] as ImageButton).setImageResource(if (oneHandedMod == KeyboardOneHandedMod.LEFT) R.drawable.ic_menu_one_hand_right else R.drawable.ic_menu_one_hand)
            val layoutParamsHoder = mOnehandHoderLayout.layoutParams
            layoutParamsHoder.width = EnvironmentSingleton.instance.holderWidth
            layoutParamsHoder.height = EnvironmentSingleton.instance.skbHeight
        }
        mLlKeyboardBottomHolder.removeAllViews()
        mLlKeyboardBottomHolder.layoutParams.width = EnvironmentSingleton.instance.skbWidth
        mInputKeyboardContainer.layoutParams.width = EnvironmentSingleton.instance.inputAreaWidth
        if (EnvironmentSingleton.instance.keyboardModeFloat) {
            mBottomPaddingKey =
                (if (EnvironmentSingleton.instance.isLandscape) getInstance().internal.keyboardBottomPaddingLandscapeFloat
                else getInstance().internal.keyboardBottomPaddingFloat)
            mRightPaddingKey =
                (if (EnvironmentSingleton.instance.isLandscape) getInstance().internal.keyboardRightPaddingLandscapeFloat
                else getInstance().internal.keyboardRightPaddingFloat)
            bottomPadding = mBottomPaddingKey.getValue()
            rightPadding = mRightPaddingKey.getValue()
            mSkbRoot.bottomPadding = 0
            mSkbRoot.rightPadding = 0
            mLlKeyboardBottomHolder.minimumHeight =
                EnvironmentSingleton.instance.heightForKeyboardMove
            val mIvKeyboardMove = ImageView(context).apply {
                setImageResource(R.drawable.ic_horizontal_line)
                isClickable = true
                isEnabled = true
            }
            mLlKeyboardBottomHolder.addView(mIvKeyboardMove)
            mIvKeyboardMove.setOnTouchListener { _, event -> onMoveKeyboardEvent(event) }
        } else {
            val fullDisplayKeyboardEnable =
                getInstance().internal.fullDisplayKeyboardEnable.getValue()
            if (fullDisplayKeyboardEnable && !EnvironmentSingleton.instance.isLandscape) {
                mFullDisplayKeyboardBar = FullDisplayKeyboardBar(context, this)
                mLlKeyboardBottomHolder.addView(mFullDisplayKeyboardBar)
                mLlKeyboardBottomHolder.minimumHeight =
                    EnvironmentSingleton.instance.heightForFullDisplayBar + EnvironmentSingleton.instance.systemNavbarWindowsBottom
            } else {
                mLlKeyboardBottomHolder.minimumHeight =
                    EnvironmentSingleton.instance.systemNavbarWindowsBottom
            }
            bottomPadding = 0
            rightPadding = 0
            mBottomPaddingKey = getInstance().internal.keyboardBottomPadding
            mRightPaddingKey = getInstance().internal.keyboardRightPadding
            mSkbRoot.bottomPadding = mBottomPaddingKey.getValue()
            mSkbRoot.rightPadding = mRightPaddingKey.getValue()
        }
        updateTheme()
        DecodingInfo.candidatesLiveData.observe(this) { _ ->
            updateCandidateBar()
            (KeyboardManager.instance.currentContainer as? CandidatesContainer)?.showCandidatesView()
        }
    }
    private var initialTouchX = 0f
    private var initialTouchY = 0f
    private var rightPaddingValue = 0
    private var bottomPaddingValue = 0
    private fun onMoveKeyboardEvent(event: MotionEvent?): Boolean {
        when (event?.action) {
            MotionEvent.ACTION_DOWN -> {
                bottomPaddingValue = mBottomPaddingKey.getValue()
                rightPaddingValue = mRightPaddingKey.getValue()
                initialTouchX = event.rawX
                initialTouchY = event.rawY
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx: Float = event.rawX - initialTouchX
                val dy: Float = event.rawY - initialTouchY
                if (dx.absoluteValue > 10) {
                    rightPaddingValue -= dx.toInt()
                    rightPaddingValue = if (rightPaddingValue < 0) 0
                    else if (rightPaddingValue > EnvironmentSingleton.instance.mScreenWidth - mSkbRoot.width) {
                        EnvironmentSingleton.instance.mScreenWidth - mSkbRoot.width
                    } else rightPaddingValue
                    initialTouchX = event.rawX
                    if (EnvironmentSingleton.instance.keyboardModeFloat) {
                        rightPadding = rightPaddingValue
                    } else {
                        mSkbRoot.rightPadding = rightPaddingValue
                    }
                }
                if (dy.absoluteValue > 10) {
                    bottomPaddingValue -= dy.toInt()
                    bottomPaddingValue = if (bottomPaddingValue < 0) 0
                    else if (bottomPaddingValue > EnvironmentSingleton.instance.mScreenHeight - mSkbRoot.height) {
                        EnvironmentSingleton.instance.mScreenHeight - mSkbRoot.height
                    } else bottomPaddingValue
                    initialTouchY = event.rawY
                    if (EnvironmentSingleton.instance.keyboardModeFloat) {
                        bottomPadding = bottomPaddingValue
                    } else {
                        mSkbRoot.bottomPadding = bottomPaddingValue
                    }
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                mRightPaddingKey.setValue(rightPaddingValue)
                mBottomPaddingKey.setValue(bottomPaddingValue)
            }
        }
        return false
    }
    fun updateTheme() {
        setBackgroundResource(android.R.color.transparent)
        val keyTextColor = ThemeManager.activeTheme.keyTextColor
        val backgrounde =
            ThemeManager.activeTheme.backgroundDrawable(ThemeManager.prefs.keyBorder.getValue())
        mSkbRoot.background = if (backgrounde is BitmapDrawable) backgrounde.bitmap.scale(
            EnvironmentSingleton.instance.skbWidth,
            EnvironmentSingleton.instance.inputAreaHeight
        ).toDrawable(context.resources) else backgrounde
        mSkbCandidatesBarView.updateTheme(keyTextColor)
        if (::mOnehandHoderLayout.isInitialized) {
            (mOnehandHoderLayout[0] as ImageButton).drawable?.setTint(keyTextColor)
            (mOnehandHoderLayout[1] as ImageButton).drawable?.setTint(keyTextColor)
        }
        mFullDisplayKeyboardBar?.updateTheme(keyTextColor)
        mAddPhrasesLayout.updateTheme(ThemeManager.activeTheme)
    }
    private fun onClick(view: View) {
        if (view.id == R.id.ib_holder_one_hand_none) {
            getInstance().keyboardSetting.oneHandedModSwitch.setValue(!getInstance().keyboardSetting.oneHandedModSwitch.getValue())
        } else {
            val oneHandedMod = getInstance().keyboardSetting.oneHandedMod.getValue()
            getInstance().keyboardSetting.oneHandedMod.setValue(if (oneHandedMod == KeyboardOneHandedMod.LEFT) KeyboardOneHandedMod.RIGHT else KeyboardOneHandedMod.LEFT)
        }
        EnvironmentSingleton.instance.initData()
        KeyboardLoaderUtil.instance.clearKeyboardMap()
        KeyboardManager.instance.clearKeyboard()
        KeyboardManager.instance.switchKeyboard()
    }
    override fun responseKeyEvent(sKey: SoftKey) {
        val keyCode = sKey.code

        // 优先检测AI补全键的开关切换逻辑
        if (checkToggleAiKey(keyCode)) return

        LogUtil.eventD(
            LogUtil.Category.INPUT,
            "softkey_event",
            "keyCode=$keyCode isKeyCodeKey=${sKey.isKeyCodeKey} isUserDef=${sKey.isUserDefKey} isUniStr=${sKey.isUniStrKey} ghostActive=$isGhostTextActive"
        )
        // 修改：将空格键的分词功能迁移到75号键（分词键）
        if (isGhostTextActive && keyCode == 75) {
            LogUtil.event(LogUtil.Category.GHOST, "ghost_confirm_key75", "action=commit")
            commitGhostText()
            return
        }
        if (sKey.isKeyCodeKey) {
            mImeState = ImeState.STATE_INPUT
            val keyEvent =
                KeyEvent(0, 0, KeyEvent.ACTION_UP, keyCode, 0, 0, 0, 0, KeyEvent.FLAG_SOFT_KEYBOARD)
            processKey(keyEvent)
        } else if (sKey.isUserDefKey || sKey.isUniStrKey) {
            if (!DecodingInfo.isAssociate && !DecodingInfo.isCandidatesListEmpty) {
                if (InputModeSwitcherManager.isChinese) chooseAndUpdate()
                else if (InputModeSwitcherManager.isEnglish) commitDecInfoText(DecodingInfo.composingStrForCommit)
            }
            if (InputModeSwitcherManager.USER_DEF_KEYCODE_SYMBOL_3 == keyCode) {
                KeyboardManager.instance.switchKeyboard(KeyboardManager.KeyboardType.SYMBOL)
                (KeyboardManager.instance.currentContainer as? SymbolContainer)?.setSymbolsView()
            } else if (InputModeSwitcherManager.USER_DEF_KEYCODE_EMOJI_4 == keyCode) {
                onSettingsMenuClick(SkbMenuMode.Emojicon)
            } else if (InputModeSwitcherManager.USER_DEF_KEYCODE_SHIFT_1 == keyCode) {
                if (InputModeSwitcherManager.isChineseT9) {
                    InputModeSwitcherManager.switchModeForUserKey(InputModeSwitcherManager.USER_DEF_KEYCODE_NUMBER_5)
                } else if (InputModeSwitcherManager.isNumberSkb) {
                    InputModeSwitcherManager.switchModeForUserKey(InputModeSwitcherManager.USER_DEF_KEYCODE_RETURN_6)
                } else {
                    InputModeSwitcherManager.switchModeForUserKey(keyCode)
                }
            } else if (keyCode in InputModeSwitcherManager.USER_DEF_KEYCODE_RETURN_6..InputModeSwitcherManager.USER_DEF_KEYCODE_SHIFT_1) {
                InputModeSwitcherManager.switchModeForUserKey(keyCode)
            } else if (keyCode in InputModeSwitcherManager.USER_DEF_KEYCODE_PASTE..InputModeSwitcherManager.USER_DEF_KEYCODE_CUT) {
                commitTextEditMenu(KeyPreset.textEditMenuPreset[keyCode])
            } else if (keyCode == InputModeSwitcherManager.USER_DEF_KEYCODE_MOVE_START) {
                service.setSelection(0, if (hasSelection) selEnd else 0)
            } else if (keyCode == InputModeSwitcherManager.USER_DEF_KEYCODE_MOVE_END) {
                if (hasSelection) {
                    val start = selStart
                    commitTextEditMenu(KeyPreset.textEditMenuPreset[InputModeSwitcherManager.USER_DEF_KEYCODE_SELECT_ALL])
                    this.postDelayed(50) { service.setSelection(start, selEnd) }
                } else {
                    commitTextEditMenu(KeyPreset.textEditMenuPreset[InputModeSwitcherManager.USER_DEF_KEYCODE_SELECT_ALL])
                    service.sendCombinationKeyEvents(KeyEvent.KEYCODE_DPAD_RIGHT)
                }
            } else if (keyCode == InputModeSwitcherManager.USER_DEF_KEYCODE_SELECT_MODE) {
                hasSelection = !hasSelection
                if (!hasSelection) service.sendCombinationKeyEvents(KeyEvent.KEYCODE_DPAD_RIGHT)
            } else if (keyCode == InputModeSwitcherManager.USER_DEF_KEYCODE_SELECT_ALL) {
                hasSelectionAll = !hasSelectionAll
                if (!hasSelectionAll) service.sendCombinationKeyEvents(KeyEvent.KEYCODE_DPAD_RIGHT)
                else commitTextEditMenu(KeyPreset.textEditMenuPreset[keyCode])
            } else if (sKey.keyLabel.isNotBlank()) {
                if (SymbolPreset.containsKey(sKey.keyLabel)) commitPairSymbol(sKey.keyLabel)
                else commitText(sKey.keyLabel)
            }
            if (mImeState != ImeState.STATE_IDLE) resetToIdleState()
        }
    }

    private fun commitGhostText() {
        if (!isGhostTextActive || currentGhostText.isEmpty()) {
            LogUtil.logSpecialOperation("InputView", "GHOST_COMMIT_SKIP", "ghost not active or empty, skipping commit")
            LogUtil.eventW(LogUtil.Category.GHOST, "ghost_commit_skip", ghostStateSummary())
            return
        }

        LogUtil.logCommitText("InputView", currentGhostText, source = "commitGhostText")
        LogUtil.event(
            LogUtil.Category.GHOST,
            "ghost_commit_start",
            "len=${currentGhostText.length} preview=\"${previewText(currentGhostText)}\""
        )
        val textToCommit = currentGhostText
        lastCommittedAiText = textToCommit
        isGhostTextActive = false
        currentGhostText = ""
        LogUtil.eventD(LogUtil.Category.INPUT, "commit_text_begin", "source=ghost")
        service.commitText(textToCommit)
        hideAiSuggestion("ghost_committed")
        LogUtil.event(LogUtil.Category.GHOST, "ghost_commit_done", "action=request_next_completion")
        LogUtil.logSpecialOperation("InputView", "GHOST_COMMIT_SUCCESS", "ghost text committed, requesting next completion")
        service.llmManager.requestCompletion(reason = "ghost_commit_next") // 重构：通过 llmManager 调用
    }

    /**
     * 检查 Ghost Text 是否处于激活状态
     * 暴露给 Service 使用，用于在提交文本前检查
     */
    fun isGhostTextActive(): Boolean = isGhostTextActive

    /**
     * 获取当前的 Ghost Text 内容
     * 暴露给 Service 使用
     */
    fun getGhostText(): String = if (isGhostTextActive) currentGhostText else ""

    /**
     * 强制清除 Ghost Text（不提交）
     * 用于在执行其他提交操作前清除 Ghost Text
     * @return true 如果 Ghost Text 被清除，false 如果没有激活的 Ghost Text
     */
    fun clearGhostTextIfNeeded(): Boolean {
        if (isGhostTextActive) {
            LogUtil.eventD(LogUtil.Category.GHOST, "ghost_clear_if_needed", "reason=pre_commit_safety")
            hideAiSuggestion("clearGhostTextIfNeeded")
            return true
        }
        return false
    }

    private var textBeforeCursor: String = ""
    override fun responseLongKeyEvent(result: Pair<PopupMenuMode, String>) {
        if (!DecodingInfo.isAssociate && !DecodingInfo.isCandidatesListEmpty) {
            if (InputModeSwitcherManager.isChinese) {
                chooseAndUpdate()
            } else if (InputModeSwitcherManager.isEnglish) {
                commitDecInfoText(DecodingInfo.composingStrForCommit)
            }
        }
        when (result.first) {
            PopupMenuMode.Text -> {
                if (SymbolPreset.containsKey(result.second)) commitPairSymbol(result.second)
                else commitText(result.second)
            }
            PopupMenuMode.SwitchIME -> InputMethodUtil.showPicker()
            PopupMenuMode.EMOJI -> {
                onSettingsMenuClick(SkbMenuMode.Emojicon)
            }
            PopupMenuMode.EnglishCell -> {
                getInstance().input.abcSearchEnglishCell.setValue(!getInstance().input.abcSearchEnglishCell.getValue())
                KeyboardManager.instance.switchKeyboard()
            }
            PopupMenuMode.Clear -> {
                if (isAddPhrases) mAddPhrasesLayout.clearPhrasesContent()
                else {
                    val clearText = service.getTextBeforeCursor(1).toString()
                    if (clearText.isNotEmpty()) {
                        textBeforeCursor = clearText
                        service.deleteSurroundingText(1000)
                    }
                }
            }
            PopupMenuMode.Revertl -> {
                commitText(textBeforeCursor)
                textBeforeCursor = ""
            }
            PopupMenuMode.Enter -> commitText("\n")
            else -> {}
        }
        if (result.first == PopupMenuMode.Text && mImeState != ImeState.STATE_PREDICT) resetToPredictState()
        else if (result.first != PopupMenuMode.None && mImeState != ImeState.STATE_IDLE) resetToIdleState()
    }
    override fun responseHandwritingResultEvent(words: Array<CandidateListItem>) {
        DecodingInfo.isAssociate = false
        DecodingInfo.cacheCandidates(words)
        mImeState = ImeState.STATE_INPUT
        updateCandidateBar()
    }
    fun processKey(event: KeyEvent): Boolean {
        val actionName = when(event.action) {
            KeyEvent.ACTION_DOWN -> "DOWN"
            KeyEvent.ACTION_UP -> "UP"
            else -> "UNKNOWN(${event.action})"
        }
        LogUtil.logKeyEvent("InputView", event.keyCode, event.unicodeChar, actionName)
        if (processFunctionKeys(event)) return true
        val englishCellDisable =
            InputModeSwitcherManager.isEnglish && !getInstance().input.abcSearchEnglishCell.getValue()
        val result = if (englishCellDisable) {
            processEnglishKey(event)
        } else if (InputModeSwitcherManager.isEnglish || InputModeSwitcherManager.isChinese) {
            processInput(event)
        } else {
            processEnglishKey(event)
        }
        return result
    }
    private fun processEnglishKey(event: KeyEvent): Boolean {
        val keyCode = event.keyCode
        var keyChar = event.unicodeChar
        val lable = keyChar.toChar().toString()
        if (keyCode == KeyEvent.KEYCODE_DEL) {
            LogUtil.logSpecialOperation("InputView", "ENGLISH_DEL", "delete key in English mode")
            sendKeyEvent(keyCode)
            if (mImeState != ImeState.STATE_IDLE) {
                LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "DEL in English mode")
                resetToIdleState()
            }
            return true
        } else if (keyCode in (KeyEvent.KEYCODE_A..KeyEvent.KEYCODE_Z)) {
            if (!InputModeSwitcherManager.isEnglishLower) keyChar = keyChar - 'a'.code + 'A'.code
            LogUtil.logSpecialOperation("InputView", "ENGLISH_CHAR", "commit char: '${keyChar.toChar()}'")
            commitText(keyChar.toChar().toString())
            return true
        } else if (keyCode != 0) {
            LogUtil.logSpecialOperation("InputView", "ENGLISH_KEYCODE", "keyCode=$keyCode")
            sendKeyEvent(keyCode)
            if (mImeState != ImeState.STATE_IDLE) {
                LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "non-zero keyCode in English")
                resetToIdleState()
            }
            return true
        } else if (lable.isNotEmpty()) {
            if (SymbolPreset.containsKey(lable)) commitPairSymbol(lable)
            else commitText(lable)
            return true
        }
        return false
    }
    private fun processFunctionKeys(event: KeyEvent): Boolean {
        val keyCode = event.keyCode
        // 关键修复：Ghost Text 的清除逻辑
        // 任何非明确确认的操作（如 Enter、Space、功能键）都应该先清除 Ghost Text
        // 只有 75 号键（分词键）或候选词点击才会提交 Ghost Text
        if (isGhostTextActive && keyCode != KeyEvent.KEYCODE_BACK) {
            LogUtil.logSpecialOperation("InputView", "GHOST_CLEAR", "ghost text cleared by function key")
            LogUtil.eventD(LogUtil.Category.GHOST, "ghost_clear_by_function_key", "keyCode=$keyCode")
            hideAiSuggestion("function_key_$keyCode")
            // 注意：这里不 return，让按键继续正常处理
        }

        if (keyCode == KeyEvent.KEYCODE_BACK) {
            LogUtil.logSpecialOperation("InputView", "BACK_KEY", "back key pressed")
            if (service.isInputViewShown) {
                requestHideSelf()
                return true
            }
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER || keyCode == KeyEvent.KEYCODE_SPACE) {
            // 空格键也应该先清除 Ghost Text（上面已处理）
            // 修改：恢复空格键的原始功能（确认选择或发送空格）
            LogUtil.logSpecialOperation("InputView", "SPACE_CENTER", "space/center key, isFinish=${DecodingInfo.isFinish}, isAssociate=${DecodingInfo.isAssociate}")
            if (DecodingInfo.isFinish || (DecodingInfo.isAssociate && !mSkbCandidatesBarView.isActiveCand())) {
                sendKeyEvent(keyCode)
                if (mImeState != ImeState.STATE_IDLE) {
                    LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "space/center key")
                    resetToIdleState()
                }
            } else {
                chooseAndUpdate()
            }
            return true
        } else if (keyCode == KeyEvent.KEYCODE_CLEAR) {
            LogUtil.logSpecialOperation("InputView", "CLEAR_KEY", "clear key pressed")
            if (mImeState != ImeState.STATE_IDLE) {
                LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "clear key")
                resetToIdleState()
            }
            return true
        } else if (keyCode == KeyEvent.KEYCODE_ENTER) {
            // Enter 键处理（Ghost Text 已在上面被清除）
            LogUtil.logSpecialOperation("InputView", "ENTER_KEY", "enter key, isFinish=${DecodingInfo.isFinish}, isAssociate=${DecodingInfo.isAssociate}")
            if (event.action == KeyEvent.ACTION_DOWN) {
                if (DecodingInfo.isFinish || DecodingInfo.isAssociate) {
                    sendKeyEvent(keyCode)
                } else {
                    commitDecInfoText(DecodingInfo.composingStrForCommit)
                }
                if (mImeState != ImeState.STATE_IDLE) {
                    LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "enter key")
                    resetToIdleState()
                }
            }
            return true
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_LEFT || keyCode == KeyEvent.KEYCODE_DPAD_RIGHT) {
            LogUtil.logSpecialOperation("InputView", "DPAD_LEFTRIGHT", "keyCode=$keyCode, flags=${event.flags}, candidatesEmpty=${DecodingInfo.isCandidatesListEmpty}")
            if (event.flags != KeyEvent.FLAG_SOFT_KEYBOARD && !DecodingInfo.isCandidatesListEmpty) {
                mSkbCandidatesBarView.updateActiveCandidateNo(keyCode)
            } else if (DecodingInfo.isFinish || DecodingInfo.isAssociate) {
                sendKeyEvent(keyCode)
            } else {
                chooseAndUpdate()
            }
            return true
        }
        return false
    }
    private fun processInput(event: KeyEvent): Boolean {
        val keyCode = event.keyCode

        // 如果不是软键盘事件（即物理按键），也需要检测开关切换
        // 软键盘事件已经在 responseKeyEvent 中检测过了
        val isSoft = (event.flags and KeyEvent.FLAG_SOFT_KEYBOARD) != 0
        if (!isSoft) {
            if (checkToggleAiKey(keyCode)) return true
        }

        if (isGhostTextActive && keyCode == KeyEvent.KEYCODE_DEL) {
            LogUtil.logSpecialOperation("InputView", "GHOST_DEL", "delete key with ghost text active, clearing all")
            LogUtil.event(LogUtil.Category.GHOST, "ghost_reject_del", "action=stop_completion_and_clear")
            service.llmManager.stopCompletion(reason = "ghost_reject_del") // 重构：通过 llmManager 调用
            hideAiSuggestion("del_reject")
            return true
        }
        if (isGhostTextActive) {
            LogUtil.logSpecialOperation("InputView", "GHOST_INTERRUPT", "user input interrupted ghost text")
            LogUtil.eventD(LogUtil.Category.GHOST, "ghost_interrupt_by_input", "keyCode=$keyCode")
            hideAiSuggestion("user_input_interrupt")
        }
        var keyChar = event.unicodeChar
        val lable = keyChar.toChar().toString()
        // 修改：75号键作为分词键的处理逻辑（在输入模式中）
        if (keyCode == 75) {
            LogUtil.logSpecialOperation("InputView", "AI_KEY_75", "AI key 75, ghostActive=$isGhostTextActive")
            LogUtil.event(LogUtil.Category.INPUT, "ai_key_75", "ghostActive=$isGhostTextActive")
            if (isGhostTextActive) {
                commitGhostText()
            } else {
                // 如果不是Ghost Text状态，可以触发新的AI生成
                service.llmManager.requestCompletion(reason = "ai_key_75") // 重构：通过 llmManager 调用
            }
            return true
        }
        if (keyCode == KeyEvent.KEYCODE_DEL) {
            LogUtil.logSpecialOperation("InputView", "DEL_KEY", "delete key, isFinish=${DecodingInfo.isFinish}, isAssociate=${DecodingInfo.isAssociate}")
            if (DecodingInfo.isFinish || DecodingInfo.isAssociate) {
                sendKeyEvent(keyCode)
                if (mImeState != ImeState.STATE_IDLE) {
                    LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "delete in finish/associate")
                    resetToIdleState()
                }
            } else {
                DecodingInfo.deleteAction()
                updateCandidate()
            }
            return true
        } else if ((Character.isLetterOrDigit(keyChar) && keyCode != KeyEvent.KEYCODE_0) || keyCode == KeyEvent.KEYCODE_APOSTROPHE || keyCode == KeyEvent.KEYCODE_SEMICOLON) {
            LogUtil.logSpecialOperation("InputView", "INPUT_CHAR", "char input: '${keyChar.toChar()}'")
            DecodingInfo.inputAction(event)
            updateCandidate()
            return true
        } else if (keyCode != 0) {
            LogUtil.logSpecialOperation("InputView", "INPUT_KEYCODE", "keyCode=$keyCode, candidatesEmpty=${DecodingInfo.isCandidatesListEmpty}")
            if (!DecodingInfo.isCandidatesListEmpty && !DecodingInfo.isAssociate) {
                chooseAndUpdate()
            }
            sendKeyEvent(keyCode)
            if (mImeState != ImeState.STATE_IDLE) {
                LogUtil.logStateChange("InputView", mImeState.name, "STATE_IDLE", "non-zero keyCode")
                resetToIdleState()
            }
            return true
        } else if (lable.isNotEmpty()) {
            if (!DecodingInfo.isCandidatesListEmpty && !DecodingInfo.isAssociate) {
                chooseAndUpdate()
            }
            if (SymbolPreset.containsKey(lable)) commitPairSymbol(lable)
            else commitText(lable)
            return true
        }
        return false
    }

    fun resetToIdleState() {
        hideAiSuggestion("reset_to_idle")
        resetCandidateWindow()
        if (hasSelectionAll) hasSelectionAll = false
        mImeState = ImeState.STATE_IDLE
    }
    private fun resetToPredictState() {
        resetCandidateWindow()
        mImeState = ImeState.STATE_PREDICT
    }
    fun chooseAndUpdate(candId: Int = mSkbCandidatesBarView.getActiveCandNo()) {
        val candidate = DecodingInfo.getCandidate(candId)
        if (candidate?.comment == "📋") {
            commitDecInfoText(candidate.text)
            if (mImeState != ImeState.STATE_PREDICT) resetToPredictState()
        } else {
            val choice = DecodingInfo.chooseDecodingCandidate(candId)
            if (DecodingInfo.isEngineFinish || DecodingInfo.isAssociate) {
                commitDecInfoText(choice)
                KeyboardManager.instance.switchKeyboard(InputModeSwitcherManager.skbLayout)
                (KeyboardManager.instance.currentContainer as? T9TextContainer)?.updateSymbolListView()
                if (mImeState != ImeState.STATE_PREDICT) resetToPredictState()
            } else {
                if (!DecodingInfo.isFinish) {
                    if (InputModeSwitcherManager.isEnglish) setComposingText(DecodingInfo.composingStrForCommit)
                    updateCandidateBar()
                    (KeyboardManager.instance.currentContainer as? T9TextContainer)?.updateSymbolListView()
                } else {
                    if (mImeState != ImeState.STATE_IDLE) resetToIdleState()
                }
            }
        }
    }
    private fun updateCandidate() {
        DecodingInfo.updateDecodingCandidate()
        if (!DecodingInfo.isFinish) {
            updateCandidateBar()
            (KeyboardManager.instance.currentContainer as? T9TextContainer)?.updateSymbolListView()
        } else {
            if (mImeState != ImeState.STATE_IDLE) resetToIdleState()
        }
        if (InputModeSwitcherManager.isEnglish) setComposingText(DecodingInfo.composingStrForCommit)
    }
    fun updateCandidateBar() {
        mSkbCandidatesBarView.showCandidates()
    }
    private fun resetCandidateWindow() {
        DecodingInfo.reset()
        updateCandidateBar()
        (KeyboardManager.instance.currentContainer as? T9TextContainer)?.updateSymbolListView()
    }
    inner class ChoiceNotifier internal constructor() : CandidateViewListener {
        override fun onClickChoice(choiceId: Int) {
            DevicesUtils.tryPlayKeyDown()
            DevicesUtils.tryVibrate(KeyboardManager.instance.currentContainer)
            chooseAndUpdate(choiceId)
        }
        override fun onClickMore(level: Int) {
            if (level == 0) {
                onSettingsMenuClick(SkbMenuMode.CandidatesMore)
            } else {
                KeyboardManager.instance.switchKeyboard()
                (KeyboardManager.instance.currentContainer as? T9TextContainer)?.updateSymbolListView()
            }
        }
        override fun onClickMenu(skbMenuMode: SkbMenuMode) {
            onSettingsMenuClick(skbMenuMode)
        }
        override fun onClickClearCandidate() {
            if (mImeState != ImeState.STATE_IDLE) resetToIdleState()
            KeyboardManager.instance.switchKeyboard()
        }
        override fun onClickClearClipBoard() {
            DataBaseKT.instance.clipboardDao().deleteAll()
            (KeyboardManager.instance.currentContainer as? ClipBoardContainer)?.showClipBoardView(
                SkbMenuMode.ClipBoard
            )
        }
    }
    fun onSettingsMenuClick(skbMenuMode: SkbMenuMode, extra: String = "") {
        when (skbMenuMode) {
            SkbMenuMode.AddPhrases -> {
                isAddPhrases = true
                DataBaseKT.instance.phraseDao().deleteByContent(extra)
                KeyboardManager.instance.switchKeyboard(InputModeSwitcherManager.skbImeLayout)
                initView(context)
                mAddPhrasesLayout.setExtraData(extra)
            }
            else -> onSettingsMenuClick(this, skbMenuMode)
        }
        mSkbCandidatesBarView.initMenuView()
    }
    enum class ImeState {
        STATE_IDLE, STATE_INPUT, STATE_PREDICT
    }
    fun selectPrefix(position: Int) {
        DevicesUtils.tryPlayKeyDown()
        DevicesUtils.tryVibrate(this)
        DecodingInfo.selectPrefix(position)
        updateCandidate()
    }
    fun showSymbols(symbols: Array<String>) {
        mImeState = ImeState.STATE_INPUT
        val list = symbols.map { symbol -> CandidateListItem("📋", symbol) }.toTypedArray()
        DecodingInfo.cacheCandidates(list)
        DecodingInfo.isAssociate = true
        updateCandidateBar()
    }
    fun requestHideSelf() {
        service.requestHideSelf(0)
    }
    private fun sendKeyEvent(keyCode: Int) {
        if (isAddPhrases) {
            mAddPhrasesLayout.sendKeyEvent(keyCode)
            when (keyCode) {
                KeyEvent.KEYCODE_ENTER -> {
                    isAddPhrases = false
                    initView(context)
                    onSettingsMenuClick(SkbMenuMode.Phrases)
                }
            }
        } else if (keyCode == KeyEvent.KEYCODE_ENTER) {
            service.sendEnterKeyEvent()
        } else if (keyCode in KeyEvent.KEYCODE_DPAD_UP..KeyEvent.KEYCODE_DPAD_RIGHT) {
            service.sendCombinationKeyEvents(keyCode, shift = hasSelection)
            if (hasSelectionAll) hasSelectionAll = false
        } else {
            service.sendCombinationKeyEvents(keyCode)
        }
    }
    private fun setComposingText(text: CharSequence) {
        if (!isAddPhrases) service.setComposingText(text)
    }
    private fun commitText(text: String) {
        if (isAddPhrases) mAddPhrasesLayout.commitText(text)
        else service.commitText(StringUtils.converted2FlowerTypeface(text))
    }
    private fun commitPairSymbol(text: String) {
        if (isAddPhrases) {
            mAddPhrasesLayout.commitText(text)
        } else {
            if (getInstance().input.symbolPairInput.getValue()) {
                service.commitText(text + SymbolPreset[text]!!)
                service.sendCombinationKeyEvents(KeyEvent.KEYCODE_DPAD_LEFT)
            } else service.commitText(text)
        }
    }
    private fun commitTextEditMenu(id: Int?) {
        if (id != null) service.commitTextEditMenu(id)
    }
    fun performEditorAction(editorAction: Int) {
        service.performEditorAction(editorAction)
    }
    private fun commitDecInfoText(resultText: String?) {
        if (resultText == null) return
        if (isAddPhrases) {
            mAddPhrasesLayout.commitText(resultText)
        } else {
            service.commitText(StringUtils.converted2FlowerTypeface(resultText))
            if (InputModeSwitcherManager.isEnglish && DecodingInfo.isEngineFinish && getInstance().input.abcSpaceAuto.getValue() && StringUtils.isEnglishWord(
                    resultText
                )
            ) {
                service.commitText(" ")
            }
        }
    }
    private fun initNavbarBackground(service: ImeService) {
        service.window.window!!.also {
            WindowCompat.setDecorFitsSystemWindows(it, false)
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
                @Suppress("DEPRECATION")
                it.navigationBarColor = Color.TRANSPARENT
            } else {
                it.insetsController?.hide(WindowInsets.Type.navigationBars())
                it.insetsController?.systemBarsBehavior =
                    WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) it.isNavigationBarContrastEnforced =
                false
        }
        ViewCompat.setOnApplyWindowInsetsListener(this) { _, insets ->
            EnvironmentSingleton.instance.systemNavbarWindowsBottom =
                insets.getInsets(WindowInsetsCompat.Type.navigationBars()).bottom
            val fullDisplayKeyboardEnable =
                getInstance().internal.fullDisplayKeyboardEnable.getValue()
            mLlKeyboardBottomHolder.minimumHeight =
                if (EnvironmentSingleton.instance.keyboardModeFloat) 0
                else if (fullDisplayKeyboardEnable) EnvironmentSingleton.instance.heightForFullDisplayBar + EnvironmentSingleton.instance.systemNavbarWindowsBottom
                else EnvironmentSingleton.instance.systemNavbarWindowsBottom
            insets
        }
    }
    @SuppressLint("SimpleDateFormat")
    fun onStartInputView(editorInfo: EditorInfo, restarting: Boolean) {
        InputModeSwitcherManager.requestInputWithSkb(editorInfo)
        KeyboardManager.instance.switchKeyboard()
        resetToIdleState()
        if (!restarting) {
            if (getInstance().clipboard.clipboardSuggestion.getValue()) {
                val lastClipboardTime = getInstance().internal.clipboardUpdateTime.getValue()
                if (System.currentTimeMillis() - lastClipboardTime <= clipboardItemTimeout * 1000) {
                    val lastClipboardContent =
                        getInstance().internal.clipboardUpdateContent.getValue()
                    if (lastClipboardContent.isNotBlank()) {
                        showSymbols(arrayOf(lastClipboardContent))
                        getInstance().internal.clipboardUpdateTime.setValue(0L)
                    }
                }
            }
        }
    }
    fun onWindowShown() {
        chinesePrediction = getInstance().input.chinesePrediction.getValue()
    }
    fun onWindowHidden() {
        hideAiSuggestion("window_hidden")
        if (isAddPhrases) {
            isAddPhrases = false
            mAddPhrasesLayout.addPhrasesHandle()
            initView(context)
        }
        if (mImeState != ImeState.STATE_IDLE) resetToIdleState()
    }
    private var selStart = 0
    private var selEnd = 0
    private var oldCandidatesEnd = 0
    fun onUpdateSelection(
        oldSelStart: Int,
        oldSelEnd: Int,
        newSelStart: Int,
        newSelEnd: Int,
        candidatesEnd: Int
    ) {
        selStart = newSelStart; selEnd = newSelEnd
        if (oldCandidatesEnd == candidatesEnd && InputModeSwitcherManager.isEnglish && !DecodingInfo.isCandidatesListEmpty && !DecodingInfo.isAssociate) {
            service.finishComposingText()
            resetToPredictState()
        }
        if (oldSelStart != oldSelEnd || newSelStart != newSelEnd) {
            if (isGhostTextActive) {
                if (LogUtil.rateLimit("ghost.selection_change", 250)) {
                    LogUtil.eventD(
                        LogUtil.Category.GHOST,
                        "ghost_clear_selection_changed",
                        "old=($oldSelStart,$oldSelEnd) new=($newSelStart,$newSelEnd)"
                    )
                }
                hideAiSuggestion("selection_changed")
            }
            return
        }
        oldCandidatesEnd = candidatesEnd

        if (isGhostTextActive) {
            if (oldSelStart != newSelStart) {
                if (LogUtil.rateLimit("ghost.cursor_move", 250)) {
                    LogUtil.eventD(
                        LogUtil.Category.GHOST,
                        "ghost_clear_cursor_moved",
                        "old=$oldSelStart new=$newSelStart"
                    )
                }
                hideAiSuggestion("cursor_moved")
                return
            }

            if (LogUtil.rateLimit("ghost.selection_ignore", 500)) {
                LogUtil.eventD(
                    LogUtil.Category.GHOST,
                    "ghost_active_ignore_prediction",
                    "sel=($newSelStart,$newSelEnd)"
                )
            }
            return
        }

        if ((chinesePrediction && InputModeSwitcherManager.isChinese && mImeState != ImeState.STATE_IDLE) || InputModeSwitcherManager.isNumberSkb) {
            val textBeforeCursor = service.getTextBeforeCursor(100)
            if (textBeforeCursor.isNotBlank()) {
                val expressionEnd = CustomEngine.parseExpressionAtEnd(textBeforeCursor)
                if (!expressionEnd.isNullOrBlank()) {
                    if (expressionEnd.length < 100) {
                        val result =
                            CustomEngine.expressionCalculator(textBeforeCursor, expressionEnd)
                        if (result.isNotEmpty()) showSymbols(result)
                    }
                } else if (StringUtils.isChineseEnd(textBeforeCursor)) {
                    DecodingInfo.isAssociate = true
                    DecodingInfo.getAssociateWord(
                        if (textBeforeCursor.length > 10) textBeforeCursor.substring(
                            textBeforeCursor.length - 10
                        ) else textBeforeCursor
                    )
                    updateCandidate()
                    updateCandidateBar()
                }
            }
        }
        if (mImeState == ImeState.STATE_IDLE || mImeState == ImeState.STATE_PREDICT) {
            if (newSelStart == newSelEnd && oldSelStart != newSelStart) {
                val text = service.getTextBeforeCursor(200)
                if (text.isNotBlank()) {
                    service.llmManager.requestCompletion(reason = "cursor_move_prediction") // 重构：通过 llmManager 调用
                }
            }
        } else if (mImeState == ImeState.STATE_INPUT) {
            service.llmManager.stopCompletion(reason = "state_input") // 重构：通过 llmManager 调用
            hideAiSuggestion("state_input_stop_completion")
        }
    }
}