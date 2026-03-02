package com.yuyan.imemodule.keyboard

import android.content.Context
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.ShapeDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.CursorAnchorInfo
import android.widget.PopupWindow
import android.widget.TextView
import android.util.Log
import android.widget.LinearLayout

/**
 * GhostText 浮动窗口
 * 使用 PopupWindow 将 GhostText 定位到输入框光标位置
 * 支持跟随光标移动、换行处理
 */
class GhostTextPopupWindow(
    private val context: Context,
    private val onBubbleClick: ((String) -> Unit)? = null,
) {
    
    companion object {
        private const val TAG = "GhostTextPopup"
    }
    
    private var popupWindow: PopupWindow? = null
    private var textView: TextView? = null
    private var containerView: LinearLayout? = null
    private var anchorDotView: View? = null
    private var guideLineView: View? = null
    private var anchorView: View? = null
    
    // 当前状态
    private var currentText: String = ""
    private var isShowing: Boolean = false
    
    // 样式参数
    private val accentSecondary = Color.parseColor("#2196F3")
    private val bubbleStroke = Color.parseColor("#1A2196F3") // 10% alpha for subtle outline

    // 屏幕尺寸缓存
    private val screenWidth: Int by lazy {
        context.resources.displayMetrics.widthPixels
    }
    private val screenHeight: Int by lazy {
        context.resources.displayMetrics.heightPixels
    }
    
    /**
     * 显示 GhostText 在光标位置
     * @param text GhostText 内容
     * @param cursorAnchorInfo 光标位置信息
     * @param anchor 锚定 View（通常是 IME 的 DecorView）
     */
    fun show(text: String, cursorAnchorInfo: CursorAnchorInfo?, anchor: View) {
        if (text.isBlank()) {
            dismiss()
            return
        }

        currentText = text
        anchorView = anchor

        val contentRoot = getOrCreateContentView()
        val tv = textView ?: return
        tv.text = text

        // 设置合理的最大宽度，超过则自动换行
        val maxWidth = (screenWidth * 0.65f).toInt().coerceAtLeast(dpToPx(200f))
        tv.maxWidth = maxWidth - dpToPx(48f) // 预留锚点和间距

        // 测量整体容器尺寸
        contentRoot.measure(
            View.MeasureSpec.makeMeasureSpec(maxWidth, View.MeasureSpec.AT_MOST),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
        )
        val contentWidth = contentRoot.measuredWidth
        val contentHeight = contentRoot.measuredHeight.coerceAtLeast(dpToPx(36f))

        val popup = getOrCreatePopupWindow(contentRoot, contentWidth, contentHeight)

        // 计算光标位置
        val position = calculatePosition(cursorAnchorInfo, contentWidth, contentHeight)

        Log.d(TAG, "show: text='${text.take(20)}...', size=(${contentWidth}x$contentHeight), pos=(${position.first}, ${position.second})")

        if (isShowing && popupWindow?.isShowing == true) {
            // 更新位置和尺寸
            popup.update(position.first, position.second, contentWidth, contentHeight)
        } else {
            // 首次显示
            contentRoot.alpha = 0f
            popup.width = contentWidth
            popup.height = contentHeight
            popup.showAtLocation(anchor, Gravity.NO_GRAVITY, position.first, position.second)
            contentRoot.animate().alpha(1f).setDuration(140L).start()
            isShowing = true
        }
    }
    
    /**
     * 仅更新位置（当光标移动时调用）
     */
    fun updatePosition(cursorAnchorInfo: CursorAnchorInfo?) {
        if (!isShowing || popupWindow?.isShowing != true || containerView == null) {
            return
        }
        
        val root = containerView!!
        val textWidth = root.measuredWidth
        val textHeight = root.measuredHeight.coerceAtLeast(dpToPx(28f))
        
        val position = calculatePosition(cursorAnchorInfo, textWidth, textHeight)
        
        Log.d(TAG, "updatePosition: pos=(${position.first}, ${position.second})")
        
        popupWindow?.update(position.first, position.second, -1, -1)
    }
    
    /**
     * 计算 GhostText 应该显示的屏幕坐标
     * @return Pair<x, y> 屏幕坐标
     */
    private fun calculatePosition(
        cursorAnchorInfo: CursorAnchorInfo?,
        textWidth: Int,
        textHeight: Int
    ): Pair<Int, Int> {
        
        if (cursorAnchorInfo == null) {
            // 没有光标信息，显示在屏幕中央偏上
            Log.w(TAG, "calculatePosition: No cursor info, using fallback position")
            return Pair(dpToPx(16f), screenHeight / 3)
        }
        
        val markerHorizontal = cursorAnchorInfo.insertionMarkerHorizontal
        val markerTop = cursorAnchorInfo.insertionMarkerTop
        val markerBaseline = cursorAnchorInfo.insertionMarkerBaseline
        val markerBottom = cursorAnchorInfo.insertionMarkerBottom
        
        // 检查光标位置是否有效
        if (markerHorizontal.isNaN() || markerTop.isNaN()) {
            Log.w(TAG, "calculatePosition: Invalid cursor coordinates (NaN)")
            return Pair(dpToPx(16f), screenHeight / 3)
        }
        
        // 获取坐标变换矩阵（处理编辑器滚动、缩放等情况）
        val matrix = cursorAnchorInfo.matrix
        val point = floatArrayOf(markerHorizontal, markerTop)
        matrix?.mapPoints(point)
        
        var x = point[0].toInt()
        var y = point[1].toInt()
        
        // 光标高度
        val cursorHeight = if (!markerBottom.isNaN() && !markerTop.isNaN()) {
            (markerBottom - markerTop).toInt()
        } else {
            dpToPx(20f) // 默认光标高度
        }
        
        Log.d(TAG, "calculatePosition: cursorX=$x, cursorY=$y, cursorH=$cursorHeight, textW=$textWidth, textH=$textHeight")
        
        // GhostText 放在光标右侧，与光标顶部对齐
        x += dpToPx(2f) // 轻微偏移，避免遮挡光标

        // 处理右边界溢出：如果超出屏幕右边，换到下一行开头
        if (x + textWidth > screenWidth - dpToPx(8f)) {
            Log.d(TAG, "calculatePosition: Wrapping to next line")
            x = dpToPx(16f) // 下一行左侧对齐
            y += cursorHeight + dpToPx(4f) // 移到下一行
        }

        // 如果靠近底部，优先向上浮以避免覆盖候选栏或键盘
        val bottomSafeMargin = dpToPx(72f)
        if (y + textHeight > screenHeight - bottomSafeMargin) {
            y = (y - textHeight - dpToPx(6f)).coerceAtLeast(dpToPx(24f))
        }
        
        // 处理左边界
        if (x < dpToPx(4f)) {
            x = dpToPx(4f)
        }
        
        // 处理上下边界（确保不超出屏幕）
        if (y < dpToPx(24f)) {
            y = dpToPx(24f) // 避免遮挡状态栏
        }
        if (y + textHeight > screenHeight - dpToPx(48f)) {
            y = screenHeight - textHeight - dpToPx(48f)
        }
        
        return Pair(x, y)
    }
    
    /**
     * 隐藏并释放 GhostText
     */
    fun dismiss() {
        if (popupWindow?.isShowing == true) {
            try {
                popupWindow?.dismiss()
            } catch (e: Exception) {
                Log.e(TAG, "dismiss: Error dismissing popup", e)
            }
        }
        isShowing = false
        currentText = ""
    }
    
    /**
     * 检查是否正在显示
     */
    fun isShowing(): Boolean = isShowing && popupWindow?.isShowing == true
    
    /**
     * 获取当前显示的文本
     */
    fun getCurrentText(): String = if (isShowing) currentText else ""
    
    /**
     * 构造整体内容视图：锚点(圆点+虚线) + 气泡文本
     */
    private fun getOrCreateContentView(): LinearLayout {
        if (containerView == null) {
            val root = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                clipToPadding = false
                setPadding(dpToPx(4f), dpToPx(2f), dpToPx(4f), dpToPx(2f))
            }

            val dot = View(context).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(8f), dpToPx(8f))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(accentSecondary)
                }
                alpha = 0.9f
            }

            val guide = View(context).apply {
                layoutParams = LinearLayout.LayoutParams(dpToPx(16f), dpToPx(2f)).apply {
                    marginStart = dpToPx(6f)
                    marginEnd = dpToPx(6f)
                }
                background = createDashedLineDrawable()
                alpha = 0.55f
            }

            val tv = getOrCreateTextView()
            val tvLp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            tvLp.marginStart = dpToPx(2f)

            root.addView(dot)
            root.addView(guide)
            root.addView(tv, tvLp)

            containerView = root
            anchorDotView = dot
            guideLineView = guide
        }
        return containerView!!
    }

    /**
     * 获取或创建 TextView (气泡)
     */
    private fun getOrCreateTextView(): TextView {
        return textView ?: TextView(context).apply {
            setTextColor(accentSecondary)
            textSize = 16f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            maxLines = Integer.MAX_VALUE
            ellipsize = null
            setPadding(dpToPx(14f), dpToPx(10f), dpToPx(14f), dpToPx(10f))
            background = createRoundedWhiteBackground()
            elevation = dpToPx(6f).toFloat()
            setShadowLayer(6f, 0f, dpToPx(2f).toFloat(), Color.parseColor("#332196F3"))

            isClickable = true
            isFocusable = false
            setOnClickListener {
                onBubbleClick?.invoke(currentText)
            }
        }.also { textView = it }
    }

    /**
     * 创建圆角白色背景，附带浅描边
     */
    private fun createRoundedWhiteBackground(): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            setColor(Color.WHITE)
            cornerRadius = dpToPx(10f).toFloat()
            setStroke(dpToPx(1f), bubbleStroke)
        }
    }

    /**
     * 创建一条短虚线，用于光标-气泡的视觉连接
     */
    private fun createDashedLineDrawable(): ShapeDrawable {
        return ShapeDrawable().apply {
            intrinsicHeight = dpToPx(2f)
            paint.apply {
                color = accentSecondary
                style = Paint.Style.STROKE
                strokeWidth = dpToPx(2f).toFloat()
                pathEffect = DashPathEffect(floatArrayOf(dpToPx(4f).toFloat(), dpToPx(3f).toFloat()), 0f)
            }
        }
    }
    
    /**
     * 获取或创建 PopupWindow
     */
    private fun getOrCreatePopupWindow(contentView: View, width: Int, height: Int): PopupWindow {
        val popup = popupWindow ?: PopupWindow(context).apply {
            this.width = width
            this.height = height
            isFocusable = false // 不获取焦点，不影响输入
            isTouchable = true // 支持点击气泡上屏
            isOutsideTouchable = false
            elevation = dpToPx(4f).toFloat()

            // 设置为不影响输入法窗口的覆盖层
            inputMethodMode = PopupWindow.INPUT_METHOD_NOT_NEEDED
        }.also { popupWindow = it }

        popup.contentView = contentView
        return popup
    }
    
    /**
     * 释放资源
     */
    fun release() {
        dismiss()
        popupWindow = null
        textView = null
        containerView = null
        anchorDotView = null
        guideLineView = null
        anchorView = null
    }
    
    private fun dpToPx(dp: Float): Int {
        return (dp * context.resources.displayMetrics.density).toInt()
    }
}
