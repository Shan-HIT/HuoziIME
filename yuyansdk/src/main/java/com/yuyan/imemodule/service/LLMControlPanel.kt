package com.yuyan.imemodule.service

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import com.yuyan.imemodule.service.data.StyleConfig
import com.yuyan.imemodule.utils.LogUtil

@SuppressLint("ViewConstructor")
class LLMControlPanel(
    context: Context,
    private val service: ImeService,
    private val targetMode: Int = MODE_STYLE,
) : LinearLayout(context) {

    companion object {
        const val MODE_STYLE = 2
        const val MODE_GEN = 3
        const val MODE_MEMORY = 4
    }

    private val TAG = "LLMControlPanel"

    init {
        LogUtil.i(TAG, "init", "初始化面板, 模式: $targetMode")

        orientation = VERTICAL
        setBackgroundColor(Color.parseColor("#F5F5F5"))
        isClickable = true
        isFocusable = true
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
        )

        val headerBar = LinearLayout(context).apply {
            orientation = HORIZONTAL
            setPadding(30, 20, 30, 20)
            setBackgroundColor(Color.WHITE)
            elevation = 5f
        }

        val titleText = TextView(context).apply {
            text = getTitleForMode(targetMode)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.BLACK)
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
            gravity = Gravity.CENTER_VERTICAL
        }

        val memoryBtn = Button(context).apply {
            text = "记忆"
            textSize = 14f
            setTextColor(Color.parseColor("#555555"))
            setBackgroundColor(Color.TRANSPARENT)
            setOnClickListener {
                if (targetMode != MODE_MEMORY) {
                    service.openMemoryPanel()
                }
            }
        }

        val closeBtn = Button(context).apply {
            text = "返回"
            textSize = 14f
            setTextColor(Color.parseColor("#555555"))
            setBackgroundColor(Color.TRANSPARENT)
            setOnClickListener {
                LogUtil.i(TAG, "Click", "点击了关闭按钮")
                service.closeControlPanel()
            }
        }

        headerBar.addView(titleText)
        headerBar.addView(memoryBtn)
        headerBar.addView(closeBtn)
        addView(headerBar)

        val contentArea = ScrollView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f)
            isFillViewport = true
        }

        val contentLayout = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(40, 40, 40, 40)
        }

        contentArea.addView(contentLayout)
        addView(contentArea)

        when (targetMode) {
            MODE_STYLE -> setupStyleSwitchView(contentLayout)
            MODE_GEN -> setupGenerationModeView(contentLayout)
            MODE_MEMORY -> setupMemoryManageView(contentLayout)
            else -> setupStyleSwitchView(contentLayout)
        }
    }

    override fun dispatchTouchEvent(ev: android.view.MotionEvent?): Boolean {
        if (ev?.action == android.view.MotionEvent.ACTION_DOWN) {
            LogUtil.d(TAG, "Touch", "Panel dispatchTouchEvent: ACTION_DOWN at (${ev.x}, ${ev.y})")
        }
        return super.dispatchTouchEvent(ev)
    }

    private fun getTitleForMode(mode: Int): String {
        return when (mode) {
            MODE_STYLE -> "人格风格切换"
            MODE_GEN -> "生成模式设置"
            MODE_MEMORY -> "记忆管理"
            else -> "LLM 面板"
        }
    }

    private fun setupMemoryManageView(container: LinearLayout) {
        val dp = { value: Int -> (value * resources.displayMetrics.density).toInt() }
        val titleColor = Color.parseColor("#1F2937")
        val bodyColor = Color.parseColor("#4B5563")
        val accentColor = Color.parseColor("#2563EB")
        val cardBackground = Color.parseColor("#FFFFFF")

        fun createCard(bgColor: Int = cardBackground, strokeColor: Int = Color.parseColor("#E5E7EB")): GradientDrawable {
            return GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dp(14).toFloat()
                setColor(bgColor)
                setStroke(dp(1), strokeColor)
            }
        }

        val introCard = LinearLayout(context).apply {
            orientation = VERTICAL
            background = GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                intArrayOf(Color.parseColor("#EAF2FF"), Color.parseColor("#F8FBFF")),
            ).apply {
                cornerRadius = dp(16).toFloat()
            }
            setPadding(dp(18), dp(16), dp(18), dp(16))
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(0, 0, 0, dp(14))
            }
        }
        introCard.addView(TextView(context).apply {
            text = "记忆总览"
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(titleColor)
        })
        introCard.addView(TextView(context).apply {
            text = "这里仅用于查看已记录内容，不提供 KV 拼接或记忆测试功能。"
            textSize = 13f
            setTextColor(bodyColor)
            setPadding(0, dp(6), 0, 0)
        })
        container.addView(introCard)

        val statsRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(0, 0, 0, dp(12))
            }
        }
        container.addView(statsRow)

        fun createStatCard(title: String): Pair<LinearLayout, TextView> {
            val valueView = TextView(context).apply {
                text = "-"
                textSize = 16f
                typeface = Typeface.DEFAULT_BOLD
                setTextColor(titleColor)
            }
            val card = LinearLayout(context).apply {
                orientation = VERTICAL
                background = createCard()
                setPadding(dp(12), dp(12), dp(12), dp(12))
                layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f).apply {
                    setMargins(dp(4), 0, dp(4), 0)
                }
                addView(TextView(context).apply {
                    text = title
                    textSize = 12f
                    setTextColor(bodyColor)
                })
                addView(valueView)
            }
            return card to valueView
        }

        val (totalCard, totalValue) = createStatCard("总记忆数")
        val (sourceCard, sourceValue) = createStatCard("最新来源")
        val (timeCard, timeValue) = createStatCard("最近时间")
        statsRow.addView(totalCard)
        statsRow.addView(sourceCard)
        statsRow.addView(timeCard)

        val actionRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(0, 0, 0, dp(10))
            }
        }
        container.addView(actionRow)

        actionRow.addView(TextView(context).apply {
            text = "已保存记忆"
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(titleColor)
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
        })

        val refreshBtn = Button(context).apply {
            text = "刷新列表"
            textSize = 13f
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dp(18).toFloat()
                setColor(accentColor)
            }
            setPadding(dp(16), dp(8), dp(16), dp(8))
        }
        actionRow.addView(refreshBtn)

        val listLayout = LinearLayout(context).apply {
            orientation = VERTICAL
        }
        container.addView(listLayout)

        val sdf = java.text.SimpleDateFormat("MM-dd HH:mm:ss", java.util.Locale.getDefault())

        fun splitDetailSegments(detail: String): List<String> {
            return detail.replace("｜", "|")
                .split("|")
                .map { it.trim() }
                .filter { it.isNotEmpty() }
        }

        fun splitKeyValue(segment: String): Pair<String, String>? {
            val idxCn = segment.indexOf('：')
            val idxEn = segment.indexOf(':')
            val splitIdx = listOf(idxCn, idxEn).filter { it > 0 }.minOrNull() ?: return null
            val key = segment.substring(0, splitIdx).trim()
            val value = segment.substring(splitIdx + 1).trim()
            if (key.isEmpty() || value.isEmpty()) return null
            return key to value
        }

        fun renderList() {
            val items = service.listMemories(200)
            totalValue.text = items.size.toString()
            sourceValue.text = items.firstOrNull()?.source?.ifBlank { "未知" } ?: "暂无"
            timeValue.text = items.firstOrNull()?.timestamp
                ?.takeIf { it > 0 }
                ?.let { sdf.format(java.util.Date(it)) }
                ?: "暂无"

            listLayout.removeAllViews()
            if (items.isEmpty()) {
                listLayout.addView(LinearLayout(context).apply {
                    orientation = VERTICAL
                    gravity = Gravity.CENTER
                    background = createCard(bgColor = Color.parseColor("#F9FAFB"), strokeColor = Color.parseColor("#E5E7EB"))
                    setPadding(dp(20), dp(24), dp(20), dp(24))
                    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                        setMargins(0, dp(4), 0, 0)
                    }
                    addView(TextView(context).apply {
                        text = "暂无可展示的记忆"
                        textSize = 14f
                        typeface = Typeface.DEFAULT_BOLD
                        setTextColor(titleColor)
                        gravity = Gravity.CENTER
                    })
                    addView(TextView(context).apply {
                        text = "当用户交互被抽取后，记忆会自动出现在这里。"
                        textSize = 12f
                        setTextColor(bodyColor)
                        gravity = Gravity.CENTER
                        setPadding(0, dp(6), 0, 0)
                    })
                })
                return
            }

            items.forEachIndexed { index, rec ->
                val card = LinearLayout(context).apply {
                    orientation = VERTICAL
                    background = createCard()
                    setPadding(dp(14), dp(12), dp(14), dp(12))
                    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                        setMargins(0, 0, 0, dp(10))
                    }
                }

                val header = LinearLayout(context).apply {
                    orientation = HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                }
                card.addView(header)

                header.addView(TextView(context).apply {
                    text = "#${index + 1}  ${if (rec.timestamp > 0) sdf.format(java.util.Date(rec.timestamp)) else "时间未知"}"
                    textSize = 12f
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(titleColor)
                    layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
                })

                header.addView(TextView(context).apply {
                    text = rec.source.ifBlank { "unknown" }
                    textSize = 11f
                    setTextColor(accentColor)
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.RECTANGLE
                        cornerRadius = dp(10).toFloat()
                        setColor(Color.parseColor("#E6F0FF"))
                    }
                    setPadding(dp(10), dp(4), dp(10), dp(4))
                })

                val whoWhat = listOf(rec.who, rec.what)
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .joinToString(" · ")
                if (whoWhat.isNotEmpty()) {
                    card.addView(TextView(context).apply {
                        text = whoWhat
                        textSize = 12f
                        setTextColor(bodyColor)
                        setPadding(0, dp(8), 0, 0)
                    })
                }

                val detailSegments = splitDetailSegments(rec.detail)
                card.addView(TextView(context).apply {
                    text = when {
                        detailSegments.isNotEmpty() -> detailSegments.first()
                        rec.detail.isNotBlank() -> rec.detail.trim()
                        else -> "（无详情）"
                    }
                    textSize = 14f
                    setTextColor(Color.parseColor("#111827"))
                    setPadding(0, dp(8), 0, 0)
                })

                val metadataSegments = detailSegments.drop(1)
                if (metadataSegments.isNotEmpty()) {
                    val metaContainer = LinearLayout(context).apply {
                        orientation = VERTICAL
                        setPadding(0, dp(8), 0, 0)
                    }
                    metadataSegments.forEach { segment ->
                        val metaRow = LinearLayout(context).apply {
                            orientation = HORIZONTAL
                            gravity = Gravity.CENTER_VERTICAL
                            background = GradientDrawable().apply {
                                shape = GradientDrawable.RECTANGLE
                                cornerRadius = dp(10).toFloat()
                                setColor(Color.parseColor("#F8FAFC"))
                            }
                            setPadding(dp(10), dp(7), dp(10), dp(7))
                            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                                setMargins(0, 0, 0, dp(6))
                            }
                        }

                        val kv = splitKeyValue(segment)
                        if (kv != null) {
                            val (key, value) = kv
                            metaRow.addView(TextView(context).apply {
                                text = key
                                textSize = 11f
                                setTextColor(Color.parseColor("#1D4ED8"))
                                background = GradientDrawable().apply {
                                    shape = GradientDrawable.RECTANGLE
                                    cornerRadius = dp(8).toFloat()
                                    setColor(Color.parseColor("#DBEAFE"))
                                }
                                setPadding(dp(8), dp(3), dp(8), dp(3))
                            })
                            metaRow.addView(TextView(context).apply {
                                text = value
                                textSize = 12f
                                setTextColor(Color.parseColor("#374151"))
                                layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f).apply {
                                    setMargins(dp(8), 0, 0, 0)
                                }
                            })
                        } else {
                            metaRow.addView(TextView(context).apply {
                                text = segment
                                textSize = 12f
                                setTextColor(Color.parseColor("#374151"))
                            })
                        }

                        metaContainer.addView(metaRow)
                    }
                    card.addView(metaContainer)
                }

                rec.vectorLabel?.let { label ->
                    card.addView(TextView(context).apply {
                        text = "label: $label"
                        textSize = 11f
                        setTextColor(Color.parseColor("#6B7280"))
                        setPadding(0, dp(8), 0, 0)
                    })
                }

                listLayout.addView(card)
            }
        }

        refreshBtn.setOnClickListener { renderList() }
        renderList()
    }

    private fun setupStyleSwitchView(container: LinearLayout) {
        val currentStyle = service.getCurrentStyle()
        val styles = StyleConfig.PROMPTS.keys.toList()
        val displayNames = mapOf(
            StyleConfig.STYLE_BUSINESS to "职场精英",
            StyleConfig.STYLE_WARM to "贴心陪伴",
            StyleConfig.STYLE_INRERNET to "社交达人",
        )

        val desc = TextView(context).apply {
            text = "请选择一个风格模型（这将触发模型热切换）："
            textSize = 14f
            setTextColor(Color.GRAY)
            setPadding(0, 0, 0, 30)
        }
        container.addView(desc)

        val radioGroup = RadioGroup(context).apply {
            orientation = RadioGroup.VERTICAL
        }

        styles.forEachIndexed { index, styleKey ->
            val rb = RadioButton(context).apply {
                text = displayNames[styleKey] ?: styleKey
                textSize = 16f
                setPadding(20, 20, 20, 20)
                tag = styleKey
                id = index
            }
            radioGroup.addView(rb)

            if (styleKey == currentStyle) {
                LogUtil.d(TAG, "StyleUI", "初始化选中风格: $styleKey (ID: $index)")
                radioGroup.check(index)
            }
        }
        container.addView(radioGroup)

        val confirmBtn = Button(context).apply {
            text = "确认切换"
            setBackgroundColor(Color.parseColor("#00796B"))
            setTextColor(Color.WHITE)
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(0, 50, 0, 0)
            }
            setOnClickListener {
                val selectedId = radioGroup.checkedRadioButtonId
                if (selectedId != -1) {
                    val selectedView = radioGroup.findViewById<View>(selectedId)
                    val newStyle = selectedView.tag as String
                    LogUtil.i(TAG, "Click", "点击确认切换风格，目标: $newStyle")
                    if (newStyle != currentStyle) {
                        service.executeStyleSwitch(newStyle)
                    } else {
                        Toast.makeText(context, "当前已经是该风格", Toast.LENGTH_SHORT).show()
                    }
                    service.closeControlPanel()
                } else {
                    LogUtil.w(TAG, "Click", "未选择任何风格")
                }
            }
        }
        container.addView(confirmBtn)
    }

    private fun setupGenerationModeView(container: LinearLayout) {
        val currentModeOrd = service.getGenerationModeOrdinal()

        val desc = TextView(context).apply {
            text = "选择文本生成策略（影响AI补全的长度和逻辑）："
            textSize = 14f
            setTextColor(Color.GRAY)
            setPadding(0, 0, 0, 30)
        }
        container.addView(desc)

        val radioGroup = RadioGroup(context).apply {
            orientation = RadioGroup.VERTICAL
        }

        val modes = listOf(
            0 to "单词补全模式 (Token)\n适合快速单词预测，响应最快",
            1 to "短语联想模式 (Phrase)\n适合短语和半句补全，平衡性好",
            2 to "整句生成模式 (Sentence)\n适合长句和逻辑补全，耗时较长",
        )

        modes.forEach { (ord, label) ->
            val rb = RadioButton(context).apply {
                text = label
                textSize = 16f
                setPadding(20, 20, 20, 20)
                id = ord
            }
            radioGroup.addView(rb)

            if (ord == currentModeOrd) {
                LogUtil.d(TAG, "GenModeUI", "初始化选中模式序号: $ord")
                radioGroup.check(ord)
            }
        }
        container.addView(radioGroup)

        val confirmBtn = Button(context).apply {
            text = "保存设置"
            setBackgroundColor(Color.parseColor("#1976D2"))
            setTextColor(Color.WHITE)
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(0, 50, 0, 0)
            }
            setOnClickListener {
                val selectedId = radioGroup.checkedRadioButtonId
                LogUtil.i(TAG, "Click", "点击保存生成模式，选中ID: $selectedId")
                if (selectedId != -1) {
                    if (selectedId != currentModeOrd) {
                        service.setGenerationModeOrdinal(selectedId)
                    } else {
                        Toast.makeText(context, "模式未变更", Toast.LENGTH_SHORT).show()
                    }
                    service.closeControlPanel()
                }
            }
        }
        container.addView(confirmBtn)
    }
}

