package com.yuyan.imemodule.prefs.behavior
import com.yuyan.imemodule.view.preference.ManagedPreference
enum class SkbMenuMode {
    SwitchKeyboard,
    KeyboardHeight,
    DarkTheme,
    Feedback,
    NumberRow,
    JianFan,
    LockEnglish,
    SymbolShow,
    CandidatesMore,
    Mnemonic,
    FlowerTypeface,
    EmojiInput,
    Handwriting,
    Custom,
    SettingsMenu,
    Settings,
    FloatKeyboard,
    OneHanded,
    PinyinT9,
    Pinyin26Jian,
    PinyinLx17,
    PinyinHandWriting,
    Pinyin26Double,
    PinyinStroke,
    ClipBoard,
    ClearClipBoard,
    Phrases,
    AddPhrases,
    CloseSKB,
    Emojicon,
    Emoticon,
    LockClipBoard,
    TextEdit,
    LLMControl,
    StyleSwitch, 
    ModeSwitch,  
    MemoryPanel,
    AutomatedTest,
    LoRAOverheadTest,
    LoRASwitchLatencyTest,
    TopKGenerationCostTest,
    ParallelDecodeEfficiencyTest;
    companion object : ManagedPreference.StringLikeCodec<SkbMenuMode> {
        override fun decode(raw: String): SkbMenuMode {
            return try {
                SkbMenuMode.valueOf(raw)
            } catch (_: IllegalArgumentException) {
                Settings
            }
        }
    }
}