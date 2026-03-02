package com.yuyan.imemodule.database
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.yuyan.imemodule.application.Launcher
import com.yuyan.imemodule.database.dao.ClipboardDao
import com.yuyan.imemodule.database.dao.PhraseDao
import com.yuyan.imemodule.database.dao.SideSymbolDao
import com.yuyan.imemodule.database.dao.SkbFunDao
import com.yuyan.imemodule.database.dao.UsedSymbolDao
import com.yuyan.imemodule.database.entry.Clipboard
import com.yuyan.imemodule.database.entry.Phrase
import com.yuyan.imemodule.database.entry.SideSymbol
import com.yuyan.imemodule.database.entry.SkbFun
import com.yuyan.imemodule.database.entry.UsedSymbol
import com.yuyan.imemodule.prefs.behavior.SkbMenuMode
import com.yuyan.imemodule.utils.thread.ThreadPoolUtils
// 更新数据库版本号为 10
@Database(entities = [SideSymbol::class, Clipboard::class, UsedSymbol::class, Phrase::class, SkbFun::class], version = 10, exportSchema = false)
abstract class DataBaseKT : RoomDatabase() {
    abstract fun sideSymbolDao(): SideSymbolDao
    abstract fun clipboardDao(): ClipboardDao
    abstract fun usedSymbolDao(): UsedSymbolDao
    abstract fun phraseDao(): PhraseDao
    abstract fun skbFunDao(): SkbFunDao
    companion object {
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("CREATE TABLE IF NOT EXISTS phrase (content TEXT PRIMARY KEY NOT NULL, isKeep INTEGER NOT NULL, t9 TEXT NOT NULL, qwerty TEXT NOT NULL, lx17 TEXT NOT NULL, time INTEGER NOT NULL)")
            }
        }
        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("CREATE TABLE IF NOT EXISTS skbfun (name TEXT KEY NOT NULL, isKeep INTEGER NOT NULL, position INTEGER NOT NULL, PRIMARY KEY (name, isKeep))")
            }
        }
        private val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("INSERT INTO skbfun (name, isKeep, position) VALUES ('TextEdit', 0, 15)")
                db.execSQL("INSERT INTO skbfun (name, isKeep, position) VALUES ('TextEdit', 1, 0)")
            }
        }
        private val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('LLMControl', 0, 16)")
            }
        }
       private val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("UPDATE skbfun SET position = 16 WHERE name = 'FloatKeyboard'")
                db.execSQL("UPDATE skbfun SET position = 11 WHERE name = 'LLMControl'")
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('LLMControl', 0, 11)")
            }
        }
        private val MIGRATION_6_7 = object : Migration(6, 7) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('StyleSwitch', 0, 17)")
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('ModeSwitch', 0, 18)")
            }
        }
        private val MIGRATION_7_8 = object : Migration(7, 8) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('AutomatedTest', 0, 22)")
            }
        }
        private val MIGRATION_8_9 = object : Migration(8, 9) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('LoRAOverheadTest', 0, 27)")
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('LoRASwitchLatencyTest', 0, 28)")
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('TopKGenerationCostTest', 0, 30)")
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('ParallelDecodeEfficiencyTest', 0, 31)")
            }
        }

        private val MIGRATION_9_10 = object : Migration(9, 10) {
            override fun migrate(db: SupportSQLiteDatabase) {
                // 移除不再需要的性能/测试入口
                db.execSQL(
                    "DELETE FROM skbfun WHERE name IN (" +
                        "'AutomatedTest'," +
                        "'LoRAOverheadTest'," +
                        "'LoRASwitchLatencyTest'," +
                        "'TopKGenerationCostTest'," +
                        "'ParallelDecodeEfficiencyTest'" +
                    ")"
                )

                // 新增“记忆”入口（与风格切换同级）
                db.execSQL("INSERT OR IGNORE INTO skbfun (name, isKeep, position) VALUES ('MemoryPanel', 0, 19)")
            }
        }

        val instance = Room.databaseBuilder(Launcher.instance.context, DataBaseKT::class.java, "ime_db")
            .allowMainThreadQueries()
            .addMigrations(MIGRATION_1_2)
            .addMigrations(MIGRATION_2_3)
            .addMigrations(MIGRATION_3_4)
            .addMigrations(MIGRATION_4_5)
            .addMigrations(MIGRATION_5_6)
            .addMigrations(MIGRATION_6_7)
            .addMigrations(MIGRATION_7_8)
            .addMigrations(MIGRATION_8_9)
            .addMigrations(MIGRATION_9_10)
            .addCallback(object :Callback(){
                override fun onCreate(db: SupportSQLiteDatabase) {
                    super.onCreate(db)
                    ThreadPoolUtils.executeSingleton {
                        initDb()
                    }
                }
                override fun onOpen(db: SupportSQLiteDatabase) {
                    super.onOpen(db)
                    ThreadPoolUtils.executeSingleton {
                        initPhrasesDb()
                    }
                }
            })
            .build()
        private fun initDb() {  //初始化数据库数据
            val symbolPinyin = listOf("，", "。", "？", "！", "……", "：", "；", ".").map {  symbolKey->
                SideSymbol(symbolKey, symbolKey)
            }
            instance.sideSymbolDao().insertAll(symbolPinyin)
            val symbolNumber = listOf("%", "/", "-", "+", "*", "#", "@").map {  symbolKey->
                SideSymbol(symbolKey, symbolKey, "number")
            }
            instance.sideSymbolDao().insertAll(symbolNumber)
        }
        private fun initPhrasesDb() {  //初始化常用语数据数据
            if(instance.phraseDao().getAll().isEmpty()) {
                val phrases = listOf(
                    Phrase(content = "我的电话是__，常联系。", t9 = "9334", qwerty = "wddh", lx17 = "wddh"),
                    Phrase(content = "抱歉，我现在不方便接电话，稍后联系。", t9 = "2799", qwerty = "bqwx", lx17 = "bqwx"),
                    Phrase(content = "我正在开会，有急事请发短信。", t9 = "9995", qwerty = "wzzk", lx17 = "wwwj"),
                    Phrase(content = "我很快就到，请稍微等一会儿。", t9 = "9455", qwerty = "whkj", lx17 = "whjj"),
                    Phrase(content = "麻烦放驿站，谢谢。", t9 = "6339", qwerty = "mffy", lx17 = "mffy"),
                )
                instance.phraseDao().insertAll(phrases)
            }
            if(instance.skbFunDao().getAllMenu().isEmpty()) {
                val skbFuns = listOf(
                    SkbFun(name = SkbMenuMode.ClipBoard.name, isKeep = 1),
                    SkbFun(name = SkbMenuMode.Emojicon.name, isKeep = 1),
                    SkbFun(name = SkbMenuMode.TextEdit.name, isKeep = 1),
                    SkbFun(name = SkbMenuMode.Emojicon.name, isKeep = 0, position = 0),
                    SkbFun(name = SkbMenuMode.SwitchKeyboard.name, isKeep = 0, position = 1),
                    SkbFun(name = SkbMenuMode.KeyboardHeight.name, isKeep = 0, position = 2),
                    SkbFun(name = SkbMenuMode.ClipBoard.name, isKeep = 0, position = 3),
                    SkbFun(name = SkbMenuMode.Phrases.name, isKeep = 0, position = 4),
                    SkbFun(name = SkbMenuMode.DarkTheme.name, isKeep = 0, position = 5),
                    SkbFun(name = SkbMenuMode.Feedback.name, isKeep = 0, position = 6),
                    SkbFun(name = SkbMenuMode.OneHanded.name, isKeep = 0, position = 7),
                    SkbFun(name = SkbMenuMode.NumberRow.name, isKeep = 0, position = 8),
                    SkbFun(name = SkbMenuMode.JianFan.name, isKeep = 0, position = 9),
                    SkbFun(name = SkbMenuMode.Mnemonic.name, isKeep = 0, position = 10),
                    SkbFun(name = SkbMenuMode.LLMControl.name, isKeep = 0, position = 11),
                    SkbFun(name = SkbMenuMode.FlowerTypeface.name, isKeep = 0, position = 12),
                    SkbFun(name = SkbMenuMode.Custom.name, isKeep = 0, position = 13),
                    SkbFun(name = SkbMenuMode.Settings.name, isKeep = 0, position = 14),
                    SkbFun(name = SkbMenuMode.TextEdit.name, isKeep = 0, position = 15),
                    SkbFun(name = SkbMenuMode.FloatKeyboard.name, isKeep = 0, position = 16),
                    SkbFun(name = SkbMenuMode.StyleSwitch.name, isKeep = 0, position = 17),
                    SkbFun(name = SkbMenuMode.ModeSwitch.name, isKeep = 0, position = 18),
                    SkbFun(name = SkbMenuMode.MemoryPanel.name, isKeep = 0, position = 19),
                )
                instance.skbFunDao().insertAll(skbFuns)
            }
        }
    }
}