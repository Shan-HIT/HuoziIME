package com.yuyan.imemodule.utils

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.annotation.StringRes
import com.yuyan.imemodule.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

fun Context.toast(string: String, duration: Int = Toast.LENGTH_SHORT) {
    val ctx = applicationContext
    if (Looper.myLooper() == Looper.getMainLooper()) {
        Toast.makeText(ctx, string, duration).show()
        return
    }
    try {
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(ctx, string, duration).show()
        }
    } catch (_: Exception) {
    }
}

fun Context.toast(@StringRes resId: Int, duration: Int = Toast.LENGTH_SHORT) {
    val ctx = applicationContext
    if (Looper.myLooper() == Looper.getMainLooper()) {
        Toast.makeText(ctx, resId, duration).show()
        return
    }
    try {
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(ctx, resId, duration).show()
        }
    } catch (_: Exception) {
    }
}

fun Context.toast(t: Throwable, duration: Int = Toast.LENGTH_SHORT) {
    toast(t.localizedMessage ?: t.stackTraceToString(), duration)
}

suspend fun <T> Context.toast(result: Result<T>, duration: Int = Toast.LENGTH_SHORT) {
    withContext(Dispatchers.Main.immediate) {
        result
            .onSuccess { toast(R.string.done, duration) }
            .onFailure { toast(it, duration) }
    }
}
