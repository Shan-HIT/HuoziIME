#include "imem_perf.h"
#include <jni.h>
#include <string>

static jclass g_PerfTrackerClass = nullptr;
static jmethodID g_OnMetricCallback = nullptr;

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_initNative(
    JNIEnv* env, jclass clazz) {
    ALOG_METRIC("Native Performance Tracker initialized");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeCreateSession(
    JNIEnv* env, jclass clazz, jint type) {
    auto session_id = PerfTracker::generate_session_id();
    auto perf_type = static_cast<PerfMetricType>(type);

    PerfTracker::instance().create_session(session_id, perf_type);

    return env->NewStringUTF(session_id.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeRecordPrefill(
    JNIEnv* env, jclass clazz,
    jstring sessionId,
    jint promptLength, jint tokenCount, jint reuseCount,
    jlong timeMs, jdouble rate,
    jlong kvCacheBefore, jlong kvCacheAfter) {

    const char* session_id = env->GetStringUTFChars(sessionId, nullptr);
    auto session = PerfTracker::instance().get_session(session_id);

    if (session) {
        session->record_prefill(
            promptLength, tokenCount, reuseCount,
            timeMs, rate, kvCacheBefore, kvCacheAfter
        );
    }

    env->ReleaseStringUTFChars(sessionId, session_id);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeRecordDecodeStep(
    JNIEnv* env, jclass clazz,
    jstring sessionId,
    jint step, jstring token, jint tokenId,
    jlong timeMs, jlong cumulativeMs, jdouble tps,
    jboolean kvHit, jint branchCount) {

    const char* session_id = env->GetStringUTFChars(sessionId, nullptr);
    const char* token_str = env->GetStringUTFChars(token, nullptr);
    auto session = PerfTracker::instance().get_session(session_id);

    if (session) {
        session->record_decode_step(
            step, token_str, tokenId,
            timeMs, cumulativeMs, tps,
            kvHit, branchCount
        );
    }

    env->ReleaseStringUTFChars(sessionId, session_id);
    env->ReleaseStringUTFChars(token, token_str);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeCompleteSession(
    JNIEnv* env, jclass clazz,
    jstring sessionId,
    jstring mode, jstring prompt,
    jobjectArray candidates,
    jlong firstTokenLatency, jlong prefillMs, jlong decodeMs,
    jint totalTokens, jboolean success) {

    const char* session_id = env->GetStringUTFChars(sessionId, nullptr);
    const char* mode_str = env->GetStringUTFChars(mode, nullptr);
    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);

    std::vector<std::string> candidates_vec;
    if (candidates != nullptr) {
        jsize len = env->GetArrayLength(candidates);
        for (jsize i = 0; i < len; i++) {
            jstring elem = (jstring)env->GetObjectArrayElement(candidates, i);
            const char* elem_str = env->GetStringUTFChars(elem, nullptr);
            candidates_vec.push_back(elem_str);
            env->ReleaseStringUTFChars(elem, elem_str);
            env->DeleteLocalRef(elem);
        }
    }

    auto session = PerfTracker::instance().get_session(session_id);
    if (session) {
        session->complete_session(
            mode_str, prompt_str, candidates_vec,
            firstTokenLatency, prefillMs, decodeMs, totalTokens, success
        );
        PerfTracker::instance().remove_session(session_id);
    }

    env->ReleaseStringUTFChars(sessionId, session_id);
    env->ReleaseStringUTFChars(mode, mode_str);
    env->ReleaseStringUTFChars(prompt, prompt_str);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeRecordContextSync(
    JNIEnv* env, jclass clazz,
    jstring sessionId,
    jstring syncType, jint historyLen, jint lastMsgLen,
    jlong prefillMs, jlong sessionLoadMs,
    jboolean success) {

    const char* session_id = env->GetStringUTFChars(sessionId, nullptr);
    const char* sync_type = env->GetStringUTFChars(syncType, nullptr);

    auto session = PerfTracker::instance().get_session(session_id);
    if (session) {
        session->record_context_sync(
            sync_type, historyLen, lastMsgLen,
            prefillMs, sessionLoadMs, success
        );
    }

    env->ReleaseStringUTFChars(sessionId, session_id);
    env->ReleaseStringUTFChars(syncType, sync_type);
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeExportJson(
    JNIEnv* env, jclass clazz) {
    std::string json = PerfTracker::instance().export_all_json();
    return env->NewStringUTF(json.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeClearAll(
    JNIEnv* env, jclass clazz) {
    PerfTracker::instance().clear_all();
    ALOG_METRIC("All performance sessions cleared");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeGetSessionCount(
    JNIEnv* env, jclass clazz) {
    return static_cast<jint>(PerfTracker::instance().session_count());
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeGetTimestampNs(
    JNIEnv* env, jclass clazz) {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    return static_cast<jlong>(ns);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeRecordCustomMetric(
    JNIEnv* env, jclass clazz,
    jstring sessionId,
    jstring metricName, jstring metricValue) {

    const char* session_id = env->GetStringUTFChars(sessionId, nullptr);
    const char* metric_name = env->GetStringUTFChars(metricName, nullptr);
    const char* metric_value = env->GetStringUTFChars(metricValue, nullptr);

    ALOG_METRIC("Session %s | Custom | %s = %s", session_id, metric_name, metric_value);

    env->ReleaseStringUTFChars(sessionId, session_id);
    env->ReleaseStringUTFChars(metricName, metric_name);
    env->ReleaseStringUTFChars(metricValue, metric_value);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeStartTimer(
    JNIEnv* env, jclass clazz) {
    auto timer = new PerfTimer();
    return reinterpret_cast<jlong>(timer);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeStopTimer(
    JNIEnv* env, jclass clazz, jlong timerPtr) {
    if (timerPtr == 0) return 0;
    PerfTimer* timer = reinterpret_cast<PerfTimer*>(timerPtr);
    jlong elapsed = timer->elapsed_ms();
    delete timer;
    return elapsed;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_performance_NativePerformanceTracker_nativeGetElapsedMs(
    JNIEnv* env, jclass clazz, jlong timerPtr) {
    if (timerPtr == 0) return 0;
    PerfTimer* timer = reinterpret_cast<PerfTimer*>(timerPtr);
    return timer->elapsed_ms();
}
