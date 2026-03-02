#include "imem_core.h"

#include "src/llama-memory.h"

#include <cstdlib>
#include <cstring>

// ===== Control-token helpers (MEM toolcall + NO_MEM) =====
static void ensure_control_tokens(ModelHandle * h, const llama_vocab * vocab) {
    if (!h || !vocab) return;

    if (h->tok_mem_open >= 0 && h->tok_mem_close >= 0 && h->tok_no_mem >= 0) return;

    std::lock_guard<std::mutex> lock(h->mtx);
    if (h->tok_mem_open >= 0 && h->tok_mem_close >= 0 && h->tok_no_mem >= 0) return;

    auto one_tok = [&](const char * s) -> llama_token {
        if (!s) return (llama_token)-1;
        std::vector<llama_token> tmp(8);
        const int n = llama_tokenize(vocab, s, (int32_t) std::strlen(s), tmp.data(), (int) tmp.size(), false, true);
        if (n == 1) return tmp[0];
        return (llama_token)-1;
    };

    if (h->tok_mem_open < 0)  h->tok_mem_open  = one_tok("<MEM_RETRIEVAL>");
    if (h->tok_mem_close < 0) h->tok_mem_close = one_tok("</MEM_RETRIEVAL>");
    if (h->tok_no_mem < 0)    h->tok_no_mem    = one_tok("<NO_MEM>");
}

static inline bool is_control_token(ModelHandle * h, llama_token t) {
    if (!h) return false;
    return (t == h->tok_mem_open) || (t == h->tok_mem_close) || (t == h->tok_no_mem);
}

static inline int token_to_piece_with_control(const llama_vocab * vocab, ModelHandle * h, llama_token t, char * buf, int buf_size) {
    const bool special = is_control_token(h, t);
    return llama_token_to_piece(vocab, t, buf, buf_size, 0, special);
}

static bool kv_splice_debug_enabled() {
    const char * v1 = std::getenv("LLAMA_KV_CACHE_DEBUG");
    if (v1 && v1[0] != '\0' && !(v1[0] == '0' && v1[1] == '\0')) return true;
    const char * v2 = std::getenv("YUYAN_KV_SPLICE_DEBUG");
    if (v2 && v2[0] != '\0' && !(v2[0] == '0' && v2[1] == '\0')) return true;
    return false;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static long long file_size_bytes(const std::string & path) {
    struct stat st{};
    if (path.empty()) return -1;
    if (stat(path.c_str(), &st) != 0) return -1;
    return (long long) st.st_size;
}

static std::string llama_meta_str(const llama_model * model, const char * key) {
    if (!model || !key) return "";
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n <= 0) return "";
    return std::string(buf);
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_tokenize(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jText) {
    ALOGI("【LogicGen】Tokenize 调用");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h || !h->model) return nullptr;
    const char* ctext = env->GetStringUTFChars(jText, nullptr);
    const int MAX_TOKS = 8192;
    std::vector<llama_token> tokens(MAX_TOKS);
    const auto* vocab = llama_model_get_vocab(h->model);
    int n = llama_tokenize(vocab, ctext, (int32_t)strlen(ctext), tokens.data(), MAX_TOKS, false, true);
    env->ReleaseStringUTFChars(jText, ctext);
    if (n < 0) return env->NewIntArray(0);
    jintArray out = env->NewIntArray(n);
    std::vector<jint> tmp(n);
    for (int i = 0; i < n; ++i) tmp[i] = (jint)tokens[i];
    env->SetIntArrayRegion(out, 0, n, tmp.data());
    return out;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_prefillPrompt(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jPrompt, jobject jCallback) {
    ALOGI("【LogicGen】prefillPrompt 调用 (实验：仅预填充，不生成候选)");
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h) return -1;
    forceStopAndJoin(h, "prefillPrompt");
    jobject gcb = env->NewGlobalRef(jCallback);
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID midOnTokenCandidates = env->GetMethodID(cbClass, "onTokenCandidates", "([Ljava/lang/String;)V");
    jmethodID midOnFinished = env->GetMethodID(cbClass, "onFinished", "()V");
    jmethodID midOnError = env->GetMethodID(cbClass, "onError", "(Ljava/lang/String;)V");
    const char * cprompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(cprompt ? cprompt : "");
    if (cprompt) env->ReleaseStringUTFChars(jPrompt, cprompt);
    auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        h->stop_flag.store(false);
        h->gen_thread = std::thread([h, gcb, midOnTokenCandidates, midOnFinished, midOnError, prompt, start_time, local_seq]() {
            auto cancelled = [&]() {
                return h->stop_flag.load(std::memory_order_relaxed) ||
                       h->generation_seq.load(std::memory_order_relaxed) != local_seq;
            };
            struct llama_context * ctx = nullptr;
            struct llama_model * model = nullptr;
            long long prefill_time_ms = 0;
            int prefill_tokens_count = 0;
            bool batch_initialized = false;
            llama_batch batch{};

            JNIEnv * threadEnv = nullptr;
            bool attached = false;
            if (gJvm->GetEnv((void **)&threadEnv, JNI_VERSION_1_6) != JNI_OK) {
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) attached = true;
            }
            if (!threadEnv) {
                ALOGE("prefillPrompt | ❌ 无法获取线程 Env");
                goto finish;
            }

            {
                std::lock_guard<std::mutex> lock(h->mtx);
                ctx = h->ctx;
                model = h->model;
            }
            if (!ctx || !model || cancelled()) {
                const char * msg = "prefillPrompt | ❌ Context 或 Model 为空";
                ALOGE("%s", msg);
                if (midOnError) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
                goto finish;
            }
            {
                const auto * vocab = llama_model_get_vocab(model);
                llama_memory_t kv_mem = llama_get_memory(ctx);

                std::vector<llama_token> tokens(std::max<size_t>(256, prompt.size() + 128));
                int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.length(), tokens.data(), (int)tokens.size(), false, true);
                if (n_tokens < 0) {
                    tokens.resize((size_t)(-n_tokens));
                    n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.length(), tokens.data(), (int)tokens.size(), false, true);
                }
                if (n_tokens <= 0) {
                    const char * msg = "prefillPrompt | ❌ tokenize 失败";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }
                tokens.resize(n_tokens);
                {
                    int n_ctx = (int) h->cparams.n_ctx;
                    if (n_ctx <= 0) n_ctx = 0;
                    int n_seq_max = (int) h->cparams.n_seq_max;
                    if (n_seq_max <= 0) n_seq_max = 1;
                    const int n_ctx_per_seq = (n_ctx > 0) ? std::max(0, n_ctx / n_seq_max) : 0;
                    if (n_ctx_per_seq > 0 && (int) tokens.size() > n_ctx_per_seq) {
                        std::stringstream ss;
                        ss << "prefillPrompt | ❌ prompt 超出 n_ctx_per_seq";
                        ss << " | tokens_total=" << tokens.size();
                        ss << " n_ctx=" << n_ctx;
                        ss << " n_seq_max=" << n_seq_max;
                        ss << " n_ctx_per_seq=" << n_ctx_per_seq;
                        ss << " (increase n_ctx or reduce n_seq_max)";
                        const std::string msg = ss.str();
                        ALOGE("%s", msg.c_str());
                        if (midOnError) {
                            jstring jerr = threadEnv->NewStringUTF(msg.c_str());
                            threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                            threadEnv->DeleteLocalRef(jerr);
                        }
                        goto finish;
                    }
                }

                size_t common_prefix = 0;
                RadixNode * match_node = nullptr;
                bool need_kv_cleanup = false;
                const llama_pos seq0_pos_max = llama_memory_seq_pos_max(kv_mem, 0);
                const size_t kv_seq_tokens = seq0_pos_max >= 0 ? (size_t) seq0_pos_max + 1 : 0;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    if (h->disable_kv_reuse.load(std::memory_order_relaxed)) {
                        // Debug/diagnostics: force cold prefill (no common-prefix reuse).
                        match_node = h->tree_root;
                        common_prefix = 0;
                        need_kv_cleanup = !h->current_tokens.empty();
                    } else {
                        RadixLookupResult lookup = radix_lookup(h->tree_root, tokens.data(), n_tokens);
                        match_node = lookup.matched_node;
                        common_prefix = (size_t)lookup.matched_length;
                    }
                    if (common_prefix > kv_seq_tokens) {
                        ALOGW("【Prefill】common_prefix (%zu) > kv_seq_tokens (%zu)，已修正",
                              common_prefix, kv_seq_tokens);
                        common_prefix = kv_seq_tokens;
                    }
                    if (common_prefix > h->current_tokens.size()) {
                        ALOGW("【Prefill】state_drift | common_prefix=%zu current_tokens=%zu kv_seq_tokens=%zu",
                              common_prefix, h->current_tokens.size(), kv_seq_tokens);
                    }
                    if (common_prefix < kv_seq_tokens) need_kv_cleanup = true;

                    int node_count = radix_count_nodes(h->tree_root);
                    if (node_count > CURSORKV_EVICTION_THRESHOLD) {
                        int to_evict = node_count - CURSORKV_EVICTION_THRESHOLD / 2;
                        int protection_threshold = std::max(h->system_prompt_token_count, h->reusable_prefix_token_count);
                        radix_evict_cold(h, h->tree_root, to_evict, protection_threshold);
                    }
                }
                if (need_kv_cleanup) {
                    llama_memory_seq_rm(kv_mem, -1, (llama_pos)common_prefix, -1);
                    std::lock_guard<std::mutex> lock(h->mtx);
                    if (common_prefix < h->current_tokens.size()) {
                        h->current_tokens.resize(common_prefix);
                    }
                    h->cursor_node = match_node ? match_node : h->tree_root;
                }

                size_t n_new = tokens.size() - common_prefix;
                auto t_prefill_start = std::chrono::high_resolution_clock::now();
                if (n_new > 0) {
                    // Avoid failures when n_new > n_batch by chunking into micro-batches.
                    int n_batch = (int) h->cparams.n_batch;
                    if (n_batch <= 0) n_batch = 256;
                    int n_ubatch = (int) h->cparams.n_ubatch;
                    if (n_ubatch <= 0) n_ubatch = std::min(n_batch, 512);
                    const int n_chunk = std::max(1, std::min(n_batch, n_ubatch));

                    for (size_t off = 0; off < n_new; off += (size_t) n_chunk) {
                        const size_t n_eval = std::min(n_new - off, (size_t) n_chunk);
                        llama_batch chunk = llama_batch_init((int) n_eval, 0, 1);
                        chunk.n_tokens = (int32_t) n_eval;
                        for (int j = 0; j < (int) n_eval; ++j) {
                            chunk.token[j] = tokens[common_prefix + off + (size_t) j];
                            chunk.pos[j] = (llama_pos) (common_prefix + off + (size_t) j);
                            chunk.n_seq_id[j] = 1;
                            chunk.seq_id[j][0] = 0;
                            // Only request logits for the final token of the entire prefill.
                            chunk.logits[j] = (off + (size_t) j + 1 == n_new);
                        }
                        const int rc = llama_decode(ctx, chunk);
                        llama_batch_free(chunk);
                        if (rc != 0) {
                            int n_ctx = (int) h->cparams.n_ctx;
                            if (n_ctx <= 0) n_ctx = 0;
                            int n_seq_max = (int) h->cparams.n_seq_max;
                            if (n_seq_max <= 0) n_seq_max = 1;
                            const int n_ctx_per_seq = (n_ctx > 0) ? std::max(0, n_ctx / n_seq_max) : 0;

                            ALOGE("prefillPrompt | ❌ llama_decode 失败 | rc=%d | off=%llu | n_eval=%llu | n_new=%llu | common_prefix=%llu | tokens_total=%llu | n_ctx=%d | n_seq_max=%d | n_ctx_per_seq=%d | n_batch=%d | n_ubatch=%d | n_chunk=%d",
                                  rc,
                                  (unsigned long long) off,
                                  (unsigned long long) n_eval,
                                  (unsigned long long) n_new,
                                  (unsigned long long) common_prefix,
                                  (unsigned long long) tokens.size(),
                                n_ctx,
                                n_seq_max,
                                n_ctx_per_seq,
                                  n_batch,
                                  n_ubatch,
                                  n_chunk);
                            const char * msg = "prefillPrompt | ❌ llama_decode 失败";
                            if (midOnError) {
                                jstring jerr = threadEnv->NewStringUTF(msg);
                                threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                                threadEnv->DeleteLocalRef(jerr);
                            }
                            goto finish;
                        }
                        if (cancelled()) break;
                    }
                }
                auto t_prefill_end = std::chrono::high_resolution_clock::now();
                prefill_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();
                prefill_tokens_count = (int)n_new;

                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    h->current_tokens = tokens;
                    if (n_new > 0 && match_node) {
                        RadixNode * new_cursor = radix_insert(
                            match_node,
                            tokens.data() + common_prefix,
                            (int)n_new,
                            (int)common_prefix
                        );
                        h->cursor_node = new_cursor;
                        refresh_tree_hot_status(h->tree_root, h->current_tokens, h->system_prompt_token_count);
                    }
                }

                double prefill_tps = (prefill_time_ms > 0) ? (prefill_tokens_count * 1000.0 / prefill_time_ms) : 0;
                long long e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - start_time).count();
                std::stringstream ss_metrics;
                ss_metrics << "__METRICS__{"
                           << "\"mode\":\"prefill\"," 
                           << "\"prefill_ms\":" << prefill_time_ms << ","
                           << "\"prefill_tokens\":" << prefill_tokens_count << ","
                           << "\"prefill_tps\":" << prefill_tps << ","
                           << "\"decode_ms\":0,"
                           << "\"decode_tokens\":0,"
                           << "\"decode_tps\":0,"
                           << "\"e2e_ms\":" << e2e_ms
                           << "}";
                std::string metrics_str = ss_metrics.str();

                jclass strCls = threadEnv->FindClass("java/lang/String");
                jobjectArray arr = threadEnv->NewObjectArray(1, strCls, nullptr);
                jstring js = threadEnv->NewStringUTF(metrics_str.c_str());
                threadEnv->SetObjectArrayElement(arr, 0, js);
                threadEnv->DeleteLocalRef(js);
                threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                threadEnv->DeleteLocalRef(arr);
                threadEnv->DeleteLocalRef(strCls);
            }

        finish:
            if (batch_initialized) {
                llama_batch_free(batch);
            }
            if (threadEnv) {
                threadEnv->CallVoidMethod(gcb, midOnFinished);
                threadEnv->DeleteGlobalRef(gcb);
            }
            if (attached && threadEnv) gJvm->DetachCurrentThread();
        });
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_generatePhraseCandidatesSpliceMemory(
        JNIEnv* env,
        jclass clazz,
        jlong handlePtr,
        jstring jPrefixBeforeMemory,
        jstring jMemory,
        jstring jSuffixAfterMemory,
        jint n_candidates,
        jobject jCallback) {
    ALOGI("【LogicGen】KV-Splice 插入 Memory 生成短语候选调用 (实验)");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return -1;
    forceStopAndJoin(h, "generatePhraseCandidatesSpliceMemory");

    jobject gcb = env->NewGlobalRef(jCallback);
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID midOnTokenCandidates = env->GetMethodID(cbClass, "onTokenCandidates", "([Ljava/lang/String;)V");
    jmethodID midOnFinished = env->GetMethodID(cbClass, "onFinished", "()V");
    jmethodID midOnError = env->GetMethodID(cbClass, "onError", "(Ljava/lang/String;)V");

    auto jstr_to_std = [&](jstring js) -> std::string {
        if (!js) return "";
        const char * c = env->GetStringUTFChars(js, nullptr);
        std::string out(c ? c : "");
        if (c) env->ReleaseStringUTFChars(js, c);
        return out;
    };
    const std::string prefix = jstr_to_std(jPrefixBeforeMemory);
    const std::string memory = jstr_to_std(jMemory);
    const std::string suffix = jstr_to_std(jSuffixAfterMemory);

    const std::string base_prompt = prefix + suffix;
    const std::string full_prompt = prefix + memory + suffix;

    auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        h->stop_flag.store(false);
        h->gen_thread = std::thread([h, gcb, midOnTokenCandidates, midOnFinished, midOnError, base_prompt, full_prompt, prefix, memory, suffix, n_candidates, start_time, local_seq]() {
            auto cancelled = [&]() {
                return h->stop_flag.load(std::memory_order_relaxed) ||
                       h->generation_seq.load(std::memory_order_relaxed) != local_seq;
            };

            struct llama_context* ctx = nullptr;
            struct llama_model* model = nullptr;
            JNIEnv* threadEnv = nullptr;
            bool attached = false;

            long long base_prefill_ms = 0;
            long long splice_shift_ms = 0;
            long long memory_prefill_ms = 0;
            long long tail_recompute_ms = 0;
            long long decode_time_ms = 0;
            int decode_tokens_count = 0;
            std::vector<float> decode_latencies;

            bool batch_initialized = false;
            llama_batch batch{};

            if (gJvm->GetEnv((void**)&threadEnv, JNI_VERSION_1_6) != JNI_OK) {
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) attached = true;
            }
            if (!threadEnv) {
                ALOGE("KV-Splice | ❌ 无法获取线程 Env");
                goto finish;
            }

            {
                std::lock_guard<std::mutex> lock(h->mtx);
                ctx = h->ctx;
                model = h->model;
            }
            if (!ctx || !model || cancelled()) {
                const char * msg = "KV-Splice | ❌ Context 或 Model 为空";
                ALOGE("%s", msg);
                if (midOnError) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
                goto finish;
            }

            {
                const auto* vocab = llama_model_get_vocab(model);
                ensure_control_tokens(h, vocab);
                llama_memory_t kv_mem = llama_get_memory(ctx);

                auto tokenize_str = [&](const std::string & s, std::vector<llama_token> & out) -> bool {
                    out.assign(s.length() + 256, 0);
                    int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.length(), out.data(), (int)out.size(), false, true);
                    if (n <= 0) return false;
                    out.resize(n);
                    return true;
                };

                std::vector<llama_token> base_tokens, full_tokens;
                if (!tokenize_str(base_prompt, base_tokens) || !tokenize_str(full_prompt, full_tokens)) {
                    const char * msg = "KV-Splice | ❌ tokenize 失败";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }
                size_t prefix_len = 0;
                while (prefix_len < base_tokens.size() && prefix_len < full_tokens.size() &&
                       base_tokens[prefix_len] == full_tokens[prefix_len]) {
                    prefix_len++;
                }
                size_t suffix_len = 0;
                while (suffix_len < (base_tokens.size() - prefix_len) && suffix_len < (full_tokens.size() - prefix_len) &&
                       base_tokens[base_tokens.size() - 1 - suffix_len] == full_tokens[full_tokens.size() - 1 - suffix_len]) {
                    suffix_len++;
                }

                const size_t insert_len = (full_tokens.size() >= prefix_len + suffix_len)
                                            ? (full_tokens.size() - prefix_len - suffix_len)
                                            : 0;
                if (insert_len == 0 && full_tokens.size() != base_tokens.size()) {
                    ALOGW("KV-Splice | ⚠️ insert_len=0 but full/base differ (prefix=%zu suffix=%zu base=%zu full=%zu). Expect weak memory effect.",
                          prefix_len, suffix_len, base_tokens.size(), full_tokens.size());
                }
                bool base_ready = false;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    base_ready = (h->current_tokens.size() == base_tokens.size()) &&
                                 std::equal(h->current_tokens.begin(), h->current_tokens.end(), base_tokens.begin());
                }
                if (!base_ready) {
                    auto t_prefill_start = std::chrono::high_resolution_clock::now();

                    size_t common_prefix = 0;
                    RadixNode* match_node = nullptr;
                    bool need_kv_cleanup = false;
                    {
                        std::lock_guard<std::mutex> lock(h->mtx);
                        if (h->disable_kv_reuse.load(std::memory_order_relaxed)) {
                            match_node = h->tree_root;
                            common_prefix = 0;
                            need_kv_cleanup = !h->current_tokens.empty();
                        } else {
                            RadixLookupResult lookup = radix_lookup(h->tree_root, base_tokens.data(), (int)base_tokens.size());
                            match_node = lookup.matched_node;
                            common_prefix = (size_t)lookup.matched_length;
                        }
                        if (common_prefix > h->current_tokens.size()) common_prefix = h->current_tokens.size();
                        if (common_prefix < h->current_tokens.size()) need_kv_cleanup = true;
                    }
                    if (need_kv_cleanup) {
                        llama_memory_seq_rm(kv_mem, -1, (llama_pos)common_prefix, -1);
                        std::lock_guard<std::mutex> lock(h->mtx);
                        h->current_tokens.resize(common_prefix);
                        h->cursor_node = match_node ? match_node : h->tree_root;
                    }
                    size_t n_new = base_tokens.size() - common_prefix;
                    llama_batch prefill_batch = llama_batch_init(std::max(1, (int)n_new), 0, 1);
                    if (n_new > 0) {
                        prefill_batch.n_tokens = (int32_t)n_new;
                        for (int j = 0; j < prefill_batch.n_tokens; ++j) {
                            prefill_batch.token[j] = base_tokens[common_prefix + j];
                            prefill_batch.pos[j] = (llama_pos)(common_prefix + j);
                            prefill_batch.n_seq_id[j] = 1;
                            prefill_batch.seq_id[j][0] = 0;
                            prefill_batch.logits[j] = (j == prefill_batch.n_tokens - 1);
                        }
                        if (llama_decode(ctx, prefill_batch) != 0) {
                            llama_batch_free(prefill_batch);
                            const char * msg = "KV-Splice | ❌ base prefill llama_decode 失败";
                            ALOGE("%s", msg);
                            if (midOnError) {
                                jstring jerr = threadEnv->NewStringUTF(msg);
                                threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                                threadEnv->DeleteLocalRef(jerr);
                            }
                            goto finish;
                        }
                    }
                    llama_batch_free(prefill_batch);

                    auto t_prefill_end = std::chrono::high_resolution_clock::now();
                    base_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();

                    {
                        std::lock_guard<std::mutex> lock(h->mtx);
                        h->current_tokens = base_tokens;
                        sync_radix_tree_from_tokens(h);
                    }
                }

                // Work on isolated sequences so seq0 remains the stable base cache.
                const llama_seq_id work_seq = (llama_seq_id) (h->cparams.n_seq_max - 1);
                const llama_seq_id temp_seq = (llama_seq_id) (h->cparams.n_seq_max - 2);

                // Candidate sequences: 1..k_pick must not collide with temp/work seqs.
                const int max_candidates = std::max(1, (int)h->cparams.n_seq_max - 3);
                const int k_pick = std::min(std::min((int)n_candidates, 20), max_candidates);
                if (k_pick <= 0) {
                    const char * msg = "KV-Splice | ❌ k_pick 计算失败";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }
                // Copy base seq0 -> work_seq
                llama_memory_seq_rm(kv_mem, work_seq, -1, -1);
                llama_memory_seq_cp(kv_mem, 0, work_seq, 0, -1);

                if (!llama_memory_can_shift(kv_mem)) {
                    ALOGW("KV-Splice | ⚠️ llama_memory_can_shift=false. Fallback to baseline full prefill.");
                    std::stringstream ss;
                    ss << "__METRICS__{\"mode\":\"splice_fallback_no_shift\",\"e2e_ms\":0}";
                    std::string metrics_str = ss.str();
                    jclass strCls = threadEnv->FindClass("java/lang/String");
                    jobjectArray arr = threadEnv->NewObjectArray(1, strCls, nullptr);
                    jstring js = threadEnv->NewStringUTF(metrics_str.c_str());
                    threadEnv->SetObjectArrayElement(arr, 0, js);
                    threadEnv->DeleteLocalRef(js);
                    threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                    threadEnv->DeleteLocalRef(arr);
                    threadEnv->DeleteLocalRef(strCls);
                    goto finish;
                }

                const llama_pos p_ins = (llama_pos) prefix_len;
                const llama_pos ins_len = (llama_pos) insert_len;
                const llama_pos del_len = (llama_pos) (base_tokens.size() - prefix_len - suffix_len);
                const llama_pos delta = ins_len - del_len;
                const llama_pos full_len = (llama_pos) full_tokens.size();

                    if (kv_splice_debug_enabled()) {
                      ALOGI("KV-Splice | lens base=%zu full=%zu prefix=%zu suffix=%zu ins=%zu del=%d delta=%d | p_ins=%d",
                          base_tokens.size(), full_tokens.size(), prefix_len, suffix_len, insert_len,
                          (int) del_len, (int) delta, (int) p_ins);
                      ALOGI("KV-Splice | seq0 pos_min=%d pos_max=%d",
                          (int) kv_mem->seq_pos_min(0), (int) kv_mem->seq_pos_max(0));
                      ALOGI("KV-Splice | after cp work_seq=%d pos_min=%d pos_max=%d",
                          (int) work_seq,
                          (int) kv_mem->seq_pos_min(work_seq), (int) kv_mem->seq_pos_max(work_seq));
                    }

                // 1) Remove the base middle span and shift the suffix by delta to create space for insertion.
                auto t_shift_start = std::chrono::high_resolution_clock::now();
                if (del_len > 0) {
                    llama_memory_seq_rm(kv_mem, work_seq, p_ins, p_ins + del_len);
                }
                if (delta != 0) {
                    llama_memory_seq_add(kv_mem, work_seq, p_ins + del_len, -1, delta);
                }
                auto t_shift_end = std::chrono::high_resolution_clock::now();
                splice_shift_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_shift_end - t_shift_start).count();
                if (kv_splice_debug_enabled()) {
                    ALOGI("KV-Splice | after shift work_seq=%d pos_min=%d pos_max=%d",
                          (int) work_seq,
                          (int) kv_mem->seq_pos_min(work_seq), (int) kv_mem->seq_pos_max(work_seq));
                }

                // 2) Prepare temp_seq = prefix-only KV so we can prefill inserted tokens consecutively.
                llama_memory_seq_rm(kv_mem, temp_seq, -1, -1);
                llama_memory_seq_cp(kv_mem, 0, temp_seq, 0, -1);
                // Drop tail from insertion point onward (keep [0, p_ins))
                llama_memory_seq_rm(kv_mem, temp_seq, p_ins, -1);
                if (kv_splice_debug_enabled()) {
                    ALOGI("KV-Splice | temp_seq=%d after prefix-only pos_min=%d pos_max=%d",
                          (int) temp_seq,
                          (int) kv_mem->seq_pos_min(temp_seq), (int) kv_mem->seq_pos_max(temp_seq));
                }

                // 3) Prefill inserted tokens into temp_seq at positions [p_ins, p_ins+ins_len)
                auto t_mem_prefill_start = std::chrono::high_resolution_clock::now();
                if (ins_len > 0) {
                    llama_batch mem_batch = llama_batch_init((int)ins_len, 0, 1);
                    mem_batch.n_tokens = (int32_t)ins_len;
                    for (int j = 0; j < mem_batch.n_tokens; ++j) {
                        mem_batch.token[j] = full_tokens[prefix_len + j];
                        mem_batch.pos[j] = p_ins + j;
                        mem_batch.n_seq_id[j] = 1;
                        mem_batch.seq_id[j][0] = temp_seq;
                        mem_batch.logits[j] = (j == mem_batch.n_tokens - 1);
                    }
                    if (llama_decode(ctx, mem_batch) != 0) {
                        llama_batch_free(mem_batch);
                        const char * msg = "KV-Splice | ❌ inserted-token prefill llama_decode 失败";
                        ALOGE("%s", msg);
                        if (midOnError) {
                            jstring jerr = threadEnv->NewStringUTF(msg);
                            threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                            threadEnv->DeleteLocalRef(jerr);
                        }
                        goto finish;
                    }
                    llama_batch_free(mem_batch);
                }
                auto t_mem_prefill_end = std::chrono::high_resolution_clock::now();
                memory_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_mem_prefill_end - t_mem_prefill_start).count();

                // 4) Splice: overlay-copy inserted KV range from temp_seq -> work_seq (preserve dst)
                if (ins_len > 0) {
                    llama_memory_seq_cp_overlay(kv_mem, temp_seq, work_seq, p_ins, p_ins + ins_len);
                }
                if (kv_splice_debug_enabled()) {
                    ALOGI("KV-Splice | after splice-cp work_seq=%d pos_min=%d pos_max=%d",
                          (int) work_seq,
                          (int) kv_mem->seq_pos_min(work_seq), (int) kv_mem->seq_pos_max(work_seq));
                }

                // 4) Recompute the last token logits in work_seq so next-token distribution can attend to memory
                if (full_len > 0) {
                    const llama_pos last_pos = full_len - 1;
                    llama_token last_token = full_tokens.back();

                    auto t_tail_start = std::chrono::high_resolution_clock::now();
                    llama_memory_seq_rm(kv_mem, work_seq, last_pos, last_pos + 1);
                    llama_batch tail_batch = llama_batch_init(1, 0, 1);
                    tail_batch.n_tokens = 1;
                    tail_batch.token[0] = last_token;
                    tail_batch.pos[0] = last_pos;
                    tail_batch.n_seq_id[0] = 1;
                    tail_batch.seq_id[0][0] = work_seq;
                    tail_batch.logits[0] = true;
                    if (llama_decode(ctx, tail_batch) != 0) {
                        llama_batch_free(tail_batch);
                        const char * msg = "KV-Splice | ❌ tail recompute llama_decode 失败";
                        ALOGE("%s", msg);
                        if (midOnError) {
                            jstring jerr = threadEnv->NewStringUTF(msg);
                            threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                            threadEnv->DeleteLocalRef(jerr);
                        }
                        goto finish;
                    }
                    llama_batch_free(tail_batch);
                    auto t_tail_end = std::chrono::high_resolution_clock::now();
                    tail_recompute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_tail_end - t_tail_start).count();
                }

                // Now sample candidates from logits of the last decode (work_seq)
                auto* logits = llama_get_logits_ith(ctx, -1);
                int n_vocab = llama_vocab_n_tokens(vocab);
                const float PROMPT_REP_PENALTY = 1.1f;
                // Light repetition penalty: apply only to user-input tokens (after reusable prefix)
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    const int rep_start = std::max(0, h->reusable_prefix_token_count);
                    for (size_t ti = rep_start; ti < h->current_tokens.size(); ++ti) {
                        llama_token t = h->current_tokens[ti];
                        if (logits[t] > 0) logits[t] /= PROMPT_REP_PENALTY;
                        else logits[t] *= PROMPT_REP_PENALTY;
                    }
                }

                // Use llama.cpp sampler chain for probabilistic first-token selection
                const uint32_t seed_base = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();

                // Prepare candidate sequences 1..k_pick from work_seq
                for (int i = 1; i <= k_pick; ++i) {
                    llama_memory_seq_rm(kv_mem, (llama_seq_id)i, -1, -1);
                    llama_memory_seq_cp(kv_mem, work_seq, (llama_seq_id)i, 0, -1);
                }

                struct SeqState {
                    std::string phrase;
                    std::vector<llama_token> gen_tokens;
                    llama_token last_token;
                    bool active = true;
                };
                std::vector<SeqState> states(k_pick);

                batch = llama_batch_init(std::max(k_pick, 1), 0, 1);
                batch_initialized = true;
                batch.n_tokens = k_pick;
                const llama_pos prompt_end_pos = full_len;
                for (int i = 0; i < k_pick; ++i) {
                    // Build a sampler chain per candidate: top_k → top_p → temp → dist
                    auto sparams = llama_sampler_chain_default_params();
                    llama_sampler* first_smpl = llama_sampler_chain_init(sparams);
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_top_k(20));
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_top_p(0.8f, 1));
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_temp(0.7f));
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_dist(seed_base + (uint32_t)i));
                    llama_token first_tok = llama_sampler_sample(first_smpl, ctx, -1);
                    llama_sampler_free(first_smpl);

                    states[i].last_token = first_tok;
                    states[i].gen_tokens.push_back(first_tok);
                    char buf[256];
                    int n_p = token_to_piece_with_control(vocab, h, first_tok, buf, 256);
                    if (n_p > 0) states[i].phrase.append(buf, n_p);

                    batch.token[i] = first_tok;
                    batch.pos[i] = prompt_end_pos;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = (llama_seq_id)(i + 1); // candidates live in seq 1..k_pick
                    batch.logits[i] = true;
                }

                auto t_decode_total_start = std::chrono::high_resolution_clock::now();
                // Build per-candidate sampler chains for the inner decode loop
                std::vector<llama_sampler*> candidate_samplers(k_pick, nullptr);
                for (int i = 0; i < k_pick; ++i) {
                    auto sparams = llama_sampler_chain_default_params();
                    candidate_samplers[i] = llama_sampler_chain_init(sparams);
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_penalties(
                        /*penalty_last_n=*/16, /*penalty_repeat=*/1.2f,
                        /*penalty_freq=*/0.0f, /*penalty_present=*/0.0f));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_top_k(20));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_top_p(0.8f, 1));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_temp(0.7f));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_dist(seed_base + 100u + (uint32_t)i));
                    // Accept the first token so penalties track it
                    llama_sampler_accept(candidate_samplers[i], states[i].gen_tokens[0]);
                }
                const int MAX_PHRASE_LEN = 16;
                int active_count = k_pick;

                for (int step = 0; step < MAX_PHRASE_LEN && active_count > 0; ++step) {
                    if (cancelled()) break;
                    auto t_step_start = std::chrono::high_resolution_clock::now();
                    if (llama_decode(ctx, batch) != 0) break;
                    auto t_step_end = std::chrono::high_resolution_clock::now();
                    decode_latencies.push_back((float) std::chrono::duration_cast<std::chrono::microseconds>(t_step_end - t_step_start).count() / 1000.0f);

                    llama_batch next_batch = llama_batch_init(k_pick, 0, 1);
                    next_batch.n_tokens = 0;

                    for (int i = 0; i < k_pick; ++i) {
                        if (!states[i].active) continue;
                        // Sample using the per-candidate sampler chain
                        llama_token next_tok = llama_sampler_sample(candidate_samplers[i], ctx, i);
                        llama_sampler_accept(candidate_samplers[i], next_tok);

                        decode_tokens_count++;
                        if (next_tok == llama_token_eos(vocab) || next_tok == llama_token_eot(vocab)) {
                            states[i].active = false;
                            active_count--;
                            continue;
                        }
                        char piece[256];
                        int n_piece = token_to_piece_with_control(vocab, h, next_tok, piece, 256);
                        if (n_piece > 0) {
                            states[i].phrase.append(piece, n_piece);
                        }
                        states[i].last_token = next_tok;
                        states[i].gen_tokens.push_back(next_tok);

                        int idx = next_batch.n_tokens;
                        next_batch.token[idx] = next_tok;
                        next_batch.pos[idx] = prompt_end_pos + step + 1;
                        next_batch.n_seq_id[idx] = 1;
                        next_batch.seq_id[idx][0] = (llama_seq_id)(i + 1);
                        next_batch.logits[idx] = true;
                        next_batch.n_tokens++;
                    }
                    llama_batch_free(batch);
                    batch = next_batch;
                }

                // Free per-candidate sampler chains
                for (int i = 0; i < k_pick; ++i) {
                    if (candidate_samplers[i]) llama_sampler_free(candidate_samplers[i]);
                }
                auto t_decode_total_end = std::chrono::high_resolution_clock::now();
                decode_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - t_decode_total_start).count();

                std::vector<std::string> result_phrases;
                for (int i = 0; i < k_pick; ++i) {
                    std::string & s = states[i].phrase;
                    if (!s.empty() && is_valid_utf8(s)) result_phrases.push_back(s);
                }

                double decode_tps = (decode_time_ms > 0) ? (decode_tokens_count * 1000.0 / decode_time_ms) : 0;
                long long e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - start_time).count();

                std::stringstream ss_metrics;
                ss_metrics << "__METRICS__{"
                           << "\"mode\":\"splice_insert\"," 
                           << "\"base_prefill_ms\":" << base_prefill_ms << ","
                           << "\"splice_shift_ms\":" << splice_shift_ms << ","
                           << "\"memory_prefill_ms\":" << memory_prefill_ms << ","
                           << "\"tail_recompute_ms\":" << tail_recompute_ms << ","
                           << "\"prefill_ms\":" << (base_prefill_ms + splice_shift_ms + memory_prefill_ms + tail_recompute_ms) << ","
                           << "\"prefill_tokens\":" << (int) insert_len << ","
                           << "\"decode_ms\":" << decode_time_ms << ","
                           << "\"decode_tokens\":" << decode_tokens_count << ","
                           << "\"decode_tps\":" << decode_tps << ","
                           << "\"e2e_ms\":" << e2e_ms << ","
                           << "\"p_ins\":" << (int)p_ins << ","
                           << "\"memory_tokens\":" << (int) insert_len << ","
                           << "\"suffix_tokens\":" << (int) suffix_len << ","
                           << "\"decode_latencies\":[";
                for (size_t i = 0; i < decode_latencies.size(); ++i) {
                    ss_metrics << decode_latencies[i] << (i == decode_latencies.size() - 1 ? "" : ",");
                }
                ss_metrics << "]}";

                result_phrases.push_back(ss_metrics.str());

                if (!result_phrases.empty()) {
                    jclass strCls = threadEnv->FindClass("java/lang/String");
                    jobjectArray arr = threadEnv->NewObjectArray(result_phrases.size(), strCls, nullptr);
                    for (size_t i = 0; i < result_phrases.size(); i++) {
                        jstring js = new_jstring_utf8_lenient(threadEnv, result_phrases[i]);
                        threadEnv->SetObjectArrayElement(arr, i, js);
                        threadEnv->DeleteLocalRef(js);
                    }
                    threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                    threadEnv->DeleteLocalRef(arr);
                    threadEnv->DeleteLocalRef(strCls);
                }
            }

        finish:
            if (batch_initialized) {
                struct llama_context* ctx_to_clear = nullptr;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    ctx_to_clear = h->ctx;
                }
                if (ctx_to_clear) {
                    llama_memory_t kv_mem2 = llama_get_memory(ctx_to_clear);
                    // Clear candidate seqs + temp/work seqs (only valid IDs: 0..n_seq_max-1)
                    for (int i = 1; i < (int)h->cparams.n_seq_max; ++i) {
                        llama_memory_seq_rm(kv_mem2, (llama_seq_id)i, -1, -1);
                    }
                }
                llama_batch_free(batch);
            }
            if (threadEnv) {
                threadEnv->CallVoidMethod(gcb, midOnFinished);
                threadEnv->DeleteGlobalRef(gcb);
            }
            if (attached && threadEnv) gJvm->DetachCurrentThread();
        });
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_generatePhraseCandidates(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jPrompt, jint n_candidates, jobject jCallback) {
    ALOGI("【LogicGen】并行 Batching 生成短语候选调用");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return -1;
    forceStopAndJoin(h, "generatePhraseCandidates");
    jobject gcb = env->NewGlobalRef(jCallback);
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID midOnTokenCandidates = env->GetMethodID(cbClass, "onTokenCandidates", "([Ljava/lang/String;)V");
    jmethodID midOnFinished = env->GetMethodID(cbClass, "onFinished", "()V");
    const char* cprompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(cprompt);
    env->ReleaseStringUTFChars(jPrompt, cprompt);
    auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        h->stop_flag.store(false);
        h->gen_thread = std::thread([h, gcb, midOnTokenCandidates, midOnFinished, prompt, n_candidates, start_time, local_seq]() {
            auto cancelled = [&]() {
                return h->stop_flag.load(std::memory_order_relaxed) ||
                       h->generation_seq.load(std::memory_order_relaxed) != local_seq;
            };
            // Predeclare all resources so any early exit via goto can still clean up safely.
            struct llama_context* ctx = nullptr;
            struct llama_model* model = nullptr;
            long long prefill_time_ms = 0;
            long long decode_time_ms = 0;
            int prefill_tokens_count = 0;
            int decode_tokens_count = 0;
            std::vector<float> decode_latencies;
            llama_pos prompt_end_pos = 0;
            bool batch_initialized = false;
            llama_batch batch{};
            JNIEnv* threadEnv = nullptr;
            bool attached = false;
            if (gJvm->GetEnv((void**)&threadEnv, JNI_VERSION_1_6) != JNI_OK) {
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) attached = true;
            }
            if (!threadEnv) {
                ALOGE("L1-Gen-Phrase | ❌ 无法获取线程 Env, 放弃本次生成");
                goto finish;
            }
            {
                std::lock_guard<std::mutex> lock(h->mtx);
                ctx = h->ctx;
                model = h->model;
            }
            if (!ctx || !model || cancelled()) {
                ALOGE("L1-Gen-Phrase | ❌ Context 或 Model 为空");
                goto finish;
            }
            {
                const auto* vocab = llama_model_get_vocab(model);
                ensure_control_tokens(h, vocab);
                llama_memory_t kv_mem = llama_get_memory(ctx);
                std::vector<llama_token> tokens(prompt.length() + 128);
                int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.length(), tokens.data(), (int)tokens.size(), false, true);
                if (n_tokens <= 0) goto finish;
                tokens.resize(n_tokens);
                size_t common_prefix = 0;
                RadixNode* match_node = nullptr;
                bool need_kv_cleanup = false;
                const llama_pos seq0_pos_max = llama_memory_seq_pos_max(kv_mem, 0);
                const size_t kv_seq_tokens = seq0_pos_max >= 0 ? (size_t) seq0_pos_max + 1 : 0;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    if (h->disable_kv_reuse.load(std::memory_order_relaxed)) {
                        match_node = h->tree_root;
                        common_prefix = 0;
                        need_kv_cleanup = !h->current_tokens.empty();
                    } else {
                        RadixLookupResult lookup = radix_lookup(h->tree_root, tokens.data(), n_tokens);
                        match_node = lookup.matched_node;
                        common_prefix = (size_t)lookup.matched_length;
                    }

                    if (common_prefix > kv_seq_tokens) {
                        ALOGW("【Prefill】common_prefix (%zu) > kv_seq_tokens (%zu)，已修正",
                              common_prefix, kv_seq_tokens);
                        common_prefix = kv_seq_tokens;
                    }
                    if (common_prefix > h->current_tokens.size()) {
                        ALOGW("【Prefill】state_drift | common_prefix=%zu current_tokens=%zu kv_seq_tokens=%zu",
                              common_prefix, h->current_tokens.size(), kv_seq_tokens);
                    }

                    if (common_prefix >= tokens.size() && !tokens.empty()) {
                        common_prefix = tokens.size() - 1;
                        if (match_node && match_node->parent) {
                            match_node = match_node->parent;
                        }
                    }

                    if (common_prefix < kv_seq_tokens) {
                        need_kv_cleanup = true;
                    }
                    int node_count = radix_count_nodes(h->tree_root);
                    if (node_count > CURSORKV_EVICTION_THRESHOLD) {
                        int to_evict = node_count - CURSORKV_EVICTION_THRESHOLD / 2;
                        int protection_threshold = std::max(h->system_prompt_token_count, h->reusable_prefix_token_count);
                        radix_evict_cold(h, h->tree_root, to_evict, protection_threshold);
                    }
                }
                if (need_kv_cleanup) {
                    llama_memory_seq_rm(kv_mem, -1, (llama_pos)common_prefix, -1);
                    std::lock_guard<std::mutex> lock(h->mtx);
                    if (common_prefix < h->current_tokens.size()) {
                        h->current_tokens.resize(common_prefix);
                    }
                    h->cursor_node = match_node ? match_node : h->tree_root;
                }
                size_t n_new = tokens.size() - common_prefix;
                batch = llama_batch_init(std::max((int)n_new, (int)n_candidates), 0, 1);
                batch_initialized = true;
                if (common_prefix > 0) {
                    std::string reuse_text = tokens_to_str(model, tokens.data(), (int)common_prefix);
                    ALOG_PFD("Prefill Start | KV Reuse: %zu tokens | Content: [%s]", common_prefix, reuse_text.c_str());
                } else {
                    ALOG_PFD("Prefill Start | No KV Reuse (Cold Start)");
                }
                auto t_prefill_start = std::chrono::high_resolution_clock::now();
                if (n_new > 0) {
                    batch.n_tokens = (int32_t)n_new;
                    for (int j = 0; j < batch.n_tokens; ++j) {
                        batch.token[j] = tokens[common_prefix + j];
                        batch.pos[j] = (llama_pos)(common_prefix + j);
                        batch.n_seq_id[j] = 1;
                        batch.seq_id[j][0] = 0;
                        batch.logits[j] = (j == batch.n_tokens - 1);
                    }
                    if (llama_decode(ctx, batch) != 0) goto finish;
                }
                auto t_prefill_end = std::chrono::high_resolution_clock::now();
                prefill_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();
                prefill_tokens_count = (int)n_new;
                {
                    double prefill_rate = prefill_time_ms > 0 ? (double)prefill_tokens_count * 1000.0 / prefill_time_ms : 0;
                    ALOG_PFD("Prefill Finish | New Tokens: %d | Time: %lld ms | Rate: %.2f T/s", prefill_tokens_count, prefill_time_ms, prefill_rate);
                }
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    h->current_tokens = tokens;
                    prompt_end_pos = (llama_pos)h->current_tokens.size();
                    if (n_new > 0 && match_node) {
                        RadixNode* new_cursor = radix_insert(
                            match_node,
                            tokens.data() + common_prefix,
                            (int)n_new,
                            (int)common_prefix
                        );
                        h->cursor_node = new_cursor;
                        refresh_tree_hot_status(h->tree_root, h->current_tokens, h->system_prompt_token_count);
                    }
                }
                if (!cancelled()) {
                    auto* logits = llama_get_logits_ith(ctx, -1);
                    int n_vocab = llama_vocab_n_tokens(vocab);
                    const float PROMPT_REP_PENALTY = 1.1f;
                    {
                        std::lock_guard<std::mutex> lock(h->mtx);
                        const int rep_start = std::max(0, h->reusable_prefix_token_count);
                        for (size_t ti = rep_start; ti < tokens.size(); ++ti) {
                            llama_token t = tokens[ti];
                            if (logits[t] > 0) logits[t] /= PROMPT_REP_PENALTY;
                            else logits[t] *= PROMPT_REP_PENALTY;
                        }
                    }
                    int k_pick = std::min((int)n_candidates, 20);
                    const uint32_t seed_base = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
                    struct SeqState {
                        std::string phrase;
                        std::vector<llama_token> gen_tokens;
                        llama_token last_token;
                        bool active = true;
                    };
                    std::vector<SeqState> states(k_pick);
                    for (int i = 1; i < k_pick; ++i) {
                        llama_memory_seq_cp(kv_mem, 0, i, 0, -1);
                    }
                    batch.n_tokens = k_pick;
                    for (int i = 0; i < k_pick; ++i) {
                        // Build a sampler chain per candidate: top_k → top_p → temp → dist
                        auto sparams = llama_sampler_chain_default_params();
                        llama_sampler* first_smpl = llama_sampler_chain_init(sparams);
                        llama_sampler_chain_add(first_smpl, llama_sampler_init_top_k(20));
                        llama_sampler_chain_add(first_smpl, llama_sampler_init_top_p(0.8f, 1));
                        llama_sampler_chain_add(first_smpl, llama_sampler_init_temp(0.7f));
                        llama_sampler_chain_add(first_smpl, llama_sampler_init_dist(seed_base + (uint32_t)i));
                        llama_token first_tok = llama_sampler_sample(first_smpl, ctx, -1);
                        llama_sampler_free(first_smpl);
                        states[i].last_token = first_tok;
                        states[i].gen_tokens.push_back(first_tok);
                        char buf[256];
                        int n_p = token_to_piece_with_control(vocab, h, first_tok, buf, 256);
                        if (n_p > 0) states[i].phrase.append(buf, n_p);
                        batch.token[i] = first_tok;
                        batch.pos[i] = prompt_end_pos;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = i; 
                        batch.logits[i] = true;
                    }
                    ALOG_PFD("Decode Start | Candidates: %d | Batch size: %d", k_pick, k_pick);
                    auto t_decode_total_start = std::chrono::high_resolution_clock::now();
                    const int MAX_PHRASE_LEN = 8;
                    int active_count = k_pick;
                    // Build per-candidate sampler chains for the inner decode loop
                    std::vector<llama_sampler*> candidate_samplers(k_pick, nullptr);
                    for (int i = 0; i < k_pick; ++i) {
                        auto sparams = llama_sampler_chain_default_params();
                        candidate_samplers[i] = llama_sampler_chain_init(sparams);
                        llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_penalties(
                            /*penalty_last_n=*/16, /*penalty_repeat=*/1.2f,
                            /*penalty_freq=*/0.0f, /*penalty_present=*/0.0f));
                        llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_top_k(20));
                        llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_top_p(0.8f, 1));
                        llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_temp(0.7f));
                        llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_dist(seed_base + 100u + (uint32_t)i));
                        // Accept the first token so penalties track it
                        llama_sampler_accept(candidate_samplers[i], states[i].gen_tokens[0]);
                    }
                    for (int step = 0; step < MAX_PHRASE_LEN && active_count > 0; ++step) {
                        if (cancelled()) break;
                        auto t_step_start = std::chrono::high_resolution_clock::now();
                        if (llama_decode(ctx, batch) != 0) break;
                        auto t_step_end = std::chrono::high_resolution_clock::now();
                        decode_latencies.push_back(std::chrono::duration<float, std::milli>(t_step_end - t_step_start).count());
                        llama_batch next_batch = llama_batch_init(active_count, 0, 1);
                        next_batch.n_tokens = 0;
                        for (int i = 0; i < k_pick; ++i) {
                            if (!states[i].active) continue;
                            int batch_idx = -1;
                            for (int b = 0; b < batch.n_tokens; ++b) {
                                if (batch.seq_id[b][0] == i) { batch_idx = b; break; }
                            }
                            if (batch_idx == -1) continue;
                            // Sample using the per-candidate sampler chain (handles rep penalty + top_k + top_p + temp + dist)
                            llama_token next_tok = llama_sampler_sample(candidate_samplers[i], ctx, batch_idx);
                            llama_sampler_accept(candidate_samplers[i], next_tok);
                            decode_tokens_count++;
                            char buf[256];
                            int n_p = token_to_piece_with_control(vocab, h, next_tok, buf, 256);
                            std::string piece(buf, n_p > 0 ? n_p : 0);
                            bool stop_branch = (next_tok == llama_vocab_eos(vocab)) ||
                                              llama_vocab_is_control(vocab, next_tok) ||
                                              (next_tok == states[i].last_token) ||
                                              (piece.find("<|im_end|>") != std::string::npos) ||
                                              (piece.find("。") != std::string::npos || piece.find("！") != std::string::npos ||
                                               piece.find("？") != std::string::npos || piece.find("\n") != std::string::npos);
                            if (stop_branch) {
                                if (llama_vocab_is_control(vocab, next_tok) && !piece.empty()) {
                                    states[i].phrase.append(piece);
                                }
                                states[i].active = false;
                                active_count--;
                            } else {
                                states[i].phrase.append(piece);
                                states[i].last_token = next_tok;
                                states[i].gen_tokens.push_back(next_tok);
                                int idx = next_batch.n_tokens;
                                next_batch.token[idx] = next_tok;
                                next_batch.pos[idx] = prompt_end_pos + step + 1;
                                next_batch.n_seq_id[idx] = 1;
                                next_batch.seq_id[idx][0] = i;
                                next_batch.logits[idx] = true;
                                next_batch.n_tokens++;
                            }
                        }
                        llama_batch_free(batch);
                        batch = next_batch;
                    }
                    auto t_decode_total_end = std::chrono::high_resolution_clock::now();
                    decode_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - t_decode_total_start).count();
                    {
                        double decode_rate = decode_time_ms > 0 ? (double)decode_tokens_count * 1000.0 / decode_time_ms : 0;
                        ALOG_PFD("Decode Finish | Total Tokens: %d | Time: %lld ms | Rate: %.2f T/s", decode_tokens_count, decode_time_ms, decode_rate);
                    }
                    for (int i = 0; i < k_pick; ++i) {
                        if (candidate_samplers[i]) llama_sampler_free(candidate_samplers[i]);
                    }
                    std::vector<std::string> result_phrases;
                    for (int i = 0; i < k_pick; ++i) {
                        std::string& s = states[i].phrase;
                        if (!s.empty() && is_valid_utf8(s)) {
                            bool is_repetitive = false;
                            size_t comma_pos = s.find("，");
                            if (comma_pos != std::string::npos) {
                                std::string part1 = s.substr(0, comma_pos);
                                std::string remaining = s.substr(comma_pos + strlen("，"));
                                if (remaining.find(part1) != std::string::npos) is_repetitive = true;
                            }
                            if (!is_repetitive) result_phrases.push_back(s);
                        }
                    }
                    double prefill_tps = (prefill_time_ms > 0) ? (prefill_tokens_count * 1000.0 / prefill_time_ms) : 0;
                    double decode_tps = (decode_time_ms > 0) ? (decode_tokens_count * 1000.0 / decode_time_ms) : 0;
                    long long e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - start_time).count();
                    std::stringstream ss_metrics;
                    ss_metrics << "__METRICS__{"
                               << "\"prefill_ms\":" << prefill_time_ms << ","
                               << "\"prefill_tokens\":" << prefill_tokens_count << ","
                               << "\"prefill_tps\":" << prefill_tps << ","
                               << "\"decode_ms\":" << decode_time_ms << ","
                               << "\"decode_tokens\":" << decode_tokens_count << ","
                               << "\"decode_tps\":" << decode_tps << ","
                               << "\"e2e_ms\":" << e2e_ms << ","
                               << "\"decode_latencies\":[";
                    for(size_t i=0; i<decode_latencies.size(); ++i) {
                        ss_metrics << decode_latencies[i] << (i == decode_latencies.size()-1 ? "" : ",");
                    }
                    ss_metrics << "]}";
                    std::string metrics_str = ss_metrics.str();
                    result_phrases.push_back(metrics_str);
                    ALOGI("【Native生成】✅ 候选词生成完成 | E2E: %lld ms | Prefill: %lld ms (%d tokens) | Decode: %lld ms (%d tokens, %.2f T/s)",
                          e2e_ms, prefill_time_ms, prefill_tokens_count, decode_time_ms, decode_tokens_count, decode_tps);
                    if (!result_phrases.empty()) {
                        jclass strCls = threadEnv->FindClass("java/lang/String");
                        jobjectArray arr = threadEnv->NewObjectArray(result_phrases.size(), strCls, nullptr);
                        for(size_t i=0; i<result_phrases.size(); i++) {
                            jstring js = new_jstring_utf8_lenient(threadEnv, result_phrases[i]);
                            threadEnv->SetObjectArrayElement(arr, i, js);
                            threadEnv->DeleteLocalRef(js);
                        }
                        threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                        threadEnv->DeleteLocalRef(arr);
                        threadEnv->DeleteLocalRef(strCls);
                    }
                }
            }
            finish:
            if (!threadEnv) {
                // Best-effort attach for cleanup to avoid leaking global refs
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) {
                    attached = true;
                }
            }
            if (batch_initialized) {
                struct llama_context* ctx_to_clear = nullptr;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    ctx_to_clear = h->ctx;
                }
                if (ctx_to_clear) {
                    for (int i = 1; i <= n_candidates + 1; ++i) {
                        llama_memory_seq_rm(llama_get_memory(ctx_to_clear), i, -1, -1);
                    }
                }
                llama_batch_free(batch);
            }
            if (threadEnv) {
                threadEnv->CallVoidMethod(gcb, midOnFinished);
                threadEnv->DeleteGlobalRef(gcb);
            } else {
                ALOGE("L1-Gen-Phrase | ❌ 清理失败：缺少有效 JNIEnv，可能存在全局引用泄漏");
            }
            if (attached && threadEnv) gJvm->DetachCurrentThread();
        });
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_generateCandidates(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jPrompt, jint n_candidates, jobject jCallback) {
    ALOGI("【LogicGen】生成词候选调用 (Smart Token)，但是我们现在先废弃了");
    return 0;
}


extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_generateSentenceCandidates(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jPrompt, jint n_candidates, jobject jCallback) {
    ALOGI("【LogicGen】生成长句候选调用，但是现在我们先废弃了");
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_generateMemoryWorker(JNIEnv * env, jclass, jlong handlePtr, jstring jPrompt, jint maxTokens, jobject jCallback) {
    ALOGI("【LogicGen】MemoryWorker 长生成调用 | maxTokens=%d", (int) maxTokens);
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !jPrompt || !jCallback) return -1;

    forceStopAndJoin(h, "generateMemoryWorker");

    jobject gcb = env->NewGlobalRef(jCallback);
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID midOnTokenCandidates = env->GetMethodID(cbClass, "onTokenCandidates", "([Ljava/lang/String;)V");
    jmethodID midOnFinished        = env->GetMethodID(cbClass, "onFinished", "()V");
    jmethodID midOnError           = env->GetMethodID(cbClass, "onError", "(Ljava/lang/String;)V");

    const char * cprompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(cprompt ? cprompt : "");
    if (cprompt) env->ReleaseStringUTFChars(jPrompt, cprompt);

    if (maxTokens <= 0) maxTokens = 128;
    if (maxTokens > 2048) maxTokens = 2048;

    {
        std::lock_guard<std::mutex> lock(h->mtx);
        const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        h->stop_flag.store(false);

        h->gen_thread = std::thread([h, gcb, midOnTokenCandidates, midOnFinished, midOnError, prompt, maxTokens, local_seq]() {
            auto cancelled = [&]() {
                return h->stop_flag.load(std::memory_order_relaxed) ||
                       h->generation_seq.load(std::memory_order_relaxed) != local_seq;
            };

            JNIEnv * threadEnv = nullptr;
            bool attached = false;
            if (gJvm->GetEnv((void **)&threadEnv, JNI_VERSION_1_6) != JNI_OK) {
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) attached = true;
            }

            // Ensure we always finish + release the global callback ref.
            struct FinishGuard {
                JNIEnv * env = nullptr;
                bool attached = false;
                jobject gcb = nullptr;
                jmethodID midOnFinished = nullptr;
                ~FinishGuard() {
                    if (env && gcb) {
                        if (midOnFinished) env->CallVoidMethod(gcb, midOnFinished);
                        env->DeleteGlobalRef(gcb);
                    }
                    if (attached && env) {
                        gJvm->DetachCurrentThread();
                    }
                }
            } guard;
            guard.env = threadEnv;
            guard.attached = attached;
            guard.gcb = gcb;
            guard.midOnFinished = midOnFinished;

            if (!threadEnv) {
                ALOGE("MemoryWorker | ❌ 无法获取线程 Env");
                return;
            }

            std::string out_accum;
            out_accum.reserve(4096);
            auto emit_piece = [&](const std::string & piece) {
                if (!midOnTokenCandidates) return;
                jclass strCls = threadEnv->FindClass("java/lang/String");
                jobjectArray arr = threadEnv->NewObjectArray(1, strCls, nullptr);
                jstring js = new_jstring_utf8_lenient(threadEnv, piece);
                threadEnv->SetObjectArrayElement(arr, 0, js);
                threadEnv->DeleteLocalRef(js);
                threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                threadEnv->DeleteLocalRef(arr);
                threadEnv->DeleteLocalRef(strCls);
            };

            llama_context * ctx = nullptr;
            llama_model   * model = nullptr;
            llama_context_params cparams{};
            {
                std::lock_guard<std::mutex> lock(h->mtx);
                ctx = h->ctx;
                model = h->model;
                cparams = h->cparams;
            }
            if (!ctx || !model || cancelled()) {
                const char * msg = "MemoryWorker | ❌ Context 或 Model 为空";
                ALOGE("%s", msg);
                if (midOnError && threadEnv) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
                return;
            }

            const auto * vocab = llama_model_get_vocab(model);
            ensure_control_tokens(h, vocab);

            // Use a non-zero sequence when available, so this doesn't evict seq0 (interactive KV cache).
            llama_seq_id work_seq = 0;
            const int n_seq_max = (int) cparams.n_seq_max;
            if (n_seq_max > 1) {
                work_seq = (llama_seq_id) (n_seq_max - 1);
                if (work_seq == 0) work_seq = 1;
            }

            llama_memory_t kv_mem = llama_get_memory(ctx);
            llama_memory_seq_rm(kv_mem, work_seq, -1, -1);

            std::vector<llama_token> toks(std::max<size_t>(256, prompt.size() + 128));
            int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int) toks.size(), false, true);
            if (n_prompt <= 0) {
                const char * msg = "MemoryWorker | ❌ tokenize 失败";
                ALOGE("%s", msg);
                if (midOnError && threadEnv) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
                return;
            }
            toks.resize(n_prompt);

            // Incremental prefill into work_seq: reuse common prefix from seq0 whenever possible.
            int n_batch = (int) cparams.n_batch;
            if (n_batch <= 0) n_batch = 256;
            size_t common_prefix = 0;
            {
                std::lock_guard<std::mutex> lock(h->mtx);
                if (h->disable_kv_reuse.load(std::memory_order_relaxed)) {
                    common_prefix = 0;
                } else {
                    RadixLookupResult lookup = radix_lookup(h->tree_root, toks.data(), (int) toks.size());
                    common_prefix = (size_t) lookup.matched_length;
                    if (common_prefix > h->current_tokens.size()) {
                        common_prefix = h->current_tokens.size();
                    }
                }
            }

            if (common_prefix > 0) {
                llama_memory_seq_cp(kv_mem, 0, work_seq, 0, -1);
                llama_memory_seq_rm(kv_mem, work_seq, (llama_pos) common_prefix, -1);
            }

            for (int i = (int) common_prefix; i < (int) toks.size(); i += n_batch) {
                if (cancelled()) return;
                int n_eval = (int) toks.size() - i;
                if (n_eval > n_batch) n_eval = n_batch;
                llama_batch batch = llama_batch_init(n_eval, 0, 1);
                batch.n_tokens = n_eval;
                for (int j = 0; j < n_eval; ++j) {
                    batch.token[j] = toks[i + j];
                    batch.pos[j] = (llama_pos) (i + j);
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = work_seq;
                    batch.logits[j] = (j == n_eval - 1);
                }
                if (llama_decode(ctx, batch) != 0) {
                    llama_batch_free(batch);
                    const char * msg = "MemoryWorker | ❌ prefill llama_decode 失败";
                    ALOGE("%s", msg);
                    if (midOnError && threadEnv) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    return;
                }
                llama_batch_free(batch);
            }

            // Greedy decode loop (explicit argmax over logits)
            llama_pos cur_pos = (llama_pos) toks.size();
            const int n_vocab = llama_vocab_n_tokens(vocab);

            for (int step = 0; step < maxTokens; ++step) {
                if (cancelled()) break;

                // logits are for the last token of the previous decode
                const float * logits = llama_get_logits_ith(ctx, -1);
                if (!logits) break;
                int best_i = 0;
                float best_v = logits[0];
                for (int i = 1; i < n_vocab; ++i) {
                    const float v = logits[i];
                    if (v > best_v) {
                        best_v = v;
                        best_i = i;
                    }
                }
                const llama_token next_tok = (llama_token) best_i;
                if (next_tok == llama_token_eos(vocab) || next_tok == llama_token_eot(vocab)) break;

                char buf[256];
                int n_p = token_to_piece_with_control(vocab, h, next_tok, buf, (int) sizeof(buf));
                std::string piece(buf, n_p > 0 ? n_p : 0);
                if (!piece.empty()) {
                    out_accum.append(piece);
                    emit_piece(piece);
                }

                // Stop once <|im_end|> is observed.
                if (out_accum.find("<|im_end|>") != std::string::npos) break;

                llama_batch batch = llama_batch_init(1, 0, 1);
                batch.n_tokens = 1;
                batch.token[0] = next_tok;
                batch.pos[0] = cur_pos++;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = work_seq;
                batch.logits[0] = true;

                if (llama_decode(ctx, batch) != 0) {
                    llama_batch_free(batch);
                    const char * msg = "MemoryWorker | ❌ decode llama_decode 失败";
                    ALOGE("%s", msg);
                    if (midOnError && threadEnv) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    break;
                }
                llama_batch_free(batch);
            }
        });
    }

    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_benchmarkDecode(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jPrompt, jint maxDecodeSteps, jobject jCallback) {
    ALOGI("【LogicGen】benchmarkDecode 调用 | maxDecodeSteps=%d", (int) maxDecodeSteps);
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h) return -1;
    forceStopAndJoin(h, "benchmarkDecode");

    jobject gcb = env->NewGlobalRef(jCallback);
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID midOnTokenCandidates = env->GetMethodID(cbClass, "onTokenCandidates", "([Ljava/lang/String;)V");
    jmethodID midOnFinished        = env->GetMethodID(cbClass, "onFinished", "()V");
    jmethodID midOnError           = env->GetMethodID(cbClass, "onError", "(Ljava/lang/String;)V");

    const char * cprompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string prompt(cprompt ? cprompt : "");
    if (cprompt) env->ReleaseStringUTFChars(jPrompt, cprompt);

    const auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        h->stop_flag.store(false);
        h->gen_thread = std::thread([h, gcb, midOnTokenCandidates, midOnFinished, midOnError, prompt, maxDecodeSteps, start_time, local_seq]() {
            auto cancelled = [&]() {
                return h->stop_flag.load(std::memory_order_relaxed) ||
                       h->generation_seq.load(std::memory_order_relaxed) != local_seq;
            };

            JNIEnv * threadEnv = nullptr;
            bool attached = false;

            auto sendError = [&](const char * msg) {
                if (threadEnv && midOnError) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
            };

            if (gJvm->GetEnv((void **) &threadEnv, JNI_VERSION_1_6) != JNI_OK) {
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) attached = true;
            }
            if (!threadEnv) {
                ALOGE("benchmarkDecode | ❌ no JNIEnv");
                goto cleanup;
            }

            // everything after this point uses structured early-exit; no jumping over initializations
            do {
                struct llama_context * ctx = nullptr;
                struct llama_model   * model = nullptr;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    ctx = h->ctx;
                    model = h->model;
                }
                if (!ctx || !model || cancelled()) {
                    sendError("invalid_handle");
                    break;
                }

                const auto * vocab = llama_model_get_vocab(model);
                llama_memory_t kv_mem = llama_get_memory(ctx);

                // Keep KV cache positions consistent with our tracked token view.
                const llama_pos kv_pos_max = llama_memory_seq_pos_max(kv_mem, 0);
                const size_t kv_tokens = kv_pos_max >= 0 ? (size_t) kv_pos_max + 1 : 0;
                if (kv_tokens > 0) {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    if (h->current_tokens.size() > kv_tokens) {
                        // Our token view is ahead of KV cache; shrink and resync radix tree.
                        h->current_tokens.resize(kv_tokens);
                        sync_radix_tree_from_tokens(h);
                    }
                }

                long long prefill_time_ms = 0;
                long long decode_time_ms = 0;
                long long ttft_ms = -1;
                int prefill_tokens_count = 0;
                int decode_tokens_count = 0;
                std::vector<float> decode_latencies;
                decode_latencies.reserve(std::max(0, (int) maxDecodeSteps));

                size_t state_before = 0;
                size_t state_after  = 0;

                std::vector<llama_token> decoded_tokens;
                decoded_tokens.reserve(std::max(0, (int) maxDecodeSteps));

                // Tokenize prompt (dynamic capacity; llama_tokenize can return negative when buffer is too small)
                std::vector<llama_token> tokens;
                tokens.resize(std::max<size_t>(256, prompt.length() + 128));
                int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.length(), tokens.data(), (int) tokens.size(), false, true);
                if (n_tokens < 0) {
                    // Required token count is -n_tokens
                    tokens.resize((size_t) (-n_tokens) + 8);
                    n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.length(), tokens.data(), (int) tokens.size(), false, true);
                }
                if (n_tokens <= 0) {
                    sendError("tokenize_failed");
                    break;
                }
                tokens.resize((size_t) n_tokens);

                const int n_ctx = (int) h->cparams.n_ctx;
                int n_seq_max = (int) h->cparams.n_seq_max;
                if (n_seq_max <= 0) n_seq_max = 1;
                const int n_ctx_per_seq = (n_ctx > 0) ? std::max(0, n_ctx / n_seq_max) : 0;
                if (n_ctx_per_seq > 0 && (int) tokens.size() > n_ctx_per_seq) {
                    std::stringstream ss;
                    ss << "prompt_too_long";
                    ss << " tokens_total=" << (unsigned long long) tokens.size();
                    ss << " n_ctx=" << n_ctx;
                    ss << " n_seq_max=" << n_seq_max;
                    ss << " n_ctx_per_seq=" << n_ctx_per_seq;
                    const std::string err = ss.str();
                    sendError(err.c_str());
                    break;
                }

                // KV reuse via radix tree
                size_t common_prefix = 0;
                RadixNode * match_node = nullptr;
                bool need_kv_cleanup = false;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    RadixLookupResult lookup = radix_lookup(h->tree_root, tokens.data(), (int) tokens.size());
                    match_node = lookup.matched_node;
                    common_prefix = (size_t) lookup.matched_length;

                    if (common_prefix > h->current_tokens.size()) {
                        common_prefix = h->current_tokens.size();
                    }
                    if (kv_tokens > 0 && common_prefix > kv_tokens) {
                        common_prefix = kv_tokens;
                    }

                    // If fully matched, roll back one token to force an eval and obtain logits.
                    if (common_prefix >= tokens.size() && !tokens.empty()) {
                        common_prefix = tokens.size() - 1;
                        need_kv_cleanup = true;
                        if (match_node && match_node->parent) {
                            match_node = match_node->parent;
                        }
                    }
                    if (common_prefix < h->current_tokens.size()) {
                        need_kv_cleanup = true;
                    }
                }
                if (need_kv_cleanup) {
                    llama_memory_seq_rm(kv_mem, 0, (llama_pos) common_prefix, -1);
                    std::lock_guard<std::mutex> lock(h->mtx);
                    h->current_tokens.resize(common_prefix);
                    h->cursor_node = match_node ? match_node : h->tree_root;
                }

                const size_t n_new = tokens.size() - common_prefix;
                // Track KV state size trend around prefill.
                // NOTE: llama_state_get_size reflects the serialized state size (best-effort proxy for KV growth).
                state_before = llama_state_get_size(ctx);
                auto t_prefill_start = std::chrono::high_resolution_clock::now();
                if (n_new > 0) {
                    int n_batch = (int) h->cparams.n_batch;
                    if (n_batch <= 0) n_batch = 256;

                    // For long prompts, decoding a single huge batch can spike memory.
                    // Prefer n_ubatch (micro-batch) when available; otherwise cap chunk size.
                    int n_ubatch = (int) h->cparams.n_ubatch;
                    if (n_ubatch <= 0) n_ubatch = std::min(n_batch, 512);
                    const int n_chunk = std::max(1, std::min(n_batch, n_ubatch));

                    bool prefill_ok = true;

                    // Avoid ggml_abort inside llama_decode when n_new > n_batch by chunking.
                    // Also avoid large single-step allocations by using n_chunk.
                    for (size_t off = 0; off < n_new; off += (size_t) n_chunk) {
                        const size_t n_eval = std::min(n_new - off, (size_t) n_chunk);
                        llama_batch batch = llama_batch_init((int) n_eval, 0, 1);
                        batch.n_tokens = (int32_t) n_eval;
                        for (int j = 0; j < (int) n_eval; ++j) {
                            batch.token[j] = tokens[common_prefix + off + (size_t) j];
                            batch.pos[j] = (llama_pos) (common_prefix + off + (size_t) j);
                            batch.n_seq_id[j] = 1;
                            batch.seq_id[j][0] = 0;
                            // Only request logits for the final token of the *entire* prefill.
                            batch.logits[j] = (off + (size_t) j + 1 == n_new);
                        }
                        const int rc = llama_decode(ctx, batch);
                        llama_batch_free(batch);
                        if (rc != 0) {
                            std::stringstream ss;
                            ss << "prefill_decode_failed";
                            ss << " rc=" << rc;
                            ss << " off=" << (unsigned long long) off;
                            ss << " n_eval=" << (unsigned long long) n_eval;
                            ss << " n_new=" << (unsigned long long) n_new;
                            ss << " common_prefix=" << (unsigned long long) common_prefix;
                            ss << " tokens_total=" << (unsigned long long) tokens.size();
                            ss << " n_batch=" << n_batch;
                            ss << " n_ubatch=" << n_ubatch;
                            ss << " n_chunk=" << n_chunk;
                            ss << " n_ctx=" << n_ctx;
                            ss << " n_seq_max=" << n_seq_max;
                            ss << " n_ctx_per_seq=" << n_ctx_per_seq;
                            const std::string err = ss.str();
                            sendError(err.c_str());
                            prefill_ok = false;
                            break;
                        }
                        if (cancelled()) break;
                    }
                    if (cancelled()) break;
                    if (!prefill_ok) {
                        // Roll back to the last known consistent prefix.
                        llama_memory_seq_rm(kv_mem, 0, (llama_pos) common_prefix, -1);
                        std::lock_guard<std::mutex> lock(h->mtx);
                        h->current_tokens.resize(common_prefix);
                        h->cursor_node = match_node ? match_node : h->tree_root;
                        break;
                    }
                }
                auto t_prefill_end = std::chrono::high_resolution_clock::now();
                prefill_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();
                prefill_tokens_count = (int) n_new;

                state_after = llama_state_get_size(ctx);

                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    h->current_tokens = tokens;
                    if (n_new > 0 && match_node) {
                        RadixNode * new_cursor = radix_insert(
                            match_node,
                            tokens.data() + common_prefix,
                            (int) n_new,
                            (int) common_prefix
                        );
                        h->cursor_node = new_cursor;
                        refresh_tree_hot_status(h->tree_root, h->current_tokens, h->system_prompt_token_count);
                    }
                }

                // Decode loop (single sequence, greedy). Clamp by remaining context (n_ctx_per_seq) to prevent invalid positions.
                const int prompt_tokens_total = (int) (common_prefix + n_new);
                int steps = std::max(0, (int) maxDecodeSteps);
                if (n_ctx_per_seq > 0) {
                    const int remaining = std::max(0, n_ctx_per_seq - prompt_tokens_total);
                    if (steps > remaining) steps = remaining;
                }
                llama_sampler * smpl = llama_sampler_init_greedy();
                auto t_decode_total_start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < steps; ++i) {
                    if (cancelled()) break;

                    const llama_token next_tok = llama_sampler_sample(smpl, ctx, -1);
                    llama_sampler_accept(smpl, next_tok);

                    if (next_tok == llama_vocab_eos(vocab) || llama_vocab_is_control(vocab, next_tok)) {
                        break;
                    }

                    decoded_tokens.push_back(next_tok);

                    llama_batch batch = llama_batch_init(1, 0, 1);
                    batch.n_tokens = 1;
                    batch.token[0] = next_tok;
                    batch.pos[0] = (llama_pos) (common_prefix + n_new + decode_tokens_count);
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = 0;
                    batch.logits[0] = true;

                    auto t_step_start = std::chrono::high_resolution_clock::now();
                    const int rc = llama_decode(ctx, batch);
                    auto t_step_end = std::chrono::high_resolution_clock::now();
                    llama_batch_free(batch);

                    if (rc != 0) {
                        sendError("decode_failed");
                        // Remove any partially decoded tokens to keep KV cache consistent.
                        llama_memory_seq_rm(kv_mem, 0, (llama_pos) (common_prefix + n_new), -1);
                        break;
                    }

                    decode_latencies.push_back(std::chrono::duration<float, std::milli>(t_step_end - t_step_start).count());
                    decode_tokens_count++;

                    if (ttft_ms < 0) {
                        ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_step_end - start_time).count();
                    }
                }
                auto t_decode_total_end = std::chrono::high_resolution_clock::now();
                decode_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - t_decode_total_start).count();
                llama_sampler_free(smpl);

                // Perf-only API: do not leave decoded tokens in KV cache, otherwise future calls see mismatched positions.
                llama_memory_seq_rm(kv_mem, 0, (llama_pos) (common_prefix + n_new), -1);

                std::string output_text_full;
                std::string output_preview;
                bool output_truncated = false;
                if (!decoded_tokens.empty()) {
                    output_text_full = tokens_to_str(model, decoded_tokens.data(), (int) decoded_tokens.size());
                    constexpr size_t kMaxPreviewBytes = 2048;
                    output_truncated = output_text_full.size() > kMaxPreviewBytes;
                    output_preview = output_text_full.substr(0, kMaxPreviewBytes);
                }

                const double prefill_tps = (prefill_time_ms > 0) ? (prefill_tokens_count * 1000.0 / prefill_time_ms) : 0.0;
                const double decode_tps  = (decode_time_ms  > 0) ? (decode_tokens_count  * 1000.0 / decode_time_ms)  : 0.0;
                const long long e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - start_time).count();

                std::stringstream ss_metrics;
                ss_metrics << "__METRICS__{";
                ss_metrics << "\"prefill_ms\":" << prefill_time_ms << ",";
                ss_metrics << "\"prefill_tokens\":" << prefill_tokens_count << ",";
                ss_metrics << "\"prefill_tps\":" << prefill_tps << ",";
                ss_metrics << "\"decode_ms\":" << decode_time_ms << ",";
                ss_metrics << "\"decode_tokens\":" << decode_tokens_count << ",";
                ss_metrics << "\"decode_tps\":" << decode_tps << ",";
                ss_metrics << "\"e2e_ms\":" << e2e_ms << ",";
                ss_metrics << "\"ttft_ms\":" << ttft_ms << ",";
                ss_metrics << "\"state_size_before\":" << (unsigned long long) state_before << ",";
                ss_metrics << "\"state_size_after\":"  << (unsigned long long) state_after  << ",";
                ss_metrics << "\"output_preview\":\"" << json_escape(output_preview) << "\",";
                ss_metrics << "\"output_bytes_total\":" << (unsigned long long) output_text_full.size() << ",";
                ss_metrics << "\"output_truncated\":" << (output_truncated ? "true" : "false") << ",";
                ss_metrics << "\"decode_latencies\":[";
                for (size_t i = 0; i < decode_latencies.size(); ++i) {
                    ss_metrics << decode_latencies[i] << (i + 1 == decode_latencies.size() ? "" : ",");
                }
                ss_metrics << "]}";

                const std::string metrics_str = ss_metrics.str();
                jclass strCls = threadEnv->FindClass("java/lang/String");
                jobjectArray arr = threadEnv->NewObjectArray(1, strCls, nullptr);
                jstring js = new_jstring_utf8_lenient(threadEnv, metrics_str);
                threadEnv->SetObjectArrayElement(arr, 0, js);
                threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                threadEnv->DeleteLocalRef(js);
                threadEnv->DeleteLocalRef(arr);
                threadEnv->DeleteLocalRef(strCls);

                if (midOnFinished) {
                    threadEnv->CallVoidMethod(gcb, midOnFinished);
                }
            } while (false);

            cleanup:
            if (threadEnv) {
                threadEnv->DeleteGlobalRef(gcb);
            }
            if (attached && threadEnv) gJvm->DetachCurrentThread();
        });
    }

    return 0;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_getModelInfoJson(JNIEnv* env, jclass clazz, jlong handlePtr) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !h->model || !h->ctx) {
        return env->NewStringUTF("{}");
    }

    struct llama_model * model = nullptr;
    struct llama_context * ctx = nullptr;
    std::string model_path;
    std::string lora_path;
    llama_context_params cparams{};
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        model = h->model;
        ctx = h->ctx;
        model_path = h->model_path_key;
        lora_path = h->current_lora_path;
        cparams = h->cparams;
    }
    if (!model || !ctx) return env->NewStringUTF("{}");

    char desc_buf[256];
    desc_buf[0] = 0;
    llama_model_desc(model, desc_buf, sizeof(desc_buf));

    const uint64_t model_bytes = llama_model_size(model);
    const uint64_t n_params = llama_model_n_params(model);
    const long long model_file_bytes = file_size_bytes(model_path);

    const auto * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Best-effort metadata keys (may be empty)
    const std::string meta_name = llama_meta_str(model, "general.name");
    const std::string meta_arch = llama_meta_str(model, "general.architecture");
    const std::string meta_file_type = llama_meta_str(model, "general.file_type");

    std::stringstream ss;
    ss << "{";
    ss << "\"model_desc\":\"" << json_escape(desc_buf) << "\",";
    ss << "\"model_path\":\"" << json_escape(model_path) << "\",";
    ss << "\"lora_path\":\"" << json_escape(lora_path) << "\",";
    ss << "\"model_file_bytes\":" << model_file_bytes << ",";
    ss << "\"model_size_bytes\":" << (unsigned long long) model_bytes << ",";
    ss << "\"model_n_params\":" << (unsigned long long) n_params << ",";
    ss << "\"vocab_size\":" << n_vocab << ",";
    ss << "\"meta\":{";
    ss << "\"general.name\":\"" << json_escape(meta_name) << "\",";
    ss << "\"general.architecture\":\"" << json_escape(meta_arch) << "\",";
    ss << "\"general.file_type\":\"" << json_escape(meta_file_type) << "\"";
    ss << "},";
    ss << "\"runtime\":{";
    ss << "\"n_ctx\":" << cparams.n_ctx << ",";
    ss << "\"n_batch\":" << cparams.n_batch << ",";
    ss << "\"n_ubatch\":" << cparams.n_ubatch << ",";
    ss << "\"n_threads\":" << cparams.n_threads << ",";
    ss << "\"n_threads_batch\":" << cparams.n_threads_batch << ",";
    ss << "\"n_seq_max\":" << cparams.n_seq_max;
    ss << "}";
    ss << "}";

    const std::string out = ss.str();
    return env->NewStringUTF(out.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_generatePhraseCandidatesSpliceMemoryFromKvFile(
        JNIEnv * env,
        jclass,
        jlong handlePtr,
        jstring jPrefixBeforeMemory,
        jstring jMemory,
        jstring jSuffixAfterMemory,
        jstring jMemoryKvPath,
        jint n_candidates,
        jobject jCallback) {
    ALOGI("【LogicGen】KV-Splice(FromFile) 插入 Memory 生成短语候选调用 (实验)");
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h) return -1;
    forceStopAndJoin(h, "generatePhraseCandidatesSpliceMemoryFromKvFile");

    jobject gcb = env->NewGlobalRef(jCallback);
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID midOnTokenCandidates = env->GetMethodID(cbClass, "onTokenCandidates", "([Ljava/lang/String;)V");
    jmethodID midOnFinished = env->GetMethodID(cbClass, "onFinished", "()V");
    jmethodID midOnError = env->GetMethodID(cbClass, "onError", "(Ljava/lang/String;)V");

    auto jstr_to_std = [&](jstring js) -> std::string {
        if (!js) return "";
        const char * c = env->GetStringUTFChars(js, nullptr);
        std::string out(c ? c : "");
        if (c) env->ReleaseStringUTFChars(js, c);
        return out;
    };

    const std::string prefix = jstr_to_std(jPrefixBeforeMemory);
    const std::string memory = jstr_to_std(jMemory);
    const std::string suffix = jstr_to_std(jSuffixAfterMemory);
    const std::string kv_path = jstr_to_std(jMemoryKvPath);

    const std::string base_prompt = prefix + suffix;
    const std::string full_prompt = prefix + memory + suffix;

    const long long kv_size = file_size_bytes(kv_path);
    ALOGI("KV-Splice(FromFile) | kv=%s size=%lldB memLen=%zu", kv_path.c_str(), kv_size, memory.size());

    auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        h->stop_flag.store(false);
        h->gen_thread = std::thread([h, gcb, midOnTokenCandidates, midOnFinished, midOnError, base_prompt, full_prompt, prefix, memory, suffix, kv_path, n_candidates, start_time, local_seq]() {
            auto cancelled = [&]() {
                return h->stop_flag.load(std::memory_order_relaxed) ||
                       h->generation_seq.load(std::memory_order_relaxed) != local_seq;
            };

            struct llama_context * ctx = nullptr;
            struct llama_model * model = nullptr;
            JNIEnv * threadEnv = nullptr;
            bool attached = false;

            long long base_prefill_ms = 0;
            long long splice_shift_ms = 0;
            long long mem_load_ms = 0;
            long long tail_recompute_ms = 0;

            bool batch_initialized = false;
            llama_batch batch{};

            if (gJvm->GetEnv((void **)&threadEnv, JNI_VERSION_1_6) != JNI_OK) {
                if (gJvm->AttachCurrentThread(&threadEnv, nullptr) == JNI_OK) attached = true;
            }
            if (!threadEnv) {
                ALOGE("KV-Splice(FromFile) | ❌ 无法获取线程 Env");
                goto finish;
            }

            {
                std::lock_guard<std::mutex> lock(h->mtx);
                ctx = h->ctx;
                model = h->model;
            }
            if (!ctx || !model || cancelled()) {
                const char * msg = "KV-Splice(FromFile) | ❌ Context 或 Model 为空";
                ALOGE("%s", msg);
                if (midOnError) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
                goto finish;
            }

            if (kv_path.empty() || !file_exists(kv_path)) {
                const char * msg = "KV-Splice(FromFile) | ❌ KV 文件不存在";
                ALOGE("%s", msg);
                if (midOnError) {
                    jstring jerr = threadEnv->NewStringUTF(msg);
                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                    threadEnv->DeleteLocalRef(jerr);
                }
                goto finish;
            }

            {
                const auto * vocab = llama_model_get_vocab(model);
                ensure_control_tokens(h, vocab);
                llama_memory_t kv_mem = llama_get_memory(ctx);

                if (!llama_memory_can_shift(kv_mem)) {
                    const char * msg = "KV-Splice(FromFile) | ❌ llama_memory_can_shift=false";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }

                auto tokenize_str = [&](const std::string & s, std::vector<llama_token> & out) -> bool {
                    out.assign(std::max<size_t>(256, s.size() + 128), 0);
                    int n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), out.data(), (int) out.size(), false, true);
                    if (n < 0) {
                        out.resize((size_t) (-n));
                        n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), out.data(), (int) out.size(), false, true);
                    }
                    if (n <= 0) return false;
                    out.resize((size_t) n);
                    return true;
                };

                std::vector<llama_token> base_tokens, full_tokens;
                if (!tokenize_str(base_prompt, base_tokens) || !tokenize_str(full_prompt, full_tokens)) {
                    const char * msg = "KV-Splice(FromFile) | ❌ tokenize 失败";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }

                // Compute insertion token span by diffing base_tokens vs full_tokens:
                size_t prefix_len = 0;
                while (prefix_len < base_tokens.size() && prefix_len < full_tokens.size() && base_tokens[prefix_len] == full_tokens[prefix_len]) {
                    prefix_len++;
                }
                size_t suffix_len = 0;
                while (suffix_len < (base_tokens.size() - prefix_len) && suffix_len < (full_tokens.size() - prefix_len) &&
                       base_tokens[base_tokens.size() - 1 - suffix_len] == full_tokens[full_tokens.size() - 1 - suffix_len]) {
                    suffix_len++;
                }
                const size_t insert_len = (full_tokens.size() >= prefix_len + suffix_len) ? (full_tokens.size() - prefix_len - suffix_len) : 0;

                // Ensure seq0 holds the base_prompt KV (cached across memory variants)
                bool base_ready = false;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    base_ready = (h->current_tokens.size() == base_tokens.size()) &&
                                 std::equal(h->current_tokens.begin(), h->current_tokens.end(), base_tokens.begin());
                }
                if (!base_ready) {
                    auto t_prefill_start = std::chrono::high_resolution_clock::now();

                    // Cold baseline prefill into seq0 using the same reuse logic as prefillPrompt.
                    size_t common_prefix = 0;
                    RadixNode * match_node = nullptr;
                    bool need_kv_cleanup = false;
                    {
                        std::lock_guard<std::mutex> lock(h->mtx);
                        if (h->disable_kv_reuse.load(std::memory_order_relaxed)) {
                            match_node = h->tree_root;
                            common_prefix = 0;
                            need_kv_cleanup = !h->current_tokens.empty();
                        } else {
                            RadixLookupResult lookup = radix_lookup(h->tree_root, base_tokens.data(), (int) base_tokens.size());
                            match_node = lookup.matched_node;
                            common_prefix = (size_t) lookup.matched_length;
                        }
                        if (common_prefix > h->current_tokens.size()) common_prefix = h->current_tokens.size();
                        if (common_prefix < h->current_tokens.size()) need_kv_cleanup = true;
                    }
                    if (need_kv_cleanup) {
                        llama_memory_seq_rm(kv_mem, -1, (llama_pos) common_prefix, -1);
                        std::lock_guard<std::mutex> lock(h->mtx);
                        h->current_tokens.resize(common_prefix);
                        h->cursor_node = match_node ? match_node : h->tree_root;
                    }

                    size_t n_new = base_tokens.size() - common_prefix;
                    if (n_new > 0) {
                        int n_batch = (int) h->cparams.n_batch;
                        if (n_batch <= 0) n_batch = 256;
                        int n_ubatch = (int) h->cparams.n_ubatch;
                        if (n_ubatch <= 0) n_ubatch = std::min(n_batch, 512);
                        const int n_chunk = std::max(1, std::min(n_batch, n_ubatch));
                        for (size_t off = 0; off < n_new; off += (size_t) n_chunk) {
                            const size_t n_eval = std::min(n_new - off, (size_t) n_chunk);
                            llama_batch chunk = llama_batch_init((int) n_eval, 0, 1);
                            chunk.n_tokens = (int32_t) n_eval;
                            for (int j = 0; j < (int) n_eval; ++j) {
                                chunk.token[j] = base_tokens[common_prefix + off + (size_t) j];
                                chunk.pos[j] = (llama_pos) (common_prefix + off + (size_t) j);
                                chunk.n_seq_id[j] = 1;
                                chunk.seq_id[j][0] = 0;
                                chunk.logits[j] = (off + (size_t) j + 1 == n_new);
                            }
                            const int rc = llama_decode(ctx, chunk);
                            llama_batch_free(chunk);
                            if (rc != 0) {
                                const char * msg = "KV-Splice(FromFile) | ❌ base prefill llama_decode 失败";
                                ALOGE("%s", msg);
                                if (midOnError) {
                                    jstring jerr = threadEnv->NewStringUTF(msg);
                                    threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                                    threadEnv->DeleteLocalRef(jerr);
                                }
                                goto finish;
                            }
                            if (cancelled()) break;
                        }
                    }
                    auto t_prefill_end = std::chrono::high_resolution_clock::now();
                    base_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();

                    {
                        std::lock_guard<std::mutex> lock(h->mtx);
                        h->current_tokens = base_tokens;
                        sync_radix_tree_from_tokens(h);
                    }
                }

                // Sequences used for splice work.
                const llama_seq_id work_seq = (llama_seq_id) (h->cparams.n_seq_max - 1);
                const llama_seq_id mem_seq = (llama_seq_id) (h->cparams.n_seq_max - 2);

                const int max_candidates = std::max(1, (int) h->cparams.n_seq_max - 3);
                const int k_pick = std::min(std::min((int) n_candidates, 20), max_candidates);
                if (k_pick <= 0) {
                    const char * msg = "KV-Splice(FromFile) | ❌ k_pick 计算失败";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }

                llama_memory_seq_rm(kv_mem, work_seq, -1, -1);
                llama_memory_seq_cp(kv_mem, 0, work_seq, 0, -1);

                const llama_pos p_ins = (llama_pos) prefix_len;
                const llama_pos ins_len = (llama_pos) insert_len;
                const llama_pos del_len = (llama_pos) (base_tokens.size() - prefix_len - suffix_len);
                const llama_pos delta = ins_len - del_len;
                const llama_pos full_len = (llama_pos) full_tokens.size();

                auto t_shift_start = std::chrono::high_resolution_clock::now();
                if (del_len > 0) {
                    llama_memory_seq_rm(kv_mem, work_seq, p_ins, p_ins + del_len);
                }
                if (delta != 0) {
                    llama_memory_seq_add(kv_mem, work_seq, p_ins + del_len, -1, delta);
                }
                auto t_shift_end = std::chrono::high_resolution_clock::now();
                splice_shift_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_shift_end - t_shift_start).count();

                auto t_load_start = std::chrono::high_resolution_clock::now();
                llama_memory_seq_rm(kv_mem, mem_seq, -1, -1);

                std::vector<llama_token> loaded_tokens;
                const int cap = std::max(64, (int) h->cparams.n_ctx);
                loaded_tokens.resize((size_t) cap);
                size_t loaded_count = 0;
                const size_t nread = llama_state_seq_load_file(ctx, kv_path.c_str(), mem_seq, loaded_tokens.data(), loaded_tokens.size(), &loaded_count);
                if (nread == 0 || loaded_count == 0) {
                    const char * msg = "KV-Splice(FromFile) | ❌ memory KV load failed";
                    ALOGE("%s", msg);
                    if (midOnError) {
                        jstring jerr = threadEnv->NewStringUTF(msg);
                        threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                        threadEnv->DeleteLocalRef(jerr);
                    }
                    goto finish;
                }

                const llama_pos use_len = (llama_pos) std::min((size_t) ins_len, loaded_count);
                if ((size_t) use_len != (size_t) ins_len) {
                    ALOGW("KV-Splice(FromFile) | ⚠️ token_count mismatch | ins_len=%d loaded=%zu use=%d", (int) ins_len, loaded_count, (int) use_len);
                }

                if (p_ins > 0) {
                    llama_memory_seq_add(kv_mem, mem_seq, 0, -1, p_ins);
                }
                auto t_load_end = std::chrono::high_resolution_clock::now();
                mem_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t_load_start).count();

                if (use_len > 0) {
                    llama_memory_seq_cp_overlay(kv_mem, mem_seq, work_seq, p_ins, p_ins + use_len);
                }

                if (kv_splice_debug_enabled()) {
                    ALOGI("KV-Splice(FromFile) | PROOF loaded=%zu use_len=%d overlay=[%d,%d) insert_prefill_calls=0",
                          loaded_count, (int) use_len, (int) p_ins, (int) (p_ins + use_len));
                }

                // 4) Tail recompute for last-token logits.
                if (full_len > 0) {
                    const llama_pos last_pos = full_len - 1;
                    llama_token last_token = full_tokens.back();

                    auto t_tail_start = std::chrono::high_resolution_clock::now();
                    llama_memory_seq_rm(kv_mem, work_seq, last_pos, last_pos + 1);
                    llama_batch tail_batch = llama_batch_init(1, 0, 1);
                    tail_batch.n_tokens = 1;
                    tail_batch.token[0] = last_token;
                    tail_batch.pos[0] = last_pos;
                    tail_batch.n_seq_id[0] = 1;
                    tail_batch.seq_id[0][0] = work_seq;
                    tail_batch.logits[0] = true;
                    if (llama_decode(ctx, tail_batch) != 0) {
                        llama_batch_free(tail_batch);
                        const char * msg = "KV-Splice(FromFile) | ❌ tail recompute llama_decode 失败";
                        ALOGE("%s", msg);
                        if (midOnError) {
                            jstring jerr = threadEnv->NewStringUTF(msg);
                            threadEnv->CallVoidMethod(gcb, midOnError, jerr);
                            threadEnv->DeleteLocalRef(jerr);
                        }
                        goto finish;
                    }
                    llama_batch_free(tail_batch);
                    auto t_tail_end = std::chrono::high_resolution_clock::now();
                    tail_recompute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_tail_end - t_tail_start).count();
                }

                auto * logits = llama_get_logits_ith(ctx, -1);
                int n_vocab = llama_vocab_n_tokens(vocab);
                const float PROMPT_REP_PENALTY = 1.1f;
                {
                    std::lock_guard<std::mutex> lock(h->mtx);
                    const int rep_start = std::max(0, h->reusable_prefix_token_count);
                    for (size_t ti = rep_start; ti < h->current_tokens.size(); ++ti) {
                        llama_token t = h->current_tokens[ti];
                        if (logits[t] > 0) logits[t] /= PROMPT_REP_PENALTY;
                        else logits[t] *= PROMPT_REP_PENALTY;
                    }
                }

                const uint32_t seed_base = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count();
                for (int i = 1; i <= k_pick; ++i) {
                    llama_memory_seq_rm(kv_mem, (llama_seq_id) i, -1, -1);
                    llama_memory_seq_cp(kv_mem, work_seq, (llama_seq_id) i, 0, -1);
                }

                struct SeqState {
                    std::string phrase;
                    std::vector<llama_token> gen_tokens;
                    llama_token last_token;
                    bool active = true;
                };
                std::vector<SeqState> states(k_pick);

                batch = llama_batch_init(std::max(k_pick, 1), 0, 1);
                batch_initialized = true;
                batch.n_tokens = k_pick;
                const llama_pos prompt_end_pos = full_len;
                for (int i = 0; i < k_pick; ++i) {
                    // Build a sampler chain per candidate: top_k → top_p → temp → dist
                    auto sparams = llama_sampler_chain_default_params();
                    llama_sampler* first_smpl = llama_sampler_chain_init(sparams);
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_top_k(20));
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_top_p(0.8f, 1));
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_temp(0.7f));
                    llama_sampler_chain_add(first_smpl, llama_sampler_init_dist(seed_base + (uint32_t)i));
                    llama_token first_tok = llama_sampler_sample(first_smpl, ctx, -1);
                    llama_sampler_free(first_smpl);

                    states[i].last_token = first_tok;
                    states[i].gen_tokens.push_back(first_tok);

                    char buf[256];
                    int n_p = token_to_piece_with_control(vocab, h, first_tok, buf, 256);
                    if (n_p > 0) states[i].phrase.append(buf, n_p);

                    batch.token[i] = first_tok;
                    batch.pos[i] = prompt_end_pos;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = (llama_seq_id) (i + 1);
                    batch.logits[i] = true;
                }

                long long decode_time_ms = 0;
                int decode_tokens_count = 0;
                std::vector<float> decode_latencies;

                auto t_decode_total_start = std::chrono::high_resolution_clock::now();
                std::vector<llama_sampler*> candidate_samplers(k_pick, nullptr);
                for (int i = 0; i < k_pick; ++i) {
                    auto sparams = llama_sampler_chain_default_params();
                    candidate_samplers[i] = llama_sampler_chain_init(sparams);
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_penalties(
                        /*penalty_last_n=*/16, /*penalty_repeat=*/1.2f,
                        /*penalty_freq=*/0.0f, /*penalty_present=*/0.0f));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_top_k(20));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_top_p(0.8f, 1));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_temp(0.7f));
                    llama_sampler_chain_add(candidate_samplers[i], llama_sampler_init_dist(seed_base + 100u + (uint32_t)i));
                    llama_sampler_accept(candidate_samplers[i], states[i].gen_tokens[0]);
                }
                const int MAX_PHRASE_LEN = 16;
                int active_count = k_pick;

                for (int step = 0; step < MAX_PHRASE_LEN && active_count > 0; ++step) {
                    if (cancelled()) break;
                    auto t_step_start = std::chrono::high_resolution_clock::now();
                    if (llama_decode(ctx, batch) != 0) break;
                    auto t_step_end = std::chrono::high_resolution_clock::now();
                    decode_latencies.push_back((float) std::chrono::duration_cast<std::chrono::microseconds>(t_step_end - t_step_start).count() / 1000.0f);

                    llama_batch next_batch = llama_batch_init(k_pick, 0, 1);
                    next_batch.n_tokens = 0;

                    for (int i = 0; i < k_pick; ++i) {
                        if (!states[i].active) continue;
                        llama_token next_tok = llama_sampler_sample(candidate_samplers[i], ctx, i);
                        llama_sampler_accept(candidate_samplers[i], next_tok);

                        decode_tokens_count++;
                        if (next_tok == llama_token_eos(vocab) || next_tok == llama_token_eot(vocab)) {
                            states[i].active = false;
                            active_count--;
                            continue;
                        }

                        char piece[256];
                        int n_piece = token_to_piece_with_control(vocab, h, next_tok, piece, 256);
                        if (n_piece > 0) states[i].phrase.append(piece, n_piece);

                        states[i].last_token = next_tok;
                        states[i].gen_tokens.push_back(next_tok);

                        int idx = next_batch.n_tokens;
                        next_batch.token[idx] = next_tok;
                        next_batch.pos[idx] = prompt_end_pos + step + 1;
                        next_batch.n_seq_id[idx] = 1;
                        next_batch.seq_id[idx][0] = (llama_seq_id) (i + 1);
                        next_batch.logits[idx] = true;
                        next_batch.n_tokens++;
                    }

                    llama_batch_free(batch);
                    batch = next_batch;
                }

                for (int i = 0; i < k_pick; ++i) {
                    if (candidate_samplers[i]) llama_sampler_free(candidate_samplers[i]);
                }
                auto t_decode_total_end = std::chrono::high_resolution_clock::now();
                decode_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - t_decode_total_start).count();

                std::vector<std::string> result_phrases;
                for (int i = 0; i < k_pick; ++i) {
                    std::string & s = states[i].phrase;
                    if (!s.empty() && is_valid_utf8(s)) result_phrases.push_back(s);
                }

                double decode_tps = (decode_time_ms > 0) ? (decode_tokens_count * 1000.0 / decode_time_ms) : 0;
                long long e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_total_end - start_time).count();

                std::stringstream ss_metrics;
                ss_metrics << "__METRICS__{";
                ss_metrics << "\"mode\":\"splice_from_file\",";
                ss_metrics << "\"base_prefill_ms\":" << base_prefill_ms << ",";
                ss_metrics << "\"splice_shift_ms\":" << splice_shift_ms << ",";
                ss_metrics << "\"mem_load_ms\":" << mem_load_ms << ",";
                ss_metrics << "\"tail_recompute_ms\":" << tail_recompute_ms << ",";
                ss_metrics << "\"prefill_ms\":" << (base_prefill_ms + splice_shift_ms + mem_load_ms + tail_recompute_ms) << ",";
                ss_metrics << "\"prefill_tokens\":" << (int) insert_len << ",";
                ss_metrics << "\"mem_kv_loaded\":" << (int) loaded_count << ",";
                ss_metrics << "\"overlay_tokens\":" << (int) use_len << ",";
                ss_metrics << "\"insert_prefill_calls\":0,";
                ss_metrics << "\"decode_ms\":" << decode_time_ms << ",";
                ss_metrics << "\"decode_tokens\":" << decode_tokens_count << ",";
                ss_metrics << "\"decode_tps\":" << decode_tps << ",";
                ss_metrics << "\"e2e_ms\":" << e2e_ms << ",";
                ss_metrics << "\"p_ins\":" << (int) p_ins << ",";
                ss_metrics << "\"memory_tokens\":" << (int) insert_len << ",";
                ss_metrics << "\"suffix_tokens\":" << (int) suffix_len << ",";
                ss_metrics << "\"decode_latencies\":[";
                for (size_t i = 0; i < decode_latencies.size(); ++i) {
                    ss_metrics << decode_latencies[i] << (i == decode_latencies.size() - 1 ? "" : ",");
                }
                ss_metrics << "]}";

                result_phrases.push_back(ss_metrics.str());

                if (!result_phrases.empty()) {
                    jclass strCls = threadEnv->FindClass("java/lang/String");
                    jobjectArray arr = threadEnv->NewObjectArray(result_phrases.size(), strCls, nullptr);
                    for (size_t i = 0; i < result_phrases.size(); i++) {
                        jstring js = new_jstring_utf8_lenient(threadEnv, result_phrases[i]);
                        threadEnv->SetObjectArrayElement(arr, i, js);
                        threadEnv->DeleteLocalRef(js);
                    }
                    threadEnv->CallVoidMethod(gcb, midOnTokenCandidates, arr);
                    threadEnv->DeleteLocalRef(arr);
                    threadEnv->DeleteLocalRef(strCls);
                }
            }

        finish:
            if (batch_initialized) {
                llama_batch_free(batch);
            }
            if (threadEnv) {
                threadEnv->CallVoidMethod(gcb, midOnFinished);
                threadEnv->DeleteGlobalRef(gcb);
            }
            if (attached && threadEnv) gJvm->DetachCurrentThread();
        });
    }
    return 0;
}