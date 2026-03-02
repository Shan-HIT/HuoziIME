#include "imem_core.h"

static std::string jstring_to_string(JNIEnv * env, jstring js) {
    if (!env || !js) return "";
    const char * c = env->GetStringUTFChars(js, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(js, c);
    return out;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_saveKVCacheSnapshot(JNIEnv* env, jclass, jlong handlePtr, jstring jText, jstring jSavePath) {
    ALOGI("【LogicCache】保存 KV Cache 调用");
    auto t_total_start = std::chrono::high_resolution_clock::now();

    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return false;

    const char* ctext = env->GetStringUTFChars(jText, nullptr);
    const char* csave = env->GetStringUTFChars(jSavePath, nullptr);
    std::string text(ctext);
    std::string save_path(csave);
    env->ReleaseStringUTFChars(jText, ctext);
    env->ReleaseStringUTFChars(jSavePath, csave);

    ALOGI("【KVSave】开始生成 KV Cache... 目标路径: %s, 文本长度: %zu", save_path.c_str(), text.length());

    forceStopAndJoin(h, "saveKVCacheSnapshot");
    std::lock_guard<std::mutex> lock(h->mtx);

    if (!h->ctx) {
        ALOGI("【KVSave】错误: ctx 为空");
        return false;
    }
    if (!h->model) {
        ALOGI("【KVSave】错误: model 为空");
        return false;
    }

    h->is_system_prompt_cached = false;
    h->current_tokens.clear();
    llama_memory_seq_rm(llama_get_memory(h->ctx), -1, 0, -1);

    const auto* vocab = llama_model_get_vocab(h->model);
    std::vector<llama_token> tokens(text.length() + 100);
    int n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int)tokens.size(), false, true);
    if (n_tokens <= 0) {
        ALOGI("【KVSave】Tokenize 失败，返回: %d", n_tokens);
        return false;
    }
    tokens.resize(n_tokens);

    llama_set_embeddings(h->ctx, false);

    int n_batch = h->cparams.n_batch;
    if (n_batch <= 0) n_batch = 256;

    ALOGI("【KVSave】开始 Decode (Tokens: %d, Batch: %d)...", n_tokens, n_batch);
    ALOG_PFD("[KVSave] Prefill Start | No KV Reuse (Cold Start)");

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = n_tokens - i;
        if (n_eval > n_batch) n_eval = n_batch;
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        batch.n_tokens = n_eval;
        for (int j = 0; j < n_eval; ++j) {
            batch.token[j] = tokens[i + j];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = false;
        }
        if (llama_decode(h->ctx, batch) != 0) {
            ALOGI("【KVSave】Decode 失败 at index %d", i);
            llama_batch_free(batch);
            return false;
        }
        llama_batch_free(batch);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    {
        long long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        double rate = time_ms > 0 ? (double)n_tokens * 1000.0 / time_ms : 0;
        ALOG_PFD("[KVSave] Prefill Finish | Total Tokens: %d | Time: %lld ms | Rate: %.2f T/s", n_tokens, time_ms, rate);
    }

    auto t_save_start = std::chrono::high_resolution_clock::now();
    size_t saved_size = llama_state_save_file(h->ctx, save_path.c_str(), tokens.data(), tokens.size());
    auto t_save_end = std::chrono::high_resolution_clock::now();
    long long save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_save_end - t_save_start).count();

    bool success = (saved_size > 0);
    if (success) {
        std::string meta_path = save_path + ".meta";
        std::ofstream mf(meta_path, std::ios::out | std::ios::trunc);
        if (mf.good()) {
            mf << "n_ctx=" << h->cparams.n_ctx << "\n";
            mf << "n_embd=" << llama_model_n_embd(h->model) << "\n";
            mf << "tokens=" << n_tokens << "\n";
            mf.close();
        }

        long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_save_end - t_total_start).count();
        ALOGI("【KVSave】✅ 生成成功 | 大小: %.2f MB | 文件写入: %lld ms | 总耗时: %lld ms",
              (double)saved_size / 1024.0 / 1024.0,
              save_ms,
              total_ms);
    } else {
        ALOGE("【KVSave】❌ llama_state_save_file 保存失败");
    }

    return (jboolean)success;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_loadSession(JNIEnv* env, jclass, jlong handlePtr, jstring jPath) {
    ALOGI("【LogicCache】Session 加载调用");

    auto t_start = std::chrono::high_resolution_clock::now();
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h || !h->ctx) return false;

    forceStopAndJoin(h, "loadSession");

    const uint64_t local_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    auto cancelled = [&]() {
        return h->stop_flag.load(std::memory_order_relaxed) ||
               h->generation_seq.load(std::memory_order_relaxed) != local_seq;
    };
    if (cancelled()) return false;

    const char* cpath = env->GetStringUTFChars(jPath, nullptr);
    std::string path(cpath);
    env->ReleaseStringUTFChars(jPath, cpath);

    std::lock_guard<std::mutex> lock(h->mtx);
    if (cancelled()) return false;

    ALOGI("L1 Prefill | 尝试从文件恢复 Session: %s", path.c_str());
    if (!file_exists(path)) {
        ALOGW("L1 Prefill | 文件不存在，跳过加载");
        return false;
    }

    std::string metaPath = path + ".session_meta";
    int meta_sys_token_count = 0;
    int meta_reusable_prefix_count = 0;
    bool hasSessionMeta = false;
    if (file_exists(metaPath)) {
        std::ifstream metaFile(metaPath);
        if (metaFile.good()) {
            std::string line;
            while (std::getline(metaFile, line)) {
                if (line.find("\"system_prompt_token_count\"") != std::string::npos) {
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        meta_sys_token_count = std::atoi(line.substr(pos + 1).c_str());
                    }
                } else if (line.find("\"reusable_prefix_token_count\"") != std::string::npos) {
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        meta_reusable_prefix_count = std::atoi(line.substr(pos + 1).c_str());
                    }
                }
            }
            metaFile.close();
            hasSessionMeta = (meta_sys_token_count > 0);
            ALOGI("L1 Prefill | 读取Session元数据 | sys_tokens: %d | reusable_prefix: %d",
                  meta_sys_token_count, meta_reusable_prefix_count);
        }
    }

    if (cancelled()) return false;

    std::vector<llama_token> session_tokens(h->cparams.n_ctx);
    size_t n_token_count_out = 0;
    size_t loaded = llama_state_load_file(h->ctx, path.c_str(), session_tokens.data(), session_tokens.size(), &n_token_count_out);
    if (loaded > 0) {
        session_tokens.resize(n_token_count_out);
        h->current_tokens = session_tokens;
        if (cancelled()) {
            sync_radix_tree_from_tokens(h);
            ALOGW("L1 Prefill | 会话加载已完成但被取消，状态已同步");
            return false;
        }
    }

    if (cancelled()) {
        ALOGW("L1 Prefill | 会话加载被取消 (seq mismatch)");
        return false;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    if (loaded <= 0) {
        ALOGE("L1 Prefill | ❌ Session 加载失败 (耗时 %lld ms)", duration);
        return false;
    }

    if (hasSessionMeta && meta_sys_token_count > 0) {
        h->system_prompt_token_count = meta_sys_token_count;
        h->reusable_prefix_token_count = meta_reusable_prefix_count;
        ALOGI("L1 Prefill | ✅ Session 加载成功 (从元数据) | Token数: %zu | sys_tokens: %d | reusable_prefix: %d | 耗时: %lld ms",
              n_token_count_out, h->system_prompt_token_count, h->reusable_prefix_token_count, duration);
    } else {
        const auto* vocab = llama_model_get_vocab(h->model);

        std::string full_text;
        int system_end_pos = -1;
        int reusable_prefix_end_pos = -1;

        for (size_t i = 0; i < session_tokens.size(); i++) {
            char buf[64];
            int n = llama_token_to_piece(vocab, session_tokens[i], buf, 64, 0, false);
            if (n > 0) {
                full_text.append(buf, n);

                if (system_end_pos == -1 && full_text.find("<|im_end|>") != std::string::npos) {
                    system_end_pos = (int)i;
                    ALOGI("L1 Prefill | 检测到系统提示词结束位置: token %d", system_end_pos);

                    std::string after_system = full_text.substr(full_text.find("<|im_end|>") + 10);
                    if (after_system.find("<|im_start|>user\n") != std::string::npos) {
                        reusable_prefix_end_pos = (int)i + 1;
                        for (size_t j = i + 1; j < session_tokens.size(); j++) {
                            char buf2[64];
                            int n2 = llama_token_to_piece(vocab, session_tokens[j], buf2, 64, 0, false);
                            if (n2 > 0) {
                                after_system.append(buf2, n2);
                                if (after_system.find("<|im_start|>user\n") != std::string::npos) {
                                    reusable_prefix_end_pos = (int)j + 1;
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }

        if (system_end_pos == -1) {
            RadixLookupResult lookup = radix_lookup(h->tree_root, session_tokens.data(), (int)session_tokens.size());
            if (lookup.matched_length > 0) {
                h->system_prompt_token_count = lookup.matched_length;
                h->reusable_prefix_token_count = lookup.matched_length;
                ALOGW("L1 Prefill | ⚠️ 未找到标记，使用radix树匹配长度: %d", lookup.matched_length);
            } else {
                int default_sys_tokens = std::min(300, (int)(n_token_count_out * 0.4));
                h->system_prompt_token_count = default_sys_tokens;
                h->reusable_prefix_token_count = default_sys_tokens;
                ALOGW("L1 Prefill | ⚠️ 无法检测边界，使用动态默认值: %d (总token数的40%%)", default_sys_tokens);
            }
        } else {
            h->system_prompt_token_count = system_end_pos + 1;
            if (reusable_prefix_end_pos > system_end_pos) {
                h->reusable_prefix_token_count = reusable_prefix_end_pos;
            } else {
                h->reusable_prefix_token_count = system_end_pos + 1;
            }
            ALOGI("L1 Prefill | ✅ 通过标记检测边界 | sys_tokens: %d | reusable_prefix: %d",
                  h->system_prompt_token_count, h->reusable_prefix_token_count);
        }

        ALOGI("L1 Prefill | ✅ Session 加载成功 (智能检测) | Token数: %zu | sys_tokens: %d | reusable_prefix: %d | 耗时: %lld ms",
              n_token_count_out, h->system_prompt_token_count, h->reusable_prefix_token_count, duration);
    }

    sync_radix_tree_from_tokens(h);
    ALOGI("【会话暂存】会话恢复完成，KV缓存已优化，可复用前缀: %d tokens", h->reusable_prefix_token_count);

    return true;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_saveSeqStateFile(JNIEnv * env, jclass, jlong handlePtr, jint seqId, jintArray jTokens, jstring jSavePath) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !h->ctx) return 0;

    const std::string save_path = jstring_to_string(env, jSavePath);
    if (save_path.empty()) return 0;

    std::vector<llama_token> tokens;
    if (jTokens) {
        const jsize n = env->GetArrayLength(jTokens);
        tokens.resize((size_t) std::max<jsize>(0, n));
        jint * ptr = env->GetIntArrayElements(jTokens, nullptr);
        if (ptr) {
            for (jsize i = 0; i < n; i++) tokens[(size_t) i] = (llama_token) ptr[i];
            env->ReleaseIntArrayElements(jTokens, ptr, JNI_ABORT);
        }
    }

    forceStopAndJoin(h, "saveSeqStateFile");
    std::lock_guard<std::mutex> lock(h->mtx);

    const size_t nwrite = llama_state_seq_save_file(
        h->ctx,
        save_path.c_str(),
        (llama_seq_id) seqId,
        tokens.empty() ? nullptr : tokens.data(),
        tokens.size());

    if (nwrite == 0) {
        ALOGW("SeqState | save failed | seq=%d path=%s", (int) seqId, save_path.c_str());
    }
    return (jlong) nwrite;
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_loadSeqStateFile(JNIEnv * env, jclass, jlong handlePtr, jint seqId, jstring jLoadPath, jint maxTokens) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !h->ctx || !env) return nullptr;

    const std::string load_path = jstring_to_string(env, jLoadPath);
    if (load_path.empty()) return nullptr;

    if (maxTokens <= 0) maxTokens = 8192;
    std::vector<llama_token> tokens;
    tokens.resize((size_t) maxTokens);
    size_t token_count_out = 0;

    forceStopAndJoin(h, "loadSeqStateFile");
    std::lock_guard<std::mutex> lock(h->mtx);

    const size_t nread = llama_state_seq_load_file(
        h->ctx,
        load_path.c_str(),
        (llama_seq_id) seqId,
        tokens.data(),
        tokens.size(),
        &token_count_out);

    if (nread == 0 || token_count_out == 0) {
        ALOGW("SeqState | load failed or empty | seq=%d path=%s", (int) seqId, load_path.c_str());
        return nullptr;
    }

    if (token_count_out > tokens.size()) token_count_out = tokens.size();
    jintArray out = env->NewIntArray((jsize) token_count_out);
    if (!out) return nullptr;
    std::vector<jint> tmp(token_count_out);
    for (size_t i = 0; i < token_count_out; i++) tmp[i] = (jint) tokens[i];
    env->SetIntArrayRegion(out, 0, (jsize) token_count_out, tmp.data());
    return out;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_buildMemoryKvBlob(JNIEnv * env, jclass, jlong handlePtr, jstring jMemoryLine, jstring jSavePath) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h) return false;

    const std::string memory_line = jstring_to_string(env, jMemoryLine);
    const std::string save_path = jstring_to_string(env, jSavePath);
    if (memory_line.empty() || save_path.empty()) return false;

    forceStopAndJoin(h, "buildMemoryKvBlob");
    std::lock_guard<std::mutex> lock(h->mtx);

    if (!h->ctx || !h->model) return false;

    llama_memory_t kv_mem = llama_get_memory(h->ctx);
    if (!llama_memory_can_shift(kv_mem)) {
        ALOGW("L1KV | cannot shift KV; buildMemoryKvBlob abort");
        return false;
    }

    const auto * vocab = llama_model_get_vocab(h->model);
    // Use stable markers around the memory line to preserve BPE boundaries at both ends.
    const std::string stub_prefix = "<memory>\n";
    const std::string stub_suffix = "\n</memory>";
    const std::string full_stub = stub_prefix + memory_line + stub_suffix;
    const std::string base_stub = stub_prefix + stub_suffix;

    auto tokenize_str = [&](const std::string & s, std::vector<llama_token> & out) -> bool {
        out.assign(std::max<size_t>(256, s.size() + 64), 0);
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
    if (!tokenize_str(base_stub, base_tokens) || !tokenize_str(full_stub, full_tokens)) {
        ALOGW("L1KV | tokenize failed | memLen=%zu", memory_line.size());
        return false;
    }

    // Diff base vs full to locate the inserted span.
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
    if (insert_len == 0) {
        ALOGW("L1KV | empty insert span; skip");
        return false;
    }

    std::vector<llama_token> mem_tokens;
    mem_tokens.reserve(insert_len);
    for (size_t i = 0; i < insert_len; i++) {
        mem_tokens.push_back(full_tokens[prefix_len + i]);
    }

    // Use a dedicated sequence to build and then strip down to memory-only KV.
    llama_seq_id mem_seq = 1;
    if (h->cparams.n_seq_max >= 2) {
        mem_seq = (llama_seq_id) (h->cparams.n_seq_max - 1);
        if (mem_seq == 0) mem_seq = 1;
    }

    llama_memory_seq_rm(kv_mem, mem_seq, -1, -1);

    // Prefill the stub prompt into mem_seq.
    int n_batch = (int) h->cparams.n_batch;
    if (n_batch <= 0) n_batch = 256;
    for (size_t off = 0; off < full_tokens.size(); off += (size_t) n_batch) {
        const size_t n_eval = std::min(full_tokens.size() - off, (size_t) n_batch);
        llama_batch batch = llama_batch_init((int) n_eval, 0, 1);
        batch.n_tokens = (int32_t) n_eval;
        for (int j = 0; j < (int) n_eval; j++) {
            batch.token[j] = full_tokens[off + (size_t) j];
            batch.pos[j] = (llama_pos) (off + (size_t) j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = mem_seq;
            batch.logits[j] = false;
        }
        const int rc = llama_decode(h->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            ALOGW("L1KV | llama_decode failed rc=%d off=%zu", rc, off);
            llama_memory_seq_rm(kv_mem, mem_seq, -1, -1);
            return false;
        }
    }

    // Remove stub prefix/suffix KV so only memory span remains.
    const llama_pos p0 = (llama_pos) prefix_len;
    const llama_pos p1 = (llama_pos) (prefix_len + insert_len);
    llama_memory_seq_rm(kv_mem, mem_seq, 0, p0);
    llama_memory_seq_rm(kv_mem, mem_seq, p1, -1);
    // Shift memory span down to start at pos=0 for portability.
    if (p0 > 0) {
        llama_memory_seq_add(kv_mem, mem_seq, p0, -1, -p0);
    }

    const size_t nwrite = llama_state_seq_save_file(h->ctx, save_path.c_str(), mem_seq, mem_tokens.data(), mem_tokens.size());
    // Clear after save to avoid polluting runtime sequences.
    llama_memory_seq_rm(kv_mem, mem_seq, -1, -1);

    if (nwrite == 0) {
        ALOGW("L1KV | save failed | path=%s", save_path.c_str());
    }
    return (jboolean) (nwrite > 0);
}
