#include "imem_core.h"

#include <cstdlib>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

struct CpuFreq {
    int cpu = -1;
    long freq = -1;
};

static long read_long_from_file(const std::string & path) {
    long value = -1;
    std::ifstream f(path);
    if (!f.good()) return -1;
    f >> value;
    return value;
}

static long get_configured_cpu_count() {
#if defined(_SC_NPROCESSORS_CONF)
    const long cpu_count_long = sysconf(_SC_NPROCESSORS_CONF);
    if (cpu_count_long > 0) {
        return cpu_count_long;
    }
#endif
    const unsigned int hc = std::thread::hardware_concurrency();
    return hc > 0 ? static_cast<long>(hc) : -1;
}

static std::vector<CpuFreq> get_cpu_freq_sorted_by_max_freq_desc() {
    const long cpu_count_long = get_configured_cpu_count();
    const int cpu_count = cpu_count_long > 0 ? static_cast<int>(cpu_count_long) : 0;

    std::vector<CpuFreq> entries;
    entries.reserve(cpu_count);

    for (int cpu = 0; cpu < cpu_count; cpu++) {
        long freq = -1;
        {
            const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
            freq = read_long_from_file(base + "/cpufreq/cpuinfo_max_freq");
            if (freq <= 0) {
                freq = read_long_from_file(base + "/cpufreq/scaling_max_freq");
            }
            if (freq <= 0) {
                freq = read_long_from_file(base + "/cpu_capacity");
            }
            if (freq <= 0) {
                freq = -1;
            }
        }
        entries.push_back({cpu, freq});
    }

    std::stable_sort(entries.begin(), entries.end(), [](const CpuFreq & a, const CpuFreq & b) {
        if (a.freq < 0 && b.freq < 0) return a.cpu < b.cpu;
        if (a.freq < 0) return false;
        if (b.freq < 0) return true;
        if (a.freq != b.freq) return a.freq > b.freq;
        return a.cpu < b.cpu;
    });

    return entries;
}

static int count_cores_by_top_freq_ratio(const std::vector<CpuFreq> & entries, double ratio) {
    if (entries.empty() || entries[0].freq <= 0) return 0;
    const double top = static_cast<double>(entries[0].freq);
    const double threshold = top * ratio;
    int count = 0;
    for (const auto & e : entries) {
        if (e.freq <= 0) break;
        if (static_cast<double>(e.freq) >= threshold) count++;
    }
    return count;
}

static int estimate_big_core_count(const std::vector<CpuFreq> & entries) {
    int big = count_cores_by_top_freq_ratio(entries, 0.85);
    if (big < 2) {
        big = count_cores_by_top_freq_ratio(entries, 0.75);
    }
    if (big < 1) big = 1;
    return big;
}

static void fill_top_freq_mask(bool * dst_mask, int n_threads) {
    memset(dst_mask, 0, GGML_MAX_N_THREADS);
    if (n_threads <= 0) return;

    const auto entries = get_cpu_freq_sorted_by_max_freq_desc();
    if (entries.empty()) return;

    int picked = 0;
    for (int i = 0; i < (int)entries.size() && picked < n_threads; i++) {
        const int cpu = entries[i].cpu;
        if (cpu < 0 || cpu >= GGML_MAX_N_THREADS) continue;
        dst_mask[cpu] = true;
        picked++;
    }

    if (picked <= 0) {
        memset(dst_mask, 0, GGML_MAX_N_THREADS);
    }
}

static int clamp_threads(int n_threads) {
    const long cpu_count_long = get_configured_cpu_count();
    const int cpu_count = cpu_count_long > 0 ? static_cast<int>(cpu_count_long) : 0;
    if (n_threads < 1) return 1;
    if (cpu_count > 0) return std::min(n_threads, cpu_count);
    return n_threads;
}

static void choose_decode_and_batch_threads(int requested_threads, int * out_decode, int * out_batch, int * out_big_cores) {
    const int requested = clamp_threads(requested_threads);
    const auto entries = get_cpu_freq_sorted_by_max_freq_desc();
    const int big = entries.empty() ? requested : estimate_big_core_count(entries);

    int decode = std::min(requested, big);
    if (decode < 1) decode = 1;

    int batch = requested;
    if (big > 0 && requested > big && big <= 4) {
        batch = big;
    }

    if (out_decode) *out_decode = decode;
    if (out_batch) *out_batch = batch;
    if (out_big_cores) *out_big_cores = big;
}

static void maybe_attach_threadpool(ModelHandle * handle, int n_threads_decode, int n_threads_batch) {
    if (!handle || !handle->ctx) return;
    n_threads_decode = clamp_threads(n_threads_decode);
    n_threads_batch = clamp_threads(n_threads_batch);
    if (n_threads_decode <= 1 && n_threads_batch <= 1) return;

    ggml_threadpool_t tp = nullptr;
    ggml_threadpool_t tp_batch = nullptr;

    const int tp_threads = std::max(n_threads_decode, 2);
    const int tp_batch_threads = std::max(n_threads_batch, 2);

    ggml_threadpool_params tpp = ggml_threadpool_params_default(tp_threads);
    tpp.strict_cpu = true;
    fill_top_freq_mask(tpp.cpumask, tp_threads);

    ggml_threadpool_params tppb = ggml_threadpool_params_default(tp_batch_threads);
    tppb.strict_cpu = true;
    fill_top_freq_mask(tppb.cpumask, tp_batch_threads);

    ggml_threadpool_params tpp_default = ggml_threadpool_params_default(tp_threads);
    ggml_threadpool_params tppb_default = ggml_threadpool_params_default(tp_batch_threads);

    const bool has_mask = memcmp(tpp.cpumask, tpp_default.cpumask, GGML_MAX_N_THREADS) != 0;
    const bool has_mask_batch = memcmp(tppb.cpumask, tppb_default.cpumask, GGML_MAX_N_THREADS) != 0;
    if (!has_mask && !has_mask_batch) {
        return;
    }

    tp = ggml_threadpool_new(&tpp);
    tp_batch = ggml_threadpool_new(&tppb);
    if (!tp || !tp_batch) {
        if (tp) ggml_threadpool_free(tp);
        if (tp_batch) ggml_threadpool_free(tp_batch);
        ALOGW("Threadpool create failed; continuing without affinity");
        return;
    }

    llama_attach_threadpool(handle->ctx, tp, tp_batch);
    handle->threadpool = tp;
    handle->threadpool_batch = tp_batch;
}

} // namespace

JavaVM* gJvm = nullptr;
std::unordered_map<std::string, ModelEntry> g_model_registry;
std::mutex g_registry_mtx;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    gJvm = vm;
    llama_log_set(llama_log_callback, nullptr);

    if (!std::getenv("OPENBLAS_WAIT_POLICY")) {
#if defined(_WIN32)
        _putenv_s("OPENBLAS_WAIT_POLICY", "PASSIVE");
#else
        setenv("OPENBLAS_WAIT_POLICY", "PASSIVE", 0);
#endif
    }

    ALOGI("系统启动 | JNI_OnLoad | 环境初始化完成 (模块化版本)");
    return JNI_VERSION_1_6;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_nativeSetLogMinPriority(JNIEnv* env, jclass clazz, jint minPriority) {
    imem_set_min_log_priority(static_cast<int>(minPriority));
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_nativeEnterPerfMode(JNIEnv* env, jclass clazz) {
    return static_cast<jint>(imem_enter_perf_mode());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_nativeExitPerfMode(JNIEnv* env, jclass clazz) {
    return static_cast<jint>(imem_exit_perf_mode());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_nativeSetDisableKvReuse(JNIEnv* env, jclass clazz, jlong handlePtr, jboolean disable) {
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return;
    h->disable_kv_reuse.store(disable == JNI_TRUE, std::memory_order_relaxed);
    ALOGI("KV reuse policy | disable_kv_reuse=%d handle=%p", (disable == JNI_TRUE) ? 1 : 0, h);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_freeModel(JNIEnv* env, jclass clazz, jlong handlePtr) {
    ALOGI("【Lifecycle】释放模型调用");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return;
    ALOGI("资源释放 | 正在销毁实例 handle=%p", h);
    forceStopAndJoin(h, "freeModel");
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        if (h->ctx) {
            llama_free(h->ctx);
            h->ctx = nullptr;
        }
        if (h->current_lora) {
            llama_adapter_lora_free(h->current_lora);
            h->current_lora = nullptr;
        }
    }

    if (h->threadpool) {
        ggml_threadpool_free(h->threadpool);
        h->threadpool = nullptr;
    }
    if (h->threadpool_batch) {
        ggml_threadpool_free(h->threadpool_batch);
        h->threadpool_batch = nullptr;
    }
    if (h->model && !h->model_path_key.empty()) {
        std::lock_guard<std::mutex> lock(g_registry_mtx);
        auto it = g_model_registry.find(h->model_path_key);
        if (it != g_model_registry.end()) {
            it->second.ref_count--;
            if (it->second.ref_count <= 0) {
                llama_model_free(it->second.model);
                g_model_registry.erase(it);
                ALOGI("资源释放 | 物理模型已卸载");
            }
        }
        h->model = nullptr;
    } else if (h->model) {
        llama_model_free(h->model);
    }
    delete h;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_stop(JNIEnv* env, jclass clazz, jlong handlePtr) {
    ALOGI("【Lifecycle】Stop 调用");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return;
    ALOGI("正在调用 stop generation。");
    forceStopAndJoin(h, "LLMBridge_stop");
    ALOGI("stop 完成 (Context 保留，仅停止线程)。");
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_createGenerationInstance(JNIEnv* env, jclass, jstring jModelPath, jstring jLoraPath, jint nThreads, jint nGpuLayers) {
    auto t_start = std::chrono::high_resolution_clock::now();
    ALOGI("【Lifecycle】创建 Generation Instance 调用");
    const char* cpath = env->GetStringUTFChars(jModelPath, nullptr);
    std::string modelPath(cpath);
    const char* cLoraPath = jLoraPath ? env->GetStringUTFChars(jLoraPath, nullptr) : nullptr;
    std::string loraPath = cLoraPath ? cLoraPath : "";
    if (cLoraPath) env->ReleaseStringUTFChars(jLoraPath, cLoraPath);
    ALOGI("创建 Generation Instance: %s, LoRA: %s, 线程=%d, GPU层数=%d", cpath, (loraPath.empty() ? "<merged/disabled>" : loraPath.c_str()), nThreads, nGpuLayers);
    llama_backend_init();
    struct llama_model* model = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_registry_mtx);
        auto it = g_model_registry.find(modelPath);
        if (it != g_model_registry.end()) {
            model = it->second.model;
            it->second.ref_count++;
            ALOGI("Generation: 复用已加载的模型。新引用计数: %d", it->second.ref_count);
        } else {
            ALOGI("Generation: 模型未加载，执行首次加载...");
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = nGpuLayers;
            mparams.use_mmap = true;
            model = llama_model_load_from_file(cpath, mparams);
            if (model) {
                g_model_registry[modelPath] = {model, 1};
            }
        }
    }
    if (!model) {
        ALOGE("Generation模型加载失败");
        env->ReleaseStringUTFChars(jModelPath, cpath);
        return 0;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_batch = 2048;
    cparams.n_ubatch = std::min((int) cparams.n_batch, (int) PREFILL_UBATCH_SIZE * 2);  // Micro-batch size

    int n_threads_decode = 1;
    int n_threads_batch = 1;
    int n_big_cores = 1;
    choose_decode_and_batch_threads(nThreads, &n_threads_decode, &n_threads_batch, &n_big_cores);
    cparams.n_threads = n_threads_decode;
    cparams.n_threads_batch = n_threads_batch;
    ALOGI("线程策略 | requested=%d | decode=%d | batch=%d | est_big=%d", nThreads, n_threads_decode, n_threads_batch, n_big_cores);

    cparams.n_seq_max = 10;
    cparams.embeddings = false;
    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("Generation上下文创建失败");
        {
            std::lock_guard<std::mutex> lock(g_registry_mtx);
            auto it = g_model_registry.find(modelPath);
            if (it != g_model_registry.end()) {
                it->second.ref_count--;
                if(it->second.ref_count <= 0) {
                    llama_model_free(model);
                    g_model_registry.erase(it);
                }
            }
        }
        env->ReleaseStringUTFChars(jModelPath, cpath);
        return 0;
    }
    struct llama_adapter_lora* adapter = nullptr;
    if (!loraPath.empty() && file_exists(loraPath)) {
        ALOGI("Generation: 正在挂载 LoRA Adapter: %s", loraPath.c_str());
        adapter = llama_adapter_lora_init(model, loraPath.c_str());
        if (adapter) {
            int32_t err = llama_set_adapter_lora(ctx, adapter, 1.0f);
            if (err == 0) {
                ALOGI("Generation: ✅ LoRA 挂载成功");
            } else {
                ALOGE("Generation: ❌ LoRA set adapter 失败, err=%d", err);
                llama_adapter_lora_free(adapter);
                adapter = nullptr;
            }
        } else {
            ALOGE("Generation: ❌ LoRA init 失败 (可能文件损坏或架构不匹配)");
        }
    } else {
        ALOGI("Generation: LoRA disabled (merged model or empty path)");
    }
    ModelHandle* handle = new ModelHandle();
    handle->model = model;
    handle->ctx = ctx;
    handle->cparams = cparams;
    handle->type = InstanceType::GENERATION;
    handle->model_path_key = modelPath;
    handle->current_lora = adapter;

    maybe_attach_threadpool(handle, cparams.n_threads, cparams.n_threads_batch);
    env->ReleaseStringUTFChars(jModelPath, cpath);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    ALOGI("Generation Instance 创建成功, Handle: %p | 耗时: %lld ms", handle, duration);
    return reinterpret_cast<jlong>(handle);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_createBenchmarkInstance(JNIEnv* env, jclass, jstring jModelPath, jstring jLoraPath, jint nThreads, jint nGpuLayers) {
    auto t_start = std::chrono::high_resolution_clock::now();
    ALOGI("【Lifecycle】创建 Benchmark Instance 调用");

    const char* cpath = env->GetStringUTFChars(jModelPath, nullptr);
    std::string modelPath(cpath ? cpath : "");

    const char* cLoraPath = jLoraPath ? env->GetStringUTFChars(jLoraPath, nullptr) : nullptr;
    std::string loraPath(cLoraPath ? cLoraPath : "");
    if (cLoraPath) env->ReleaseStringUTFChars(jLoraPath, cLoraPath);

    ALOGI("创建 Benchmark Instance: %s, LoRA: %s, 线程=%d, GPU层数=%d", cpath ? cpath : "", (loraPath.empty() ? "<merged/disabled>" : loraPath.c_str()), nThreads, nGpuLayers);

    llama_backend_init();

    struct llama_model* model = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_registry_mtx);
        auto it = g_model_registry.find(modelPath);
        if (it != g_model_registry.end()) {
            model = it->second.model;
            it->second.ref_count++;
            ALOGI("Benchmark: 复用已加载的模型。新引用计数: %d", it->second.ref_count);
        } else {
            ALOGI("Benchmark: 模型未加载，执行首次加载...");
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = nGpuLayers;
            mparams.use_mmap = true;
            model = llama_model_load_from_file(cpath, mparams);
            if (model) {
                g_model_registry[modelPath] = {model, 1};
            }
        }
    }

    if (!model) {
        ALOGE("Benchmark模型加载失败");
        if (cpath) env->ReleaseStringUTFChars(jModelPath, cpath);
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 1024;
    cparams.n_ubatch = PREFILL_UBATCH_SIZE;

    int n_threads_decode = 1;
    int n_threads_batch = 1;
    int n_big_cores = 1;
    choose_decode_and_batch_threads(nThreads, &n_threads_decode, &n_threads_batch, &n_big_cores);
    cparams.n_threads = n_threads_decode;
    cparams.n_threads_batch = n_threads_batch;
    ALOGI("线程策略 | requested=%d | decode=%d | batch=%d | est_big=%d", nThreads, n_threads_decode, n_threads_batch, n_big_cores);

    cparams.n_seq_max = 1;
    cparams.embeddings = false;

    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("Benchmark上下文创建失败");
        {
            std::lock_guard<std::mutex> lock(g_registry_mtx);
            auto it = g_model_registry.find(modelPath);
            if (it != g_model_registry.end()) {
                it->second.ref_count--;
                if (it->second.ref_count <= 0) {
                    llama_model_free(model);
                    g_model_registry.erase(it);
                }
            }
        }
        if (cpath) env->ReleaseStringUTFChars(jModelPath, cpath);
        return 0;
    }

    struct llama_adapter_lora* adapter = nullptr;
    if (!loraPath.empty() && file_exists(loraPath)) {
        ALOGI("Benchmark: 正在挂载 LoRA Adapter: %s", loraPath.c_str());
        adapter = llama_adapter_lora_init(model, loraPath.c_str());
        if (adapter) {
            int32_t err = llama_set_adapter_lora(ctx, adapter, 1.0f);
            if (err == 0) {
                ALOGI("Benchmark: ✅ LoRA 挂载成功");
            } else {
                ALOGE("Benchmark: ❌ LoRA set adapter 失败, err=%d", err);
                llama_adapter_lora_free(adapter);
                adapter = nullptr;
            }
        } else {
            ALOGE("Benchmark: ❌ LoRA init 失败 (可能文件损坏或架构不匹配)");
        }
    } else {
        ALOGI("Benchmark: LoRA disabled (merged model or empty path)");
    }

    ModelHandle* handle = new ModelHandle();
    handle->model = model;
    handle->ctx = ctx;
    handle->cparams = cparams;
    handle->type = InstanceType::GENERATION;
    handle->model_path_key = modelPath;
    handle->current_lora = adapter;

    maybe_attach_threadpool(handle, cparams.n_threads, cparams.n_threads_batch);

    if (cpath) env->ReleaseStringUTFChars(jModelPath, cpath);

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    ALOGI("Benchmark Instance 创建成功, Handle: %p | 耗时: %lld ms", handle, duration);
    return reinterpret_cast<jlong>(handle);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_loadLora(JNIEnv* env, jclass, jlong handlePtr, jstring jLoraPath) {
    ALOGI("【Lifecycle】LoRA 加载/卸载 调用");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h || h->type != InstanceType::GENERATION) return;
    (void)env;
    (void)jLoraPath;
    ALOGI("L3 LoRA | NO-OP (merged model)");
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_unloadLora(JNIEnv* env, jclass clazz, jlong handlePtr) {
    Java_com_yuyan_imemodule_llm_LLMBridge_loadLora(env, clazz, handlePtr, nullptr);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_triggerNightMode(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jSwapPath) {
    auto t_start = std::chrono::high_resolution_clock::now();
    ALOGI("【Lifecycle】Night Mode 触发");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return false;
    forceStopAndJoin(h, "NightMode_Entry");
    const char* cPath = env->GetStringUTFChars(jSwapPath, nullptr);
    std::string swapPath(cPath);
    env->ReleaseStringUTFChars(jSwapPath, cPath);
    std::lock_guard<std::mutex> lock(h->mtx);
    if (!h->ctx || !h->model) {
        ALOGW("【NightMode】上下文为空，无法执行休眠序列化");
        return false;
    }
    ALOGI("【NightMode】>>> 进入休眠模式 (Phase: Serialization) <<<");
    {
        std::string sessionMetaPath = swapPath + ".session_meta";
        std::ofstream metaFile(sessionMetaPath, std::ios::out | std::ios::trunc);
        if (metaFile.good()) {
            metaFile << "{\n";
            metaFile << "  \"system_prompt_token_count\": " << h->system_prompt_token_count << ",\n";
            metaFile << "  \"reusable_prefix_token_count\": " << h->reusable_prefix_token_count << ",\n";
            metaFile << "  \"total_token_count\": " << h->current_tokens.size() << ",\n";
            metaFile << "  \"session_signature\": \"" << h->session_signature << "\",\n";
            metaFile << "  \"style_cache_path\": \"" << h->cache_path << "\"\n";
            metaFile << "}\n";
            metaFile.close();
            ALOGI("【NightMode】✅ 会话元数据已保存 | sys_tokens: %d | reusable_prefix: %d | total: %zu",
                  h->system_prompt_token_count, h->reusable_prefix_token_count, h->current_tokens.size());
        }
    }
    size_t saved_size = llama_state_save_file(h->ctx, swapPath.c_str(), h->current_tokens.data(), h->current_tokens.size());
    auto t_end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    if (saved_size > 0) {
        ALOGI("【NightMode】✅ 热数据序列化成功 | Size: %.2f MB | Token数: %zu | 耗时: %lld ms",
              (double)saved_size / 1024.0 / 1024.0,
              h->current_tokens.size(),
              duration);
        llama_memory_seq_rm(llama_get_memory(h->ctx), -1, 0, -1);

        if (h->current_tokens.size() > static_cast<size_t>(h->system_prompt_token_count)) {
            h->current_tokens.resize(h->system_prompt_token_count);
        }
        ALOGI("【NightMode】保留系统提示词 tokens: %zu，用于 DayMode 验证", h->current_tokens.size());

        if (h->tree_root) delete h->tree_root;
        h->tree_root = new RadixNode(-1, -1, nullptr);
        h->tree_root->is_hot = true;
        h->cursor_node = h->tree_root;

        ALOGI("【NightMode】✅ KV Cache已释放，Radix Tree 已重置，进入低功耗待机状态");
        return true;
    } else {
        ALOGE("【NightMode】❌ 序列化失败 (耗时 %lld ms)，取消休眠操作", duration);
        return false;
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_triggerDayMode(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jSwapPath) {
    ALOGI("【Lifecycle】Day Mode 触发");
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) {
        ALOGE("【DayMode】错误: Handle 为空");
        return false;
    }
    const char* cPath = env->GetStringUTFChars(jSwapPath, nullptr);
    std::string swapPath(cPath);
    env->ReleaseStringUTFChars(jSwapPath, cPath);
    ALOGI("【DayMode】>>> 开始唤醒流程 | 目标路径: %s <<<", swapPath.c_str());
    bool contextExists = false;
    bool isFreshStart = false;
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        contextExists = (h->ctx != nullptr);
    }
    if (contextExists) {
        ALOGI("【DayMode】Context 已存在 (Hot Standby)，无需重建");
        std::lock_guard<std::mutex> lock(h->mtx);
        if (h->system_prompt_token_count > 0 && h->current_tokens.size() > 0) {
            bool systemTokensValid = true;
            for (int i = 0; i < std::min((int)h->system_prompt_token_count, (int)h->current_tokens.size()); ++i) {
                if (h->current_tokens[i] == 0) {
                    systemTokensValid = false;
                    break;
                }
            }
            if (!systemTokensValid) {
                ALOGW("【DayMode】检测到系统提示词Token异常，尝试清理并恢复");
                llama_memory_seq_rm(llama_get_memory(h->ctx), -1, h->system_prompt_token_count, -1);
                h->current_tokens.resize(h->system_prompt_token_count);
                isFreshStart = true;
            } else {
                ALOGI("【DayMode】系统提示词验证通过 (Token数: %d | 可复用前缀: %d)",
                      h->system_prompt_token_count, h->reusable_prefix_token_count);

                if (file_exists(swapPath)) {
                    std::string sessionMetaPath = swapPath + ".session_meta";
                    int meta_sys_token_count = 0;
                    int meta_reusable_prefix_count = 0;
                    std::string meta_session_signature;
                    bool scene_changed = false;

                    if (file_exists(sessionMetaPath)) {
                        std::ifstream metaFile(sessionMetaPath);
                        if (metaFile.good()) {
                            std::string line;
                            while (std::getline(metaFile, line)) {
                                if (line.find("\"system_prompt_token_count\"") != std::string::npos) {
                                    size_t pos = line.find(':');
                                    if (pos != std::string::npos) meta_sys_token_count = std::atoi(line.substr(pos + 1).c_str());
                                } else if (line.find("\"reusable_prefix_token_count\"") != std::string::npos) {
                                    size_t pos = line.find(':');
                                    if (pos != std::string::npos) meta_reusable_prefix_count = std::atoi(line.substr(pos + 1).c_str());
                                } else if (line.find("\"session_signature\"") != std::string::npos) {
                                    size_t start = line.find(": \"");
                                    size_t end = line.rfind("\"");
                                    if (start != std::string::npos && end > start + 3) {
                                        meta_session_signature = line.substr(start + 3, end - start - 3);
                                    }
                                }
                            }
                            metaFile.close();
                        }
                    }

                    if (meta_sys_token_count > 0 && meta_sys_token_count != h->system_prompt_token_count) {
                        ALOGW("【DayMode】⚠️ 检测到对话场景变化 | Swap系统提示词: %d -> 当前: %d",
                              meta_sys_token_count, h->system_prompt_token_count);
                        scene_changed = true;
                    }

                    if (!scene_changed && !meta_session_signature.empty() && !h->session_signature.empty() && meta_session_signature != h->session_signature) {
                        ALOGW("【DayMode】⚠️ 检测到会话签名变化，跳过 swap 恢复");
                        ALOGW("【DayMode】swap_sig=%s cur_sig=%s", meta_session_signature.c_str(), h->session_signature.c_str());
                        scene_changed = true;
                    }

                    if (scene_changed) {
                        llama_memory_seq_rm(llama_get_memory(h->ctx), -1, h->system_prompt_token_count, -1);
                        if (h->current_tokens.size() > static_cast<size_t>(h->system_prompt_token_count)) {
                            h->current_tokens.resize(h->system_prompt_token_count);
                        }
                        if (h->tree_root) delete h->tree_root;
                        h->tree_root = new RadixNode(-1, -1, nullptr);
                        h->tree_root->is_hot = true;
                        h->cursor_node = h->tree_root;
                        ALOGW("【DayMode】⚠️ 已跳过 swap 恢复（场景切换），已清理用户KV");
                    } else {
                        size_t token_count_out = 0;
                        size_t max_tokens = h->cparams.n_ctx;
                        std::vector<llama_token> restored_tokens(max_tokens);
                        size_t loaded = llama_state_load_file(h->ctx, swapPath.c_str(), restored_tokens.data(), restored_tokens.size(), &token_count_out);
                        if (loaded > 0 && token_count_out > 0) {
                            restored_tokens.resize(token_count_out);
                            h->current_tokens = restored_tokens;
                            ALOGI("【DayMode】✅ Hot Standby 下从 swap 文件恢复 KV Cache | Token数: %zu", token_count_out);
                            if (meta_reusable_prefix_count > 0) {
                                h->reusable_prefix_token_count = meta_reusable_prefix_count;
                                ALOGI("【DayMode】更新 reusable_prefix: %d", h->reusable_prefix_token_count);
                            }
                        } else {
                            ALOGW("【DayMode】⚠️ swap 文件加载失败或为空，保持当前状态");
                        }
                    }
                } else {
                    ALOGW("【DayMode】⚠️ swap 文件不存在，无法恢复 KV Cache");
                }
            }
        } else {
            ALOGW("【DayMode】警告: 系统提示词Token数异常 (%d)，标记为新鲜启动", h->system_prompt_token_count);
            isFreshStart = true;
        }
        sync_radix_tree_from_tokens(h);
        return true;
    }
    ALOGI("【DayMode】需要重建Context (Cold Start)");
    auto start = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(h->mtx);
    h->ctx = llama_init_from_model(h->model, h->cparams);
    if (!h->ctx) {
        ALOGE("【DayMode】❌ 唤醒失败：无法重建 Context");
        return false;
    }
    
    std::string sessionMetaPath = swapPath + ".session_meta";
    int meta_sys_token_count = 0;
    int meta_reusable_prefix_count = 0;
    std::string styleCachePath = "";
    std::string meta_session_signature = "";
    bool hasSessionMeta = false;
    
    if (file_exists(sessionMetaPath)) {
        std::ifstream metaFile(sessionMetaPath);
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
                } else if (line.find("\"style_cache_path\"") != std::string::npos) {
                    size_t start = line.find(": \"");
                    size_t end = line.rfind("\"");
                    if (start != std::string::npos && end > start + 3) {
                        styleCachePath = line.substr(start + 3, end - start - 3);
                    }
                } else if (line.find("\"session_signature\"") != std::string::npos) {
                    size_t start = line.find(": \"");
                    size_t end = line.rfind("\"");
                    if (start != std::string::npos && end > start + 3) {
                        meta_session_signature = line.substr(start + 3, end - start - 3);
                    }
                }
            }
            metaFile.close();
            hasSessionMeta = (meta_sys_token_count > 0 || meta_reusable_prefix_count > 0);
            ALOGI("【DayMode】读取会话元数据 | sys_tokens: %d | reusable_prefix: %d | style_path: %s | sig: %s",
                  meta_sys_token_count, meta_reusable_prefix_count, styleCachePath.c_str(), meta_session_signature.c_str());
        }
    }
    
    if (!hasSessionMeta && h->cache_path.empty() && file_exists(swapPath + ".style_meta")) {
        std::ifstream oldMetaFile(swapPath + ".style_meta");
        if (oldMetaFile.good()) {
            std::getline(oldMetaFile, styleCachePath);
            oldMetaFile.close();
        }
    }
    
    bool restoredFromSwap = false;
    bool restoredFromStyleCache = false;
    bool allowSwapRestore = true;
    if (!meta_session_signature.empty() && !h->session_signature.empty() && meta_session_signature != h->session_signature) {
        ALOGW("【DayMode】⚠️ session_signature 不匹配，跳过 swap 恢复 (Cold Start)");
        ALOGW("【DayMode】swap_sig=%s cur_sig=%s", meta_session_signature.c_str(), h->session_signature.c_str());
        allowSwapRestore = false;
    }

    if (allowSwapRestore && file_exists(swapPath)) {
        size_t token_count_out = 0;
        size_t max_tokens = h->cparams.n_ctx;
        std::vector<llama_token> restored_tokens(max_tokens);
        size_t loaded = llama_state_load_file(h->ctx, swapPath.c_str(), restored_tokens.data(), restored_tokens.size(), &token_count_out);
        if (loaded > 0 && token_count_out > 0) {
            restored_tokens.resize(token_count_out);
            h->current_tokens = restored_tokens;
            
            if (hasSessionMeta && meta_sys_token_count > 0) {
                h->system_prompt_token_count = meta_sys_token_count;
                h->reusable_prefix_token_count = meta_reusable_prefix_count;
                ALOGI("【DayMode】从元数据恢复 | sys_tokens: %d | reusable_prefix: %d",
                      h->system_prompt_token_count, h->reusable_prefix_token_count);
            } else {
                const auto* vocab = llama_model_get_vocab(h->model);
                std::string systemEndMarker = "<|im_end|>";
                std::string currentText = "";
                for (size_t i = 0; i < restored_tokens.size(); ++i) {
                    char buf[64];
                    int n = llama_token_to_piece(vocab, restored_tokens[i], buf, 64, 0, false);
                    if (n > 0) {
                        currentText.append(buf, n);
                        if (currentText.find(systemEndMarker) != std::string::npos) {
                            h->system_prompt_token_count = (int)i + 1;
                            h->reusable_prefix_token_count = h->system_prompt_token_count;
                            ALOGI("【DayMode】(Fallback) 检测到系统提示词结束位置: Token %d", h->system_prompt_token_count);
                            break;
                        }
                    }
                }
                if (h->system_prompt_token_count == 0) {
                    h->system_prompt_token_count = std::min(200, (int)restored_tokens.size());
                    h->reusable_prefix_token_count = h->system_prompt_token_count;
                    ALOGW("【DayMode】(Fallback) 未找到系统提示词结束标记，使用默认值: %d", h->system_prompt_token_count);
                }
            }
            restoredFromSwap = true;
            ALOGI("【DayMode】✅ 从Swap文件恢复成功 | Token数: %zu | 系统Token数: %d | 可复用前缀: %d",
                  token_count_out, h->system_prompt_token_count, h->reusable_prefix_token_count);
        }
    }
    if (!restoredFromSwap && !styleCachePath.empty() && file_exists(styleCachePath)) {
        ALOGI("【DayMode】尝试从风格缓存恢复: %s", styleCachePath.c_str());
        size_t token_count_out = 0;
        size_t max_tokens = h->cparams.n_ctx;
        std::vector<llama_token> style_tokens(max_tokens);
        size_t loaded = llama_state_load_file(h->ctx, styleCachePath.c_str(), style_tokens.data(), style_tokens.size(), &token_count_out);
        if (loaded > 0 && token_count_out > 0) {
            style_tokens.resize(token_count_out);
            h->current_tokens = style_tokens;
            h->system_prompt_token_count = (int)token_count_out;
            h->reusable_prefix_token_count = (int)token_count_out;
            h->is_system_prompt_cached = true;
            restoredFromStyleCache = true;
            ALOGI("【DayMode】✅ 从风格缓存恢复成功 | Token数: %zu | reusable_prefix: %d", token_count_out, h->reusable_prefix_token_count);
        }
    }
    if (!restoredFromSwap && !restoredFromStyleCache) {
        ALOGW("【DayMode】⚠️ 所有恢复方式都失败，执行冷启动");
        llama_memory_seq_rm(llama_get_memory(h->ctx), -1, 0, -1);
        h->current_tokens.clear();
        h->system_prompt_token_count = 0;
        h->reusable_prefix_token_count = 0;
        h->is_system_prompt_cached = false;
        isFreshStart = true;
    } else {
        int protection_threshold = std::max(h->system_prompt_token_count, h->reusable_prefix_token_count);
        refresh_tree_hot_status(h->tree_root, h->current_tokens, protection_threshold);
        
        if (!h->current_tokens.empty()) {
            RadixLookupResult lookup = radix_lookup(h->tree_root, h->current_tokens.data(), (int)h->current_tokens.size());
            if (!lookup.full_match) {
                int start_pos = lookup.matched_length;
                RadixNode* insert_point = lookup.matched_node ? lookup.matched_node : h->tree_root;
                h->cursor_node = radix_insert(
                    insert_point,
                    h->current_tokens.data() + start_pos,
                    (int)h->current_tokens.size() - start_pos,
                    start_pos
                );
            } else {
                h->cursor_node = lookup.matched_node;
            }
            ALOGI("【DayMode】CursorKV: 树路径已恢复 | cursor_node ID: %d", 
                  h->cursor_node ? h->cursor_node->id : -1);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    ALOGI("【DayMode】唤醒完成 | 耗时: %lld ms | 恢复方式: %s%s | FreshStart: %s",
          duration,
          restoredFromSwap ? "Swap" : "",
          restoredFromStyleCache ? "StyleCache" : "",
          isFreshStart ? "是" : "否");
    return !isFreshStart;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_setSessionSignature(JNIEnv* env, jclass clazz, jlong handlePtr, jstring jSignature) {
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return;
    const char* cSig = jSignature ? env->GetStringUTFChars(jSignature, nullptr) : nullptr;
    std::string sig = cSig ? cSig : "";
    if (cSig) env->ReleaseStringUTFChars(jSignature, cSig);
    std::lock_guard<std::mutex> lock(h->mtx);
    h->session_signature = sig;
    ALOGI("【Session】setSessionSignature len=%zu", h->session_signature.size());
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_clearKvKeepSystem(JNIEnv* env, jclass clazz, jlong handlePtr) {
    (void)env;
    (void)clazz;
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return false;
    forceStopAndJoin(h, "clearKvKeepSystem");
    std::lock_guard<std::mutex> lock(h->mtx);
    if (!h->ctx) return false;

    const int keep = std::max(0, h->system_prompt_token_count);
    llama_memory_seq_rm(llama_get_memory(h->ctx), -1, keep, -1);
    if (h->current_tokens.size() > static_cast<size_t>(keep)) {
        h->current_tokens.resize(keep);
    }
    if (h->reusable_prefix_token_count > keep) {
        h->reusable_prefix_token_count = keep;
    }

    if (h->tree_root) delete h->tree_root;
    h->tree_root = new RadixNode(-1, -1, nullptr);
    h->tree_root->is_hot = true;
    h->cursor_node = h->tree_root;

    ALOGI("【KV】clearKvKeepSystem keep=%d", keep);
    return true;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_warmup(JNIEnv* env, jclass clazz, jlong handlePtr) {
    ALOGI("【Lifecycle】Warmup 已跳过 (逻辑已移除)");
}

extern "C"
JNIEXPORT void JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_setReusablePrefixTokenCount(JNIEnv* env, jclass clazz, jlong handlePtr, jint tokenCount) {
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return;
    std::lock_guard<std::mutex> lock(h->mtx);
    h->reusable_prefix_token_count = tokenCount;
    ALOGI("【Config】设置可复用前缀Token数: %d (系统提示词: %d)", tokenCount, h->system_prompt_token_count);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_getReusablePrefixTokenCount(JNIEnv* env, jclass clazz, jlong handlePtr) {
    ModelHandle* h = reinterpret_cast<ModelHandle*>(handlePtr);
    if (!h) return 0;
    std::lock_guard<std::mutex> lock(h->mtx);
    return h->reusable_prefix_token_count;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_yuyan_imemodule_llm_LLMBridge_createEmbeddingInstance(JNIEnv* env, jclass, jstring jModelPath, jint nThreads, jint nGpuLayers) {
    auto t_start = std::chrono::high_resolution_clock::now();
    ALOGI("【Lifecycle】创建 Embedding Instance 调用");

    const char* cpath = env->GetStringUTFChars(jModelPath, nullptr);
    std::string modelPath(cpath ? cpath : "");
    ALOGI("创建 Embedding Instance: %s, 线程=%d, GPU层数=%d", cpath ? cpath : "", nThreads, nGpuLayers);

    llama_backend_init();

    struct llama_model* model = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_registry_mtx);
        auto it = g_model_registry.find(modelPath);
        if (it != g_model_registry.end()) {
            model = it->second.model;
            it->second.ref_count++;
            ALOGI("Embedding: 复用已加载的模型。新引用计数: %d", it->second.ref_count);
        } else {
            ALOGI("Embedding: 模型未加载，执行首次加载...");
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = nGpuLayers;
            mparams.use_mmap = true;
            model = llama_model_load_from_file(cpath, mparams);
            if (model) {
                g_model_registry[modelPath] = {model, 1};
            }
        }
    }

    if (!model) {
        ALOGE("Embedding模型加载失败");
        if (cpath) env->ReleaseStringUTFChars(jModelPath, cpath);
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_ubatch = 256;

    int n_threads_decode = 1;
    int n_threads_batch = 1;
    int n_big_cores = 1;
    choose_decode_and_batch_threads(nThreads, &n_threads_decode, &n_threads_batch, &n_big_cores);
    cparams.n_threads = n_threads_decode;
    cparams.n_threads_batch = n_threads_batch;
    ALOGI("Embedding线程策略 | requested=%d | decode=%d | batch=%d | est_big=%d", nThreads, n_threads_decode, n_threads_batch, n_big_cores);

    cparams.n_seq_max = 1;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        ALOGE("Embedding上下文创建失败");
        {
            std::lock_guard<std::mutex> lock(g_registry_mtx);
            auto it = g_model_registry.find(modelPath);
            if (it != g_model_registry.end()) {
                it->second.ref_count--;
                if (it->second.ref_count <= 0) {
                    llama_model_free(model);
                    g_model_registry.erase(it);
                }
            }
        }
        if (cpath) env->ReleaseStringUTFChars(jModelPath, cpath);
        return 0;
    }

    ModelHandle* handle = new ModelHandle();
    handle->model = model;
    handle->ctx = ctx;
    handle->cparams = cparams;
    handle->type = InstanceType::GENERATION;
    handle->model_path_key = modelPath;
    handle->current_lora = nullptr;

    maybe_attach_threadpool(handle, cparams.n_threads, cparams.n_threads_batch);

    if (cpath) env->ReleaseStringUTFChars(jModelPath, cpath);

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    ALOGI("Embedding Instance 创建成功, Handle: %p | 耗时: %lld ms", handle, duration);
    return reinterpret_cast<jlong>(handle);
}
