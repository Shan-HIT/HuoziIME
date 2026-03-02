#pragma once

#include <jni.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <memory>
#include <android/log.h>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <utility>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <list>

extern "C" {
#include "llama.h"
#include "ggml.h"
}

#define PREFILL_UBATCH_SIZE 512

#define SHIELD_ORIGINAL_LOGS 0
#define LOG_TAG "IMEM-OS-Native"
#define ALOG_METRIC(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "PERF_METRIC", "📊 " fmt, ##__VA_ARGS__)

#ifndef IMEM_DEFAULT_MIN_LOG_PRIORITY
#if defined(NDEBUG)
#define IMEM_DEFAULT_MIN_LOG_PRIORITY ANDROID_LOG_ERROR
#else
#define IMEM_DEFAULT_MIN_LOG_PRIORITY ANDROID_LOG_DEBUG
#endif
#endif

inline std::atomic<int> g_imem_min_log_priority{IMEM_DEFAULT_MIN_LOG_PRIORITY};
inline std::atomic<int> g_imem_perf_mode_depth{0};

inline int imem_get_min_log_priority() {
    return g_imem_min_log_priority.load(std::memory_order_relaxed);
}

inline void imem_set_min_log_priority(int min_priority) {
    g_imem_min_log_priority.store(min_priority, std::memory_order_relaxed);
}

inline int imem_get_perf_mode_depth() {
    return g_imem_perf_mode_depth.load(std::memory_order_relaxed);
}

inline int imem_enter_perf_mode() {
    return g_imem_perf_mode_depth.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline int imem_exit_perf_mode() {
    int cur = g_imem_perf_mode_depth.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (g_imem_perf_mode_depth.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed)) {
            return cur - 1;
        }
    }
    return 0;
}

inline bool imem_should_log(int prio) {
    // Perf-mode silences non-error logs.
    if (g_imem_perf_mode_depth.load(std::memory_order_relaxed) > 0 && prio < ANDROID_LOG_ERROR) {
        return false;
    }
    return prio >= g_imem_min_log_priority.load(std::memory_order_relaxed);
}

#if SHIELD_ORIGINAL_LOGS
    #define ALOGI(fmt, ...) ((void)0)
    #define ALOGW(fmt, ...) ((void)0)
    #define ALOGD(fmt, ...) ((void)0)
    #define ALOGE(fmt, ...) do { if (imem_should_log(ANDROID_LOG_ERROR)) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "❌ " fmt, ##__VA_ARGS__); } while (0)
#else
    #define ALOGI(fmt, ...) do { if (imem_should_log(ANDROID_LOG_INFO)) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "ℹ️ " fmt, ##__VA_ARGS__); } while (0)
    #define ALOGW(fmt, ...) do { if (imem_should_log(ANDROID_LOG_WARN)) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "⚠️ " fmt, ##__VA_ARGS__); } while (0)
    #define ALOGD(fmt, ...) do { if (imem_should_log(ANDROID_LOG_DEBUG)) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "🐛 " fmt, ##__VA_ARGS__); } while (0)
    #define ALOGE(fmt, ...) do { if (imem_should_log(ANDROID_LOG_ERROR)) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "❌ " fmt, ##__VA_ARGS__); } while (0)
#endif

#define ALOG_FORCE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "IMEM-FORCE", "🔥 " fmt, ##__VA_ARGS__)
#define ALOG_PFD(fmt, ...)   __android_log_print(ANDROID_LOG_ERROR, "IMEM-PFD", "🚀 " fmt, ##__VA_ARGS__)

extern JavaVM* gJvm;
extern std::string g_debug_log_path;

void debug_log_plaintext(const std::string& event_type, const std::string& content);
void llama_log_callback(ggml_log_level level, const char * text, void * user_data);
bool file_exists(const std::string& name);
void normalize_vector(std::vector<float>& vec);
bool is_valid_utf8(const std::string& str);
std::string sanitize_token_display(const std::string& input);
std::string tokens_to_str(const struct llama_model* model, const llama_token* tokens, int n_tokens);

// Creates a Java string from (potentially invalid) UTF-8 bytes.
jstring new_jstring_utf8_lenient(JNIEnv * env, const std::string & s);

struct LogBuffer {
    std::stringstream ss;
    void append(const std::string& str) {
        ss << str;
        if (ss.tellp() > 3000) {
            ALOGD("%s", ss.str().c_str());
            ss.str("");
            ss.clear();
        }
    }
    void flush() {
        if (ss.tellp() > 0) {
            ALOGD("%s", ss.str().c_str());
            ss.str("");
            ss.clear();
        }
    }
};

struct RadixNode {
    static int _global_id_counter;
    int id;
    llama_token token;
    int kv_pos;                          
    std::vector<RadixNode*> children;
    RadixNode* parent;
    long long last_access;               
    bool is_hot;                         
    bool is_soft_deleted;                
    llama_token user_preference_token = -1;

    RadixNode(llama_token t, int pos, RadixNode* p) : token(t), kv_pos(pos), parent(p) {
        id = ++_global_id_counter;
        last_access = std::chrono::system_clock::now().time_since_epoch().count();
        is_hot = false;
        is_soft_deleted = false;
        user_preference_token = -1;
    }
    ~RadixNode() {
        for (auto c : children) delete c;
    }
    
    void touch() {
        last_access = std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    RadixNode* find_child(llama_token t) const {
        for (auto c : children) {
            if (c->token == t) return c;
        }
        return nullptr;
    }
};

struct RadixLookupResult {
    RadixNode* matched_node;    
    int matched_length;         
    int kv_pos;                 
    bool full_match;            
    
    RadixLookupResult() : matched_node(nullptr), matched_length(0), kv_pos(-1), full_match(false) {}
};

enum class InstanceType { GENERATION };

constexpr int CURSORKV_MAX_BRANCHES = 3;        
constexpr int CURSORKV_EVICTION_THRESHOLD = 512; 

struct ModelHandle {
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;
    std::atomic_bool stop_flag{false};
    std::atomic<uint64_t> generation_seq{0};
    std::thread gen_thread;
    std::mutex mtx;
    std::vector<llama_token> current_tokens;
    RadixNode* tree_root = nullptr;
    RadixNode* cursor_node = nullptr;           
    llama_context_params cparams;
    InstanceType type = InstanceType::GENERATION;
    struct llama_adapter_lora* current_lora = nullptr;
    std::string current_lora_path = "None";
    bool is_system_prompt_cached = false;
    int system_prompt_token_count = 0;
    int reusable_prefix_token_count = 0;
    std::string cache_path = "";
    std::string model_path_key = "";
    std::string session_signature = "";

    std::atomic_bool disable_kv_reuse{false};

    llama_token tok_mem_open  = (llama_token)-1;   // <MEM_RETRIEVAL>
    llama_token tok_mem_close = (llama_token)-1;   // </MEM_RETRIEVAL>
    llama_token tok_no_mem    = (llama_token)-1;   // <NO_MEM>

    ModelHandle() {
        tree_root = new RadixNode(-1, -1, nullptr);
        tree_root->is_hot = true;
        cursor_node = tree_root;                
    }
    ~ModelHandle() {
        if (tree_root) delete tree_root;
    }
};

struct ModelEntry {
    llama_model* model;
    int ref_count;
};

void print_tree_recursive(RadixNode* node, std::string prefix, bool is_last, LogBuffer& logger, const struct llama_model* model, int depth, int system_token_count);
void reset_hot_status_recursive(RadixNode* node, int system_threshold);
void activate_current_path(RadixNode* root, const std::vector<llama_token>& active_tokens);
void refresh_tree_hot_status(RadixNode* root, const std::vector<llama_token>& active_tokens, int system_threshold);
void print_lru_queue(RadixNode* root, int system_threshold);
void forceStopAndJoin(ModelHandle* h, const char* source = "unknown");

RadixLookupResult radix_lookup(RadixNode* root, const llama_token* tokens, int n_tokens);

RadixNode* radix_insert(RadixNode* parent, const llama_token* tokens, int n_tokens, int start_kv_pos);

RadixNode* radix_soft_delete(RadixNode* node);

bool radix_restore(RadixNode* node);

int radix_evict_cold(ModelHandle* h, RadixNode* root, int max_evict, int system_threshold);

int radix_count_nodes(RadixNode* root);

void radix_get_kv_range(RadixNode* node, int* p0_out, int* p1_out);

void sync_radix_tree_from_tokens(ModelHandle* h);