#pragma once

#include "imem_core.h"
#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <atomic>
#include <memory>
#include <mutex>

enum class PerfMetricType : int {
    PREFILL = 0,
    DECODE_STEP = 1,
    GENERATION_SESSION = 2,
    CONTEXT_SYNC = 5,
};

struct PerfMetric {
    PerfMetricType type;
    int64_t timestamp_ns;
    std::string session_id;

    
    std::unordered_map<std::string, std::string> string_fields;
    std::unordered_map<std::string, int64_t> int_fields;
    std::unordered_map<std::string, double> double_fields;
    std::unordered_map<std::string, bool> bool_fields;

    
    void set_string(const std::string& key, const std::string& value) {
        string_fields[key] = value;
    }

    void set_int(const std::string& key, int64_t value) {
        int_fields[key] = value;
    }

    void set_double(const std::string& key, double value) {
        double_fields[key] = value;
    }

    void set_bool(const std::string& key, bool value) {
        bool_fields[key] = value;
    }

    std::string to_json() const {
        std::stringstream ss;
        ss << "{";
        ss << "\"type\":" << static_cast<int>(type) << ",";
        ss << "\"timestamp_ns\":" << timestamp_ns << ",";
        ss << "\"session_id\":\"" << session_id << "\",";

        
        ss << "\"strings\":{";
        bool first = true;
        for (const auto& kv : string_fields) {
            if (!first) ss << ",";
            ss << "\"" << kv.first << "\":\"" << kv.second << "\"";
            first = false;
        }
        ss << "},";

        
        ss << "\"ints\":{";
        first = true;
        for (const auto& kv : int_fields) {
            if (!first) ss << ",";
            ss << "\"" << kv.first << "\":" << kv.second;
            first = false;
        }
        ss << "},";

        
        ss << "\"doubles\":{";
        first = true;
        for (const auto& kv : double_fields) {
            if (!first) ss << ",";
            ss << "\"" << kv.first << "\":" << kv.second;
            first = false;
        }
        ss << "},";

        
        ss << "\"bools\":{";
        first = true;
        for (const auto& kv : bool_fields) {
            if (!first) ss << ",";
            ss << "\"" << kv.first << "\":" << (kv.second ? "true" : "false");
            first = false;
        }
        ss << "}";

        ss << "}";
        return ss.str();
    }
};


class PerfTimer {
public:
    PerfTimer() : start_time(std::chrono::high_resolution_clock::now()) {}

    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    
    int64_t elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_time).count();
    }

    
    int64_t elapsed_ms() const {
        return elapsed_ns() / 1000000;
    }

    
    int64_t elapsed_us() const {
        return elapsed_ns() / 1000;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};


class PerfSession {
public:
    PerfSession(const std::string& session_id, PerfMetricType session_type)
        : session_id(session_id), session_type(session_type), start_ns(0) {

        auto now = std::chrono::high_resolution_clock::now();
        start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }

    std::string get_session_id() const { return session_id; }

    int64_t get_elapsed_ns() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count() - start_ns;
    }

    int64_t get_elapsed_ms() const {
        return get_elapsed_ns() / 1000000;
    }

    
    void record_prefill(int prompt_length, int token_count, int reuse_count,
                        int64_t time_ms, double rate, int64_t kv_cache_before, int64_t kv_cache_after) {
        PerfMetric metric;
        metric.type = PerfMetricType::PREFILL;
        metric.timestamp_ns = start_ns;
        metric.session_id = session_id;

        metric.set_int("prompt_length", prompt_length);
        metric.set_int("token_count", token_count);
        metric.set_int("reuse_token_count", reuse_count);
        metric.set_int("time_ms", time_ms);
        metric.set_double("rate_tokens_per_sec", rate);
        metric.set_int("kv_cache_size_before", kv_cache_before);
        metric.set_int("kv_cache_size_after", kv_cache_after);

        add_metric(metric);
    }

    
    void record_decode_step(int step, const std::string& token, int token_id,
                           int64_t time_ms, int64_t cumulative_ms, double tps,
                           bool kv_hit, int branch_count) {
        PerfMetric metric;
        metric.type = PerfMetricType::DECODE_STEP;
        metric.timestamp_ns = start_ns;
        metric.session_id = session_id;

        metric.set_int("step_number", step);
        metric.set_string("token_generated", token);
        metric.set_int("token_id", token_id);
        metric.set_int("time_ms", time_ms);
        metric.set_int("cumulative_time_ms", cumulative_ms);
        metric.set_double("tokens_per_sec", tps);
        metric.set_bool("kv_cache_hit", kv_hit);
        metric.set_int("candidate_branch_count", branch_count);

        add_metric(metric);
    }

    
    void complete_session(const std::string& mode, const std::string& prompt,
                         const std::vector<std::string>& candidates,
                         int64_t first_token_latency_ms, int64_t prefill_ms,
                         int64_t decode_ms, int total_tokens, bool success) {
        PerfMetric metric;
        metric.type = PerfMetricType::GENERATION_SESSION;
        metric.timestamp_ns = start_ns;
        metric.session_id = session_id;

        metric.set_string("mode", mode);
        metric.set_string("prompt", prompt);
        metric.set_int("total_time_ms", get_elapsed_ms());
        metric.set_int("first_token_latency_ms", first_token_latency_ms);
        metric.set_int("prefill_time_ms", prefill_ms);
        metric.set_int("decode_time_ms", decode_ms);
        metric.set_int("total_tokens_generated", total_tokens);

        double avg_tps = (decode_ms > 0) ? (total_tokens * 1000.0) / decode_ms : 0.0;
        metric.set_double("avg_tokens_per_sec", avg_tps);

        metric.set_bool("success", success);

        add_metric(metric);
    }

    
    void record_context_sync(const std::string& sync_type, int history_len,
                            int last_msg_len, int64_t prefill_ms,
                            int64_t session_load_ms, bool success) {
        PerfMetric metric;
        metric.type = PerfMetricType::CONTEXT_SYNC;
        metric.timestamp_ns = start_ns;
        metric.session_id = session_id;

        metric.set_string("sync_type", sync_type);
        metric.set_int("history_length", history_len);
        metric.set_int("last_msg_length", last_msg_len);
        metric.set_int("prefill_time_ms", prefill_ms);
        metric.set_int("session_load_time_ms", session_load_ms);
        metric.set_int("total_time_ms", prefill_ms + session_load_ms);
        metric.set_bool("success", success);

        add_metric(metric);
    }


    
    const std::vector<PerfMetric>& get_metrics() const {
        return metrics;
    }

    
    void clear() {
        metrics.clear();
    }

private:
    std::string session_id;
    PerfMetricType session_type;
    int64_t start_ns;
    std::vector<PerfMetric> metrics;

    void add_metric(const PerfMetric& metric) {
        metrics.push_back(metric);

        
        ALOG_METRIC("Session %s | %s | %s",
                   session_id.c_str(),
                   metric_type_to_string(metric.type).c_str(),
                   metric_summary(metric).c_str());
    }

    static std::string metric_type_to_string(PerfMetricType type) {
        switch (type) {
            case PerfMetricType::PREFILL: return "PREFILL";
            case PerfMetricType::DECODE_STEP: return "DECODE";
            case PerfMetricType::GENERATION_SESSION: return "SESSION";
            case PerfMetricType::CONTEXT_SYNC: return "SYNC";
            default: return "UNKNOWN";
        }
    }

    static std::string metric_summary(const PerfMetric& metric) {
        std::stringstream ss;
        switch (metric.type) {
            case PerfMetricType::PREFILL: {
                auto it = metric.int_fields.find("time_ms");
                if (it != metric.int_fields.end()) {
                    ss << "time=" << it->second << "ms";
                }
                break;
            }
            case PerfMetricType::DECODE_STEP: {
                auto it_step = metric.int_fields.find("step_number");
                auto it_time = metric.int_fields.find("time_ms");
                if (it_step != metric.int_fields.end() && it_time != metric.int_fields.end()) {
                    ss << "step=" << it_step->second << " time=" << it_time->second << "ms";
                }
                break;
            }
            case PerfMetricType::GENERATION_SESSION: {
                auto it = metric.int_fields.find("total_time_ms");
                if (it != metric.int_fields.end()) {
                    ss << "total=" << it->second << "ms";
                }
                break;
            }
            default:
                ss << "recorded";
                break;
        }
        return ss.str();
    }
};


class PerfTracker {
public:
    static PerfTracker& instance() {
        static PerfTracker tracker;
        return tracker;
    }

    
    std::shared_ptr<PerfSession> create_session(const std::string& session_id, PerfMetricType type) {
        auto session = std::make_shared<PerfSession>(session_id, type);
        std::lock_guard<std::mutex> lock(sessions_mutex);
        sessions[session_id] = session;
        return session;
    }

    
    std::shared_ptr<PerfSession> get_session(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        auto it = sessions.find(session_id);
        if (it != sessions.end()) {
            return it->second;
        }
        return nullptr;
    }

    
    void remove_session(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        sessions.erase(session_id);
    }

    
    static std::string generate_session_id() {
        static std::atomic<uint64_t> counter{0};
        auto now = std::chrono::system_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
        return "sess_" + std::to_string(ns) + "_" + std::to_string(counter.fetch_add(1));
    }

    
    std::string export_all_json() {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        std::stringstream ss;
        ss << "[\n";

        bool first = true;
        for (const auto& kv : sessions) {
            const auto& metrics = kv.second->get_metrics();
            for (const auto& metric : metrics) {
                if (!first) ss << ",\n";
                ss << "  " << metric.to_json();
                first = false;
            }
        }

        ss << "\n]\n";
        return ss.str();
    }

    
    void clear_all() {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        sessions.clear();
    }

    
    size_t session_count() {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        return sessions.size();
    }

private:
    PerfTracker() = default;
    ~PerfTracker() = default;

    std::unordered_map<std::string, std::shared_ptr<PerfSession>> sessions;
    std::mutex sessions_mutex;
};


#define PERF_SCOPED_TIMER(name) PerfTimer __perf_timer_##name
#define PERF_GET_ELAPSED_MS(name) __perf_timer_##name.elapsed_ms()
#define PERF_GET_ELAPSED_US(name) __perf_timer_##name.elapsed_us()
#define PERF_GET_ELAPSED_NS(name) __perf_timer_##name.elapsed_ns()


#define PERF_CREATE_SESSION(type) \
    PerfTracker::instance().create_session(PerfTracker::generate_session_id(), PerfMetricType::type)

#define PERF_GET_SESSION(id) \
    PerfTracker::instance().get_session(id)
