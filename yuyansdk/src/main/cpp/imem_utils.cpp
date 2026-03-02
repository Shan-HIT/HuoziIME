#include "imem_core.h"

std::string g_debug_log_path = "";
int RadixNode::_global_id_counter = 0;

void debug_log_plaintext(const std::string& event_type, const std::string& content) {
    if (g_debug_log_path.empty()) return;
    std::ofstream f(g_debug_log_path, std::ios::app);
    if (f.good()) {
        std::string escaped;
        escaped.reserve(content.size() * 1.1);
        for (char c : content) {
            if (c == '"') escaped += "\\\"";
            else if (c == '\\') escaped += "\\\\";
            else if (c == '\n') escaped += "\\n";
            else if (c == '\r') escaped += "";
            else escaped += c;
        }
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
        f << "{\"time\": \"" << time_buf << "\", \"event\": \"" << event_type << "\", \"content\": \"" << escaped << "\"}\n";
    }
}

void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) __android_log_print(ANDROID_LOG_ERROR, "LLAMA_CPP", "%s", text);
}

void normalize_vector(std::vector<float>& vec) {
    float sum = 0.0f;
    for (float v : vec) sum += v * v;
    float norm = std::sqrt(sum);
    if (norm > 1e-5f) { 
        float inv_norm = 1.0f / norm;
        for (float& v : vec) v *= inv_norm;
    }
}

bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

bool is_valid_utf8(const std::string& str) {
    const unsigned char* bytes = (const unsigned char*)str.data();
    size_t len = str.length();
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = bytes[i];
        if (c < 0x80) continue;
        int extra = 0;
        if ((c & 0xE0) == 0xC0) extra = 1;
        else if ((c & 0xF0) == 0xE0) extra = 2;
        else if ((c & 0xF8) == 0xF0) extra = 3;
        else return false;
        if (i + extra >= len) return false;
        for (int j = 0; j < extra; ++j) {
            if ((bytes[++i] & 0xC0) != 0x80) return false;
        }
    }
    return true;
}

std::string sanitize_token_display(const std::string& input) {
    if (input == "\n") return "\\n";
    if (input == "\r") return "\\r";
    if (input == "\t") return "\\t";
    if (is_valid_utf8(input)) {
        return input;
    } else {
        std::stringstream ss;
        ss << "[HEX";
        for (unsigned char c : input) {
            ss << ":" << std::hex << std::setw(2) << std::setfill('0') << (int)c;
        }
        ss << "]";
        return ss.str();
    }
}

std::string tokens_to_str(const struct llama_model* model, const llama_token* tokens, int n_tokens) {
    if (!model || !tokens || n_tokens <= 0) return "";
    std::string result;
    const auto* vocab = llama_model_get_vocab(model);
    char buf[256];
    for (int i = 0; i < n_tokens; ++i) {
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, false);
        if (n > 0) {
            result += std::string(buf, n);
        }
    }
    return result;
}

void print_tree_recursive(RadixNode* node, std::string prefix, bool is_last, LogBuffer& logger, const struct llama_model* model, int depth, int system_token_count) {
    bool is_system_node = (node->kv_pos != -1 && node->kv_pos < system_token_count);
    bool is_system_end = (node->kv_pos == system_token_count - 1);
    bool print_this_node = true;
    std::string display_token = "ROOT";
    if (node->kv_pos == -1 && node->token == -1) {
    } else if (is_system_node) {
        if (is_system_end) {
            display_token = "KV[System][0:" + std::to_string(system_token_count) + "]";
        } else {
            print_this_node = false;
        }
    } else if (node->token != -1 && model) {
        const auto* vocab = llama_model_get_vocab(model);
        char buf[64];
        int n = llama_token_to_piece(vocab, node->token, buf, 64, 0, false);
        if (n > 0) {
            std::string s(buf, n);
            display_token = sanitize_token_display(s);
        } else {
            display_token = "[UNK]";
        }
    }
    if (print_this_node) {
        std::stringstream line_ss;
        line_ss << prefix;
        line_ss << (is_last ? "└── " : "├── ");
        std::string status = node->is_hot ? "HOT " : "COLD";
        if (display_token.find("KV[System]") != std::string::npos) {
            line_ss << "[" << status << "] 🔥🔥 " << display_token << " (Aggregate Root)";
        } else {
            line_ss << "[" << status << "] Token: \"" << display_token << "\" "
                    << "(ID:" << node->id << ", Pos:" << node->kv_pos << ")";
        }
        line_ss << "\n";
        logger.append(line_ss.str());
    }
    std::string next_prefix = prefix;
    if (print_this_node) {
        next_prefix += (is_last ? "    " : "│   ");
    }
    for (size_t i = 0; i < node->children.size(); ++i) {
        print_tree_recursive(node->children[i], next_prefix, i == node->children.size() - 1, logger, model, depth + 1, system_token_count);
    }
}

void reset_hot_status_recursive(RadixNode* node, int system_threshold) {
    if (node->kv_pos != -1 && node->kv_pos < system_threshold) {
        node->is_hot = true;
    } else {
        node->is_hot = false; 
    }
    if (node->kv_pos == -1) node->is_hot = true;
    for (auto child : node->children) {
        reset_hot_status_recursive(child, system_threshold);
    }
}

void activate_current_path(RadixNode* root, const std::vector<llama_token>& active_tokens) {
    RadixNode* curr = root;
    curr->is_hot = true;
    for (auto t : active_tokens) {
        bool found = false;
        for (auto child : curr->children) {
            if (child->token == t) {
                child->is_hot = true; 
                curr = child;
                found = true;
                break;
            }
        }
        if (!found) break; 
    }
}

void refresh_tree_hot_status(RadixNode* root, const std::vector<llama_token>& active_tokens, int system_threshold) {
    reset_hot_status_recursive(root, system_threshold);
    activate_current_path(root, active_tokens);
}

void print_lru_queue(RadixNode* root, int system_threshold) {
    std::vector<RadixNode*> all_nodes;
    std::list<RadixNode*> q;
    q.push_back(root);
    while(!q.empty()) {
        RadixNode* curr = q.front(); q.pop_front();
        if(curr->token != -1) all_nodes.push_back(curr);
        for(auto c : curr->children) q.push_back(c);
    }
    std::sort(all_nodes.begin(), all_nodes.end(), [](RadixNode* a, RadixNode* b){
        return a->last_access > b->last_access;
    });
    std::stringstream ss;
    ss << "\n=== Output 2: LRU Zone Verification (Phase 2) ===\n";
    ss << std::left << std::setw(8) << "[Rank]"
       << std::setw(15) << "[Zone]"
       << std::setw(10) << "[Status]"
       << std::setw(12) << "[TimeDelta]"
       << "Node Info\n";
    ss << "----------------------------------------------------------------\n";
    long long now = std::chrono::system_clock::now().time_since_epoch().count();
    for(size_t i=0; i<all_nodes.size(); ++i) {
        if(i > 25) { ss << "... (truncated for brevity)\n"; break; }
        RadixNode* n = all_nodes[i];
        long long diff_ms = (now - n->last_access) / 10000;
        std::string zone = n->is_hot ? "PROTECTION" : "EVICTION";
        std::string status = n->is_hot ? "HOT" : "COLD";
        ss << "[" << std::setw(6) << i << "] "
           << std::setw(15) << zone
           << std::setw(10) << status
           << "T-" << std::setw(6) << diff_ms << "ms | "
           << "ID:" << std::setw(4) << n->id
           << " Pos:" << n->kv_pos << "\n";
    }
    ss << "================================================================";
    ALOGD("%s", ss.str().c_str());
}

void forceStopAndJoin(ModelHandle* h, const char* source) {
    if (!h) return;
    h->stop_flag.store(true);
    uint64_t cancel_seq = h->generation_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    std::thread thread_to_join;
    {
        std::lock_guard<std::mutex> lock(h->mtx);
        if (h->gen_thread.joinable()) {
            thread_to_join = std::move(h->gen_thread);
            ALOGI("forceStopAndJoin | 正在停止线程 (来源: %s, seq=%llu)", source, (unsigned long long)cancel_seq);
        }
    }
    if (thread_to_join.joinable()) {
        if (thread_to_join.get_id() == std::this_thread::get_id()) {
            ALOGW("forceStopAndJoin | 检测到同线程 join，改为 detach 避免死锁 (来源: %s, seq=%llu)", source, (unsigned long long)cancel_seq);
            thread_to_join.detach();
        } else {
            thread_to_join.join();
            ALOGI("forceStopAndJoin | 线程已安全结束");
        }
    }
    h->stop_flag.store(false);
}

void sync_radix_tree_from_tokens(ModelHandle* h) {
    if (!h || h->current_tokens.empty()) {
        ALOGI("【RadixSync】跳过同步 (tokens为空)");
        return;
    }
    
    int protection_threshold = std::max(h->system_prompt_token_count, h->reusable_prefix_token_count);
    refresh_tree_hot_status(h->tree_root, h->current_tokens, protection_threshold);
    
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
        ALOGI("【RadixSync】树路径已重建 | 已匹配: %d | 新插入: %zu | cursor_node ID: %d",
              lookup.matched_length, h->current_tokens.size() - start_pos,
              h->cursor_node ? h->cursor_node->id : -1);
    } else {
        h->cursor_node = lookup.matched_node;
        ALOGI("【RadixSync】树路径完全匹配 | Token数: %zu | cursor_node ID: %d",
              h->current_tokens.size(), h->cursor_node ? h->cursor_node->id : -1);
    }
}

RadixLookupResult radix_lookup(RadixNode* root, const llama_token* tokens, int n_tokens) {
    RadixLookupResult result;
    if (!root || n_tokens <= 0) {
        result.matched_node = root;
        return result;
    }
    
    RadixNode* current = root;
    int matched = 0;
    
    for (int i = 0; i < n_tokens; ++i) {
        RadixNode* child = current->find_child(tokens[i]);
        if (child && !child->is_soft_deleted) {
            child->touch();
            current = child;
            matched++;
        } else {
            if (child && child->is_soft_deleted) {
                result.matched_node = current;
                result.matched_length = matched;
                result.kv_pos = current->kv_pos;
                result.full_match = false;
                return result;
            }
            break;
        }
    }
    
    result.matched_node = current;
    result.matched_length = matched;
    result.kv_pos = current->kv_pos;
    result.full_match = (matched == n_tokens);
    
    return result;
}

RadixNode* radix_insert(RadixNode* parent, const llama_token* tokens, int n_tokens, int start_kv_pos) {
    if (!parent || n_tokens <= 0) return parent;
    RadixNode* current = parent;
    for (int i = 0; i < n_tokens; ++i) {
        llama_token tok = tokens[i];
        int kv_pos = start_kv_pos + i;
        RadixNode* existing = current->find_child(tok);
        if (existing) {
            existing->is_soft_deleted = false;
            existing->touch();
            current = existing;
        } else {
            RadixNode* new_node = new RadixNode(tok, kv_pos, current);
            new_node->is_hot = true;
            current->children.push_back(new_node);
            current = new_node;
        }
    }
    
    return current;
}

RadixNode* radix_soft_delete(RadixNode* node) {
    if (!node || node->parent == nullptr) return node;    
    node->is_soft_deleted = true;
    node->is_hot = false;    
    return node->parent;
}

bool radix_restore(RadixNode* node) {
    if (!node) return false;
    
    RadixNode* current = node;
    while (current) {
        current->is_soft_deleted = false;
        current->touch();
        current = current->parent;
    }
    
    return true;
}

static void collect_cold_leaves(RadixNode* node, std::vector<RadixNode*>& leaves, int system_threshold) {
    if (!node) return;
    
    if (node->kv_pos >= 0 && node->kv_pos < system_threshold) return;
    
    if (node->is_hot) return;
    
    if (node->is_soft_deleted) return;
    
    if (node->children.empty()) {
        leaves.push_back(node);
    } else {
        for (auto child : node->children) {
            collect_cold_leaves(child, leaves, system_threshold);
        }
    }
}

static void delete_subtree(RadixNode* node) {
    if (!node) return;
    for (auto child : node->children) delete_subtree(child);
    delete node;
}

int radix_evict_cold(ModelHandle* h, RadixNode* root, int max_evict, int system_threshold) {
    if (!h || !root || max_evict <= 0 || !h->ctx) return 0;

    llama_memory_t kv_mem = llama_get_memory(h->ctx);
    std::vector<RadixNode*> cold_leaves;
    collect_cold_leaves(root, cold_leaves, system_threshold);

    if (cold_leaves.empty()) return 0;

    std::sort(cold_leaves.begin(), cold_leaves.end(), [](RadixNode* a, RadixNode* b) {
        return a->last_access < b->last_access;
    });

    int evicted = 0;
    int min_cut_pos = -1;
    for (RadixNode* leaf : cold_leaves) {
        if (evicted >= max_evict) break;
        if (!leaf->parent) continue;

        int p0 = 0, p1 = 0;
        radix_get_kv_range(leaf, &p0, &p1);
        if (p0 < 0) continue;
        if (kv_mem) {
            llama_memory_seq_rm(kv_mem, -1, p0, p1);
        }

        auto& siblings = leaf->parent->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), leaf), siblings.end());
        delete_subtree(leaf);

        if (min_cut_pos == -1 || p0 < min_cut_pos) min_cut_pos = p0;
        evicted++;
    }

    if (min_cut_pos >= 0) {
        std::lock_guard<std::mutex> lock(h->mtx);
        if ((size_t)min_cut_pos < h->current_tokens.size()) {
            h->current_tokens.resize((size_t)min_cut_pos);
        }
        h->cursor_node = root;
        int protection_threshold = std::max(h->system_prompt_token_count, h->reusable_prefix_token_count);
        refresh_tree_hot_status(root, h->current_tokens, protection_threshold);
    }

    ALOGI("CursorKV | Evicted %d cold leaf subtrees", evicted);
    return evicted;
}

int radix_count_nodes(RadixNode* root) {
    if (!root) return 0;
    
    int count = 1;  // Count self
    for (auto child : root->children) {
        count += radix_count_nodes(child);
    }
    return count;
}

void radix_get_kv_range(RadixNode* node, int* p0_out, int* p1_out) {
    if (!node || !p0_out || !p1_out) return;
    
    *p0_out = node->kv_pos;
    
    int max_pos = node->kv_pos;
    std::list<RadixNode*> queue;
    queue.push_back(node);
    
    while (!queue.empty()) {
        RadixNode* curr = queue.front();
        queue.pop_front();
        
        if (curr->kv_pos > max_pos) {
            max_pos = curr->kv_pos;
        }
        for (auto child : curr->children) {
            queue.push_back(child);
        }
    }
    
    *p1_out = max_pos + 1;
}

static inline void append_replacement(std::vector<jchar> & out) {
    out.push_back((jchar) 0xFFFD);
}

jstring new_jstring_utf8_lenient(JNIEnv * env, const std::string & s) {
    if (!env) return nullptr;
    if (s.empty()) return env->NewString(reinterpret_cast<const jchar*>(u""), 0);

    std::vector<jchar> out;
    out.reserve(s.size());

    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(s.data());
    const size_t len = s.size();
    size_t i = 0;

    while (i < len) {
        const uint8_t c0 = bytes[i];

        if (c0 < 0x80) {
            out.push_back((jchar) c0);
            ++i;
            continue;
        }

        // 2-byte
        if ((c0 & 0xE0) == 0xC0) {
            if (i + 1 >= len) {
                append_replacement(out);
                break;
            }
            const uint8_t c1 = bytes[i + 1];
            if ((c1 & 0xC0) != 0x80) {
                append_replacement(out);
                ++i;
                continue;
            }
            const uint32_t cp = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(c1 & 0x3F);
            if (cp < 0x80) {
                append_replacement(out);
                i += 2;
                continue;
            }
            out.push_back((jchar) cp);
            i += 2;
            continue;
        }

        // 3-byte
        if ((c0 & 0xF0) == 0xE0) {
            if (i + 2 >= len) {
                append_replacement(out);
                break;
            }
            const uint8_t c1 = bytes[i + 1];
            const uint8_t c2 = bytes[i + 2];
            if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80)) {
                append_replacement(out);
                ++i;
                continue;
            }
            const uint32_t cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(c1 & 0x3F) << 6) | (uint32_t)(c2 & 0x3F);
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
                append_replacement(out);
                i += 3;
                continue;
            }
            out.push_back((jchar) cp);
            i += 3;
            continue;
        }

        // 4-byte
        if ((c0 & 0xF8) == 0xF0) {
            if (i + 3 >= len) {
                append_replacement(out);
                break;
            }
            const uint8_t c1 = bytes[i + 1];
            const uint8_t c2 = bytes[i + 2];
            const uint8_t c3 = bytes[i + 3];
            if (((c1 & 0xC0) != 0x80) || ((c2 & 0xC0) != 0x80) || ((c3 & 0xC0) != 0x80)) {
                append_replacement(out);
                ++i;
                continue;
            }
            const uint32_t cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(c1 & 0x3F) << 12) | ((uint32_t)(c2 & 0x3F) << 6) | (uint32_t)(c3 & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) {
                append_replacement(out);
                i += 4;
                continue;
            }
            const uint32_t u = cp - 0x10000;
            const jchar high = (jchar)(0xD800 + (u >> 10));
            const jchar low  = (jchar)(0xDC00 + (u & 0x3FF));
            out.push_back(high);
            out.push_back(low);
            i += 4;
            continue;
        }

        append_replacement(out);
        ++i;
    }

    if (out.empty()) return env->NewString(reinterpret_cast<const jchar*>(u""), 0);
    return env->NewString(out.data(), (jsize) out.size());
}