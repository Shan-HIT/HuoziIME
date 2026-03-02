#include "imem_vector_db.h"
#include "imem_core.h"

#include <android/log.h>
#include <mutex>
#include <fstream>
#include <unordered_map>

#include "hnswlib/hnswlib.h"

namespace imem {

static std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

static void l2_normalize(std::vector<float> & v) {
    normalize_vector(v);
}

struct VectorDb {
    int dim = 0;
    int max_elements = 0;
    std::string index_dir;

    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;

    std::vector<std::string> texts; // label -> text
    std::mutex mtx;

    long long next_label = 1;
};

static std::unordered_map<ModelHandle*, std::unique_ptr<VectorDb>> g_vdb;
static std::mutex g_vdb_mtx;

static bool ensure_dir_exists(const std::string & path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
        return (st.st_mode & S_IFDIR) != 0;
    }
#if defined(_WIN32)
    return false;
#else
    return mkdir(path.c_str(), 0777) == 0;
#endif
}

static bool save_texts(const VectorDb & db) {
    const std::string path = join_path(db.index_dir, "texts.jsonl");
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.good()) return false;

    for (size_t i = 1; i < db.texts.size(); ++i) {
        std::string t = db.texts[i];
        size_t pos = 0;
        while ((pos = t.find('"', pos)) != std::string::npos) {
            t.insert(pos, "\\");
            pos += 2;
        }
        out << "{\"id\":" << (long long)i << ",\"text\":\"" << t << "\"}\n";
    }
    return true;
}

static bool load_texts(VectorDb & db) {
    const std::string path = join_path(db.index_dir, "texts.jsonl");
    if (!file_exists(path)) return false;

    std::ifstream in(path);
    if (!in.good()) return false;

    db.texts.clear();
    db.texts.resize(1);

    std::string line;
    while (std::getline(in, line)) {
        const auto id_pos = line.find("\"id\"");
        const auto text_pos = line.find("\"text\"");
        if (id_pos == std::string::npos || text_pos == std::string::npos) continue;

        const auto colon = line.find(':', id_pos);
        const auto comma = line.find(',', colon);
        if (colon == std::string::npos) continue;
        const std::string id_str = line.substr(colon + 1, (comma == std::string::npos ? std::string::npos : comma - colon - 1));
        const long long id = std::atoll(id_str.c_str());
        if (id <= 0) continue;

        const auto text_colon = line.find(':', text_pos);
        const auto first_quote = line.find('"', text_colon + 1);
        if (first_quote == std::string::npos) continue;
        const auto second_quote = line.rfind('"');
        if (second_quote == std::string::npos || second_quote <= first_quote) continue;
        std::string text = line.substr(first_quote + 1, second_quote - first_quote - 1);
        size_t p = 0;
        while ((p = text.find("\\\"", p)) != std::string::npos) {
            text.erase(p, 1);
            p += 1;
        }

        if ((size_t)id >= db.texts.size()) db.texts.resize((size_t)id + 1);
        db.texts[(size_t)id] = std::move(text);
        db.next_label = std::max(db.next_label, id + 1);
    }

    return true;
}

static bool vdb_has(ModelHandle * h) {
    std::lock_guard<std::mutex> lock(g_vdb_mtx);
    return g_vdb.find(h) != g_vdb.end();
}

static VectorDb * vdb_get(ModelHandle * h) {
    std::lock_guard<std::mutex> lock(g_vdb_mtx);
    auto it = g_vdb.find(h);
    return it == g_vdb.end() ? nullptr : it->second.get();
}

static std::vector<float> embed_text(ModelHandle * h, const std::string & text) {
    std::vector<float> result;
    if (!h || !h->ctx || !h->model) return result;

    std::lock_guard<std::mutex> lock(h->mtx);

    const auto * vocab = llama_model_get_vocab(h->model);
    std::vector<llama_token> tokens(text.size() + 32);
    int n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), tokens.data(), (int)tokens.size(), false, true);
    if (n_tokens <= 0) return result;
    tokens.resize(n_tokens);

    const int dim = llama_model_n_embd(h->model);
    if (dim <= 0) return result;

    llama_seq_id embed_seq = 0;
    const int n_seq_max = (int) h->cparams.n_seq_max;
    if (n_seq_max > 1) {
        embed_seq = (llama_seq_id) (n_seq_max - 1);
        if (embed_seq == 0) embed_seq = 1;
    }

    llama_memory_t kv_mem = llama_get_memory(h->ctx);
    llama_memory_seq_rm(kv_mem, embed_seq, -1, -1);

    llama_set_embeddings(h->ctx, true);

    int n_batch = (int)h->cparams.n_batch;
    if (n_batch <= 0) n_batch = 256;

    const enum llama_pooling_type pooling_type = llama_pooling_type(h->ctx);

    std::vector<float> sum((size_t)dim, 0.0f);
    int pooled_count = 0;

    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = n_tokens - i;
        if (n_eval > n_batch) n_eval = n_batch;
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        batch.n_tokens = n_eval;
        for (int j = 0; j < n_eval; ++j) {
            batch.token[j] = tokens[i + j];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = embed_seq;
            batch.logits[j] = true;
        }
        if (llama_decode(h->ctx, batch) != 0) {
            llama_batch_free(batch);
            llama_set_embeddings(h->ctx, false);
            return {};
        }

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int j = 0; j < n_eval; ++j) {
                const int global_tok_idx = i + j;
                if (global_tok_idx == 0) continue;
                const float * v = llama_get_embeddings_ith(h->ctx, j);
                if (!v) continue;
                for (int d = 0; d < dim; ++d) {
                    sum[(size_t)d] += v[d];
                }
                pooled_count++;
            }
        }

        llama_batch_free(batch);
    }

    if (pooling_type != LLAMA_POOLING_TYPE_NONE) {
        float * embd_seq = llama_get_embeddings_seq(h->ctx, embed_seq);
        if (!embd_seq) {
            llama_set_embeddings(h->ctx, false);
            return {};
        }
        result.resize(dim);
        std::memcpy(result.data(), embd_seq, (size_t)dim * sizeof(float));
    } else {
        if (pooled_count <= 0) {
            llama_set_embeddings(h->ctx, false);
            return {};
        }
        result.resize(dim);
        const float inv = 1.0f / (float)pooled_count;
        for (int d = 0; d < dim; ++d) {
            result[(size_t)d] = sum[(size_t)d] * inv;
        }
    }

    llama_set_embeddings(h->ctx, false);

    l2_normalize(result);
    return result;
}

bool vector_db_init(ModelHandle * h, const std::string & index_dir, int max_elements, int m, int ef_construction) {
    if (!h || !h->model || !h->ctx) return false;
    if (max_elements <= 0) max_elements = 2048;
    if (m <= 0) m = 16;
    if (ef_construction <= 0) ef_construction = 200;

    const int dim = llama_model_n_embd(h->model);
    if (dim <= 0) return false;

    std::lock_guard<std::mutex> lock(g_vdb_mtx);
    if (g_vdb.find(h) != g_vdb.end()) return true;

    auto db = std::make_unique<VectorDb>();
    db->dim = dim;
    db->max_elements = max_elements;
    db->index_dir = index_dir;

    ensure_dir_exists(index_dir);

    db->space = std::make_unique<hnswlib::InnerProductSpace>(dim);

    const std::string index_path = join_path(index_dir, "hnsw.index");
    const bool has_index = file_exists(index_path);
    try {
        if (has_index) {
            try {
                db->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(db->space.get(), index_path);
            } catch (...) {
                db->index.reset();
            }
        }
        if (!db->index) {
            db->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(db->space.get(), (size_t)max_elements, m, ef_construction);
            db->index->setEf(64);
        }
    } catch (...) {
        return false;
    }

    db->texts.resize(1);
    load_texts(*db);

    g_vdb[h] = std::move(db);
    return true;
}

static bool vdb_rebuild_from_texts_locked(ModelHandle * h, VectorDb * db, int max_elements, int m, int ef_construction) {
    if (!h || !db) return false;
    if (max_elements <= 0) max_elements = db->max_elements > 0 ? db->max_elements : 2048;
    if (m <= 0) m = 16;
    if (ef_construction <= 0) ef_construction = 200;
    if (db->dim <= 0) return false;

    db->space = std::make_unique<hnswlib::InnerProductSpace>(db->dim);
    db->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(db->space.get(), (size_t)max_elements, m, ef_construction);
    db->index->setEf(64);

    const size_t n = db->texts.size();
    if (n <= 1) {
        db->next_label = 1;
        return true;
    }

    for (size_t lab = 1; lab < n; ++lab) {
        const std::string & text = db->texts[lab];
        if (text.empty()) continue;
        auto vec = embed_text(h, text);
        if (vec.empty()) continue;
        try {
            db->index->addPoint(vec.data(), (hnswlib::labeltype)lab);
        } catch (...) {
        }
    }

    db->next_label = (long long)db->texts.size();
    if (db->next_label < 1) db->next_label = 1;

    const std::string index_path = join_path(db->index_dir, "hnsw.index");
    try {
        if (db->index) db->index->saveIndex(index_path);
    } catch (...) {
    }
    save_texts(*db);
    return true;
}

bool vector_db_rebuild_from_texts(ModelHandle * h, const std::string & index_dir, int max_elements, int m, int ef_construction) {
    if (!h || !h->model || !h->ctx) return false;
    if (!vdb_has(h)) {
        if (!vector_db_init(h, index_dir, max_elements, m, ef_construction)) return false;
    }
    VectorDb * db = vdb_get(h);
    if (!db) return false;
    std::lock_guard<std::mutex> lock(db->mtx);
    db->index_dir = index_dir;
    load_texts(*db);
    db->dim = llama_model_n_embd(h->model);
    return vdb_rebuild_from_texts_locked(h, db, max_elements, m, ef_construction);
}

void vector_db_close(ModelHandle * h) {
    if (!h) return;
    std::unique_ptr<VectorDb> db;
    {
        std::lock_guard<std::mutex> lock(g_vdb_mtx);
        auto it = g_vdb.find(h);
        if (it == g_vdb.end()) return;
        db = std::move(it->second);
        g_vdb.erase(it);
    }

    if (!db) return;
    std::lock_guard<std::mutex> lock(db->mtx);

    const std::string index_path = join_path(db->index_dir, "hnsw.index");
    try {
        if (db->index) {
            db->index->saveIndex(index_path);
        }
    } catch (...) {
    }

    save_texts(*db);
}

long long vector_db_add_text(ModelHandle * h, const std::string & text) {
    VectorDb * db = vdb_get(h);
    if (!db) return 0;

    std::lock_guard<std::mutex> lock(db->mtx);
    const long long label = db->next_label++;
    if ((size_t)label >= db->texts.size()) db->texts.resize((size_t)label + 1);
    db->texts[(size_t)label] = text;

    auto vec = embed_text(h, text);
    if (vec.empty()) return 0;

    try {
        db->index->addPoint(vec.data(), (hnswlib::labeltype)label);
    } catch (...) {
        return 0;
    }

    return label;
}

std::vector<long long> vector_db_search(ModelHandle * h, const std::string & query_text, int k) {
    std::vector<long long> out;
    VectorDb * db = vdb_get(h);
    if (!db) return out;

    if (k <= 0) k = 3;

    auto q = embed_text(h, query_text);
    if (q.empty()) return out;

    std::lock_guard<std::mutex> lock(db->mtx);
    try {
        auto top = db->index->searchKnn(q.data(), (size_t)k);
        while (!top.empty()) {
            out.push_back((long long)top.top().second);
            top.pop();
        }
        std::reverse(out.begin(), out.end());
    } catch (...) {
        return {};
    }

    return out;
}

std::vector<std::pair<long long, float>> vector_db_search_scored(ModelHandle * h, const std::string & query_text, int k) {
    std::vector<std::pair<long long, float>> out;
    VectorDb * db = vdb_get(h);
    if (!db) return out;

    if (k <= 0) k = 3;

    auto q = embed_text(h, query_text);
    if (q.empty()) return out;

    std::lock_guard<std::mutex> lock(db->mtx);
    try {
        auto top = db->index->searchKnn(q.data(), (size_t)k);
        while (!top.empty()) {
            const float dist = top.top().first;
            float cos = 1.0f - dist;
            if (cos > 1.0f) cos = 1.0f;
            if (cos < -1.0f) cos = -1.0f;
            const auto lab = (long long)top.top().second;
            out.emplace_back(lab, cos);
            top.pop();
        }
        std::reverse(out.begin(), out.end());
    } catch (...) {
        return {};
    }
    return out;
}

std::string vector_db_get_text(ModelHandle * h, long long label) {
    VectorDb * db = vdb_get(h);
    if (!db) return "";

    std::lock_guard<std::mutex> lock(db->mtx);
    if (label <= 0) return "";
    if ((size_t)label >= db->texts.size()) return "";
    return db->texts[(size_t)label];
}

int vector_db_count(ModelHandle * h) {
    VectorDb * db = vdb_get(h);
    if (!db) return 0;
    std::lock_guard<std::mutex> lock(db->mtx);
    try {
        return db->index ? (int)db->index->getCurrentElementCount() : 0;
    } catch (...) {
        return 0;
    }
}

}

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbInit(
        JNIEnv * env, jclass, jlong handlePtr, jstring jIndexDir, jint maxElements, jint m, jint efConstruction) {
    (void)env;
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !jIndexDir) return JNI_FALSE;
    const char * cdir = env->GetStringUTFChars(jIndexDir, nullptr);
    std::string dir = cdir ? cdir : "";
    if (cdir) env->ReleaseStringUTFChars(jIndexDir, cdir);

    const bool ok = imem::vector_db_init(h, dir, (int)maxElements, (int)m, (int)efConstruction);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbClose(
        JNIEnv * env, jclass, jlong handlePtr) {
    (void)env;
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    imem::vector_db_close(h);
}

JNIEXPORT jlong JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbAddText(
        JNIEnv * env, jclass, jlong handlePtr, jstring jText) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !jText) return 0;

    const char * ctext = env->GetStringUTFChars(jText, nullptr);
    std::string text = ctext ? ctext : "";
    if (ctext) env->ReleaseStringUTFChars(jText, ctext);

    return (jlong)imem::vector_db_add_text(h, text);
}

JNIEXPORT jlongArray JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbSearch(
        JNIEnv * env, jclass, jlong handlePtr, jstring jQueryText, jint k) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !jQueryText) return env->NewLongArray(0);

    const char * ctext = env->GetStringUTFChars(jQueryText, nullptr);
    std::string text = ctext ? ctext : "";
    if (ctext) env->ReleaseStringUTFChars(jQueryText, ctext);

    auto labels = imem::vector_db_search(h, text, (int)k);
    jlongArray arr = env->NewLongArray((jsize)labels.size());
    if (!arr) return nullptr;
    if (!labels.empty()) {
        std::vector<jlong> tmp;
        tmp.reserve(labels.size());
        for (auto v : labels) tmp.push_back((jlong)v);
        env->SetLongArrayRegion(arr, 0, (jsize)tmp.size(), tmp.data());
    }
    return arr;
}

JNIEXPORT jlongArray JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbSearchScored(
        JNIEnv * env, jclass, jlong handlePtr, jstring jQueryText, jint k) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !jQueryText) return env->NewLongArray(0);

    const char * ctext = env->GetStringUTFChars(jQueryText, nullptr);
    std::string text = ctext ? ctext : "";
    if (ctext) env->ReleaseStringUTFChars(jQueryText, ctext);

    auto pairs = imem::vector_db_search_scored(h, text, (int)k);

    jlongArray arr = env->NewLongArray((jsize)pairs.size());
    if (!arr) return nullptr;
    if (!pairs.empty()) {
        std::vector<jlong> tmp;
        tmp.reserve(pairs.size());
        for (const auto & it : pairs) {
            const uint64_t lab = (uint64_t)(uint32_t)it.first;
            uint32_t bits = 0;
            static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
            std::memcpy(&bits, &it.second, sizeof(uint32_t));
            const uint64_t packed = (lab << 32) | (uint64_t)bits;
            tmp.push_back((jlong)packed);
        }
        env->SetLongArrayRegion(arr, 0, (jsize)tmp.size(), tmp.data());
    }
    return arr;
}

JNIEXPORT jstring JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbGetText(
        JNIEnv * env, jclass, jlong handlePtr, jlong label) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h) return env->NewStringUTF("");
    std::string t = imem::vector_db_get_text(h, (long long)label);
    return new_jstring_utf8_lenient(env, t);
}

JNIEXPORT jint JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbCount(
        JNIEnv * env, jclass, jlong handlePtr) {
    (void)env;
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h) return 0;
    return (jint)imem::vector_db_count(h);
}

JNIEXPORT jboolean JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbRebuildFromTexts(
        JNIEnv * env, jclass, jlong handlePtr, jstring jIndexDir, jint maxElements, jint m, jint efConstruction) {
    ModelHandle * h = reinterpret_cast<ModelHandle *>(handlePtr);
    if (!h || !jIndexDir) return JNI_FALSE;

    const char * cdir = env->GetStringUTFChars(jIndexDir, nullptr);
    std::string dir = cdir ? cdir : "";
    if (cdir) env->ReleaseStringUTFChars(jIndexDir, cdir);

    const bool ok = imem::vector_db_rebuild_from_texts(h, dir, (int)maxElements, (int)m, (int)efConstruction);
    return ok ? JNI_TRUE : JNI_FALSE;
}

}
