#pragma once

#include <jni.h>
#include <string>
#include <vector>
#include <utility>

struct ModelHandle;

namespace imem {

struct VectorDb;

bool vector_db_init(ModelHandle * h, const std::string & index_dir, int max_elements, int m, int ef_construction);
void vector_db_close(ModelHandle * h);

bool vector_db_rebuild_from_texts(ModelHandle * h, const std::string & index_dir, int max_elements, int m, int ef_construction);

long long vector_db_add_text(ModelHandle * h, const std::string & text);

std::vector<long long> vector_db_search(ModelHandle * h, const std::string & query_text, int k);

std::vector<std::pair<long long, float>> vector_db_search_scored(ModelHandle * h, const std::string & query_text, int k);

std::string vector_db_get_text(ModelHandle * h, long long label);

int vector_db_count(ModelHandle * h);

}

extern "C" {
JNIEXPORT jboolean JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbInit(
        JNIEnv * env, jclass, jlong handlePtr, jstring jIndexDir, jint maxElements, jint m, jint efConstruction);

JNIEXPORT void JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbClose(
        JNIEnv * env, jclass, jlong handlePtr);

JNIEXPORT jlong JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbAddText(
        JNIEnv * env, jclass, jlong handlePtr, jstring jText);

JNIEXPORT jlongArray JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbSearch(
        JNIEnv * env, jclass, jlong handlePtr, jstring jQueryText, jint k);

JNIEXPORT jlongArray JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbSearchScored(
        JNIEnv * env, jclass, jlong handlePtr, jstring jQueryText, jint k);

JNIEXPORT jstring JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbGetText(
        JNIEnv * env, jclass, jlong handlePtr, jlong label);

JNIEXPORT jint JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbCount(
        JNIEnv * env, jclass, jlong handlePtr);

JNIEXPORT jboolean JNICALL Java_com_yuyan_imemodule_llm_LLMBridge_vectorDbRebuildFromTexts(
        JNIEnv * env, jclass, jlong handlePtr, jstring jIndexDir, jint maxElements, jint m, jint efConstruction);
}
