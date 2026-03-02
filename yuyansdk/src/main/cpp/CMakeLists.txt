cmake_minimum_required(VERSION 3.18)
project(yuyan_llama)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(GGML_LTO OFF CACHE BOOL "" FORCE)

if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    message(STATUS "为 arm64-v8a 添加性能优化编译标志")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -ffast-math -funroll-loops -pthread -march=armv8.6-a+dotprod+i8mm+fp16 -fno-finite-math-only")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -funroll-loops -pthread -march=armv8.6-a+dotprod+i8mm+fp16 -fno-finite-math-only")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG")
endif()

set(BLAS_FOUND ON CACHE BOOL "" FORCE)
set(BLAS_LIBRARIES ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libopenblas.so CACHE FILEPATH "" FORCE)
set(BLAS_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../../libs/include CACHE PATH "" FORCE)

set(GGML_BLAS ON CACHE BOOL "" FORCE)
set(GGML_BLAS_VENDOR "OpenBLAS" CACHE STRING "" FORCE)
message(STATUS "强制设置 BLAS 选项:")
message(STATUS "  BLAS_FOUND = ${BLAS_FOUND}")
message(STATUS "  BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
message(STATUS "  BLAS_INCLUDE_DIRS = ${BLAS_INCLUDE_DIRS}")
message(STATUS "  GGML_BLAS = ${GGML_BLAS}")
message(STATUS "  GGML_BLAS_VENDOR = ${GGML_BLAS_VENDOR}")
add_subdirectory(llama.cpp llama_build)

add_library(llama_jni SHARED
        native-lib.cpp
        imem_utils.cpp
        imem_logic_gen.cpp
        imem_logic_cache.cpp
        imem_perf_jni.cpp
        imem_vector_db.cpp
)
target_include_directories(llama_jni PRIVATE
        ${CMAKE_SOURCE_DIR}/llama.cpp
        ${CMAKE_SOURCE_DIR}/../../../libs/include
)

target_link_libraries(llama_jni PRIVATE
        llama
        ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libopenblas.so
        m
        log
)

if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    target_compile_definitions(llama_jni PRIVATE __ANDROID_API__=21)
endif()