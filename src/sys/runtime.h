#pragma once

#include <memory>

#include <tensorrt_llm/executor/runtime.h>

#include "rust/cxx.h"

namespace tlr = tensorrt_llm::executor;

struct CUstream_st;

inline std::shared_ptr<tlr::CudaStream> cudaStream(
    CUstream_st* stream
) {
    return std::make_shared<tlr::CudaStream>(stream);
}

inline std::unique_ptr<tlr::BufferManager> bufferManager(
    std::shared_ptr<tlr::CudaStream> stream
) {
    return std::make_unique<tlr::BufferManager>(stream);
}