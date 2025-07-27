#pragma once

#include <memory>

#include <tensorrt_llm/executor/executor.h>

#include "rust/cxx.h"

namespace tle = tensorrt_llm::executor;

struct Result;

inline std::unique_ptr<tle::ExecutorConfig> executor_config(
) {
    return std::make_unique<tle::ExecutorConfig>(
    );
}

inline std::unique_ptr<tle::Request> request(
    const rust::Slice<const std::uint32_t> input_token_ids,
    const std::uint32_t max_tokens
) {
    return std::make_unique<tle::Request>(
        tle::VecTokens(input_token_ids.begin(), input_token_ids.end()),
        max_tokens
    );
}

inline std::unique_ptr<tle::Executor> executor(
    const rust::Str model_path,
    const tle::ModelType model_type,
    const tle::ExecutorConfig& executor_config
) {
    auto path = std::filesystem::path(static_cast<std::string>(model_path));
    return std::make_unique<tle::Executor>(
        path,
        model_type,
        executor_config
    );
}

inline std::unique_ptr<std::vector<tle::Response>> await_responses(
    tle::Executor& executor,
    const std::uint64_t request_id
) {
    return std::make_unique<std::vector<tle::Response>>(
        executor.awaitResponses(request_id)
    );
}

inline std::unique_ptr<tle::Response> await_response(
    tle::Executor& executor,
    const std::uint64_t request_id
) {
    auto responses = executor.awaitResponses(request_id);
    return std::make_unique<tle::Response>(
        responses[0]
    );
}

inline uint32_t get_num_responses_ready(
    const tle::Executor& executor,
    const std::uint64_t request_id
) {
    return executor.getNumResponsesReady(request_id);
}

Result get_result(const tle::Response& response);