#include "trtllm/src/sys/executor.rs.h"

std::unique_ptr<tle::Tensor> tensor(
    tle::DataType data_type,
    TensorData* data,
    std::unique_ptr<tle::Shape> shape
) {
    return std::make_unique<tle::Tensor>(
        tle::Tensor::of(data_type, data, *shape)
    );
}

Result get_result(
    const tle::Response& response
) {
    auto result = response.getResult();

    rust::Vec<VecTokens> output_token_ids;
    output_token_ids.reserve(result.outputTokenIds.size());
    for (const auto& token_ids : result.outputTokenIds) {
        rust::Vec<uint32_t> v;
        v.reserve(token_ids.size());
        for (const auto& token_id : token_ids) {
            v.push_back(token_id);
        }
        output_token_ids.push_back(VecTokens { v });
    }

    return Result {
        is_final: result.isFinal,
        output_token_ids: output_token_ids
    };
}