#pragma once

#include "tensorrt_llm/plugins/api/tllmPlugin.h"

inline bool init_trtllm_plugins() {
    return initTrtLlmPlugins();
}