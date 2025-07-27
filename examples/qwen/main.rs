use anyhow::{Result, anyhow};
use std::fs::File;
use trtllm::{
    executor::{Executor, ExecutorConfig, ModelType},
    init_trtllm_plugins,
};

fn main() -> Result<()> {
    if !init_trtllm_plugins()? {
        return Err(anyhow!("Failed to initialize TensorRT-LLM plugins"));
    }

    let config = ExecutorConfig::new();
    let mut executor = Executor::open(
        "/home/coder/trtllm-rs/examples/qwen/engines/qwen2.5_3b_1gpu_bf16",
        ModelType::DecoderOnly,
        &config,
    )?;
    Ok(())
}
