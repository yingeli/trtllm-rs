use anyhow::{Result, anyhow};
use std::fs::File;
use trtllm::{
    executor::{Executor, ExecutorConfig, ModelType, Request},
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

    let mut request = Request::new(&[1, 2, 3, 4], 100); // Example token IDs and max tokens
    request.set_streaming(true)?;

    let request_id = executor.enqueue_request(&request)?;
    println!("Request ID: {}", request_id);

    let mut is_final = false;
    while !is_final {
        while executor.get_num_responses_ready(request_id)? == 0 {
            // Wait for responses to be ready
            std::thread::sleep(std::time::Duration::from_millis(1)); // Sleep to avoid busy waiting
        }

        let responses = executor.await_responses(request_id)?;
        println!("Received responses");

        for response in responses {
            let result = response.get_result()?;
            for token_ids in result.output_token_ids() {
                println!("Response Token IDs: {:?}", token_ids.as_slice());
            }
            if result.is_final() {
                is_final = true;
                break;
            }
        }
    }

    Ok(())
}
