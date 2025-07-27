use anyhow::{Result, anyhow};
use tokenizers::{Encoding, Tokenizer};
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

    let tokenizer = Tokenizer::from_file(
        "/home/coder/trtllm-rs/examples/qwen/models/Qwen2.5-3B-Instruct/tokenizer.json",
    )
    .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    let encoding: Encoding = tokenizer
        .encode("Who are you?", true)
        .map_err(|e| anyhow!("Failed to encode input: {}", e))?;
    let input_token_ids = encoding.get_ids();

    let mut request = Request::new(&input_token_ids, 10000);
    // request.set_streaming(true)?;
    request.set_end_id(151645)?;
    request.set_pad_id(151643)?;

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
                // println!("Result Token IDs: {:?}", token_ids.as_slice());
                let output = tokenizer
                    .decode(token_ids.as_slice(), false)
                    .map_err(|e| anyhow!("Failed to decode token IDs: {}", e))?;
                println!("Output: {}", output);
            }
            if result.is_final() {
                is_final = true;
                break;
            }
        }
    }

    Ok(())
}
