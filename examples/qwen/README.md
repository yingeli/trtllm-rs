python /home/coder/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir ./models/Qwen3-30B-A3B-Instruct-2507/ \
    --output_dir ./checkpoints/qwen3_30b_a3b_instruct_2507 \
    --dtype bfloat16

/home/ubuntu/.local/bin/trtllm-build --checkpoint_dir ./checkpoints/qwen3_30b_a3b_instruct_2507 \
    --output_dir ./engines/qwen3_30b_a3b_instruct_2507_1gpu_bf16

python3 /home/coder/TensorRT-LLM/examples/run.py \
    --input_text "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nIs Microsoft a great company?|im_end|>\n<|im_start|>assistant\n" \
    --max_output_len=500 \
    --tokenizer_dir ./models/Qwen3-30B-A3B-Instruct-2507/ \
    --engine_dir=./engines/qwen3_30b_a3b_instruct_2507_1gpu_bf16 \
    --streaming \
    --run_profiling


python3 /home/coder/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir ./models/Qwen2.5-3B-Instruct/ \
    --output_dir ./checkpoints/qwen2.5_3b_1gpu_bf16

/home/ubuntu/.local/bin/trtllm-build --checkpoint_dir ./checkpoints/qwen2.5_3b_1gpu_bf16 \
    --output_dir ./engines/qwen2.5_3b_1gpu_bf16 \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16

python3 /home/coder/TensorRT-LLM/examples/run.py --input_text "你好，请问你叫什么？" \
    --max_output_len=50 \
    --tokenizer_dir ./models/Qwen2.5-3B-Instruct/ \
    --engine_dir=./engines/qwen2.5_3b_1gpu_bf16