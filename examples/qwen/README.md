python convert_checkpoint.py --model_dir ./models/Qwen3-30B-A3B/ \
                              --output_dir ./checkpoints/qwen3_30b_a3b_1gpu_bf16 \
                              --dtype bfloat16

python convert_checkpoint.py --model_dir ./models/Qwen3-4B/ \
                              --output_dir ./checkpoints/qwen3_4b_1gpu_bf16 \
                              --dtype bfloat16

python convert_checkpoint.py --model_dir ./models/Qwen2.5-3B-Instruct/ \
                              --output_dir ./checkpoints/qwen2.5_3b_1gpu_bf16

trtllm-build --checkpoint_dir ./checkpoints/qwen2.5_3b_1gpu_bf16 \
            --output_dir ./engines/qwen2.5_3b_1gpu_bf16 \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16

python3 /app/tensorrt_llm/examples/run.py --input_text "你好，请问你叫什么？" \
    --max_output_len=50 \
    --tokenizer_dir ./models/Qwen2.5-3B-Instruct/ \
    --engine_dir=./engines/qwen2.5_3b_1gpu_bf16