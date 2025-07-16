SGLANG_USE_MODELSCOPE=true \
python3 -m sglang.bench_serving \
    --backend sglang --dataset-name random --random-input 1024 --random-output 1024 \
    --num-prompts 10 --port 8188 --max-concurrency 20