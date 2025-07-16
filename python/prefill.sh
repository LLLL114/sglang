PORT=30000
BASEGPUID=0
DISPORT=8998
if [ -n "$1" ]; then
    PORT=$1
fi

if [ -n "$2" ]; then
    BASEGPUID=$2
fi

if [ -n "$3" ]; then
    DISPORT=$3
fi

echo "Starting prefill server: port $PORT / GPU ID $BASEGPUID / Disaggregation Bootstrap Port $DISPORT"

SGLANG_USE_MODELSCOPE=true \
python -m sglang.launch_server \
    --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct --disaggregation-mode prefill \
    --port $PORT --base-gpu-id $BASEGPUID --disaggregation-bootstrap-port $DISPORT \
    --trust-remote-code --disable-radix-cache --tp-size 1

# deepseek-ai/DeepSeek-V2-Lite
# Qwen/Qwen2.5-0.5B-Instruct
# /root/.cache/modelscope/hub/models/