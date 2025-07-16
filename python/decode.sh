PORT=31000
BASEGPUID=4

if [ -n "$1" ]; then
    PORT=$1
fi

if [ -n "$2" ]; then
    BASEGPUID=$2
fi

echo "Starting decode server: port $PORT / GPU ID $BASEGPUID"

SGLANG_USE_MODELSCOPE=true \
python -m sglang.launch_server \
    --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct --disaggregation-mode decode \
    --port $PORT --base-gpu-id $BASEGPUID \
    --trust-remote-code --disable-radix-cache --tp-size 1