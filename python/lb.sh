HOST=0.0.0.0
python3 -m sglang.srt.disaggregation.mini_lb \
    --prefill http://$HOST:30000 http://$HOST:30001 \
    --decode http://$HOST:31000 http://$HOST:31001 \
    --host $HOST --port 8188 \
    --prefill-bootstrap-ports 8998 8999