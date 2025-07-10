#!/bin/bash
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo ""
    echo "=== System Memory ==="
    free -h | grep -E "^Mem|^Swap"
    echo ""
    echo "=== Python Processes ==="
    ps aux | grep python | grep -v grep | awk '{printf "PID: %s, CPU: %s%%, MEM: %s%%, CMD: %s\n", $2, $3, $4, $11}'
    sleep 1
done
