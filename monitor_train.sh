#!/bin/bash
# monitor_train.sh - 每20分钟检查训练状态

LOG_FILE="/tmp/claude-1000/-home-lbp-projects-code-sapiens/tasks/bx3jnsa80.output"
MODEL_PATH="/home/lbp/projects/code-sapiens/best_model.pt"

while true; do
    # 检查训练进程是否还在运行
    if ps aux | grep -q "[p]ython train.py"; then
        echo "=== $(date) ==="
        echo "训练进行中..."
        tail -5 "$LOG_FILE" 2>/dev/null | grep "Step"
        echo "---"
    else
        echo "=== $(date) ==="
        echo "训练已完成!"
        tail -10 "$LOG_FILE"

        # 保存最终模型大小
        ls -lh "$MODEL_PATH"

        echo ""
        echo "训练完成，开始测试模型..."

        # 运行测试
        cd /home/lbp/projects/code-sapiens
        source /home/lbp/.local/bin/mamba activate mamba 2>/dev/null || source $(which mamba activate mamba) 2>/dev/null || true
        python test_model.py

        break
    fi

    sleep 1200  # 20分钟
done
