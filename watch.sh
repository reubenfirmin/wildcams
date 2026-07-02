#!/bin/bash
# Manually start or restart the SD card watcher daemon

PROJECT_ROOT="$(pwd)"
PID_FILE="$PROJECT_ROOT/.sd_watcher.pid"
LOG_FILE="$PROJECT_ROOT/.sd_watcher.log"

# Stop existing watcher if running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "🔄 Stopping existing watcher (PID: $PID)..."
        "$PROJECT_ROOT/stop-watcher.sh"
        sleep 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Start new watcher
echo "🎥 Starting SD card watcher daemon..."
"$PROJECT_ROOT/sd_watcher.py" --daemon --pid-file "$PID_FILE" --log-file "$LOG_FILE" &

sleep 0.5

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    echo "✅ SD card watcher started (PID: $PID)"
    echo "📋 Watch activity: tail -f $LOG_FILE"
    echo "🛑 Stop with: ./stop-watcher.sh"
else
    echo "❌ Failed to start daemon"
    exit 1
fi
