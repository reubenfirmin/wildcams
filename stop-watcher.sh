#!/bin/bash
# Stop the SD card watcher daemon

PROJECT_ROOT="${1:-$(pwd)}"
PID_FILE="$PROJECT_ROOT/.sd_watcher.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "⚠️  No watcher PID file found"
    exit 1
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "🛑 Stopping watcher (PID: $PID)..."
    kill "$PID"
    sleep 1

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "⚠️  Force stopping..."
        kill -9 "$PID"
    fi

    rm -f "$PID_FILE"
    echo "✅ Watcher stopped"
else
    echo "⚠️  Watcher not running (stale PID file)"
    rm -f "$PID_FILE"
fi
