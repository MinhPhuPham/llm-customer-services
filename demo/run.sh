#!/bin/bash
# ──────────────────────────────────────────────
# Start/Stop the Support AI demo server
#
# Usage:
#   ./demo/run.sh start    # Start server on port 5050
#   ./demo/run.sh stop     # Stop server
#   ./demo/run.sh restart  # Restart
#   ./demo/run.sh status   # Check if running
# ──────────────────────────────────────────────

PORT=5050
PID_FILE="demo/.server.pid"
LOG_FILE="demo/.server.log"

cd "$(dirname "$0")/.." || exit 1

start_server() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "⚠️  Server already running (PID $(cat "$PID_FILE"))"
        echo "   → http://localhost:$PORT"
        return 1
    fi

    echo "🚀 Starting Support AI demo..."
    python3 demo/app.py > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"

    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
            echo "✅ Server ready (PID $pid)"
            echo "   → http://localhost:$PORT"
            return 0
        fi
        sleep 1
    done

    echo "❌ Server failed to start. Check $LOG_FILE"
    cat "$LOG_FILE" | tail -10
    return 1
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f "$PID_FILE"
            echo "🛑 Server stopped (PID $pid)"
        else
            rm -f "$PID_FILE"
            echo "⚠️  Server was not running (stale PID)"
        fi
    else
        # Try to find and kill by port
        local pid=$(lsof -ti :$PORT 2>/dev/null)
        if [ -n "$pid" ]; then
            kill "$pid"
            echo "🛑 Server stopped (PID $pid)"
        else
            echo "⚠️  No server running on port $PORT"
        fi
    fi
}

case "${1:-start}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "✅ Running (PID $(cat "$PID_FILE")) → http://localhost:$PORT"
        else
            echo "🛑 Not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
