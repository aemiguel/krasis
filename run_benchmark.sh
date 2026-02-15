#!/usr/bin/env bash
# Run a Krasis benchmark and exit (don't start the server).
# Usage: ./run_benchmark.sh <server.py args...>
#
# Runs server.py --benchmark in background, waits for benchmark completion,
# then kills the process (server would otherwise block on uvicorn).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi

LOGFILE=$(mktemp /tmp/krasis_bench_XXXXXX.log)
echo "Benchmark output: $LOGFILE"

# Run server with --benchmark, capture output
"$PYTHON" -m krasis.server --benchmark "$@" > >(tee "$LOGFILE") 2>&1 &
PID=$!

echo "Server PID: $PID"

# Wait for benchmark to complete (look for "Benchmark archived" or server start)
while kill -0 "$PID" 2>/dev/null; do
    if grep -q "Benchmark archived to\|starting server on" "$LOGFILE" 2>/dev/null; then
        echo ""
        echo "Benchmark complete. Stopping server..."
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        echo "Done. Full output in: $LOGFILE"
        exit 0
    fi
    sleep 2
done

# Process exited on its own (error?)
EXITCODE=$?
echo "Process exited with code $EXITCODE"
echo "Full output in: $LOGFILE"
exit $EXITCODE
