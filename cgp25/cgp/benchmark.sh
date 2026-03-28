#!/bin/bash

# Number of times to run (defaults to 5 if no argument is passed)
N=${1:-5}
CMD="./cgp"
TOTAL_TIME=0

echo "Benchmarking '$CMD' ($N runs)..."
echo "----------------------------------------"

for ((i=1; i<=N; i++)); do
    # 1. { time -p ... } uses the POSIX time format.
    # 2. >/dev/null 2>&1 silences `make run` so it doesn't flood the terminal.
    # 3. The outer 2>&1 captures the output of `time` so we can parse it.
    # 4. awk extracts just the numeric value next to "real".
    RUN_TIME=$( { time -p $CMD >/dev/null 2>&1 ; } 2>&1 | awk '/^real/ {print $2}' )
    
    echo "Run $i: ${RUN_TIME}s"
    
    # Accumulate total time using awk (Bash doesn't do floats)
    TOTAL_TIME=$(awk "BEGIN {print $TOTAL_TIME + $RUN_TIME}")
done

echo "----------------------------------------"

# Calculate average and format to 4 decimal places
AVERAGE=$(awk "BEGIN {printf \"%.4f\", $TOTAL_TIME / $N}")

echo "Total Time:   ${TOTAL_TIME}s"
echo "Average Time: ${AVERAGE}s"
