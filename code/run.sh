#!/bin/bash
echo 'hello csbgpu4'
# Check if the third argument is provided
if [ -n "$3" ]; then
    echo '3 args'
    python -u main.py --config $1 --feat "$2" > $3 2>&1
else
    echo '2 args'
    python -u main.py --config $1 > $2 2>&1
fi

