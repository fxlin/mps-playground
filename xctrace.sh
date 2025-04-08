PROG=test-fp16.py
# PROG=amp-receipe-mps-old.py

# trace filname. must end with .trace
TRACE="$(basename "$PROG" .py)_$(date +%Y%m%d%H%M%S).trace"

xctrace record --template "FL-gpu-counters1" \
--output /tmp/${TRACE} \
--launch -- \
/Users/felixlin/workspace-mps/myenv-python312/bin/python3 ${PROG}


