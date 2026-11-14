#!/bin/bash
mkdir -p /logs/verifier
python /tests/replay.py /trace/trace.jsonl /logs/verifier/reward.txt 2>&1 | tee /logs/verifier/replay.log
[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt
