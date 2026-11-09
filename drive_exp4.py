#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXP4 driver (resumable): wait for shard A -> pull -> merge -> build judge_input
-> judge (same-source DeepSeek, error-clean loop) -> analyze.

Run in background; safe to re-run (pull resumes, build overwrites, judge resumes).
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_server import run, pull

PY = "/c/Users/<LOCAL_USER>/<WORKBUDDY_DIR>/binaries/python/envs/default/Scripts/python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

FUSED_A = "/home/<SERVER_USER>/mmhal_inject_v3/fused_1000_hallu_a.jsonl"
LOCAL_A = "fused_1000_hallu_a.jsonl"
LOCAL_B = "fused_1000_hallu_b.jsonl"
MERGED = "fused_1000_merged.jsonl"
JUDGE_IN = "judge_input_exp4.jsonl"
JUDGE_OUT = "judge_output_exp4.jsonl"
KEY = "sk-REDACTED="
MODEL = "DeepSeek-V4.1-Flash-1017-Event"
BASE = "https://api.scnet.cn/api/llm/v1/chat/completions"
TARGET_A_LINES = 500
MAX_WAIT_S = 14 * 3600

def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

# ---------- Phase 1: wait for shard A ----------
log("PHASE1: waiting for shard A to finish (process exit + 500 lines)")
t0 = time.time()
while True:
    rc, out, err = run(
        "cd /home/<SERVER_USER>/mmhal_inject_v3 && wc -l < fused_1000_hallu_a.jsonl; "
        "ps aux | grep run_qwen_fused | grep fused_1000_hallu_a | grep -v grep | wc -l",
        timeout=120)
    parts = out.strip().split("\n")
    try:
        nlines = int(parts[0].strip())
        nproc = int(parts[1].strip())
    except Exception:
        nlines, nproc = 0, 99
    log(f"  a.jsonl lines={nlines} shardA_procs={nproc} elapsed={int(time.time()-t0)}s")
    if nproc == 0 and nlines >= TARGET_A_LINES:
        log("shard A finished."); break
    if nproc == 0 and nlines < TARGET_A_LINES:
        log(f"ERROR: shard A process gone but only {nlines}/{TARGET_A_LINES} lines — needs attention. Stopping.")
        sys.exit(3)
    if time.time() - t0 > MAX_WAIT_S:
        log("ERROR: exceeded MAX_WAIT_S waiting for shard A. Stopping."); sys.exit(4)
    time.sleep(240)

# ---------- Phase 2: pull shard A (resumable) ----------
log("PHASE2: pulling shard A (resumable)")
pull(FUSED_A, LOCAL_A)

# ---------- Phase 3: merge ----------
log("PHASE3: merge a+b -> merged")
with open(MERGED, "w", encoding="utf-8") as m:
    for p in (LOCAL_A, LOCAL_B):
        with open(p, encoding="utf-8") as f:
            for l in f:
                m.write(l)
log("merged written")

# ---------- Phase 4: build judge_input ----------
log("PHASE4: build judge_input")
subprocess.run([PY, "build_judge_input_exp4.py", "--inputs", LOCAL_A, LOCAL_B,
                "--out", JUDGE_IN], check=True)

# ---------- Phase 5: judge with error-clean loop ----------
def count_errors(path):
    n = 0
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                if json.loads(l).get("verdict") == "error":
                    n += 1
            except Exception:
                pass
    return n

def strip_errors(path):
    if not os.path.exists(path):
        return 0
    kept = 0; removed = 0
    tmp = path + ".tmp"
    with open(path, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
        for l in f:
            try:
                if json.loads(l).get("verdict") == "error":
                    removed += 1; continue
            except Exception:
                pass
            g.write(l); kept += 1
    os.replace(tmp, path)
    return removed

log("PHASE5: judge (resumable, error-clean loop)")
for it in range(10):
    subprocess.run([PY, "run_glm_judge.py", "--inp", JUDGE_IN, "--out", JUDGE_OUT,
                    "--workers", "24", "--model", MODEL, "--key", KEY, "--base", BASE],
                   check=True)
    ne = count_errors(JUDGE_OUT)
    log(f"  judge pass {it+1}: residual errors = {ne}")
    if ne == 0:
        break
    rem = strip_errors(JUDGE_OUT)
    log(f"  stripped {rem} error rows for re-judge")
    if it == 9:
        log("WARN: still errors after 10 passes — manual check needed")

# ---------- Phase 6: verify 0 missing ----------
n_in = sum(1 for _ in open(JUDGE_IN, encoding="utf-8"))
n_out = sum(1 for _ in open(JUDGE_OUT, encoding="utf-8"))
log(f"PHASE6: judge_input={n_in} judge_output={n_out} errors={count_errors(JUDGE_OUT)}")
if n_in != n_out:
    log(f"ERROR: missing {(n_in - n_out)} verdicts. Stopping before analysis.")
    sys.exit(5)

# ---------- Phase 7: analyze ----------
log("PHASE7: analyze -> fig_mega_1000.pdf + EXP4_MEGA_ANALYSIS.md")
subprocess.run([PY, "analyze_exp4.py", "--inp", JUDGE_IN, "--out", JUDGE_OUT], check=True)
log("ALL DONE ✅")
