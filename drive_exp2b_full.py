# -*- coding: utf-8 -*-
"""
drive_exp2b_full.py -- EXP2b (InternVL2-4B) 全阶段统一驱动器（本地驱动，远程在 4090 服务器）。

阶段：download(兜底监测/启动) -> smoke(12) -> generate(648) -> pull -> judge(本地) -> analyze+写账本。

设计要点：
- 远程长任务（下载/冒烟/生成）一律 setsid nohup 脱离 SSH 独立跑；本驱动器只做「检查 + 必要时启动 + 轮询」。
- 每阶段启动前先查远端进程防重；每循环重连 SSH（应对链路抖动）。
- 状态落在 drive_exp2b_full.state.json，进程重启可续。
- 单进程长后台跑（可能 1-3h），完成打印 COMPLETE 并写 EXP2b 行进 EXPERIMENT_LOG.md。
"""
import os, sys, time, json, io, subprocess, argparse

LOCAL = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\<LOCAL_USER>\<WORKBUDDY_DIR>\binaries\python\envs\default\Scripts\python.exe"
HOST, PORT, USER, PASS = "<REDACTED-SERVER-IP>", 22, "<SERVER_USER>", "<REDACTED-SERVER-PASSWORD>"
REMOTE = "/home/<SERVER_USER>/mmhal_inject_v3"
PYREMOTE = "/home/<SERVER_USER>/miniconda3/envs/train/bin/python"

JUDGE_KEY = "sk-REDACTED="
JUDGE_MODEL = "DeepSeek-V4.1-Flash-1017-Event"
JUDGE_BASE = "https://api.scnet.cn/api/llm/v1/chat/completions"

GEN_OUT = "cross_model_judge_input_internvl.jsonl"
SMOKE_OUT = "cross_model_judge_input_internvl_smoke.jsonl"
JUDGE_OUT = "cross_model_judge_output_internvl.jsonl"
ANALYSIS_MD = "cross_model_analysis_internvl.md"
EXPECTED_ROWS = 648
SHARD2_BYTES = 3_000_000_000

STATE_FILE = os.path.join(LOCAL, "drive_exp2b_full.state.json")
LOG = os.path.join(LOCAL, "drive_exp2b_full.log")
DEFAULT_STATE = {"downloaded": False, "smoked": False, "generated": False,
                 "pulled": False, "judged": False, "analyzed": False}

SERVER_SCRIPT = r'''import os, time, sys, glob
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# 服务器网络对 hf-mirror 做 MITM 自签证书拦截 -> 彻底禁用 SSL 验证（requests / urllib3 / aiohttp 全覆盖）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import urllib3
    urllib3.disable_warnings()
    _ocs = urllib3.util.ssl_.create_urllib3_context
    def _create(*a, **k):
        ctx = _ocs(*a, **k)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    urllib3.util.ssl_.create_urllib3_context = _create
    _ohc = urllib3.connection.HTTPSConnection.connect
    def _pc(self, *a, **k):
        try:
            self.cert_reqs = "CERT_NONE"; self.ca_certs = None; self.ca_certs_dir = None
        except Exception:
            pass
        return _ohc(self, *a, **k)
    urllib3.connection.HTTPSConnection.connect = _pc
except Exception:
    pass
import requests
_orig = requests.Session.request
def _patched(self, *a, **k):
    k.setdefault("verify", False)
    return _orig(self, *a, **k)
requests.Session.request = _patched
# fsspec 底层用 aiohttp；强制其 TCPConnector 不验证（应对服务器网络对 hf-mirror 的 MITM 自签证书）
try:
    import aiohttp
    _oc = aiohttp.TCPConnector.__init__
    def _pc(self, *a, **k):
        k["ssl"] = False
        _oc(self, *a, **k)
    aiohttp.TCPConnector.__init__ = _pc
except Exception:
    pass
from huggingface_hub import snapshot_download
REPO = "OpenGVLab/InternVL2-4B"
CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--OpenGVLab--InternVL2-4B")
def shard2_present():
    for s in glob.glob(CACHE + "/snapshots/*/model-00002-of-00002.safetensors"):
        try:
            if os.path.getsize(s) > 3_000_000_000:
                return True
        except OSError:
            pass
    return False
if shard2_present():
    print("SHARD2_PRESENT skip", flush=True); sys.exit(0)
for attempt in range(40):
    try:
        path = snapshot_download(REPO, repo_type="model", max_workers=8,
            allow_patterns=["*.json","*.txt","*.py","*.safetensors","*.model","*.bin"],
            ignore_patterns=["examples/*","*.msgpack","*.h5","*.ot"])
        print("DONE", path, flush=True); break
    except Exception as e:
        print("ATTEMPT", attempt, "ERR", repr(e)[:300], flush=True); time.sleep(20)
'''


def log(*a):
    line = " ".join(str(x) for x in a)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def connect():
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=25,
              allow_agent=False, look_for_keys=False)
    return c


def run(c, cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    return o.channel.recv_exit_status(), out, err


def remote_lines(c, path):
    _, out, _ = run(c, f"wc -l < {path} 2>/dev/null || echo 0")
    try:
        return int((out or "0").strip() or 0)
    except Exception:
        return 0


def proc_alive(c, name):
    _, out, _ = run(c, f"ps aux | grep '{name}' | grep -v grep | wc -l")
    return int((out or "0").strip() or 0) > 0


def pick_free_gpu(c):
    _, out, _ = run(c,
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits", t=30)
    best, best_free = 1, -1
    for ln in out.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 3:
            try:
                idx = int(parts[0]); used = int(parts[1]); total = int(parts[2])
                if total - used > best_free:
                    best_free, best = total - used, idx
            except ValueError:
                pass
    log(f"   free GPU = {best} (free={best_free}MiB)")
    return best


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE, encoding="utf-8"))
            for k, v in DEFAULT_STATE.items():
                s.setdefault(k, v)
            return s
        except Exception:
            pass
    return dict(DEFAULT_STATE)


def save_state(s):
    json.dump(s, open(STATE_FILE, "w", encoding="utf-8"), indent=2)


# ---------- stages ----------
def shard2_size(c):
    _, out, _ = run(c,
        "for f in $(find /home/<SERVER_USER>/.cache/huggingface/hub/models--OpenGVLab--InternVL2-4B/snapshots "
        "-name 'model-00002-of-00002.safetensors' 2>/dev/null); do stat -L -c %s \"$f\"; done")
    vals = [int(x) for x in out.split() if x.strip().isdigit()]
    return max(vals) if vals else 0


def df_free(c):
    _, out, _ = run(c, "df -m / | tail -1 | awk '{print $4}'")
    try:
        return int(out.split()[0])
    except Exception:
        return -1


def launch_download(c):
    if proc_alive(c, "dl_internvl_v2"):
        log("[dl] already running -> skip"); return
    sftp = c.open_sftp()
    sftp.putfo(io.BytesIO(SERVER_SCRIPT.encode()), f"{REMOTE}/dl_internvl_v2.py")
    sftp.close()
    c.exec_command(f"chmod +x {REMOTE}/dl_internvl_v2.py")
    c.exec_command(f"setsid nohup {PYREMOTE} {REMOTE}/dl_internvl_v2.py >> {REMOTE}/dl_internvl_v2.out 2>&1 < /dev/null &")
    log("[dl] launched (detached)")


def launch_smoke(c, gpu):
    if proc_alive(c, "server_cross_model_internvl.py"):
        log("[smoke] gen proc running -> skip"); return
    sh = ("#!/bin/bash\n"
          f"export CUDA_VISIBLE_DEVICES={gpu}\n"
          "export OMP_NUM_THREADS=8\n"
          f"cd {REMOTE}\n"
          f"{PYREMOTE} server_cross_model_internvl.py "
          f"--fused fused_50_out_v3.jsonl --parquet_dir /home/<SERVER_USER>/MMHalSnowball "
          f"--model OpenGVLab/InternVL2-4B --include_file cross_model_selection_ids.json "
          f"--n_images 12 --first_traj_only --resume --device cuda:0 "
          f"--limit 12 --out {SMOKE_OUT} 2>&1 | tee {SMOKE_OUT}.log\n"
          f"echo SMOKE_DONE >> {SMOKE_OUT}.log\n")
    sftp = c.open_sftp(); sftp.putfo(io.BytesIO(sh.encode()), f"{REMOTE}/run_internvl_smoke.sh"); sftp.close()
    c.exec_command(f"chmod +x {REMOTE}/run_internvl_smoke.sh")
    c.exec_command(f"setsid nohup bash {REMOTE}/run_internvl_smoke.sh > /dev/null 2>&1 < /dev/null &")
    log("[smoke] launched (detached)")


def launch_generate(c, gpu):
    if proc_alive(c, "server_cross_model_internvl.py"):
        log("[gen] gen proc running -> skip"); return
    sh = ("#!/bin/bash\n"
          f"export CUDA_VISIBLE_DEVICES={gpu}\n"
          "export OMP_NUM_THREADS=8\n"
          f"cd {REMOTE}\n"
          f"{PYREMOTE} server_cross_model_internvl.py "
          f"--fused fused_50_out_v3.jsonl --parquet_dir /home/<SERVER_USER>/MMHalSnowball "
          f"--model OpenGVLab/InternVL2-4B --include_file cross_model_selection_ids.json "
          f"--n_images 12 --first_traj_only --resume --device cuda:0 "
          f"--out {GEN_OUT} 2>&1 | tee {GEN_OUT}.log\n"
          f"echo GEN_DONE >> {GEN_OUT}.log\n")
    sftp = c.open_sftp(); sftp.putfo(io.BytesIO(sh.encode()), f"{REMOTE}/run_internvl_full.sh"); sftp.close()
    c.exec_command(f"chmod +x {REMOTE}/run_internvl_full.sh")
    c.exec_command(f"setsid nohup bash {REMOTE}/run_internvl_full.sh > /dev/null 2>&1 < /dev/null &")
    log("[gen] launched (detached)")


def do_pull(c):
    sftp = c.open_sftp(); sftp.get(f"{REMOTE}/{GEN_OUT}", os.path.join(LOCAL, GEN_OUT)); sftp.close()
    rows = sum(1 for _ in open(os.path.join(LOCAL, GEN_OUT), encoding="utf-8"))
    log(f"[pull] {GEN_OUT} rows={rows}")
    return rows


def do_judge():
    env = os.environ.copy()
    log("[judge] DeepSeek judge locally (resumable)")
    r = subprocess.run([PY, os.path.join(LOCAL, "run_glm_judge.py"),
                        "--key", JUDGE_KEY, "--model", JUDGE_MODEL, "--base", JUDGE_BASE,
                        "--workers", "16", "--max_tokens", "4096",
                        "--inp", os.path.join(LOCAL, GEN_OUT),
                        "--out", os.path.join(LOCAL, JUDGE_OUT)],
                       capture_output=True, text=True, env=env, timeout=50 * 60)
    log("[judge] rc", r.returncode, "stderr", (r.stderr or "")[-400:])
    outp = os.path.join(LOCAL, JUDGE_OUT)
    rows = sum(1 for _ in open(outp, encoding="utf-8")) if os.path.exists(outp) else 0
    return rows


def do_analyze_and_log():
    env = os.environ.copy()
    r = subprocess.run([PY, os.path.join(LOCAL, "analyze_cross_model.py"),
                        "--jin", os.path.join(LOCAL, GEN_OUT),
                        "--jout", os.path.join(LOCAL, JUDGE_OUT),
                        "--out", os.path.join(LOCAL, ANALYSIS_MD)],
                       capture_output=True, text=True, env=env, timeout=600)
    log("[analyze] rc", r.returncode, "stdout", (r.stdout or "").strip()[-600:])
    recovery = None
    per_round = None
    for ln in (r.stdout or "").splitlines():
        if "recovery H->F" in ln:
            recovery = ln.split("=")[-1].strip()
        if "LLaVA per-round" in ln:
            per_round = ln.split(":", 1)[-1].strip()
    if recovery is None and os.path.exists(os.path.join(LOCAL, ANALYSIS_MD)):
        import re
        for ln in open(os.path.join(LOCAL, ANALYSIS_MD), encoding="utf-8"):
            if "Self-recovery rate H->F" in ln:
                m = re.search(r"= ([\d.]+%)", ln)
                if m:
                    recovery = m.group(1)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    row = (f"\n### EXP2b 实测（{stamp}，统一驱动器自动写入）\n"
           f"- InternVL2-4B：OVERALL/逐轮见 `cross_model_analysis_internvl.md`；"
           f"self-recovery(H→F) = **{recovery or 'TBD'}**；逐轮 = {per_round or 'TBD'}\n"
           f"- 引用写法：`EXP2b (InternVL2-4B, 12 images, 648 units)`\n"
           f"- 与 EXP1/EXP2a/EXP2c-e 并列，作开源多架构多样性第三块（ViT-300M 动态分块 + Phi-3-mini-3.8B）。")
    log("[log] append EXP2b row")
    log(row)
    with open(os.path.join(LOCAL, "EXPERIMENT_LOG.md"), "a", encoding="utf-8") as f:
        f.write(row + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-reset", action="store_true")
    args = ap.parse_args()
    if args.force_reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    log("=== drive_exp2b_full", time.strftime("%Y-%m-%d %H:%M:%S"))
    st = load_state()
    if st.get("analyzed"):
        log("[done] already analyzed -> exit 0"); sys.exit(0)

    deadline = time.time() + 240 * 60  # 4h 总预算
    while time.time() < deadline:
        try:
            c = connect()
        except Exception as e:
            log(f"[connect-fail] {e} -> retry in 60s"); time.sleep(60); continue
        try:
            # ---- 1. download ----
            if not st["downloaded"]:
                sz = shard2_size(c); free = df_free(c)
                log(f"[dl] shard2={sz} free={free}MB alive={proc_alive(c,'dl_internvl_v2')}")
                if sz >= SHARD2_BYTES:
                    st["downloaded"] = True; save_state(st); log("[dl] verified")
                elif free >= 4000:
                    launch_download(c)
                else:
                    log("[dl] disk low, wait"); time.sleep(90); c.close(); continue
                if not st["downloaded"]:
                    time.sleep(90); c.close(); continue

            # ---- 2. smoke ----
            if not st["smoked"]:
                rows = remote_lines(c, f"{REMOTE}/{SMOKE_OUT}")
                log(f"[smoke] rows={rows} alive={proc_alive(c,'server_cross_model_internvl.py')}")
                if rows >= 12:
                    st["smoked"] = True; save_state(st); log("[smoke] verified")
                else:
                    gpu = pick_free_gpu(c); launch_smoke(c, gpu)
                    time.sleep(90); c.close(); continue

            # ---- 3. generate ----
            if not st["generated"]:
                rows = remote_lines(c, f"{REMOTE}/{GEN_OUT}")
                log(f"[gen] rows={rows} alive={proc_alive(c,'server_cross_model_internvl.py')}")
                if rows >= EXPECTED_ROWS:
                    st["generated"] = True; save_state(st); log("[gen] verified")
                else:
                    gpu = pick_free_gpu(c); launch_generate(c, gpu)
                    time.sleep(90); c.close(); continue

            # ---- 4. pull (local) ----
            if not st["pulled"]:
                rows = do_pull(c)
                if rows >= EXPECTED_ROWS:
                    st["pulled"] = True; save_state(st)
                else:
                    log(f"[pull] only {rows} -> retry"); c.close(); continue

            # ---- 5. judge (local) ----
            if not st["judged"]:
                rows = do_judge()
                if rows >= EXPECTED_ROWS:
                    st["judged"] = True; save_state(st)
                else:
                    log(f"[judge] only {rows}/{EXPECTED_ROWS} -> retry"); c.close(); continue

            # ---- 6. analyze + log ----
            if not st["analyzed"]:
                do_analyze_and_log()
                st["analyzed"] = True; save_state(st)
            c.close()
            log("=== drive_exp2b_full COMPLETE ===")
            return 0
        except Exception as e:
            log(f"[loop-err] {repr(e)[:300]}")
            try:
                c.close()
            except Exception:
                pass
            time.sleep(30)
    log("=== TIMEOUT 4h ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
