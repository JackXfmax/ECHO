# -*- coding: utf-8 -*-
"""本地驱动 EXP2b 下载阶段：上传带重试的续传脚本到服务器，脱离 SSH 后台跑，
精确轮询 shard2 落地 + 磁盘余量，直到完成。DNS 已恢复，shard1 会自动跳过。"""
import paramiko, time, os, sys, io

HOST, PORT, USER, PASS = "<REDACTED-SERVER-IP>", 22, "<SERVER_USER>", "<REDACTED-SERVER-PASSWORD>"
REMOTE = "/home/<SERVER_USER>/mmhal_inject_v3"
PYREMOTE = "/home/<SERVER_USER>/miniconda3/envs/train/bin/python"
SHARD2_BYTES = 3_000_000_000  # shard2 完整应 > 3.0G（实际 ~3.34G）

SERVER_SCRIPT = r'''import os, time, sys, glob
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
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
    print("SHARD2_PRESENT skip", flush=True)
    sys.exit(0)

for attempt in range(40):
    try:
        path = snapshot_download(
            REPO, repo_type="model", max_workers=8,
            allow_patterns=["*.json", "*.txt", "*.py", "*.safetensors", "*.model", "*.bin"],
            ignore_patterns=["examples/*", "*.msgpack", "*.h5", "*.ot"])
        print("DONE", path, flush=True)
        break
    except Exception as e:
        print("ATTEMPT", attempt, "ERR", repr(e)[:300], flush=True)
        time.sleep(20)
'''


def connect():
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


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def shard2_size(c):
    rc, out, _ = run(c,
        "for f in $(find /home/<SERVER_USER>/.cache/huggingface/hub/models--OpenGVLab--InternVL2-4B/snapshots "
        "-name 'model-00002-of-00002.safetensors' 2>/dev/null); do stat -L -c %s \"$f\"; done")
    vals = [int(x) for x in out.split() if x.strip().isdigit()]
    return max(vals) if vals else 0


def df_free(c):
    rc, out, _ = run(c, "df -m / | tail -1 | awk '{print $4}'")
    try:
        return int(out.split()[0])
    except Exception:
        return -1


def proc_alive(c):
    rc, out, _ = run(c, "ps aux | grep dl_internvl_v2 | grep -v grep | wc -l")
    return int((out or "0").strip() or 0) > 0


def upload_and_launch(c):
    sftp = c.open_sftp()
    sftp.putfo(io.BytesIO(SERVER_SCRIPT.encode()), f"{REMOTE}/dl_internvl_v2.py")
    sftp.close()
    c.exec_command(f"chmod +x {REMOTE}/dl_internvl_v2.py")
    c.exec_command(
        f"setsid nohup {PYREMOTE} {REMOTE}/dl_internvl_v2.py >> {REMOTE}/dl_internvl_v2.out 2>&1 < /dev/null &")
    log("[launch] dl_internvl_v2.py started (detached)")


def main():
    deadline = time.time() + 120 * 60
    launched_once = False
    while time.time() < deadline:
        try:
            c = connect()
        except Exception as e:
            log(f"[connect-fail] {e} -> retry in 60s")
            time.sleep(60)
            continue
        try:
            sz = shard2_size(c)
            free = df_free(c)
            alive = proc_alive(c)
            log(f"[check {time.strftime('%H:%M:%S')}] shard2_bytes={sz} free_MB={free} dl_alive={alive}")
            if sz >= SHARD2_BYTES:
                log("=== SHARD2 COMPLETE ===")
                c.close()
                return 0
            if free < 4000:
                log(f"[warn] disk free={free}MB < 4G, pausing launch (will retry after proc frees space)")
            if not alive and free >= 4000:
                upload_and_launch(c)
                launched_once = True
            elif not alive and not launched_once and sz < SHARD2_BYTES:
                upload_and_launch(c)
                launched_once = True
        except Exception as e:
            log(f"[loop-err] {repr(e)[:200]}")
        try:
            c.close()
        except Exception:
            pass
        time.sleep(90)
    log("=== TIMEOUT: shard2 not complete within 120min ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
