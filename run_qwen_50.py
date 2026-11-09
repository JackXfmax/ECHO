import paramiko, os, time

HOST, PORT = "<REDACTED-SERVER-IP>", 22
USER, PASS = "<SERVER_USER>", "<REDACTED-SERVER-PASSWORD>"
remote_dir = "/home/<SERVER_USER>/mmhal_inject_v3"
local_out = r"C:\Users\<LOCAL_USER>\Desktop\MMHalSnowball_inject_v3"

local_run = r"<WORKSPACE>\run_qwen_test.py"
local_sh = r"<WORKSPACE>\run_qwen_50.sh"

def connect():
    ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return ssh

# 1) upload latest scripts
ssh = connect(); sftp = ssh.open_sftp()
print("[upload] run_qwen_test.py ...", flush=True)
sftp.put(local_run, os.path.join(remote_dir, "run_qwen_test.py"))
sftp.put(local_sh, os.path.join(remote_dir, "run_qwen_50.sh"))
sftp.close(); ssh.close()
print("[upload] done", flush=True)

# 2) launch detached background run via setsid
ssh = connect()
launch = ("chmod +x %s/run_qwen_50.sh && "
          "setsid bash %s/run_qwen_50.sh > %s/qwen_50_run.log 2>&1 < /dev/null &"
          % (remote_dir, remote_dir, remote_dir))
print("[launch] ", launch, flush=True)
ssh.exec_command(launch, timeout=60)
time.sleep(6)
# confirm it started
stdin, stdout, stderr = ssh.exec_command("tail -n 3 %s/qwen_50_run.log; echo '--- ps ---'; ps aux | grep run_qwen_test | grep -v grep | head" % remote_dir)
print(stdout.read().decode(errors="replace"))
ssh.close()

# 3) poll log until DONE or timeout
deadline = time.time() + 65 * 60
last = ""
while time.time() < deadline:
    try:
        ssh = connect()
        cmd = ("echo '=== LOG TAIL ==='; tail -n 4 %s/qwen_50_run.log; "
               "echo '=== OUTPUT LINES ==='; wc -l < %s/qwen_50_out.jsonl 2>/dev/null || echo 0"
               % (remote_dir, remote_dir))
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode(errors="replace")
        ssh.close()
        # only print if changed
        if out != last:
            print(out, flush=True)
            last = out
        if "DONE" in out:
            print("[monitor] DONE detected.", flush=True)
            break
    except Exception as e:
        print("[monitor] poll error:", e, flush=True)
    time.sleep(45)

# 4) pull result
try:
    ssh = connect(); sftp = ssh.open_sftp()
    rpath = os.path.join(remote_dir, "qwen_50_out.jsonl")
    lpath = os.path.join(local_out, "qwen_50_out.jsonl")
    try:
        with sftp.open(rpath, "rb") as fr:
            data = fr.read()
        with open(lpath, "wb") as fw:
            fw.write(data)
        print("[pull] qwen_50_out.jsonl ->", lpath, len(data), "bytes", flush=True)
    except Exception as e:
        print("[pull] FAILED:", repr(e), flush=True)
    sftp.close(); ssh.close()
except Exception as e:
    print("[pull] connect FAILED:", e, flush=True)

print("[monitor] exit", flush=True)
