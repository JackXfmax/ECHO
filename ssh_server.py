#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SSH/SFTP helper for the MMHalSnowball 4090 server (<REDACTED-SERVER-IP>)."""
import sys, os, io, time

HOST = "<REDACTED-SERVER-IP>"
PORT = 22
USER = "<SERVER_USER>"
PASS = "<REDACTED-SERVER-PASSWORD>"

def _client():
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=30, banner_timeout=30,
              look_for_keys=False, allow_agent=False)
    return c

def run(cmd, timeout=600):
    c = _client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err

def push(local, remote):
    """SFTP upload a single local file to the server."""
    import paramiko
    c = _client()
    sftp = c.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    c.close()
    return os.path.getsize(local)

def pull(remote, local, max_bytes=None, chunk_cb=None):
    """SFTP pull a single file. Returns (size_bytes, done)."""
    import paramiko
    c = _client()
    sftp = c.open_sftp()
    # ensure local dir
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    # resume support: get existing local size
    try:
        local_size = os.path.getsize(local)
    except OSError:
        local_size = 0
    rsize = sftp.stat(remote).st_size
    if max_bytes:
        rsize = min(rsize, max_bytes)
    if local_size >= rsize and rsize > 0:
        sftp.close(); c.close()
        return rsize, True
    # remote read from offset
    with sftp.open(remote, "rb") as rf:
        rf.seek(local_size)
        mode = "ab" if local_size > 0 else "wb"
        with open(local, mode) as lf:
            lf.seek(local_size)
            remaining = rsize - local_size
            got = 0
            while remaining > 0:
                buf = rf.read(min(1024*1024, remaining))
                if not buf:
                    break
                lf.write(buf)
                got += len(buf)
                remaining -= len(buf)
                if chunk_cb:
                    chunk_cb(local_size + got, rsize)
    sftp.close(); c.close()
    return rsize, True

def main():
    mode = sys.argv[1]
    if mode == "run":
        rc, out, err = run(sys.argv[2], timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 600)
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        sys.exit(rc)
    elif mode == "pull":
        local = sys.argv[3] if len(sys.argv) > 3 else None
        remote = sys.argv[2]
        if not local:
            local = os.path.basename(remote)
        size, done = pull(remote, local)
        print(f"PULLED {remote} -> {local} ({size} bytes, done={done})")
    elif mode == "push":
        local = sys.argv[2]
        remote = sys.argv[3]
        size = push(local, remote)
        print(f"PUSHED {local} -> {remote} ({size} bytes)")
    else:
        print("usage: ssh_server.py run '<cmd>' [timeout] | pull <remote> [local] | push <local> <remote>")
        sys.exit(2)

if __name__ == "__main__":
    main()
