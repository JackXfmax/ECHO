#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the MMHalSnowball total freeze.

Usage:  python verify_frozen.py
Checks, against frozen_registry.json:
  1. every registered file still exists
  2. MD5 unchanged (ANY edit to an experiment artifact is detected)
  3. byte size / line count unchanged
  4. judge outputs: uid coverage complete + ZERO error verdicts (EXP1/EXP2/EXP4)
  5. EXP4 undetermined == 1 (the single safety refusal, intentionally excluded)

Exit code 0 = ALL PASS, 1 = something changed. Run this before citing any number.
"""
import os, json, hashlib, sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
REG = "frozen_registry.json"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nlines(p):
    if not p.lower().endswith((".jsonl", ".json", ".md", ".py", ".txt")):
        return None
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def judge_audit(inp, out, label):
    """Return (ok, message) for a judge input/output pair."""
    I = [json.loads(l) for l in open(inp, encoding="utf-8")]
    O, parse_err = [], 0
    for l in open(out, encoding="utf-8"):
        try:
            O.append(json.loads(l))
        except Exception:
            parse_err += 1
    iu = {r["uid"] for r in I}
    ou = {r.get("uid") for r in O}
    v = Counter(r.get("verdict") for r in O)
    err = v.get("error", 0)
    und = v.get("undetermined", 0)
    miss, extra = len(iu - ou), len(ou - iu)
    # allowed: EXP4 has exactly 1 undetermined (safety refusal)
    allowed_und = 1 if label == "EXP4" else 0
    ok = (miss == 0 and extra == 0 and parse_err == 0 and err == 0
          and und == allowed_und)
    return ok, (f"in={len(I)} out={len(O)} miss={miss} extra={extra} "
                f"H={v.get('hallucination', 0)} F={v.get('faithful', 0)} "
                f"error={err} undetermined={und} parse_err={parse_err}")


def main():
    if not os.path.isfile(REG):
        print("FATAL: frozen_registry.json missing -- freeze baseline lost!")
        return 1
    reg = json.load(open(REG, encoding="utf-8"))
    fails, checked = [], 0

    print("=" * 68)
    print("MMHalSnowball TOTAL FREEZE VERIFICATION")
    print("frozen_at:", reg["frozen_at"])
    print("=" * 68)

    drift = []
    for r, info in reg["rounds"].items():
        n_bad = 0
        for e in info["files"]:
            p, checked = e["frozen_path"], checked + 1
            if not os.path.isfile(p):
                fails.append(f"[{r}] MISSING FILE {p}")
                n_bad += 1
                continue
            m, b, l = md5(p), os.path.getsize(p), nlines(p)
            if m != e["md5"]:
                fails.append(f"[{r}] MD5 CHANGED {p}")
                n_bad += 1
            elif b != e["bytes"]:
                fails.append(f"[{r}] SIZE CHANGED {p}")
                n_bad += 1
            elif l != e["lines"]:
                fails.append(f"[{r}] LINE COUNT CHANGED {p}")
                n_bad += 1
            # live-file drift: the working copy must still equal its frozen twin
            src = e.get("src")
            if e.get("copied") and src and os.path.isfile(src) and md5(src) != e["md5"]:
                drift.append(f"[{r}] LIVE FILE DRIFT {src} != frozen {p}")
        print(f"  {r:<32} {len(info['files']):>3} files  {'OK' if n_bad == 0 else str(n_bad) + ' BAD'}")
    for e in reg["shared"]:
        p, checked = e["frozen_path"], checked + 1
        if not os.path.isfile(p):
            fails.append(f"[shared] MISSING FILE {p}")
            continue
        if md5(p) != e["md5"]:
            fails.append(f"[shared] MD5 CHANGED {p}")
    print(f"  {'shared infrastructure':<32} {len(reg['shared']):>3} files  "
          f"{'OK' if not any('shared' in x for x in fails) else 'BAD'}")
    print(f"  files checked: {checked}")
    if drift:
        print("-" * 68)
        print(f"LIVE WORKING-FILE DRIFT: {len(drift)} file(s) differ from frozen twins")
        for d in drift[:10]:
            print("   ~", d)
        print("   (frozen artifacts are intact; but someone edited a live copy.")
        print("    Re-run freeze_all_experiments.py to re-freeze, or restore from frozen dir.)")
        fails += drift

    print("-" * 68)
    print("JUDGE OUTPUT INTEGRITY")
    pairs = [
        ("EXP1", "EXP1_main_qwen_9000/data/judge_input.jsonl",
         "EXP1_main_qwen_9000/data/judge_output.jsonl"),
        ("EXP2/llava", "cross_model_judge_input_llava.jsonl", "cross_model_judge_output_llava.jsonl"),
        ("EXP2/internvl", "cross_model_judge_input_internvl.jsonl", "cross_model_judge_output_internvl.jsonl"),
        ("EXP2/gpt4o", "cross_model_judge_input_gpt4o.jsonl", "cross_model_judge_output_gpt4o.jsonl"),
        ("EXP2/gemini", "cross_model_judge_input_gemini25flash.jsonl", "cross_model_judge_output_gemini25flash.jsonl"),
        ("EXP2/claude", "cross_model_judge_input_claudesonnet46.jsonl", "cross_model_judge_output_claudesonnet46.jsonl"),
        ("EXP4", "judge_input_exp4.jsonl", "judge_output_exp4.jsonl"),
    ]
    for label, i, o in pairs:
        ok, msg = judge_audit(i, o, label)
        print(f"  {label:<16} {'OK ' if ok else 'BAD'}  {msg}")
        if not ok:
            fails.append(f"[judge] {label}: {msg}")
    print("-" * 68)

    if fails:
        print(f"RESULT: {len(fails)} PROBLEM(S) -- DO NOT cite numbers before resolving:")
        for f in fails:
            print("   !", f)
        return 1
    print("RESULT: ALL CHECKS PASSED -- freeze intact, numbers safe to cite.")
    print("Reminder: EXP1/EXP2/EXP3/EXP4 numbers are ISOLATED; never merge or average.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
