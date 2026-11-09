#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-off total freeze for MMHalSnowball EXP1-EXP4.

Rules enforced here (from EXPERIMENT_LOG.md hard constraints):
  * each round gets its own EXP<n>_<codename>/ directory
  * COPY, never MOVE -- live working files stay where they are
  * every frozen file gets MD5 + bytes + line count into a machine-readable registry
  * EXP1 is already frozen: it is VERIFIED only, never rewritten
"""
import os, json, hashlib, shutil, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

REG = "frozen_registry.json"

# (relative source path, destination subdir inside the round dir)
EXP2 = {
    "data": [
        "cross_model_judge_input_llava.jsonl", "cross_model_judge_output_llava.jsonl",
        "cross_model_judge_input_internvl.jsonl", "cross_model_judge_output_internvl.jsonl",
        "cross_model_judge_input_gpt4o.jsonl", "cross_model_judge_output_gpt4o.jsonl",
        "cross_model_judge_input_gemini25flash.jsonl", "cross_model_judge_output_gemini25flash.jsonl",
        "cross_model_judge_input_claudesonnet46.jsonl", "cross_model_judge_output_claudesonnet46.jsonl",
        "cross_model_selection.json", "cross_model_selection_attn.json",
        "cross_model_selection_ids.json",
    ],
    "results": ["cross_model_analysis_internvl.md",
                "figs/figA_five_model_oscillation.png", "figs/figB_self_recovery_bar.png"],
    "assets": sorted("cross_images/" + f for f in os.listdir("cross_images") if f.endswith(".jpg")),
    "code": ["server_cross_model_probe.py", "server_cross_model_internvl.py",
             "run_api_vlm_local.py", "analyze_cross_model.py", "analyze_frontier.py"],
}

EXP3 = {
    "data": ["attn_perlayer_hf.jsonl", "attn_perlayer_ff.jsonl", "attn_decompose_hf.jsonl",
             "attn_selection_hf.jsonl", "attn_selection_ff.jsonl",
             "attn_selection_expanded.json", "attn_smoke.jsonl"],
    "results": ["figs/figC_layer_var.png", "figs/figC_layer_delta.png",
                "figs/figC_layer_delta_lift.png", "figs/figC_rounds_r3_r7.png",
                "figs/figD_head_signature.png", "figs/figD_mechanism_decompose.png"],
    "code": ["build_attn_selection_hf.py", "build_attn_selection_ff.py",
             "server_attn_perlayer.py", "server_attn_decompose.py",
             "analyze_attn_perlayer.py", "analyze_attn_controlled.py", "analyze_decompose.py",
             "make_figC.py", "make_figD_heads.py",
             "diag_permutation.py", "diag_splithalf.py", "diag_residualize.py"],
}

EXP4 = {
    "data": ["fused_1000_hallu_a.jsonl", "fused_1000_hallu_b.jsonl",
             "fused_1000_merged.jsonl", "judge_input_exp4.jsonl",
             "judge_output_exp4.jsonl", "test_set_1000.json", "test_set_1000.jsonl"],
    "results": ["fig_mega_1000.pdf", "EXP4_MEGA_ANALYSIS.md", "EXP4_MEGA_PLAN.md"],
    "code": ["construct_inject_v3_1000.py", "build_judge_input_exp4.py",
             "analyze_exp4.py", "run_glm_judge.py", "judge_prompt.py",
             "run_qwen_fused_batch.py", "ssh_server.py"],
}

# shared infrastructure: verified in place, not copied (used by every round)
SHARED = ["run_qwen_fused_batch.py", "construct_inject_v3_server.py",
          "export_judge_input.py", "judge_prompt.py", "run_glm_judge.py",
          "audit_main_experiment.py", "freeze_snapshot.py",
          "_server_run_qwen_fused_batch.py",
          "EXPERIMENT_LOG.md", "PAPER_WRITING_KIT.md",
          "FROZEN_MANIFEST.md", "verify_frozen.py", "freeze_all_experiments.py"]
# NOTE: frozen_registry.json / FROZEN_FILES.md are intentionally NOT registered
# (self-referential: regenerated on every freeze). FROZEN_FILES.md is a view of
# the registry, and the registry cannot contain its own hash.

# EXP1 (already frozen 2026-10-26): verify only, never rewrite
EXP1_FILES = ["EXP1_main_qwen_9000/MANIFEST.md",
              "EXP1_main_qwen_9000/AUDIT_REPORT.txt",
              "EXP1_main_qwen_9000/data/fused_50_out_v3.jsonl",
              "EXP1_main_qwen_9000/data/judge_input.jsonl",
              "EXP1_main_qwen_9000/data/judge_output.jsonl",
              "EXP1_main_qwen_9000/results/result_judge_analysis.md",
              "EXP1_main_qwen_9000/results/self_recovery_analysis.md",
              "EXP1_main_qwen_9000/code/run_qwen_fused_batch.py",
              "EXP1_main_qwen_9000/code/construct_inject_v3_server.py",
              "EXP1_main_qwen_9000/code/export_judge_input.py",
              "EXP1_main_qwen_9000/code/judge_prompt.py",
              "EXP1_main_qwen_9000/code/run_glm_judge.py",
              "EXP1_main_qwen_9000/code/analyze_results_fast.py",
              "EXP1_main_qwen_9000/code/audit_main_experiment.py"]

ROUNDS = {"EXP2_crossmodel": EXP2,
          "EXP3_attention_mechanism": EXP3,
          "EXP4_mega_qwen_1000": EXP4}


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


def entry(src, dst, copied):
    return {"src": src, "frozen_path": dst, "copied": copied,
            "md5": md5(dst), "bytes": os.path.getsize(dst), "lines": nlines(dst)}


def main():
    registry = {
        "frozen_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": ROOT,
        "note": "MD5 baseline. Run verify_frozen.py to detect ANY change. "
                "copied=false means verified in place (already-frozen EXP1 or shared infra).",
        "rounds": {}, "shared": [], "provenance_notes": [],
    }
    missing = []

    # EXP1: verify only
    e1 = []
    for p in EXP1_FILES:
        if os.path.isfile(p):
            e1.append(entry(p, p, False))
        else:
            missing.append("EXP1:" + p)
    registry["rounds"]["EXP1_main_qwen_9000"] = {"status": "FROZEN_2026-10-26 (verify-only)",
                                                  "files": e1}

    # EXP2/3/4: copy into own dirs
    for rdir, groups in ROUNDS.items():
        files = []
        for sub, paths in groups.items():
            for src in paths:
                if not os.path.isfile(src):
                    missing.append(rdir + ":" + src)
                    continue
                dst = os.path.join(rdir, sub, os.path.basename(src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                files.append(entry(src, dst, True))
        registry["rounds"][rdir] = {"status": "FROZEN_2026-11-07 (copied)", "files": files}

    # shared infra
    for p in SHARED:
        if os.path.isfile(p):
            registry["shared"].append(entry(p, p, False))
        else:
            missing.append("SHARED:" + p)

    registry["provenance_notes"] = [
        "EXP1 frozen 2026-10-26 23:26, MANIFEST.md MD5 table verified unchanged.",
        "Actual generation engine = run_qwen_fused_batch.py; server copy (LF) is byte-identical "
        "to the frozen CRLF copy after stripping CR (md5 differs only by line endings).",
        "7 v3 shard scripts run_v3_t0.sh..t42.sh all call run_qwen_fused_batch.py "
        "--conditions baseline,injection.",
        "cross_model_judge_input.jsonl (0 bytes, root) is an EMPTY stale stub - DO NOT USE.",
        "judge_output_mixed_backup.jsonl is historical provenance only - NOT a result source.",
        "figs/figATT*_selftest.png and *_smoke.png are synthetic/smoke - DO NOT CITE.",
    ]

    with open(REG, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=1)

    # human-readable file listing
    lines = ["# 冻结文件清单（自动生成，勿手改）", "",
             f"冻结时间：{registry['frozen_at']}　|　基线：`{REG}`　|　校验：`python verify_frozen.py`", ""]
    for r, info in registry["rounds"].items():
        lines += [f"## {r} — {info['status']}", "",
                  "| 冻结路径 | bytes | lines | md5 |", "|---|---:|---:|---|"]
        for e in info["files"]:
            ln = "" if e["lines"] is None else str(e["lines"])
            lines.append(f"| `{e['frozen_path']}` | {e['bytes']} | {ln} | `{e['md5']}` |")
        lines.append("")
    lines += ["## 共享基础设施（原地校验，不复制）", "",
              "| 路径 | bytes | lines | md5 |", "|---|---:|---:|---|"]
    for e in registry["shared"]:
        ln = "" if e["lines"] is None else str(e["lines"])
        lines.append(f"| `{e['frozen_path']}` | {e['bytes']} | {ln} | `{e['md5']}` |")
    if missing:
        lines += ["", "## ⚠ 缺失文件（未纳入冻结）", ""] + [f"- `{m}`" for m in missing]
    open("FROZEN_FILES.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("frozen_at:", registry["frozen_at"])
    for r, info in registry["rounds"].items():
        print(f"  {r}: {len(info['files'])} files")
    print(f"  shared: {len(registry['shared'])} files")
    print("missing:", missing if missing else "NONE")
    print("registry ->", REG, "| listing -> FROZEN_FILES.md")


if __name__ == "__main__":
    main()
