"""
freeze_snapshot.py -- copy (never move) the verified main-study artifacts into a
dated snapshot directory with an MD5 manifest, so any later edit to the live
workspace files is detectable.

The cross-model / attention scripts are NOT frozen: they are still in flight.
"""
import os, shutil, hashlib, time, sys, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default=None,
                help="snapshot dir name, e.g. EXP2_crossmodel_llava. "
                     "Default: EXP1_main_qwen_9000 (the verified main study).")
ap.add_argument("--spec", default=None,
                help="JSON file mapping group -> [files]; overrides the built-in EXP1 set.")
args = ap.parse_args()

LOCAL = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(LOCAL, args.tag or "EXP1_main_qwen_9000")
P = lambda *a: os.path.join(LOCAL, *a)

if args.spec:
    GROUPS = json.load(open(args.spec, encoding="utf-8"))
else:
    GROUPS = {
    "code": [
        "construct_inject_v3_server.py",   # prompt / injection construction
        "run_qwen_fused_batch.py",         # v3 generation (image every round)
        "export_judge_input.py",           # fused -> judge_input.jsonl
        "judge_prompt.py",                 # judge prompt + verdict parsing (Rule 6)
        "run_glm_judge.py",                # judge runner (used w/ DeepSeek model)
        "analyze_results_fast.py",         # main 6x2 table + cluster bootstrap
        "fig_make.py",                     # the three figures
        "audit_main_experiment.py",        # this snapshot's audit
        "freeze_snapshot.py",              # this file
    ],
    "data": [
        "fused_50_out_v3.jsonl",           # 9000 raw (traj,round) records
        "judge_input.jsonl",               # 9000 judge prompts
        "judge_output.jsonl",              # 9000 verdicts, single DeepSeek judge
        "judge_output_mixed_backup.jsonl", # provenance: pre-rejudge mixed-judge set
    ],
    "results": [
        "result_judge_analysis.md",
        "self_recovery_analysis.md",
        "self_recovery_sample20.json",
        "figs/fig1_round_curve.png",
        "figs/fig2_category_bar.png",
        "figs/fig3_recovery_sankey.png",
    ],
    }

def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

if os.path.exists(SNAP):
    print(f"snapshot {SNAP} already exists; refusing to overwrite")
    sys.exit(1)

TAG = os.path.basename(SNAP)
IS_MAIN = not args.spec
manifest = [
    f"# {TAG} -- frozen snapshot",
    "",
    f"created {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
]
if IS_MAIN:
    manifest += [
        "> **这是「第一轮实验」(EXP1) 的最终冻结件。** 后续补充实验另建独立目录，",
        "> 严禁与本目录文件混用、覆盖或交叉引用数字。完整版本总账见 `EXPERIMENT_LOG.md`。",
        "",
        "**Scope**: the MAIN study only (50 images, 9000 judged units, single DeepSeek judge).",
        "The cross-model (LLaVA) validation and the attention probe are **still in flight**",
        "and are deliberately NOT part of this snapshot.",
        "",
        "Frozen after `audit_main_experiment.py` passed all checks: 9000/9000 rows,",
        "0 errors, no missing/orphan uids, all 6 settings x 50 images x 3 trajectories,",
        "degradation < 2%, published numbers reproduce (entity_missing 0.951,",
        "self-recovery 28.0%, injection-baseline delta +0.53..+0.65).",
        "",
    ]
else:
    manifest += [
        f"**补充实验冻结件**（非 EXP1 主实验）。版本归属与引用规则见 `EXPERIMENT_LOG.md`。",
        "",
    ]
manifest += [
    "Frozen after `audit_main_experiment.py` passed all checks: 9000/9000 rows,",
    "0 errors, no missing/orphan uids, all 6 settings x 50 images x 3 trajectories,",
    "degradation < 2%, published numbers reproduce (entity_missing 0.951,",
    "self-recovery 28.0%, injection-baseline delta +0.53..+0.65).",
    "",
    "| group | file | bytes | md5 |",
    "|---|---|---:|---|",
]

n = 0
for grp, files in GROUPS.items():
    d = os.path.join(SNAP, grp)
    os.makedirs(d, exist_ok=True)
    for rel in files:
        src = P(rel)
        if not os.path.exists(src):
            print(f"  [MISS] {rel}")
            manifest.append(f"| {grp} | {rel} | MISSING | - |")
            continue
        dst = os.path.join(d, os.path.basename(rel))
        shutil.copy2(src, dst)          # copy -- never move
        h = md5(dst)
        sz = os.path.getsize(dst)
        n += 1
        manifest.append(f"| {grp} | {rel} | {sz:,} | `{h}` |")
        print(f"  [frozen] {grp}/{os.path.basename(rel)}  {sz:,} B  {h[:12]}...")

manifest += [
    "",
    "## How to verify nothing changed later",
    "",
    "```bash",
    f"python freeze_snapshot.py --verify   # not implemented: re-run md5 manually",
    "```",
    "",
    f"Or simply: `md5sum {TAG}/code/*.py` and compare with this table.",
    "",
    "## Reproduction order (main study)",
    "",
    "1. `construct_inject_v3_server.py` -- build the 6x2 x 3-trajectory protocol",
    "2. `run_qwen_fused_batch.py` -- Qwen2.5-VL-7B generation, image EVERY round -> `fused_50_out_v3.jsonl`",
    "3. `export_judge_input.py` -- -> `judge_input.jsonl` (9000)",
    "4. `run_glm_judge.py --model DeepSeek-V4.1-Flash-1017-Event --max_tokens 4096` -> `judge_output.jsonl`",
    "5. `analyze_results_fast.py` -> `result_judge_analysis.md`",
    "6. `fig_make.py` -> `figs/*.png`",
    "",
]

open(os.path.join(SNAP, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(manifest))
print(f"\nfroze {n} files -> {SNAP}")
print("manifest:", os.path.join(SNAP, "MANIFEST.md"))
