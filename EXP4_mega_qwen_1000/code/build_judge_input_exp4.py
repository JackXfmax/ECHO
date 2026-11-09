#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flatten EXP4 fused outputs into per-(sample,condition,traj,round) judge input.

EXP4 = 1000 images, Qwen2.5-VL-7B, halluconv_formatting setting only,
conditions = baseline(1 round) + injection(9 rounds), K=3 trajectories.

Each fused line = one sample with `cells` = [baseline, injection].
Each cell has `trajectories` (K=3) and `n_rounds_per_traj`.
Each trajectory has `trajectory_id`, `seed`, `rounds` (list of round dicts)
with keys: round, category_key, mode, current_question, ground_truth,
premise_false, model_output.

Output: one judge_input record per (sample, condition, traj, round).
Fields kept for the judge: current_question, ground_truth, model_output.
Metadata carried along for stratified analysis.
"""
import json, os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True,
                help="fused_*.jsonl files (shard a / b / merged)")
ap.add_argument("--out", required=True)
ap.add_argument("--only_condition", default=None,
                help="if set, only emit this condition (baseline/injection)")
ap.add_argument("--limit_records", type=int, default=None)
args = ap.parse_args()

n = 0
with open(args.out, "w", encoding="utf-8") as fout:
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sid = d.get("sample_id")
                meta = {
                    "sample_id": sid,
                    "hallucination_type": d.get("hallucination_type"),
                    "truth_token": d.get("truth_token"),
                    "false_token": d.get("false_token"),
                    "truth_noun": d.get("truth_noun"),
                    "fact": d.get("fact"),
                    "hallucinatory_fact": d.get("hallucinatory_fact"),
                    "vision_mode": d.get("vision_mode"),
                    "temperature": d.get("temperature"),
                    "top_p": d.get("top_p"),
                    "num_samples": d.get("num_samples"),
                }
                for cell in d.get("cells", []):
                    cond = cell.get("condition")
                    if args.only_condition and cond != args.only_condition:
                        continue
                    for traj in cell.get("trajectories", []):
                        tid = traj.get("trajectory_id")
                        tseed = traj.get("seed")
                        for rnd in traj.get("rounds", []):
                            rec = dict(meta)
                            rec.update({
                                "condition": cond,
                                "trajectory_id": tid,
                                "traj_seed": tseed,
                                "round": rnd.get("round"),
                                "category_key": rnd.get("category_key"),
                                "mode": rnd.get("mode"),
                                "premise_false": rnd.get("premise_false"),
                                "current_question": rnd.get("current_question"),
                                "ground_truth": rnd.get("ground_truth"),
                                "model_output": rnd.get("model_output"),
                                "uid": f"{sid}|{cond}|t{tid}|r{rnd.get('round')}",
                            })
                            # skip empty model outputs (shouldn't happen, but safe)
                            if not rec.get("model_output"):
                                continue
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            n += 1
                            if args.limit_records and n >= args.limit_records:
                                print(f"wrote {n} records -> {args.out}")
                                sys.exit(0)
print(f"wrote {n} records -> {args.out}")
