# ECHO: Multi-turn Conversational Hallucination Snowball in LVLMs

Code, experiment data, and analysis for studying the **conversational hallucination snowball effect** in Large Vision-Language Models (LVLMs).

When an LVLM is pushed into hallucinating in turn 1 of a multi-turn dialogue, injected false premises do not stay isolated: they **snowball** across subsequent turns, and the model only partially self-recovers.

## Experiment Suite

| Directory | Model | Scale | What it shows |
|---|---|---|---|
| `EXP1_main_qwen_9000/` | Qwen2.5-VL-7B-Instruct | 50 images × 9,000 judge units | Main experiment: injection raises hallucination rate from ~0.05–0.13 to ~0.65–0.71 (+0.53–0.65); 28.0% H→F self-recovery |
| `EXP2_crossmodel/` | LLaVA-1.5-7B, InternVL2-4B (+ GPT-4o / Gemini-2.5-Flash / Claude-Sonnet via API) | 12 images × 648 units each | Cross-architecture generality; open-source models are markedly more severe and less self-recovering |
| `EXP3_attention_mechanism/` | Qwen2.5-VL-7B (eager attention) | 72 trajectories × 648 forwards | Stratified attention analysis: recovery is **not** driven by a global visual-attention rebound; only sparse, sign-heterogeneous head re-weighting |
| `EXP4_mega_qwen_1000/` | Qwen2.5-VL-7B-Instruct | 1,000 images × 30,000 judge units | Scale-up replication: injection hallucination rate 0.736 vs baseline 0.182 (+0.555); self-recovery 0.242 |

Each experiment directory follows the same layout:

```
EXP<n>_<codename>/
├── code/       # generation, judging, analysis scripts (frozen copies)
└── data/       # trajectories, judge inputs/outputs (JSONL)
```

Root-level scripts mirror the same pipeline (dialogue construction, generation, judging, attention analysis); `data/examples/` holds the sample release (see below).

## Data

**Examples (released now).** `data/examples/` contains 3 example samples, each with the source image and its full annotated dialogue:

- `04724052.jpg` — injected false premise *"car"* vs. ground truth *"bus"* (existence-type injection)
- `10768688.jpg` — injected false premise *"building"* vs. ground truth *"bus"*
- `131064738.jpg` — attribute-reversal injection on an elephants image (the sample used for the attention visualizations)

`data/examples/annotations.json` provides, per sample: the dataset's own image description, the ground-truth fact, the injected hallucinatory fact, and the **full 9-round injection dialogue** (question, model response, and per-round hallucination/faithful verdict by the LLM judge) from EXP4 (Qwen2.5-VL-7B-Instruct, temperature 0.7, trajectory t0).

**Full dataset (coming soon).** The complete set of 1,000 images and all annotated multi-turn dialogues (EXP1: 50 images × 9,000 judged units; EXP2: 5 models × 648; EXP4: 1,000 images × 30,000 judged units) will be **released progressively** — we plan to open up the full annotated data in stages as the paper progresses through review/publication. The judge input/output files already included in the `EXP*/data/` directories contain the complete questions, responses, and verdicts; the remaining images accompany them in a later batch.

## Pipeline Overview

1. **Dialogue construction** — `construct_inject_v3_server.py` / `construct_inject_v3_1000.py` build multi-turn dialogues with 8 categories of injected false premises (entity_missing, attribute_reversal, background_fabrication, context_pollution), with lead/escalate scheduling across rounds.
2. **Generation** — `run_qwen_fused_batch.py` runs the LVLM over rounds (image passed every round, temperature 0.7, K=3 trajectories).
3. **Judging** — `judge_prompt.py` + `run_glm_judge.py` call an LLM judge (memoryless, 3-field verdict: hallucination / faithful / undetermined) on every round.
4. **Analysis** — `analyze_*.py` compute per-round curves, transition matrices, self-recovery rates, and bootstrap CIs.
5. **Attention analysis (EXP3)** — `server_attn_perlayer.py` / `analyze_attn_controlled.py`: stratified matching on exact bitmask of prior H/F history, permutation tests with max-|delta| statistic, split-half replication.

## A Note on Credentials

All scripts were developed against a private GPU server and LLM-judge API endpoints. For this public release, **all credentials have been removed** and replaced with placeholders:

- `<REDACTED-SERVER-IP>`, `<REDACTED-SERVER-PASSWORD>`, `<SERVER_USER>` — SSH endpoints
- `sk-REDACTED` — LLM API keys
- `<REDACTED-PROXY-HOST>` — third-party API proxy

To re-run, set these via environment variables or your own configuration.

## Requirements

See `requirements.txt`. Development used Python 3.10+, PyTorch 2.x, Transformers (4.x for InternVL compatibility notes).

## License

For academic use. If this code or data is useful for your research, please cite the corresponding paper (to appear).
