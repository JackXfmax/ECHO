# Cross-Model Validation: LLaVA-1.5-7B vs Qwen2.5-VL

rows judged: 648   samples: 12   errors skipped: 0

## 1. Per-round hallucination rate (injection, rounds 1-9)

| round | LLaVA (this run) | Qwen (reference) |
|---|---|---|
| 1 (entity_missing) | 1.000 [1.000,1.000] | 0.923 |
| 2 (entity_missing) | 1.000 [1.000,1.000] | 0.979 |
| 3 (attribute_reversal) | 0.917 [0.750,1.000] | 0.570 |
| 4 (attribute_reversal) | 1.000 [1.000,1.000] | 0.639 |
| 5 (background_fabrication) | 1.000 [1.000,1.000] | 0.500 |
| 6 (background_fabrication) | 1.000 [1.000,1.000] | 0.826 |
| 7 (context_pollution) | 0.500 [0.250,0.750] | 0.659 |
| 8 (context_pollution) | 0.611 [0.375,0.833] | 0.631 |
| 9 (dataset_final) | 0.333 [0.083,0.583] | 0.479 |

## 2. Self-recovery transition (per consecutive round)

HH=430  HF=76  FH=28  FF=42   (H=hallucination, F=faithful)

**Self-recovery rate H->F = 76/506 = 15.0%**

Qwen reference self-recovery rate = 28.0%

## 3. Interpretation

- If LLaVA's per-round curve ALSO shows non-monotonic oscillation (e.g. a spike at the entity-missing rounds 1-2 and a dip/recovery later), the snowball+self-recovery phenomenon is architectural, not a Qwen quirk.
- The absolute rates differ across architectures (expected); the STRUCTURAL pattern (oscillation + partial self-recovery) is the claim.
- Note: LLaVA here runs only the first trajectory per (sample,setting) to bound compute; Qwen's reference aggregates all trajectories.