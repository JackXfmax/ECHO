# MMHalSnowball 判定结果分析（单一裁判版）
- Judge: **DeepSeek-V4.1-Flash-1017-Event**（新 key，全程同一裁判）
- judged rows: 9000   errors: 0 (0.0%)
- 说明：此前 2560 条由 GLM-5.3-Flash 判定；为消除跨裁判噪声，已用 DeepSeek 重判这 2560 条，现全量 9000 条为单一裁判。混合版保留在 `judge_output_mixed_backup.jsonl`。

================ 1. Hallucination rate by setting x condition (ALL rounds) ================
setting                       baseline       injection           delta
cleanconv_formatting     0.133[0.060,0.213]   0.689[0.649,0.728]           0.556
cleanconv_question       0.113[0.053,0.180]   0.690[0.646,0.731]           0.576
factconv_formatting      0.120[0.053,0.200]   0.650[0.605,0.693]           0.530
halluconv_question       0.133[0.073,0.200]   0.704[0.666,0.740]           0.570
halluconv_formatting     0.127[0.060,0.200]   0.711[0.674,0.750]           0.584
irrconv_formatting       0.047[0.013,0.087]   0.694[0.650,0.734]           0.647

========= 2. FINAL round (dataset core question): baseline r1 vs injection r9 =============
setting                    baseline r1    injection r9         delta (paired CI)
cleanconv_formatting     0.133[0.060,0.220]   0.473[0.387,0.567]       0.340 [0.240,0.460]
cleanconv_question       0.113[0.053,0.180]   0.440[0.347,0.540]       0.327 [0.233,0.420]
factconv_formatting      0.120[0.053,0.193]   0.393[0.293,0.493]       0.273 [0.180,0.380]
halluconv_question       0.133[0.073,0.200]   0.527[0.440,0.613]       0.393 [0.293,0.487]
halluconv_formatting     0.127[0.060,0.200]   0.547[0.453,0.647]       0.420 [0.313,0.520]
irrconv_formatting       0.047[0.013,0.087]   0.493[0.400,0.587]       0.447 [0.353,0.547]

=============== 3. Injection collapse curve: hallucination rate by round 1..9 ==============
  round 1 (entity_missing        ): 0.923  95% CI [0.876, 0.966]
  round 2 (entity_missing        ): 0.979  95% CI [0.948, 0.997]
  round 3 (attribute_reversal    ): 0.570  95% CI [0.516, 0.623]
  round 4 (attribute_reversal    ): 0.639  95% CI [0.588, 0.691]
  round 5 (background_fabrication): 0.500  95% CI [0.436, 0.571]
  round 6 (background_fabrication): 0.826  95% CI [0.773, 0.871]
  round 7 (context_pollution     ): 0.659  95% CI [0.608, 0.710]
  round 8 (context_pollution     ): 0.631  95% CI [0.568, 0.696]
  round 9 (dataset_final         ): 0.479  95% CI [0.408, 0.543]

===== 4. Interaction: does a hallucinated prefix (halluconv) amplify injection damage? =====
  cleanconv_formatting   clean-prefix  delta=+0.336 [+0.213,+0.447]
  cleanconv_question     clean-prefix  delta=+0.332 [+0.233,+0.433]
  factconv_formatting    other         delta=+0.274 [+0.173,+0.373]
  halluconv_question     HALLU-prefix  delta=+0.399 [+0.293,+0.500]
  halluconv_formatting   HALLU-prefix  delta=+0.423 [+0.320,+0.527]
  irrconv_formatting     other         delta=+0.445 [+0.346,+0.547]

  halluconv prefix mean delta = +0.407 [+0.323,+0.490]
  cleanconv prefix mean delta = +0.336 [+0.250,+0.420]
  INTERACTION (hallu - clean) = +0.070  (CI from separate bootstraps; overlap of [0.323,0.490] and [0.250,0.420] is the conservative read)

=================== 5. Self-recovery inside injected trajectories ===================
  rounds following a hallucination: 5154   recovered next round: 1443 (28.0%)

=================== 6. By injection category (rounds 1-8) ===================
  entity_missing           n=1800  rate=0.951 [0.909,0.978]
  attribute_reversal       n=1800  rate=0.604 [0.558,0.651]
  background_fabrication   n=1800  rate=0.663 [0.616,0.710]
  context_pollution        n=1800  rate=0.645 [0.595,0.698]
