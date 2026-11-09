# 冻结文件清单（自动生成，勿手改）

冻结时间：2026-11-07 18:54:32　|　基线：`frozen_registry.json`　|　校验：`python verify_frozen.py`

## EXP1_main_qwen_9000 — FROZEN_2026-10-26 (verify-only)

| 冻结路径 | bytes | lines | md5 |
|---|---:|---:|---|
| `EXP1_main_qwen_9000/MANIFEST.md` | 5003 | 90 | `7b8affe588fdeffecaa1aa5baadee7e1` |
| `EXP1_main_qwen_9000/AUDIT_REPORT.txt` | 4319 | 77 | `ea750183afdcebc67c7c11fe466578eb` |
| `EXP1_main_qwen_9000/data/fused_50_out_v3.jsonl` | 8098130 | 50 | `ebf5e0ed376eb3aba6386a726675a191` |
| `EXP1_main_qwen_9000/data/judge_input.jsonl` | 9482947 | 9000 | `8eaa8285f6bbbe5cbefb9c58efbb2f34` |
| `EXP1_main_qwen_9000/data/judge_output.jsonl` | 4701163 | 9000 | `e0ddba3903605f586cfaac467bb6eaae` |
| `EXP1_main_qwen_9000/results/result_judge_analysis.md` | 3721 | 54 | `a4d667d4a464d920fff611cbcbad0a8a` |
| `EXP1_main_qwen_9000/results/self_recovery_analysis.md` | 4157 | 43 | `d71a477f1d72b599dedf0c2bbd86cbcf` |
| `EXP1_main_qwen_9000/code/run_qwen_fused_batch.py` | 18699 | 391 | `373e260ee9ed47e6df2097c537b23c39` |
| `EXP1_main_qwen_9000/code/construct_inject_v3_server.py` | 26206 | 464 | `03eb1876427b3d05671396758fc850ed` |
| `EXP1_main_qwen_9000/code/export_judge_input.py` | 2312 | 51 | `447aa527b73ce9fb828485bda3f9a218` |
| `EXP1_main_qwen_9000/code/judge_prompt.py` | 6394 | 132 | `4ea7bb320d0eb7ce36429895e9c1f90a` |
| `EXP1_main_qwen_9000/code/run_glm_judge.py` | 4979 | 118 | `c5f9c486358781520b431ddc6e5b9e09` |
| `EXP1_main_qwen_9000/code/analyze_results_fast.py` | 9477 | 222 | `f73c72c2a15f39a93df4cf778d00be76` |
| `EXP1_main_qwen_9000/code/audit_main_experiment.py` | 6923 | 168 | `188fd30ce3036f9b6522b5363948d208` |

## EXP2_crossmodel — FROZEN_2026-11-07 (copied)

| 冻结路径 | bytes | lines | md5 |
|---|---:|---:|---|
| `EXP2_crossmodel\data\cross_model_judge_input_llava.jsonl` | 457844 | 648 | `a14c55f27e74a0b795e539a2112b685c` |
| `EXP2_crossmodel\data\cross_model_judge_output_llava.jsonl` | 325711 | 648 | `623cec5bc4c87dec8f3c0d14356bb34a` |
| `EXP2_crossmodel\data\cross_model_judge_input_internvl.jsonl` | 403728 | 648 | `00bcb52aef6450d25e2582b1b36e0002` |
| `EXP2_crossmodel\data\cross_model_judge_output_internvl.jsonl` | 322266 | 648 | `037c1e86e5dc7d3017e85dac175363d4` |
| `EXP2_crossmodel\data\cross_model_judge_input_gpt4o.jsonl` | 408630 | 648 | `20f015d0b956e4e6101b935df3a55800` |
| `EXP2_crossmodel\data\cross_model_judge_output_gpt4o.jsonl` | 329960 | 648 | `35e098df4bd6cbadda17f73d16168fd5` |
| `EXP2_crossmodel\data\cross_model_judge_input_gemini25flash.jsonl` | 430962 | 648 | `8a4025edc4a16761bd116ff6995b5bb9` |
| `EXP2_crossmodel\data\cross_model_judge_output_gemini25flash.jsonl` | 334678 | 648 | `eda1bbe10cd9df294a6b82814973be95` |
| `EXP2_crossmodel\data\cross_model_judge_input_claudesonnet46.jsonl` | 589454 | 648 | `890513469ed6474b02545fb8a8520be2` |
| `EXP2_crossmodel\data\cross_model_judge_output_claudesonnet46.jsonl` | 341824 | 648 | `c4d8b0e5c86604a2c246099791037b71` |
| `EXP2_crossmodel\data\cross_model_selection.json` | 5422 | 182 | `461d01353433af88eefefd88c718c389` |
| `EXP2_crossmodel\data\cross_model_selection_attn.json` | 6837 | 362 | `f614713fbff096c1c78484e2c2d54c2c` |
| `EXP2_crossmodel\data\cross_model_selection_ids.json` | 229 | 17 | `b572bf6a2cb83c546734c25fc90aa89b` |
| `EXP2_crossmodel\results\cross_model_analysis_internvl.md` | 1477 | 31 | `5f0c6409b1321d5e48c9e630f8b3abd5` |
| `EXP2_crossmodel\results\figA_five_model_oscillation.png` | 241628 |  | `b256d4da3c59645affdc735f657f626b` |
| `EXP2_crossmodel\results\figB_self_recovery_bar.png` | 104280 |  | `e1b0a88ca5186634dd44ece8a9425440` |
| `EXP2_crossmodel\assets\00107236.jpg` | 55166 |  | `a18e21ee140117977b8f06147323e558` |
| `EXP2_crossmodel\assets\01938383.jpg` | 44623 |  | `9e177cdf11a7b506046f26984522449e` |
| `EXP2_crossmodel\assets\03799686.jpg` | 58460 |  | `b8ddd0876197295e20306c2121affef1` |
| `EXP2_crossmodel\assets\04554655.jpg` | 45778 |  | `506a48c398ad9088e7f51c36e7746ac9` |
| `EXP2_crossmodel\assets\071003205.jpg` | 61923 |  | `cbfaa70dfd5a62e2dcf2ec2c617ca285` |
| `EXP2_crossmodel\assets\07684600.jpg` | 69660 |  | `2b448238f5c0f5dfa5fb089d104ece37` |
| `EXP2_crossmodel\assets\10560550.jpg` | 41949 |  | `0620842f97ed500a2fb178ffe16f43b1` |
| `EXP2_crossmodel\assets\10839120.jpg` | 31232 |  | `ad9889db64d18f9df7d80ebbae098686` |
| `EXP2_crossmodel\assets\12281968.jpg` | 52141 |  | `44f8bca141e28f55a6ad12a568f9d9ad` |
| `EXP2_crossmodel\assets\12387507.jpg` | 49277 |  | `bcba25a46fb1b1dfb66a29bc1bfa949b` |
| `EXP2_crossmodel\assets\13325312.jpg` | 93002 |  | `d7ae564ad6f480caeba02683a3f7bc9c` |
| `EXP2_crossmodel\assets\13755465.jpg` | 169400 |  | `93fad3e569e633a269aad643758d8ce0` |
| `EXP2_crossmodel\code\server_cross_model_probe.py` | 6911 | 151 | `790c22eadfc03734e8436ae6834cb6e8` |
| `EXP2_crossmodel\code\server_cross_model_internvl.py` | 10832 | 238 | `7e1df2bd2f32101813f5e4b55c3b9637` |
| `EXP2_crossmodel\code\run_api_vlm_local.py` | 6686 | 172 | `bbe5ae183888de3272a37f927fe369d1` |
| `EXP2_crossmodel\code\analyze_cross_model.py` | 4902 | 118 | `aaba70784bcf453bee0c10d18444493c` |
| `EXP2_crossmodel\code\analyze_frontier.py` | 2906 | 72 | `3ed1e851c5d4a97005b31341bbb84f2b` |

## EXP3_attention_mechanism — FROZEN_2026-11-07 (copied)

| 冻结路径 | bytes | lines | md5 |
|---|---:|---:|---|
| `EXP3_attention_mechanism\data\attn_perlayer_hf.jsonl` | 21715502 | 2763 | `21e104983e69fd4fa89b44384b687c0d` |
| `EXP3_attention_mechanism\data\attn_perlayer_ff.jsonl` | 5249850 | 668 | `23177365e15c5e9e04e1df140a3be633` |
| `EXP3_attention_mechanism\data\attn_decompose_hf.jsonl` | 12545563 | 2763 | `ff266578568a0c94dc84493510420bd5` |
| `EXP3_attention_mechanism\data\attn_selection_hf.jsonl` | 581334 | 2763 | `fa272866a1ff78df9174384e7c0175c3` |
| `EXP3_attention_mechanism\data\attn_selection_ff.jsonl` | 144533 | 668 | `bd80f794821c07e668cdea9b52697d25` |
| `EXP3_attention_mechanism\data\attn_selection_expanded.json` | 6261 | 362 | `3d4e528a28e549db1bc72e7881babccc` |
| `EXP3_attention_mechanism\data\attn_smoke.jsonl` | 94579 | 12 | `a26ab3820c99cf805d547cdfd8a723e0` |
| `EXP3_attention_mechanism\results\figC_layer_var.png` | 342898 |  | `2e4ed3913ee2dfbf5d4f2367eacef312` |
| `EXP3_attention_mechanism\results\figC_layer_delta.png` | 129255 |  | `68655b5645e5954149747d69d8afd5fe` |
| `EXP3_attention_mechanism\results\figC_layer_delta_lift.png` | 132648 |  | `5d7cb9b8dc633a4d53c463d5d5e43423` |
| `EXP3_attention_mechanism\results\figC_rounds_r3_r7.png` | 232695 |  | `709881c0edbec7c570eaa36f1f467818` |
| `EXP3_attention_mechanism\results\figD_head_signature.png` | 76783 |  | `9393d9ac9dfd38324aa74cd6d90af6c7` |
| `EXP3_attention_mechanism\results\figD_mechanism_decompose.png` | 218237 |  | `37748bf1899b072307d9736f0664275d` |
| `EXP3_attention_mechanism\code\build_attn_selection_hf.py` | 6942 | 183 | `2a5428a3d34a9c06ec32c22208308b52` |
| `EXP3_attention_mechanism\code\build_attn_selection_ff.py` | 3467 | 96 | `d07e640af59af14b2648ee76b0a5e636` |
| `EXP3_attention_mechanism\code\server_attn_perlayer.py` | 11881 | 308 | `24c097d206620483288f6416e513f927` |
| `EXP3_attention_mechanism\code\server_attn_decompose.py` | 11105 | 303 | `948a87603c394bc261a5e556d42bdb3b` |
| `EXP3_attention_mechanism\code\analyze_attn_perlayer.py` | 9584 | 249 | `7d1d01d9ec164560585a62f770a2b2f0` |
| `EXP3_attention_mechanism\code\analyze_attn_controlled.py` | 12717 | 300 | `79554d1d0da8272a7571bc0c1d5f7917` |
| `EXP3_attention_mechanism\code\analyze_decompose.py` | 5477 | 143 | `31f351c8f638fd85eaa5c9e328bdbda1` |
| `EXP3_attention_mechanism\code\make_figC.py` | 6929 | 182 | `04988e4ea86b6779da09e3f1432af6af` |
| `EXP3_attention_mechanism\code\make_figD_heads.py` | 4184 | 104 | `c07e04287aa7c4ec4f3c084abbe18285` |
| `EXP3_attention_mechanism\code\diag_permutation.py` | 3299 | 83 | `75eff908ca744113a6075b7a9033f3db` |
| `EXP3_attention_mechanism\code\diag_splithalf.py` | 2744 | 72 | `bebd8dd0a9b3c53a78872d9d40bd83ec` |
| `EXP3_attention_mechanism\code\diag_residualize.py` | 3350 | 92 | `5dae11b14df71e0a8e7d44d8305ece38` |

## EXP4_mega_qwen_1000 — FROZEN_2026-11-07 (copied)

| 冻结路径 | bytes | lines | md5 |
|---|---:|---:|---|
| `EXP4_mega_qwen_1000\data\fused_1000_hallu_a.jsonl` | 13783878 | 500 | `7236be09343d555e900b6ac42d11809c` |
| `EXP4_mega_qwen_1000\data\fused_1000_hallu_b.jsonl` | 13991721 | 500 | `e07d5185044ed88b5d0b59e011f03995` |
| `EXP4_mega_qwen_1000\data\fused_1000_merged.jsonl` | 27776599 | 1000 | `2285a594212790ae72a2e9f156cac094` |
| `EXP4_mega_qwen_1000\data\judge_input_exp4.jsonl` | 39618023 | 30000 | `ac42d876b12bd082af8cbecb035399d5` |
| `EXP4_mega_qwen_1000\data\judge_output_exp4.jsonl` | 15888651 | 30000 | `8be609b9e4917216389534685586fa84` |
| `EXP4_mega_qwen_1000\data\test_set_1000.json` | 9177239 | 210002 | `6048b2834b3a9af52a47dc2207bb75d3` |
| `EXP4_mega_qwen_1000\data\test_set_1000.jsonl` | 8058236 | 1000 | `f700e75d6b16ce853fa25ed28d18dfc5` |
| `EXP4_mega_qwen_1000\results\fig_mega_1000.pdf` | 26417 |  | `c31cc024651deb173c7c068fa3327647` |
| `EXP4_mega_qwen_1000\results\EXP4_MEGA_ANALYSIS.md` | 2240 | 46 | `d30b467763a0745bf6df66a999da4a5c` |
| `EXP4_mega_qwen_1000\results\EXP4_MEGA_PLAN.md` | 5489 | 100 | `a26b645a8ec6ee1cef7f7a6583e65816` |
| `EXP4_mega_qwen_1000\code\construct_inject_v3_1000.py` | 26310 | 465 | `484f63acf2f7c66d1ae48d0fa9ea0562` |
| `EXP4_mega_qwen_1000\code\build_judge_input_exp4.py` | 4011 | 82 | `e03ef4c177c78f3bc52e65da51052069` |
| `EXP4_mega_qwen_1000\code\analyze_exp4.py` | 10382 | 250 | `0965f1666d053170c35bda263399ce51` |
| `EXP4_mega_qwen_1000\code\run_glm_judge.py` | 4980 | 118 | `5909a0f3db1d94a847caf4e8e3909175` |
| `EXP4_mega_qwen_1000\code\judge_prompt.py` | 6394 | 132 | `4ea7bb320d0eb7ce36429895e9c1f90a` |
| `EXP4_mega_qwen_1000\code\run_qwen_fused_batch.py` | 18699 | 391 | `373e260ee9ed47e6df2097c537b23c39` |
| `EXP4_mega_qwen_1000\code\ssh_server.py` | 3182 | 101 | `03da37479605d3e497160243f76ea887` |

## 共享基础设施（原地校验，不复制）

| 路径 | bytes | lines | md5 |
|---|---:|---:|---|
| `run_qwen_fused_batch.py` | 18699 | 391 | `373e260ee9ed47e6df2097c537b23c39` |
| `construct_inject_v3_server.py` | 26206 | 464 | `03eb1876427b3d05671396758fc850ed` |
| `export_judge_input.py` | 2312 | 51 | `447aa527b73ce9fb828485bda3f9a218` |
| `judge_prompt.py` | 6394 | 132 | `4ea7bb320d0eb7ce36429895e9c1f90a` |
| `run_glm_judge.py` | 4980 | 118 | `5909a0f3db1d94a847caf4e8e3909175` |
| `audit_main_experiment.py` | 6923 | 168 | `188fd30ce3036f9b6522b5363948d208` |
| `freeze_snapshot.py` | 5803 | 146 | `494be86efaf6ad7be018df88816e4a06` |
| `_server_run_qwen_fused_batch.py` | 18308 | 391 | `35642037cbb2126a7dc833ec9b737da2` |
| `EXPERIMENT_LOG.md` | 20540 | 266 | `6130fec647bb058e2bf3eb68afc7eec0` |
| `PAPER_WRITING_KIT.md` | 19749 | 304 | `8170fe4f1f675e59277e597d6bd9f201` |
| `FROZEN_MANIFEST.md` | 9087 | 127 | `d3e7c80965c300a3fbcf2bb3a3759826` |
| `verify_frozen.py` | 5864 | 150 | `c3fd278ec91b380bfdbb48e9b5dedd2d` |
| `freeze_all_experiments.py` | 9236 | 198 | `ccd96550cb0d32780bf9d7452092c40b` |
