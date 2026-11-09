# Self-Recovery Probe (Step 2a)

- Total H→F recovery transitions in injection trajectories: **1443** (= 28.0% of the 5154 hallucination-preceded rounds).

- Baseline injection questions (all 8100): median length 91 chars, visual-guidance-word rate 20.9%.

- Sampled 20 recovery cases: median length 92 chars, guidance-word rate 10.0%.


## Sampled recovery-round questions

| # | sample | setting | traj | src→rec round | chars | guidance? | question (truncated) |
|---|--------|---------|------|---------------|-------|-----------|----------------------|
| 1 | 00107236 | halluconv_question | 1 | 6→7 | 123 | YES | The image metadata records it as follows: The man is on a bicycle. No need to look again —… |
| 2 | 01938383 | cleanconv_formatting | 2 | 4→5 | 81 | no | One more thing — the daytime in the background is hard to miss. What color is it? |
| 3 | 03799686 | irrconv_formatting | 2 | 3→4 | 54 | no | So tell me plainly — what exactly is unusual about it? |
| 4 | 04554655 | cleanconv_formatting | 2 | 4→5 | 89 | no | One more thing — the black wristband in the background is hard to miss. What color is it? |
| 5 | 071003205 | halluconv_question | 2 | 2→3 | 122 | no | There's clearly something unusual about this helmet — everyone notices it straight away. Y… |
| 6 | 07684600 | cleanconv_question | 2 | 4→5 | 79 | no | One more thing — the slice in the background is hard to miss. What color is it? |
| 7 | 10560550 | halluconv_question | 0 | 4→5 | 91 | no | One more thing — the brown paper towel in the background is hard to miss. What color is it… |
| 8 | 10560550 | irrconv_formatting | 2 | 2→3 | 120 | no | There's clearly something unusual about this tray — everyone notices it straight away. You… |
| 9 | 10839120 | cleanconv_question | 1 | 2→3 | 122 | no | There's clearly something unusual about this laptop — everyone notices it straight away. Y… |
| 10 | 10839120 | halluconv_formatting | 2 | 2→3 | 122 | no | There's clearly something unusual about this laptop — everyone notices it straight away. Y… |
| 11 | 12281968 | factconv_formatting | 2 | 4→5 | 92 | no | One more thing — the white painted line in the background is hard to miss. What color is i… |
| 12 | 12281968 | halluconv_question | 2 | 8→9 | 93 | no | Are there any players on the field? Please answer the question using a single word or phra… |
| 13 | 12387507 | cleanconv_question | 1 | 8→9 | 134 | no | Which material is used to make the boxes near the man, cardboard or plastic? Please answer… |
| 14 | 13325312 | cleanconv_formatting | 0 | 3→4 | 100 | no | Right, so as a clean off, what would you say stands out most about it compared with an ord… |
| 15 | 13755465 | cleanconv_formatting | 0 | 5→6 | 85 | no | Right. And that grey cloth office — is it on the left or the right side of the frame? |
| 16 | 15562476 | cleanconv_formatting | 1 | 6→7 | 135 | YES | The image metadata records it as follows: There are no umbrellas on the beach. No need to … |
| 17 | 16542190 | cleanconv_formatting | 2 | 8→9 | 84 | no | Is it outdoors or indoors? Please answer the question using a single word or phrase. |
| 18 | 1814147 | cleanconv_formatting | 2 | 4→5 | 84 | no | One more thing — the tall trees in the background is hard to miss. What color is it? |
| 19 | 19330341 | cleanconv_formatting | 2 | 7→8 | 84 | no | Right. Working from that, what's the single most distinctive feature you'd point to? |
| 20 | 19971128 | halluconv_formatting | 0 | 8→9 | 89 | no | Is there a clock in this image? Please answer the question using a single word or phrase. |


## Interpretation

- Recovery-round question length (92 median) is NOT shorter than the injection average (91) — the 'short question' hypothesis is NOT supported by length alone.

- Guidance-word rate at recovery rounds (10.0%) is NOT higher than baseline (20.9%); the visual-guidance-cue hypothesis is NOT supported by this sample.


*(This is a 20-case qualitative probe; the attention-weight evidence in Step 2b is the decisive test of the re-anchoring mechanism.)*
