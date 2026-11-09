# 参考文献核查报告

**核查日期**：2026-11-07　|　**条目总数**：54
**方法**：逐条按标题查 OpenAlex + Crossref；13 个 arXiv 号用 arXiv API 精确回查；对 API 未命中／弱命中的条目再用网页检索逐条人工确认。
**结论**：**3 条必须删除（虚构/重复）、13 条元数据有误、38 条完全正确**。

---

## A. 🔴 必须删除 / 无法索引（3 条）

| key | 问题 | 证据 |
|---|---|---|
| `retmask2026` | **同一篇论文被拆成两条 + 标题是杜撰的**。`Optimizing Retrieval Heads for Long-Context LMs` 找不到独立文献。 | 真实文献只有一篇：Youmi Ma & Naoaki Okazaki, *From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context **Language Models***, **Findings of ACL 2026**, pp.27731–27747, arXiv:2601.11020。即 `optimizing2026retrieval` 那一条。两条应合并为一。 |
| `survey2024theory` | **与 `liu2024survey` 完全重复**，且作者写成 `Anonymous`、无 arXiv 号。 | `liu2024survey` 已正确给出 Liu Hanchao et al. + arXiv:2402.00253，标题一字不差。删此条，统一引用 `liu2024survey`。 |
| `cot2025reasoning` | **无法定位到任何独立文献**（泛化标题 + `Anonymous` + `arXiv preprint` 无编号，2025）。 | 多轮精确检索只命中主题综述（如 ACL 2026 *Revealing the Seen, Imagining the Beyond*、arXiv:2511.12861 等），没有题为 *Chain-of-Thought Reasoning in Multimodal LLMs* 的 2025 预印本。建议删除或换成真实存在的具体文献。 |

> 这 3 条都是 `author={Anonymous}` 的占位条目，投稿前务必清掉——审稿人一查就会命中。

---

## B. 🟠 文献真实存在，但元数据写错（13 条）

### B1. arXiv 号是占位符（1 条）

| key | 你写的 | 实际 |
|---|---|---|
| `neuhaus2025repope` | `arXiv preprint arXiv:2505.xxxxx` | **arXiv:2504.15707**（Neuhaus & Hein, 2025）。文献真实存在，`xxxxx` 必须补。 |

### B2. 作者被写成 `Anonymous`（8 条，真实作者如下）

| key | 真实作者 | 并需同步修正 |
|---|---|---|
| `unveiling2024induction` | Siyu Chen, Heejune Sheen, Tianhao Wang, Zhuoran Yang | 标题补全为 `Unveiling Induction Heads: Provable Training Dynamics **and Feature Learning in Transformers**`（NeurIPS 2024） |
| `kvshift2025` | Mingyu Xu, Bingning Wang, Weipeng Chen | ICML 2025 正确，仅补作者 |
| `causallava2024` | Xinmiao Hu, Chun Wang, Ruihe An, ChenYu Shao, Xiaojun Ye, Sheng Zhou, Liangcheng Li | **年份 2024 → 2025**，arXiv:2505.19474；标题补全 `...Hallucination **in Multimodal Large Language Models**` |
| `modalityprior2025` | Guanyu Zhou, Yibo Yan, Xin Zou, Kun Wang, Aiwei Liu, Xuming Hu | 真实为 **ICLR 2025**（arXiv:2410.04780），标题补全 `...Hallucinations **in Multimodal Large Language Models** via...` |
| `vlncausal2024` | Liuyi Wang, Zongtao He, Ronghao Dang, Mengjiao Shen, Chengju Liu, Qijun Chen | CVPR 2024 正确，补 arXiv:2404.10241 |
| `mixture2025` | Yifei Gao, Ning Xu, Hongshuo Tian, Bolun Zheng, Jinbo Cao, An-An Liu | **期刊写错**：不是 `Neurocomputing`，是 **Expert Systems with Applications, 280:127422 (2025)**；标题补全 `Mixture of Causal Experts: A Causal Perspective **to Build Dual-level Mixture-of-experts Models**` |
| `cimbmvqa2026` | （PubMed 41172593） | 标题补全 `CIMB-MVQA: Causal Intervention on Modality-specific Biases **for Medical Visual Question Answering**`；期刊 `Medical Image Analysis, 107(Part B):103850, 2026` 正确 |
| `optimizing2026retrieval` | **Youmi Ma, Naoaki Okazaki** | venue 应为 **Findings of ACL 2026**（非 ACL 主会）；标题补全 `...for Long-Context **Language Models**`；arXiv:2601.11020 |

### B3. 其他元数据问题（4 条）

| key | 问题 | 修正 |
|---|---|---|
| `yu2025rlaifv` | 标题用了 arXiv 版，但 venue 标 CVPR 2025。CVPR 版正式题名是 *RLAIF-V: **Open-Source AI Feedback Leads to** Super GPT-4V Trustworthiness* | 按所引版本统一；arXiv 版为 arXiv:2405.17220（2024） |
| `zhang2026evaluating` | 标题与 CVF 官方 PDF 文件名不一致（CVF 为 `Exposing and Evaluating Hallucinations for GUI Grounding`） | 核对最终版题名后再定；CVPR 2026 论文确实存在 |
| `liu2024llavanext` | `journal` 直接填了博客 URL，用 `@article` 类型不规范 | 改 `@misc` + `howpublished`/`url`，或改引正式论文 |
| `pearl2009causality` | 年份争议：*Causality* 初版 **2000**，2009 为再版 | 若按通行引用保留 2009 亦可，但建议注明版本或改 2000 |

> ⚠️ 另需注意：`causallava2024` 的 key 含 2024 而真实年份是 2025，`modalityprior2025` 的真实预印本是 2024 年——key 与年份不一致会让正文 `\cite{}` 产生误导，建议同步改名。

---

## C. ✅ 完全正确（38 条）

13 个 arXiv 号经 arXiv API 回查 **标题相似度全部 1.000**（一一精确对应）：

```
2407.00569 zhong2024investigating      2404.18930 bai2024hallucination
2402.00253 liu2024survey               2311.07397 wang2023amber
2309.14525 sun2023aligning             2310.16045 yin2023woodpecker
2311.16839 zhao2023beyond              2209.11895 olsson2022context
2404.15574 wu2024retrieval             2502.13923 bai2025qwen25vl
2308.12966 bai2023qwen                2304.14178 ye2023mplugowl
2306.14824 peng2023kosmos2
```

其余 25 篇会议/期刊论文经 OpenAlex（多数 sim=1.000）或 Crossref 精确命中，年份与 venue 相符，包括：
`li2023evaluating` (EMNLP 2023)、`guan2024hallusionbench` (CVPR 2024)、`rohrbach2018object` (EMNLP 2018)、
`wang2024haloquest` (ECCV 2024)、`augustin2025dash` (ICCV 2025, arXiv:2503.23573)、`wang2024crpe` (ECCV 2024)、
`xiao2026finer` (CVPR 2026, Oral)、`leng2024mitigating` (CVPR 2024)、`huang2024opera` (CVPR 2024)、
`chen2024halc` (ICML 2024)、`chuang2024dola` (ICLR 2024)、`yang2025nullu` (CVPR 2025)、
`yu2024rlhfv` (CVPR 2024)、`rafailov2023dpo` (NeurIPS 2023)、`lin2026hulluedit` (CVPR 2026)、
`dastmalchi2026cipher` (CVPR 2026, arXiv:2603.10470)、`vaswani2017attention`、`li2023inference`、
`liu2023llava`、`chen2024internvl`、`alayrac2022flamingo`、`li2023blip2`、`zhu2024minigpt4`、
`dai2023instructblip`、`liu2024improved`。

---

## 优先处理顺序

1. **删 3 条**：`retmask2026`、`survey2024theory`、`cot2025reasoning`。
2. **补占位号**：`neuhaus2025repope` → 2504.15707。
3. **8 条 Anonymous 补真实作者**（B2 表），顺带修 `causallava2024` 年份、`mixture2025` 期刊、`modalityprior2025` venue。
4. **统一 `yu2025rlaifv` 题名版本**、核对 `zhang2026evaluating`、把 `liu2024llavanext` 改 `@misc`。

> 注：核查中 OpenAlex/Crossref 未命中不等于文献不存在（如 `bai2024hallucination` 未入库但 arXiv 确认无误）；本报告中所有"不存在"判定均已用网页检索二次确认。
