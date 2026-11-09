# -*- coding: utf-8 -*-
"""Generate an HTML preview of the FUSED design for one real sample:
   6 前置语境 x (基线 / 注入) = 12 条独立对话, 每条完整展开。"""
import json, html, base64, io, os, sys

SRC = r"C:\Users\<LOCAL_USER>\Desktop\MMHalSnowball_inject_v3\constructed_50.jsonl"
OUT = r"C:\Users\<LOCAL_USER>\Desktop\MMHalSnowball_inject_v3\fused_design_preview.html"

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
s = rows[0]

SETTINGS = ["cleanconv_formatting", "cleanconv_question",
            "factconv_formatting", "halluconv_question",
            "halluconv_formatting", "irrconv_formatting"]

# image: pull from parquet? we don't have parquet locally -> try cached b64 from earlier preview json
img_tag = ""
try:
    alt = r"C:\Users\<LOCAL_USER>\Desktop\MMHalSnowball_inject_v3\constructed.json"
    old = json.load(open(alt, encoding="utf-8"))
    for o in old:
        if o.get("sample_id") == s["sample_id"] and o.get("image_b64"):
            img_tag = f"<img class='img' src='data:image/jpeg;base64,{o['image_b64']}'/>"
            break
except Exception as e:
    img_tag = f"<div class='noimg'>(图片未内联: {e})</div>"

def esc(x):
    return html.escape(str(x)) if x is not None else ""

final_gt = s.get("truth_token", "")

def bubbles_for(setting, cond):
    """Return list of (kind, text, gt) for one conversation.
       kind: 'prefix-user' | 'prefix-agent' | 'inject' | 'final'"""
    bs = s["original_conversations"][setting]
    out = []
    for role, ctype, content in bs[:-1]:
        if ctype == "image":
            out.append(("image", "[图片]", None))
        elif role == "user":
            out.append(("prefix-user", content, None))
        else:
            out.append(("prefix-agent", content, None))
    if cond == "injection":
        for it in s["injections"]:
            out.append(("inject", it["current_question"], it["ground_truth"]))
    out.append(("final", bs[-1][2], final_gt))
    return out

LABEL = {
    "image": ("图片", "#888"),
    "prefix-user": ("前置·问", "#9aa5b1"),
    "prefix-agent": ("前置·答(数据集自带)", "#7f8c99"),
    "inject": ("注入轮", "#ff9b6a"),
    "final": ("核心题", "#8fe38f"),
}

parts = ["<!doctype html><html lang='zh'><head><meta charset='utf-8'><style>"
"body{background:#0f1117;color:#e6e6e6;font-family:system-ui,'Segoe UI',Arial,sans-serif;margin:0;padding:22px;}"
"h1{font-size:19px;margin:0 0 6px;} h2{font-size:15px;color:#9ad;margin:26px 0 8px;}"
".img{max-width:340px;border-radius:8px;margin:10px 0;border:1px solid #333;}"
".legend{background:#14171f;border:1px solid #2a2f3a;border-radius:8px;padding:12px;font-size:12.5px;color:#a8b0bd;line-height:1.7;margin:12px 0;}"
".meta{font-size:12.5px;color:#8b93a1;margin:4px 0;}"
".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px;}"
".card{background:#171a23;border:1px solid #2a2f3a;border-radius:10px;padding:12px 14px;}"
".ctitle{font-size:13px;font-weight:600;color:#ffd27a;margin-bottom:2px;}"
".csub{font-size:11.5px;color:#7f8c99;margin-bottom:8px;}"
".bub{margin:5px 0;font-size:12.5px;line-height:1.5;}"
".tag{display:inline-block;min-width:112px;font-size:10.5px;padding:1px 6px;border-radius:4px;"
"background:#232833;color:#9aa5b1;margin-right:6px;vertical-align:top;}"
".txt{display:inline-block;width:calc(100% - 130px);vertical-align:top;}"
".gt{color:#8fe38f;font-size:11.5px;margin:2px 0 0 122px;}"
".hint{font-size:11px;color:#6b7583;margin:0 0 10px;}"
"</style></head><body>"]

parts.append("<h1>融合设计 · 单样本全展开（6 前置语境 × 基线/注入 = 12 条独立对话）</h1>")
parts.append(f"<div class='meta'>样本 <b>{esc(s['sample_id'])}</b> · 类型 {esc(s['hallucination_type'])} · "
             f"真相 <b style='color:#8fe38f'>{esc(s['truth_token'])}</b> · 幻觉词 <b style='color:#ff9b6a'>{esc(s['false_token'])}</b></div>")
parts.append(f"<div class='meta'>fact: {esc(s['fact'])}</div>")
parts.append(img_tag)
parts.append("<div class='legend'>"
"<b>怎么读：</b>每张卡片 = <b>一条独立对话</b>（都从这张图重新开始，彼此不通气）。<br>"
"· <span style='color:#9aa5b1'>前置·问 / 前置·答</span> = 原数据集自带的开头几轮，答案由数据集给定，Qwen 不生成；<br>"
"· <span style='color:#ff9b6a'>注入轮</span> = 我们造的 8 轮渐进逼问（4 类 × 2 轮）；<br>"
"· <span style='color:#8fe38f'>核心题</span> = 原数据集的灵魂问题，GT 用数据集自带 answer。<br>"
"<b>基线组</b>只有「前置 + 核心题」；<b>注入组</b>在中间插入 8 轮逼问。两者对比 = 主效应B；"
"6 种前置横向对比 = 主效应A；两相乘 = 交互效应（幻觉前置是否放大注入诱导）。</div>")

for cond, title, hint in (("baseline", "基线组 · 6 条对话（只有前置 + 核心题，不注入）",
                           "模型每条只回答 1 次 → 小计 6 次"),
                          ("injection", "注入组 · 6 条对话（前置 + 8 轮注入 + 核心题）",
                           "模型每条回答 9 次 → 小计 54 次")):
    parts.append(f"<h2>{title}</h2><div class='hint'>{hint}</div><div class='grid'>")
    for setting in SETTINGS:
        bs = bubbles_for(setting, cond)
        n_qa = sum(1 for k, _, _ in bs if k in ("prefix-user", "inject", "final"))
        parts.append("<div class='card'>")
        parts.append(f"<div class='ctitle'>{esc(setting)}</div>")
        parts.append(f"<div class='csub'>条件={cond} · 共 {n_qa} 轮问答 · Qwen 生成 "
                     f"{sum(1 for k,_,_ in bs if k in ('inject','final'))} 次</div>")
        for kind, text, gt in bs:
            lab, col = LABEL[kind]
            parts.append("<div class='bub'>")
            parts.append(f"<span class='tag' style='color:{col}'>{lab}</span>")
            parts.append(f"<span class='txt'>{esc(text)}</span>")
            parts.append("</div>")
            if gt:
                parts.append(f"<div class='gt'>GT → {esc(gt)}</div>")
        parts.append("</div>")
    parts.append("</div>")

parts.append("<div class='legend' style='margin-top:20px'><b>总计：</b>基线 6×1 + 注入 6×9 = <b>60 次</b>模型回答/轮；"
             "× K=3 重复 = <b>180 次/图</b>；× 50 图 = <b>9000 次</b>（双卡约 3 小时）。</div>")
parts.append("</body></html>")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(parts))
print("WROTE", OUT, os.path.getsize(OUT), "bytes")
print("sample:", s["sample_id"], "| settings:", len(s["original_conversations"]), "| injections:", len(s["injections"]))
