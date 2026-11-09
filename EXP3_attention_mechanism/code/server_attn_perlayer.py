"""
server_attn_perlayer.py -- EXP3 mechanism probe (run on the 4090 server, cuda:1)

WHAT IT MEASURES
    For every selected (trajectory, round r) we replay the EXACT generation prompt
    of EXP1 (system + original-conversation prefix + Q1/A1 ... Q_{r-1}/A_{r-1} + Q_r)
    and teacher-force the real answer A_r. From the EAGER attention maps we compute,
    for each of the 28 transformer layers:

        VAR_l(r) = mean over answer query positions, mean over heads, of
                   sum_{k in image tokens} softmax_attn[q, k]

    i.e. the fraction of the answer's attention mass that lands on the image.

WHY THIS DESIGN
    * attention is read at the ANSWER tokens of the CURRENT round only -> the measure
      does not mechanically shrink as the dialogue grows (the old whole-sequence
      average did, which is what made the naive "hallucination vs recovery" contrast
      uninterpretable).
    * attn_implementation="eager" -> real softmax maps (sdpa/flash return none).
    * hooks consume each layer's (H, L, L) map immediately and drop it, so peak
      memory stays at ~1 layer instead of all 28.

THE CAUSAL CONTRAST (built by build_attn_selection_hf.py, joined in analysis)
    self-heal  : H at r-1 -> F at r
    persist    : H at r-1 -> H at r
    both stratified by the EXACT H/F bitmask of rounds 1..r-1, so history is fixed.

RUN
    CUDA_VISIBLE_DEVICES=1 conda run -n train python server_attn_perlayer.py \
        --fused /home/<SERVER_USER>/mmhal_inject_v3/fused_50_out_v3.jsonl \
        --data  /home/<SERVER_USER>/mmhal_inject_v3/constructed_50.jsonl \
        --sel   attn_selection_hf.jsonl \
        --parquet_dir /home/<SERVER_USER>/MMHalSnowball \
        --out   attn_perlayer_hf.jsonl
"""
import argparse, json, os, sys, time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import server_common as sc

ap = argparse.ArgumentParser()
ap.add_argument("--fused", default="/home/<SERVER_USER>/mmhal_inject_v3/fused_50_out_v3.jsonl")
ap.add_argument("--data", default="/home/<SERVER_USER>/mmhal_inject_v3/constructed_50.jsonl")
ap.add_argument("--sel", default="attn_selection_hf.jsonl")
ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
ap.add_argument("--model", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--out", default="attn_perlayer_hf.jsonl")
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_pixels", type=int, default=320000)
args = ap.parse_args()

# ----------------------------------------------------------------------------- data
samples = {}
for line in open(args.data, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    samples[r["sample_id"]] = r
print(f"[data] {len(samples)} samples with original_conversations", flush=True)

fused = sc.load_fused_index(args.fused)
print(f"[fused] {len(fused)} trajectory keys", flush=True)

sel = [json.loads(l) for l in open(args.sel, encoding="utf-8") if l.strip()]
if args.limit:
    sel = sel[:args.limit]
print(f"[sel] {len(sel)} forward passes queued", flush=True)

# ----------------------------------------------------------------------------- model
processor = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    attn_implementation="eager")
model.eval()
tok = processor.tokenizer
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
print(f"[model] eager attention, IMG_PAD id={IMG_PAD}, device={next(model.parameters()).device}",
      flush=True)

# locate the 28 self-attention modules, in layer order
attn_mods = [(n, m) for n, m in model.named_modules() if n.endswith("self_attn")]
print(f"[model] {len(attn_mods)} self_attn modules hooked (expect 28)", flush=True)
if len(attn_mods) == 0:
    sys.exit("no self_attn modules found -- aborting")

loader = sc.ImageLoader(args.parquet_dir)

# --------------------------------------------------------------------------- hooks
_sink = {}          # layer_idx -> list of per-head ratios


def make_hook(lidx):
    def hook(module, inp, out):
        if not isinstance(out, tuple):
            return out
        new = []
        for o in out:
            if isinstance(o, torch.Tensor) and o.dim() == 4:
                new.append(_consume(lidx, o))
            else:
                new.append(o)
        return tuple(new)
    return hook


_qspan = (0, 0)
_imgpos = None


def _consume(lidx, att):
    """att: (B, H, L, L). Returns a tiny placeholder so the big map is freed."""
    global _sink
    try:
        a = att[0]                      # (H, L, L)
        qs, qe = _qspan
        sub = a[:, qs:qe, :]            # (H, nq, L)
        img = sub[:, :, _imgpos]        # (H, nq, nimg)
        ratio = img.sum(-1) / (sub.sum(-1) + 1e-9)     # (H, nq)
        _sink[lidx] = ratio.mean(dim=1)                # (H,) per-head, keep on gpu
        full = a[:, :, _imgpos].sum(-1) / (a.sum(-1) + 1e-9)
        _sink[("full", lidx)] = full.mean().item()
    except Exception as e:
        print("[hook err]", e, flush=True)
    return att.new_zeros(1, 1, 1, 1)


for lidx, (name, mod) in enumerate(attn_mods):
    mod.register_forward_hook(make_hook(lidx))


# ------------------------------------------------------------------- prompt rebuild
def build_prefix_messages(bubbles, pil):
    msgs = []
    for role, ctype, content in bubbles:
        r = "user" if role == "user" else "assistant"
        part = {"type": "image", "image": pil} if ctype == "image" else {"type": "text", "text": content}
        if msgs and msgs[-1]["role"] == r:
            if isinstance(msgs[-1]["content"], list):
                msgs[-1]["content"].append(part)
            else:
                msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]}, part]
        else:
            msgs.append({"role": r, "content": [part]})
    return msgs


def build_messages(sample, setting, traj, upto, pil, with_answer):
    bubbles = sample["original_conversations"][setting]
    msgs = [{"role": "system", "content": "You are a helpful vision-language assistant."}]
    msgs += build_prefix_messages(bubbles[:-1], pil)
    for rd in sorted(traj["rounds"], key=lambda x: x["round"]):
        if rd["round"] > upto:
            break
        msgs.append({"role": "user", "content": [{"type": "text", "text": rd["current_question"]}]})
        if rd["round"] < upto or (with_answer and rd["round"] == upto):
            msgs.append({"role": "assistant", "content": rd["model_output"]})
    return msgs


def find_span(full_ids, ans_ids, fallback_start):
    """locate the answer token span inside full_ids"""
    n, m = len(full_ids), len(ans_ids)
    for i in range(n - m, max(fallback_start - 5, -1), -1):
        if i >= 0 and full_ids[i:i + m] == ans_ids:
            return i, i + m
    return fallback_start, n


# ------------------------------------------------------------------------- forward
@torch.no_grad()
def one_pass(sample, setting, traj, r, pil, mp):
    global _qspan, _imgpos, _sink
    _sink = {}
    m_noans = build_messages(sample, setting, traj, r, pil, with_answer=False)
    m_ans = build_messages(sample, setting, traj, r, pil, with_answer=True)
    t_prompt = processor.apply_chat_template(m_noans, tokenize=False, add_generation_prompt=True)
    t_full = processor.apply_chat_template(m_ans, tokenize=False, add_generation_prompt=False)
    ans_text = traj_by_round(traj, r)
    ans_ids = tok(ans_text, add_special_tokens=False)["input_ids"]
    if not ans_ids:
        return None

    enc_p = processor(text=[t_prompt], images=[pil], return_tensors="pt", max_pixels=mp)
    enc_f = processor(text=[t_full], images=[pil], return_tensors="pt", max_pixels=mp)
    p_ids = enc_p["input_ids"][0].tolist()
    f_ids = enc_f["input_ids"][0].tolist()
    if len(f_ids) <= len(p_ids) + 2:
        return None
    qs, qe = find_span(f_ids, ans_ids, len(p_ids))
    if qe <= qs:
        return None

    inputs = {k: v.to(model.device) for k, v in enc_f.items()}
    f_ids_t = inputs["input_ids"][0]
    imgpos = (f_ids_t == IMG_PAD).nonzero(as_tuple=True)[0]
    if imgpos.numel() == 0:
        return None
    _imgpos = imgpos
    _qspan = (qs, qe)
    model(**inputs, output_attentions=True, return_dict=True)

    layers = []
    for lidx in range(len(attn_mods)):
        v = _sink.get(lidx)
        if v is None:
            layers.append(None)
        else:
            layers.append(round(float(v.mean().item()), 6))
    heads = []
    for lidx in range(len(attn_mods)):
        v = _sink.get(lidx)
        heads.append([round(float(x), 5) for x in v.tolist()] if v is not None else None)
    fulls = [_sink.get(("full", lidx)) for lidx in range(len(attn_mods))]
    return {
        "layers": layers,
        "heads": heads,
        "full_layers": [round(float(x), 6) if x is not None else None for x in fulls],
        "n_image_tokens": int(imgpos.numel()),
        "n_answer_tokens": int(qe - qs),
        "seq_len": int(len(f_ids)),
        "max_pixels": mp,
    }


def traj_by_round(traj, r):
    for rd in traj["rounds"]:
        if rd["round"] == r:
            return rd["model_output"] or ""
    return ""


def robust(sample, setting, traj, r, pil):
    for mp in (args.max_pixels, 200000, 128000):
        try:
            out = one_pass(sample, setting, traj, r, pil, mp)
            torch.cuda.empty_cache()
            if out is not None:
                return out
        except torch.cuda.OutOfMemoryError:
            print("  OOM -> shrink", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            print("  ERR", repr(e)[:200], flush=True)
            torch.cuda.empty_cache()
            return None
    return None


# ---------------------------------------------------------------------------- main
done = set()
if os.path.exists(args.out):
    for line in open(args.out, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            done.add((r["sample_id"], r["setting"], r["trajectory_id"], r["round"]))
        except Exception:
            pass
    print(f"[resume] {len(done)} already done", flush=True)

sel = [s for s in sel if (s["sample_id"], s["setting"], s["trajectory_id"], s["round"]) not in done]
sel.sort(key=lambda s: (s["sample_id"], s["setting"], s["trajectory_id"], s["round"]))
print(f"[run] {len(sel)} passes to do", flush=True)

t0 = time.time()
n_ok = n_fail = 0
last_sid = None
pil = None
with open(args.out, "a", encoding="utf-8") as fo:
    for i, s in enumerate(sel):
        sid, st, tid, r = s["sample_id"], s["setting"], s["trajectory_id"], s["round"]
        key = (sid, st, tid)
        inj = fused.get(key, {}).get("injection")
        if inj is None:
            n_fail += 1
            continue
        sample = samples.get(sid)
        if sample is None:
            n_fail += 1
            continue
        if sid != last_sid:
            pil = loader[sid]
            last_sid = sid
        if pil is None:
            n_fail += 1
            continue
        out = robust(sample, st, inj, r, pil)
        if out is None:
            n_fail += 1
            print(f"[{i}] FAIL {sid} {st} t{tid} r{r}", flush=True)
            continue
        rec = dict(s)
        rec.update(out)
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        n_ok += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(sel) - i - 1) / 60
            print(f"[{i+1}/{len(sel)}] ok={n_ok} fail={n_fail} "
                  f"{el/60:.1f}min elapsed eta {eta:.1f}min", flush=True)

print(f"DONE ok={n_ok} fail={n_fail} in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)
