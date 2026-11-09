"""
server_attn_decompose.py -- EXP3b: where does the answer's attention actually GO?

The pre-registered metric (visual attention ratio over the answer tokens) came out
flat: self-heal and persist are indistinguishable once history is held fixed, under
every stratification scheme. If the picture is not what changes, the conversational
context must be -- the paper is about a SNOWBALL, i.e. the model leaning on its own
contaminated dialogue history instead of the image.

So we decompose the answer tokens' attention mass into disjoint key groups:

    img        image tokens
    hist_asst  the model's OWN previous answers (rounds 1..r-1)   <- snowball carrier
    hist_user  previous user turns (includes the injected premises) + prefix
    cur_q      the current round's question
    self_a     the answer's own earlier tokens (teacher-forced self-attention)
    other      everything else (system, template markers)

and read it at two query positions: the CURRENT QUESTION (before answering) and the
ANSWER (while answering), plus the first 8 answer tokens (the decision point).

Same selection, same model, same eager attention -> directly joinable with
attn_perlayer_hf.jsonl.
"""
import argparse, json, os, time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import server_common as sc

ap = argparse.ArgumentParser()
ap.add_argument("--fused", default="/home/<SERVER_USER>/mmhal_inject_v3/fused_50_out_v3.jsonl")
ap.add_argument("--data", default="/home/<SERVER_USER>/mmhal_inject_v3/constructed_50.jsonl")
ap.add_argument("--sel", default="attn_selection_hf.jsonl")
ap.add_argument("--parquet_dir", default="/home/<SERVER_USER>/MMHalSnowball")
ap.add_argument("--model", default="/home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--out", default="attn_decompose_hf.jsonl")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_pixels", type=int, default=320000)
args = ap.parse_args()

samples = {}
for line in open(args.data, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    samples[r["sample_id"]] = r
fused = sc.load_fused_index(args.fused)
sel = [json.loads(l) for l in open(args.sel, encoding="utf-8") if l.strip()]
if args.limit:
    sel = sel[:args.limit]
print(f"[sel] {len(sel)} passes", flush=True)

processor = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    attn_implementation="eager")
model.eval()
tok = processor.tokenizer
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
print(f"[model] eager, IMG_PAD={IMG_PAD}", flush=True)

attn_mods = [(n, m) for n, m in model.named_modules() if n.endswith("self_attn")]
print(f"[model] {len(attn_mods)} self_attn modules", flush=True)
loader = sc.ImageLoader(args.parquet_dir)

_sink = {}
_state = {}


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


KEYS = ["img", "hist_asst", "hist_user", "cur_q", "self_a"]


def _consume(lidx, att):
    a = att[0]                       # (H, L, L)
    L = a.shape[-1]
    res = {}
    for qname, (qs, qe) in _state["qsets"].items():
        if qe <= qs:
            continue
        sub = a[:, qs:qe, :]         # (H, nq, L)
        tot = sub.sum(-1) + 1e-9
        for kname in KEYS:
            idx = _state["keys"].get(kname)
            if idx is None or idx.numel() == 0:
                res[(qname, kname)] = 0.0
                continue
            m = sub[:, :, idx].sum(-1) / tot
            res[(qname, kname)] = float(m.mean().item())
    _sink[lidx] = res
    return att.new_zeros(1, 1, 1, 1)


for lidx, (name, mod) in enumerate(attn_mods):
    mod.register_forward_hook(make_hook(lidx))


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


SYSTEM = "You are a helpful vision-language assistant."


def segments(sample, setting, traj, r):
    """ordered (label, text) pairs exactly as the chat template renders them"""
    segs = [("sys", SYSTEM)]
    bubbles = sample["original_conversations"][setting][:-1]
    for role, ctype, content in bubbles:
        if ctype == "image":
            continue
        segs.append(("hist_user" if role == "user" else "hist_asst", content))
    for rd in sorted(traj["rounds"], key=lambda x: x["round"]):
        if rd["round"] > r:
            break
        segs.append(("cur_q" if rd["round"] == r else "hist_user", rd["current_question"]))
        if rd["model_output"]:
            segs.append(("cur_a" if rd["round"] == r else "hist_asst", rd["model_output"]))
    return segs


def build_messages(sample, setting, traj, r, pil):
    bubbles = sample["original_conversations"][setting][:-1]
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += build_prefix_messages(bubbles, pil)
    for rd in sorted(traj["rounds"], key=lambda x: x["round"]):
        if rd["round"] > r:
            break
        msgs.append({"role": "user", "content": [{"type": "text", "text": rd["current_question"]}]})
        if rd["model_output"]:
            msgs.append({"role": "assistant", "content": rd["model_output"]})
    return msgs


def find_sub(full, pat, start):
    n, m = len(full), len(pat)
    if m == 0:
        return None
    i = start
    while i + m <= n:
        if full[i:i + m] == pat:
            return i
        i += 1
    return None


@torch.no_grad()
def one_pass(sample, setting, traj, r, pil, mp):
    global _sink, _state
    _sink = {}
    msgs = build_messages(sample, setting, traj, r, pil)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = processor(text=[text], images=[pil], return_tensors="pt", max_pixels=mp)
    ids = enc["input_ids"][0].tolist()

    # ---- locate every segment by sequential subsequence search -------------
    pos = 0
    spans = {k: [] for k in KEYS + ["sys", "cur_a"]}
    for label, txt in segments(sample, setting, traj, r):
        pat = tok(txt, add_special_tokens=False)["input_ids"]
        i = find_sub(ids, pat, pos)
        if i is None:
            continue
        spans.setdefault(label, []).append((i, i + len(pat)))
        pos = i + len(pat)

    def to_idx(pairs):
        out = []
        for a, b in pairs:
            out.extend(range(a, b))
        return out

    dev = model.device
    idt = torch.tensor(ids, device=dev)
    keys = {
        "img": (idt == IMG_PAD).nonzero(as_tuple=True)[0],
        "hist_asst": torch.tensor(to_idx(spans.get("hist_asst", [])), device=dev) if spans.get("hist_asst") else None,
        "hist_user": torch.tensor(to_idx(spans.get("hist_user", [])), device=dev) if spans.get("hist_user") else None,
        "cur_q": torch.tensor(to_idx(spans.get("cur_q", [])), device=dev) if spans.get("cur_q") else None,
        "self_a": torch.tensor(to_idx(spans.get("cur_a", [])), device=dev) if spans.get("cur_a") else None,
    }
    ca = spans.get("cur_a", [])
    cq = spans.get("cur_q", [])
    if not ca or not cq:
        return None
    qsets = {
        "q": (cq[0][0], cq[-1][1]),
        "a": (ca[0][0], ca[-1][1]),
        "a8": (ca[0][0], min(ca[0][0] + 8, ca[-1][1])),
    }
    _state = {"keys": keys, "qsets": qsets}
    inputs = {k: v.to(dev) for k, v in enc.items()}
    model(**inputs, output_attentions=True, return_dict=True)

    out = {}
    for qn in ("q", "a", "a8"):
        for kn in KEYS:
            arr = []
            for lidx in range(len(attn_mods)):
                v = _sink.get(lidx, {}).get((qn, kn))
                arr.append(round(float(v), 6) if v is not None else None)
            out[f"{kn}_{qn}"] = arr
    out["n_image_tokens"] = int(keys["img"].numel())
    out["n_hist_asst"] = int(len(to_idx(spans.get("hist_asst", []))))
    out["n_cur_a"] = int(len(to_idx(ca)))
    out["seq_len"] = int(len(ids))
    out["max_pixels"] = mp
    if any(out[f"img_{qn}"][0] is None for qn in ("q", "a", "a8")):
        return None
    return out


def robust(sample, setting, traj, r, pil):
    for mp in (args.max_pixels, 200000, 128000):
        try:
            o = one_pass(sample, setting, traj, r, pil, mp)
            torch.cuda.empty_cache()
            if o is not None:
                return o
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:
            print("  ERR", repr(e)[:160], flush=True)
            torch.cuda.empty_cache()
            return None
    return None


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
    print(f"[resume] {len(done)} done", flush=True)
sel = [s for s in sel if (s["sample_id"], s["setting"], s["trajectory_id"], s["round"]) not in done]
sel.sort(key=lambda s: (s["sample_id"], s["setting"], s["trajectory_id"], s["round"]))
print(f"[run] {len(sel)} to do", flush=True)

t0 = time.time()
ok = fail = 0
last = None
pil = None
with open(args.out, "a", encoding="utf-8") as fo:
    for i, s in enumerate(sel):
        sid, st, tid, r = s["sample_id"], s["setting"], s["trajectory_id"], s["round"]
        inj = fused.get((sid, st, tid), {}).get("injection")
        sample = samples.get(sid)
        if inj is None or sample is None:
            fail += 1
            continue
        if sid != last:
            pil = loader[sid]
            last = sid
        if pil is None:
            fail += 1
            continue
        o = robust(sample, st, inj, r, pil)
        if o is None:
            fail += 1
            print(f"[{i}] FAIL {sid} {st} t{tid} r{r}", flush=True)
            continue
        rec = dict(s)
        rec.update(o)
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        ok += 1
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(sel)}] ok={ok} fail={fail} {el/60:.1f}min "
                  f"eta {el/(i+1)*(len(sel)-i-1)/60:.1f}min", flush=True)
print(f"DONE ok={ok} fail={fail} in {(time.time()-t0)/60:.1f} min", flush=True)
