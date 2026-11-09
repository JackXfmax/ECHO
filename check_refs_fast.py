#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast reference check: arXiv bulk id_list + OpenAlex + Crossref title lookup.

For EVERY entry we query at least one bibliographic API by exact title, so each
citation is actually searched (not just pattern-matched).
"""
import json, re, time, difflib
import requests

E = json.load(open("refs_parsed.json", encoding="utf-8"))
S = requests.Session()
S.headers.update({"User-Agent": "ref-verify/1.0 (mailto:research@example.org)"})


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


# ---------- 1) arXiv: one request for ALL ids ----------
ids = []
for e in E:
    m = re.search(r"arXiv[:/]\s*(\d{4}\.\d{4,5})", e["venue"])
    if m and "xxxxx" not in e["venue"]:
        ids.append(m.group(1))
print("arXiv ids to verify:", len(ids), flush=True)
arxiv_map = {}
u = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=200"
try:
    r = S.get(u, timeout=60)
    print("arxiv api:", r.status_code, len(r.content), flush=True)
    for block in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
        im = re.search(r"<id>http[s]?://arxiv.org/abs/([^<]+)", block)
        tm = re.search(r"<title>(.*?)</title>", block, re.S)
        pm = re.search(r"<published>(\d{4})", block)
        if im and tm:
            aid = im.group(1).strip()
            title = re.sub(r"\s+", " ", tm.group(1)).strip()
            arxiv_map[aid] = (title, pm.group(1) if pm else "")
except Exception as ex:
    print("arxiv EXC", ex, flush=True)

for e in E:
    m = re.search(r"arXiv[:/]\s*(\d{4}\.\d{4,5})", e["venue"])
    e["arxiv_id"] = m.group(1) if m and "xxxxx" not in e["venue"] else ""
    e["arxiv_check"] = "n/a"
    if "xxxxx" in e["venue"]:
        e["arxiv_check"] = "PLACEHOLDER_ID"
        continue
    if not e["arxiv_id"]:
        continue
    hit = arxiv_map.get(e["arxiv_id"])
    if not hit:
        for k, v in arxiv_map.items():
            if k.startswith(e["arxiv_id"]) or e["arxiv_id"] in k:
                hit = v
                break
    if not hit:
        e["arxiv_check"] = "ID_NOT_FOUND"
    else:
        s = sim(e["title"], hit[0])
        e["arxiv_real_title"] = hit[0]
        e["arxiv_real_year"] = hit[1]
        e["arxiv_sim"] = round(s, 3)
        e["arxiv_check"] = "MATCH" if s > 0.85 else "TITLE_MISMATCH"


# ---------- 2) OpenAlex + Crossref by exact title ----------
def openalex(title):
    try:
        r = S.get("https://api.openalex.org/works",
                  params={"filter": "title.search:" + title[:150], "per-page": 5,
                          "select": "title,publication_year,primary_location,ids,authorships,type"},
                  timeout=30)
        if r.status_code != 200:
            return []
        return r.json().get("results", [])
    except Exception:
        return []


def crossref(title):
    try:
        r = S.get("https://api.crossref.org/works",
                  params={"query.bibliographic": title[:150], "rows": 3,
                          "select": "title,published,container-title,DOI,type"},
                  timeout=30)
        if r.status_code != 200:
            return []
        return r.json()["message"]["items"]
    except Exception:
        return []


for i, e in enumerate(E, 1):
    oa = openalex(e["title"])
    best, bs = None, 0.0
    for w in oa:
        s = sim(e["title"], w.get("title", ""))
        if s > bs:
            best, bs = w, s
    e["oa_sim"] = round(bs, 3)
    if best:
        e["oa_title"] = best.get("title")
        e["oa_year"] = best.get("publication_year")
        loc = best.get("primary_location") or {}
        src = loc.get("source") or {}
        e["oa_venue"] = src.get("display_name") or ""
        au = best.get("authorships") or []
        e["oa_first_author"] = (au[0].get("author", {}) or {}).get("display_name", "") if au else ""

    cr, cbest, cbs = crossref(e["title"]), None, 0.0
    for w in cr:
        t = (w.get("title") or [""])[0]
        s = sim(e["title"], t)
        if s > cbs:
            cbest, cbs = w, s
    e["cr_sim"] = round(cbs, 3)
    if cbest:
        e["cr_title"] = (cbest.get("title") or [""])[0]
        ct = (cbest.get("container-title") or [""])
        e["cr_venue"] = ct[0] if ct else ""
        pub = cbest.get("published", {}) or {}
        dp = (pub.get("date-parts") or [[None]])[0]
        e["cr_year"] = dp[0] if dp else ""

    top = max(bs, cbs)
    e["status"] = "FOUND" if top > 0.86 else ("WEAK" if top > 0.62 else "NOT_FOUND")
    print(f"[{i:2d}/54] {e['key']:<28} arxiv={e['arxiv_check']:<16} "
          f"api={e['status']:<10} oa={e['oa_sim']} cr={e['cr_sim']} "
          f"yr={e.get('oa_year') or e.get('cr_year') or ''} "
          f"venue={str(e.get('oa_venue') or e.get('cr_venue') or '')[:20]}", flush=True)
    time.sleep(0.25)

json.dump(E, open("refs_checked.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nDONE -> refs_checked.json", flush=True)
