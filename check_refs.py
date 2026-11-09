#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check every bibliography entry against arXiv API and Semantic Scholar."""
import json, re, time, difflib, sys
import requests

ENTRIES = json.load(open("refs_parsed.json", encoding="utf-8"))
UA = {"User-Agent": "Mozilla/5.0 (ref-verification; research use)"}
S = requests.Session()
S.headers.update(UA)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def arxiv_by_id(aid):
    """Return (found, real_title, real_year) for an arXiv id."""
    url = f"http://export.arxiv.org/api/query?id_list={aid}&max_results=1"
    try:
        r = S.get(url, timeout=25)
        if r.status_code != 200:
            return None
        m = re.search(r"<entry>.*?<title>(.*?)</title>.*?<published>(.*?)</published>", r.text, re.S)
        if not m:
            return None
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        year = m.group(2)[:4]
        return (title, year)
    except Exception:
        return None


def s2_search(title, retries=3):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 5,
              "fields": "title,year,venue,externalIds,authors,publicationVenue"}
    for k in range(retries):
        try:
            r = S.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json().get("data", [])
            if r.status_code == 429:
                time.sleep(4 + 3 * k)
                continue
            return []
        except Exception:
            time.sleep(2)
    return []


def main():
    out = []
    for i, e in enumerate(ENTRIES, 1):
        row = dict(e)
        row["arxiv_id"] = ""
        m = re.search(r"arXiv[:/]\s*(\d{4}\.\d{4,5})", e["venue"])
        if m:
            row["arxiv_id"] = m.group(1)

        # 1) arXiv id verification
        row["arxiv_check"] = "n/a"
        if row["arxiv_id"] and "xxxxx" not in e["venue"]:
            a = arxiv_by_id(row["arxiv_id"])
            if a is None:
                row["arxiv_check"] = "ID_NOT_FOUND"
            else:
                s = sim(e["title"], a[0])
                row["arxiv_real_title"] = a[0]
                row["arxiv_real_year"] = a[1]
                row["arxiv_sim"] = round(s, 3)
                row["arxiv_check"] = "TITLE_MATCH" if s > 0.85 else "TITLE_MISMATCH"
        elif "xxxxx" in e["venue"]:
            row["arxiv_check"] = "PLACEHOLDER_ID"
        time.sleep(0.4)

        # 2) Semantic Scholar title search
        hits = s2_search(e["title"][:180])
        best, best_s = None, 0.0
        for h in hits:
            s = sim(e["title"], h.get("title", ""))
            if s > best_s:
                best, best_s = h, s
        row["s2_sim"] = round(best_s, 3)
        if best:
            row["s2_title"] = best.get("title")
            row["s2_year"] = best.get("year")
            row["s2_venue"] = best.get("venue") or ""
            ext = best.get("externalIds") or {}
            row["s2_arxiv"] = ext.get("ArXiv", "")
            authors = best.get("authors") or []
            row["s2_first_author"] = authors[0].get("name", "") if authors else ""
        row["s2_status"] = ("FOUND" if best_s > 0.86 else
                            "WEAK" if best_s > 0.65 else "NOT_FOUND")
        out.append(row)
        print(f"[{i:2d}/54] {e['key']:<28} arxiv={row['arxiv_check']:<16} "
              f"s2={row['s2_status']:<10} sim={row['s2_sim']} "
              f"year={row.get('s2_year','')} venue={str(row.get('s2_venue',''))[:22]}", flush=True)
        time.sleep(1.1)

    json.dump(out, open("refs_checked.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nDONE -> refs_checked.json")


if __name__ == "__main__":
    main()
