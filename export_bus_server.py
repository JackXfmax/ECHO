#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export bus-related sample images from MMHalSnowball parquet to JPEGs (server side).

Usage (on server):
  python export_bus_server.py '<json list of sample_id>'
Images are saved to /home/<SERVER_USER>/mmhal_inject_v3/bus_images/<sample_id>.jpg
Original resolution, no resize (user wants the raw image).
"""
import json, os, sys

sys.path.insert(0, "/home/<SERVER_USER>/mmhal_inject_v3")
import server_common as sc

OUT = "/home/<SERVER_USER>/mmhal_inject_v3/bus_images"
os.makedirs(OUT, exist_ok=True)

ids = json.loads(sys.argv[1])
loader = sc.ImageLoader("/home/<SERVER_USER>/MMHalSnowball")
ok = 0
for sid in ids:
    try:
        img = loader[sid]
    except Exception as e:
        print("ERR", sid, str(e)[:120])
        continue
    if img is None:
        print("MISS", sid)
        continue
    img = img.convert("RGB")
    p = os.path.join(OUT, f"{sid}.jpg")
    img.save(p, format="JPEG", quality=92)
    print("OK", sid, img.size, os.path.getsize(p), "B")
    ok += 1
print(f"exported {ok}/{len(ids)} -> {OUT}")
