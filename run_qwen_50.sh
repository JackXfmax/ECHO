#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate train
cd /home/<SERVER_USER>
python -u /home/<SERVER_USER>/mmhal_inject_v3/run_qwen_test.py \
  --model_path /home/<SERVER_USER>/models/Qwen2.5-VL-7B-Instruct \
  --data /home/<SERVER_USER>/mmhal_inject_v3/constructed_50.jsonl \
  --out /home/<SERVER_USER>/mmhal_inject_v3/qwen_50_out.jsonl \
  --parquet_dir /home/<SERVER_USER>/MMHalSnowball \
  --device cuda:0 --limit 50
