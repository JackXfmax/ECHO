SET_PREFIX = {"cleanconv_formatting":0,"cleanconv_question":0,
              "factconv_formatting":1,"halluconv_question":1,
              "halluconv_formatting":1,"irrconv_formatting":1}
INJ, FINAL = 8, 1
K = 3

print("=== 单条轨迹（trajectory）：一张图跑一个条件的轮次 ===")
print(f"{'条件':<28}{'前置问答':>8}{'Qwen生成':>9}{'对话总轮次':>10}")
tot_qwen = 0
for cond in ("baseline", "injection"):
    gen = FINAL if cond == "baseline" else INJ + FINAL
    for s, pre in SET_PREFIX.items():
        tot = pre + gen
        tot_qwen += gen
        print(f"{cond+' · '+s:<28}{pre:>8}{gen:>9}{tot:>10}")
print(f"\n12 个条件全跑完：一条轨迹 Qwen 共回答 {tot_qwen} 次")
print(f"最长单次对话 = {max(SET_PREFIX.values())+INJ+FINAL} 轮 | 最短 = {FINAL} 轮")
print(f"\n=== 一张图（K={K} 条轨迹）===")
print(f"Qwen 总生成次数 = {tot_qwen} x {K} = {tot_qwen*K} 次")
print(f"总问答轮次(含数据集自带前置) = {sum(p+ (FINAL if c=='baseline' else INJ+FINAL) for c in ('baseline','injection') for p in SET_PREFIX.values())} x {K} = "
      f"{sum(p+(FINAL if c=='baseline' else INJ+FINAL) for c in ('baseline','injection') for p in SET_PREFIX.values())*K} 轮")
print(f"\n=== 50 张图 ===")
print(f"Qwen 总生成 = {tot_qwen*K*50} 次 | 预计单卡 ~{tot_qwen*K*50*2.4/3600:.1f} 小时 / 双卡 ~{tot_qwen*K*50*2.4/3600/2:.1f} 小时")
