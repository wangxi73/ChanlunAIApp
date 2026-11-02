import json
from pathlib import Path

base = Path(__file__).resolve().parent
with open(base / "backtest_results_5y_100w.json", "r", encoding="utf-8") as f:
    data = json.load(f)

p1 = data["proportional_25w"]["quant_hs300"]
p2 = data["proportional_25w"]["quant_all_a"]
d1 = data["diversified_2_5w"]["quant_hs300"]
d2 = data["diversified_2_5w"]["quant_all_a"]

h_p1 = {x["date"]: x for x in p1["history"]}
h_p2 = {x["date"]: x for x in p2["history"]}
h_d1 = {x["date"]: x for x in d1["history"]}
h_d2 = {x["date"]: x for x in d2["history"]}

dates = [x["date"] for x in p1["history"]]
checkpoints = []
for target in ["2021-10-29", "2022-10-31", "2023-10-31", "2024-10-31", "2025-10-31", "2026-10-29"]:
    cand = [d for d in dates if d <= target][-1]
    checkpoints.append(cand)

print("Checkpoints:", checkpoints)
for i in range(1, len(checkpoints)):
    d_prev, d_cur = checkpoints[i-1], checkpoints[i]
    b_ret = (h_p1[d_cur]["hs300_level"] - h_p1[d_prev]["hs300_level"]) / h_p1[d_prev]["hs300_level"] * 100.0
    p1_ret = (h_p1[d_cur]["portfolio_value"] - h_p1[d_prev]["portfolio_value"]) / h_p1[d_prev]["portfolio_value"] * 100.0
    p2_ret = (h_p2[d_cur]["portfolio_value"] - h_p2[d_prev]["portfolio_value"]) / h_p2[d_prev]["portfolio_value"] * 100.0
    d1_ret = (h_d1[d_cur]["portfolio_value"] - h_d1[d_prev]["portfolio_value"]) / h_d1[d_prev]["portfolio_value"] * 100.0
    d2_ret = (h_d2[d_cur]["portfolio_value"] - h_d2[d_prev]["portfolio_value"]) / h_d2[d_prev]["portfolio_value"] * 100.0
    print(f"{d_prev} -> {d_cur} | HS300:{b_ret:+.2f}% | P_HS300:¥{h_p1[d_cur]['portfolio_value']:,.0f}({p1_ret:+.2f}%) | P_AllA:¥{h_p2[d_cur]['portfolio_value']:,.0f}({p2_ret:+.2f}%) | D_HS300:¥{h_d1[d_cur]['portfolio_value']:,.0f}({d1_ret:+.2f}%) | D_AllA:¥{h_d2[d_cur]['portfolio_value']:,.0f}({d2_ret:+.2f}%)")
