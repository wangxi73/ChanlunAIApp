import bisect
import json
import math
from pathlib import Path
from universe import get_universes
from chanlun_engine import evaluate_asset, get_stage1_candidates
from data_loader import load_all_data

def prepare_indexed_data(raw_data):
    indexed = {}
    for sym, d in raw_data.items():
        day_dates = [b["time"] for b in d["day"]]
        m60_dates = [b["date"] for b in d["m60"]]
        m15_dates = [b["date"] for b in d["m15"]]
        indexed[sym] = {
            "day": d["day"], "day_dates": day_dates,
            "m60": d["m60"], "m60_dates": m60_dates,
            "m15": d["m15"], "m15_dates": m15_dates,
            "snaps": d["snaps"]
        }
    return indexed

def get_bars_up_to(idx_entry, dt):
    i_d = bisect.bisect_right(idx_entry["day_dates"], dt)
    i_60 = bisect.bisect_right(idx_entry["m60_dates"], dt)
    i_15 = bisect.bisect_right(idx_entry["m15_dates"], dt)
    day_slice = idx_entry["day"][max(0, i_d - 250):i_d]
    m60_slice = idx_entry["m60"][max(0, i_60 - 280):i_60]
    m15_slice = idx_entry["m15"][max(0, i_15 - 280):i_15]
    return day_slice, m60_slice, m15_slice

def simulate_account(account_id, account_name, scope_label, universe, indexed_data, trade_dates, bench_map):
    cash = 100000.0
    initial_capital = 100000.0
    holdings = {}
    history = []
    trades = []
    closed_trades = []
    init_bench = bench_map[trade_dates[0]]

    for dt in trade_dates:
        # 1. Update holdings with today's close price
        for sym, h in holdings.items():
            snap = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if snap and snap["lastPrice"] > 0:
                h["last_price"] = snap["lastPrice"]

        # 2. Build snapshots for universe on date dt
        day_snaps = {}
        for sym, _, _ in universe:
            s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if s:
                day_snaps[sym] = s

        # 3. Stage 1 funnel (top_k=44) + Stage 2 Chanlun 3-period evaluation
        stage1_selected = get_stage1_candidates(universe, day_snaps, holdings, top_k=44)
        evaluated = []
        for meta in stage1_selected:
            sym = meta[0]
            idx_e = indexed_data.get(sym)
            if not idx_e:
                continue
            d_bars, m60_bars, m15_bars = get_bars_up_to(idx_e, dt)
            if not d_bars or not m15_bars:
                continue
            ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, day_snaps.get(sym))
            evaluated.append(ev)

        evaluated.sort(key=lambda x: (
            x["score"],
            x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0,
            x["snapshot"]["amountYi"] if x["snapshot"] else 0.0
        ), reverse=True)
        eval_map = {e["symbol"]: e for e in evaluated}
        day_actions = []
        day_reasons = []

        # 4. Pure Quant SELL rules on existing holdings (auto-approved at close price)
        for sym in list(holdings.keys()):
            h = holdings[sym]
            ev = eval_map.get(sym)
            if not ev:
                continue
            p_m15 = ev["periods"].get("15m")
            if not p_m15:
                continue
            zs15 = p_m15.get("lastZs")
            bi15 = p_m15.get("lastBi")
            div15 = p_m15.get("divRatio")
            sell_reason = None
            stop_ref = round(ev["price"] * 0.96, 3)

            if zs15 and ev["price"] < zs15["dd"]:
                stop_ref = zs15["dd"]
                sell_reason = f"[纯量化规则] {h['name']}({sym})跌破15m中枢极值DD({zs15['dd']})触发结构破位止损，卖出全部{h['shares']}股。"
            elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                if zs15:
                    stop_ref = round(zs15["zd"] * 0.96, 3)
                sell_reason = f"[纯量化规则] {h['name']}({sym})出现15m向上笔盘整顶背驰一卖(面积比={div15})，卖出{h['shares']}股。"

            if sell_reason:
                price = ev["price"]
                shares = h["shares"]
                gross = shares * price
                fee = max(gross * 0.0005, 5.0)
                net = gross - fee
                cash += net
                pnl_amt = net - (shares * h["avg_cost"])
                pnl_pct = (price - h["avg_cost"]) / h["avg_cost"] * 100.0
                closed_trades.append({
                    "symbol": sym, "name": h["name"], "buy_date": h["buy_date"],
                    "sell_date": dt, "shares": shares, "buy_price": h["avg_cost"],
                    "sell_price": price, "pnl_amt": round(pnl_amt, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": sell_reason
                })
                trades.append({
                    "date": dt, "action": "SELL", "symbol": sym, "name": h["name"],
                    "shares": shares, "price": price, "amount": round(gross, 2),
                    "fee": round(fee, 2), "stop_loss": stop_ref, "reason": sell_reason
                })
                day_actions.append(f"卖出{h['name']}({sym}) {shares}股@¥{price:.2f}")
                day_reasons.append(sell_reason)
                del holdings[sym]
        # 5. Pure Quant BUY rules on unheld candidates (auto-approved at close price)
        budget = cash
        buy_count = 0
        for cand in evaluated:
            if buy_count >= 2:
                break
            sym = cand["symbol"]
            if sym in holdings:
                continue
            if cand["score"] < 45 or budget < 15000.0 or cand["price"] <= 0:
                continue
            target_amt = 25000.0 if cand["kind"] == "stock" else 30000.0
            alloc = min(budget * 0.95, target_amt)
            lots = int(alloc // (cand["price"] * 100.0))
            shares = lots * 100
            if shares < 100:
                continue
            p_m15 = cand["periods"].get("15m")
            zs15 = p_m15.get("lastZs") if p_m15 else None
            stop_loss = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
            gross = shares * cand["price"]
            fee = max(gross * 0.00025, 5.0)
            total_cost = gross + fee
            if total_cost > cash:
                continue
            cash -= total_cost
            budget -= gross
            buy_count += 1
            holdings[sym] = {
                "name": cand["name"], "shares": shares,
                "avg_cost": round(total_cost / shares, 4),
                "last_price": cand["price"], "buy_date": dt
            }
            reason = f"[纯量化·{scope_label}] {cand['name']}({sym})呈{'、'.join(cand['tags'])}，建仓{shares}股(约¥{gross:.2f})，止损位¥{stop_loss}。"
            trades.append({
                "date": dt, "action": "BUY", "symbol": sym, "name": cand["name"],
                "shares": shares, "price": cand["price"], "amount": round(gross, 2),
                "fee": round(fee, 2), "stop_loss": stop_loss, "score": cand["score"], "reason": reason
            })
            day_actions.append(f"买入{cand['name']}({sym}) {shares}股@¥{cand['price']:.2f}")
            day_reasons.append(reason)

        mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
        port_val = round(cash + mkt_val, 2)
        port_ret = round((port_val - initial_capital) / initial_capital * 100.0, 2)
        hs300_lvl = bench_map[dt]
        hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
        alpha = round(port_ret - hs300_ret, 2)
        history.append({
            "date": dt, "portfolio_value": port_val, "cash": round(cash, 2),
            "holdings_count": len(holdings), "portfolio_return_pct": port_ret,
            "hs300_level": hs300_lvl, "hs300_return_pct": hs300_ret,
            "alpha_pct": alpha,
            "decision": " / ".join(day_actions) if day_actions else "HOLD",
            "reason": day_reasons[0] if day_reasons else f"沪深300报{hs300_lvl:.2f}点，维持现有仓位。"
        })

    return {
        "account_id": account_id, "account_name": account_name,
        "scope_label": scope_label, "initial_capital": initial_capital,
        "final_value": history[-1]["portfolio_value"], "final_cash": round(cash, 2),
        "final_holdings": holdings, "history": history,
        "trades": trades, "closed_trades": closed_trades
    }

def compute_metrics(res):
    hist = res["history"]
    vals = [h["portfolio_value"] for h in hist]
    peak = vals[0]
    max_dd = 0.0
    max_dd_date = hist[0]["date"]
    for h in hist:
        v = h["portfolio_value"]
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
            max_dd_date = h["date"]

    daily_rets = []
    for i in range(1, len(vals)):
        daily_rets.append((vals[i] - vals[i - 1]) / vals[i - 1])
    mean_r = sum(daily_rets) / len(daily_rets) if daily_rets else 0.0
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in daily_rets) / len(daily_rets)) if daily_rets else 0.0
    sharpe = ((mean_r * 242 - 0.015) / (std_r * math.sqrt(242))) if std_r > 0 else 0.0

    ct = res["closed_trades"]
    wins = [t for t in ct if t["pnl_amt"] > 0]
    losses = [t for t in ct if t["pnl_amt"] <= 0]
    win_rate = (len(wins) / len(ct) * 100.0) if ct else 0.0
    avg_win = (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (abs(sum(t["pnl_pct"] for t in losses) / len(losses))) if losses else 0.0
    pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    return {
        "start_date": hist[0]["date"], "end_date": hist[-1]["date"],
        "trading_days": len(hist),
        "initial_capital": res["initial_capital"],
        "final_value": res["final_value"],
        "total_return_pct": hist[-1]["portfolio_return_pct"],
        "hs300_return_pct": hist[-1]["hs300_return_pct"],
        "alpha_pct": hist[-1]["alpha_pct"],
        "max_drawdown_pct": round(max_dd, 2),
        "max_drawdown_date": max_dd_date,
        "sharpe_ratio": round(sharpe, 2),
        "total_orders": len(res["trades"]),
        "buy_orders": sum(1 for t in res["trades"] if t["action"] == "BUY"),
        "sell_orders": sum(1 for t in res["trades"] if t["action"] == "SELL"),
        "closed_roundtrips": len(ct),
        "win_count": len(wins), "loss_count": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 2), "avg_loss_pct": round(avg_loss, 2),
        "profit_loss_ratio": round(pl_ratio, 2)
    }

if __name__ == "__main__":
    hs300_u, all_a_u = get_universes()
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]
    print(f"Fetching K-line data for {len(all_syms)} symbols (HS300={len(hs300_u)}, AllA={len(all_a_u)})...")
    raw_data = load_all_data(all_syms, max_workers=12)
    indexed = prepare_indexed_data(raw_data)

    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2025-10-29" <= b["time"] <= "2026-10-29"]
    print(f"Running 1-year backtest over {len(trade_dates)} trading days ({trade_dates[0]} -> {trade_dates[-1]})...")

    res_hs300 = simulate_account(
        "quant_hs300", "账户①：无AI介入 · 仅限沪深300 (¥100,000)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map
    )
    res_all_a = simulate_account(
        "quant_all_a", "账户②：无AI介入 · 全市场沪深A股 (¥100,000)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map
    )

    m_hs300 = compute_metrics(res_hs300)
    m_all_a = compute_metrics(res_all_a)
    res_hs300["metrics"] = m_hs300
    res_all_a["metrics"] = m_all_a

    out_path = Path(__file__).resolve().parent / "backtest_results_1y.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"quant_hs300": res_hs300, "quant_all_a": res_all_a}, f, ensure_ascii=False, indent=2)

    print("HS300 Metrics:", json.dumps(m_hs300, ensure_ascii=False))
    print("All-A Metrics:", json.dumps(m_all_a, ensure_ascii=False))
