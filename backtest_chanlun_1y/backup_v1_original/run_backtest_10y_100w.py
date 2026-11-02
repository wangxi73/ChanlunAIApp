import bisect
import json
import pickle
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from universe import get_universes
from chanlun_engine import evaluate_asset, get_stage1_candidates
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import fetch_hfq_factors, build_period_dataset, CACHE_FULL_DIR

CACHE_10Y_DIR = Path(__file__).resolve().parent / "cache_10y_hfq"
CACHE_10Y_DIR.mkdir(parents=True, exist_ok=True)

def fetch_10y_symbol_with_hfq(symbol):
    cf = CACHE_10Y_DIR / f"{symbol}.pkl"
    if cf.exists():
        try:
            with open(cf, "rb") as f:
                d = pickle.load(f)
                if d.get("raw_day"):
                    return symbol, d
        except Exception:
            pass

    old_cf = CACHE_FULL_DIR / f"{symbol}.pkl"
    f_dates, f_vals = [], []
    if old_cf.exists():
        try:
            with open(old_cf, "rb") as f:
                old_d = pickle.load(f)
                f_dates = old_d.get("f_dates", [])
                f_vals = old_d.get("f_vals", [])
        except Exception:
            pass
    if not f_dates and symbol != "sh000300":
        f_dates, f_vals = fetch_hfq_factors(symbol)

    raw_bars = []
    try:
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=3000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        arr = json.loads(urllib.request.urlopen(req, timeout=12).read().decode())
        if isinstance(arr, list):
            for r in arr:
                dt = str(r["day"])[:10]
                c = float(r["close"])
                if c > 0:
                    raw_bars.append({
                        "symbol": symbol, "time": dt,
                        "open": float(r["open"]), "close": c,
                        "high": float(r["high"]), "low": float(r["low"]),
                        "volume": float(r["volume"]) / 100.0
                    })
    except Exception:
        pass

    payload = {"raw_day": raw_bars, "f_dates": f_dates, "f_vals": f_vals}
    if raw_bars:
        with open(cf, "wb") as f:
            pickle.dump(payload, f)
    return symbol, payload

def simulate_10y(account_id, account_name, scope_label, universe, indexed_data, trade_dates, bench_map, eval_cache, scale_factor=10.0):
    initial_capital = 1000000.0
    cash = initial_capital
    min_cash_threshold = 15000.0 * scale_factor
    stock_cap = 25000.0 * scale_factor
    etf_cap = 30000.0 * scale_factor

    holdings = {}
    history = []
    trades = []
    closed_trades = []
    init_bench = bench_map[trade_dates[0]]

    for dt in trade_dates:
        for sym, h in holdings.items():
            snap = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if snap and snap["lastPrice"] > 0:
                h["last_price"] = snap["lastPrice"]

        day_snaps = {}
        for sym, _, _ in universe:
            s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if s:
                day_snaps[sym] = s

        stage1_selected = get_stage1_candidates(universe, day_snaps, holdings, top_k=44)
        evaluated = []
        for meta in stage1_selected:
            sym = meta[0]
            ck = (sym, dt)
            if ck in eval_cache:
                ev = eval_cache[ck]
                if ev is not None:
                    evaluated.append(ev)
                continue
            idx_e = indexed_data.get(sym)
            if not idx_e:
                eval_cache[ck] = None
                continue
            i_d = bisect.bisect_right(idx_e["day_dates"], dt)
            i_60 = bisect.bisect_right(idx_e["m60_dates"], dt)
            i_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
            d_bars = idx_e["day"][max(0, i_d - 250):i_d]
            m60_bars = idx_e["m60"][max(0, i_60 - 280):i_60]
            m15_bars = idx_e["m15"][max(0, i_15 - 280):i_15]
            if not d_bars or not m15_bars:
                eval_cache[ck] = None
                continue
            ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, day_snaps.get(sym))
            eval_cache[ck] = ev
            evaluated.append(ev)

        evaluated.sort(key=lambda x: (
            x["score"],
            x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0,
            x["snapshot"]["amountYi"] if x["snapshot"] else 0.0
        ), reverse=True)
        eval_map = {e["symbol"]: e for e in evaluated}

        day_actions = []
        day_reasons = []

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

        budget = cash
        buy_count = 0
        for cand in evaluated:
            if buy_count >= 2:
                break
            sym = cand["symbol"]
            if sym in holdings:
                continue
            if cand["score"] < 45 or budget < min_cash_threshold or cand["price"] <= 0:
                continue
            target_amt = stock_cap if cand["kind"] == "stock" else etf_cap
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

if __name__ == "__main__":
    hs300_u, all_a_u = get_universes()
    cache_dir = Path(__file__).resolve().parent / "cache"
    cached_syms = {p.stem for p in cache_dir.glob("*.pkl")}
    if cached_syms:
        all_a_u = [m for m in all_a_u if m[0] in cached_syms]
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]
    print(f"Fetching 10Y K-line + HFQ data for {len(all_syms)} symbols...")

    raw_cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_10y_symbol_with_hfq, s): s for s in all_syms}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, d = fut.result()
            raw_cache[sym] = d
            if i % 50 == 0 or i == len(all_syms):
                print(f"Loaded 10Y {i}/{len(all_syms)} symbols...")

    dataset = build_period_dataset(raw_cache, "2015-10-01", "2026-10-29", "2026-10-29")
    indexed = prepare_indexed_data(dataset)

    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2016-10-28" <= b["time"] <= "2026-10-29"]
    print(f"Running 10-year 100w backtest over {len(trade_dates)} trading days ({trade_dates[0]} -> {trade_dates[-1]})...")

    eval_cache = {}
    res_hs300_10x = simulate_10y(
        "quant_hs300_10y_100w", "账户①：无AI介入 · 仅限沪深300 (¥1,000,000 · 单笔25万)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map, eval_cache, scale_factor=10.0
    )
    res_alla_10x = simulate_10y(
        "quant_all_a_10y_100w", "账户②：无AI介入 · 全市场沪深A股 (¥1,000,000 · 单笔25万)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map, eval_cache, scale_factor=10.0
    )
    res_hs300_10x["metrics"] = compute_metrics(res_hs300_10x)
    res_alla_10x["metrics"] = compute_metrics(res_alla_10x)

    res_hs300_div = simulate_10y(
        "quant_hs300_10y_100w_div", "账户①：无AI介入 · 仅限沪深300 (¥1,000,000 · 单笔2.5万分散)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map, eval_cache, scale_factor=1.0
    )
    res_alla_div = simulate_10y(
        "quant_all_a_10y_100w_div", "账户②：无AI介入 · 全市场沪深A股 (¥1,000,000 · 单笔2.5万分散)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map, eval_cache, scale_factor=1.0
    )
    res_hs300_div["metrics"] = compute_metrics(res_hs300_div)
    res_alla_div["metrics"] = compute_metrics(res_alla_div)

    out_path = Path(__file__).resolve().parent / "backtest_results_10y_100w.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "proportional_25w": {"quant_hs300": res_hs300_10x, "quant_all_a": res_alla_10x},
            "diversified_2_5w": {"quant_hs300": res_hs300_div, "quant_all_a": res_alla_div}
        }, f, ensure_ascii=False, indent=2)

    print("10Y 100w (25w/trade) HS300:", json.dumps(res_hs300_10x["metrics"], ensure_ascii=False))
    print("10Y 100w (25w/trade) All-A:", json.dumps(res_alla_10x["metrics"], ensure_ascii=False))
    print("10Y 100w (2.5w/trade) HS300:", json.dumps(res_hs300_div["metrics"], ensure_ascii=False))
    print("10Y 100w (2.5w/trade) All-A:", json.dumps(res_alla_div["metrics"], ensure_ascii=False))
