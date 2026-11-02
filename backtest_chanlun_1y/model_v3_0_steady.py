"""
ChanlunAIApp V3.0 模型（稳健统一漏斗 · 8仓分散版）
------------------------------------------------
核心特征：
1. 选股池：当期真实 (沪深300 + 创业板指 + 科创50) 合并为统一精选池（无幸存者偏差 PIT，2019.08前无科创板自动由沪深300+创业板组成）；
2. 初筛漏斗 (Stage 1)：三大板块合并后按全池成交额与主买优势统一筛选 Top-52（天然要求双创入选标的具备全市场一线大资金容量）；
3. 牛熊智能路由 (Stage 2)：
   - BULL (强牛主升)：20cm双创龙头 +8分弹性优先；
   - RANGE (结构震荡)：按原生缠论三周期得分公平竞价；
   - WEAK (弱势/高位急跌防守)：沪深300低波蓝筹红利 +15分避险优先，单笔仓位缩至60%、每日限买1只>=60分；
   - CRASH (熊市反弹末端诱多 / 高位非背驰闪崩)：全员暂停新开仓 (0开仓)；
4. 仓位与风控：100万本金下单笔上限 12.5万 (ETF 15万，约8仓分散)，启用高位非背驰急跌保护 + 熊市锁死初始止损。
"""
from run_backtest_v3_unified import build_unified_pit_rosters, simulate_v3_unified

def run_v3_0_steady(rosters_by_date, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                    initial_capital=1000000.0, use_ratchet_stop=True):
    scale = initial_capital / 1000000.0
    u_rosters = build_unified_pit_rosters(rosters_by_date)
    return simulate_v3_unified(
        "V3.0 模型 (统一漏斗 · 单笔12.5万稳健8仓版)",
        u_rosters, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
        initial_capital=initial_capital,
        single_stock_cap=125000.0 * scale,
        single_etf_cap=150000.0 * scale,
        min_cash=75000.0 * scale,
        use_ratchet_stop=use_ratchet_stop
    )
