"""Целевой поиск: винрейт >= 60%, сделок больше, прибыль после комиссий и на in-sample, и вне выборки.
Конструкция: вход (откат / пробой / любой момент) + близкий тейк + широкий стоп по ATR + лимит по времени, с фильтром режима и без."""
import itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import engine
from engine import run, summarize
from intraday import load15, prepare
from btlib import sma, ema, rsi as rsi_f
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
C = dict(c_mkt=0.00075, c_stop=0.00075, c_lim=0.0002)
d15 = load15()
dd = d15.resample("1D").agg(dict(close="last")).dropna()
sma100 = (dd.close > sma(dd.close, 100)).shift(1)
rows = []
for tf in ["1h", "4h"]:
    x = prepare(d15, tf); c = x.close
    reg = {"ансамбль": (x.up.values, x.dn.values), "SMA100 дн": (sma100.reindex(x.index.floor("1D")).fillna(False).values.astype(bool), (~sma100.reindex(x.index.floor("1D")).fillna(True).values.astype(bool))), "нет": (np.ones(len(x), bool), np.ones(len(x), bool))}
    r2, r4, r14 = rsi_f(c, 2), rsi_f(c, 4), rsi_f(c, 14); mid = ema(c, 20)
    dn3 = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3)); up3 = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (c.shift(2) > c.shift(3))
    ent = {"RSI2<10": (r2 < 10, r2 > 90), "RSI2<5": (r2 < 5, r2 > 95), "RSI4<20": (r4 < 20, r4 > 80), "RSI14<30": (r14 < 30, r14 > 70), "RSI14<40": (r14 < 40, r14 > 60),
           "Кельтнер 1.5": (c < mid - 1.5 * x.atr, c > mid + 1.5 * x.atr), "Кельтнер 2.5": (c < mid - 2.5 * x.atr, c > mid + 2.5 * x.atr), "3 свечи против": (dn3, up3),
           "в любой момент": (pd.Series(True, index=x.index), pd.Series(True, index=x.index)),
           "пробой 24": (c > x.high.rolling(24).max().shift(1), c < x.low.rolling(24).min().shift(1)), "пробой 72": (c > x.high.rolling(72).max().shift(1), c < x.low.rolling(72).min().shift(1))}
    for (rn, (ru, rdn)), (en, (el, es)), slm, tpr, mh, side in itertools.product(reg.items(), ent.items(), [2.0, 3.0, 5.0], [0.33, 0.5, 1.0], [24, 72], ["long", "both"]):
        tr = run(x, sig_long=el.values & ru, sig_short=(es.values & rdn) if side == "both" else None, exit_long=~ru if rn != "нет" else None, exit_short=~rdn if rn != "нет" else None,
                 sl_mult=slm, tp_r=tpr, max_hold=mh, **C)
        row, _ = summarize(tr, x.index, tf=tf, regime=rn, entry=en, sl=slm, tp=tpr, hold=mh, side=side)
        rows.append(row)
    print(tf, "done", len(rows), flush=True)
T = pd.DataFrame(rows)
for k in ["is_cagr", "oos_cagr", "is_mdd", "oos_mdd"]: T[k] *= 100
T["oos_tpy"] = T.oos_tpd * 365; T["is_tpy"] = T.is_tpd * 365
T.round(3).to_csv("hiwin_grid.csv", index=False)
print("всего конфигураций:", len(T))
hw = T[(T.is_win >= 0.60) & (T.oos_win >= 0.60)]
print(f"винрейт >= 60% на обоих периодах: {len(hw)}; из них в плюсе на IS: {(hw.is_cagr > 0).sum()}, в плюсе на OOS: {(hw.oos_cagr > 0).sum()}, в плюсе на обоих: {((hw.is_cagr > 0) & (hw.oos_cagr > 0)).sum()}")
print("\n--- винрейт>=60%: медианы по группам ---")
print(hw.groupby(["tf", "side", "regime"]).agg(n=("oos_sharpe", "size"), is_sh=("is_sharpe", "median"), oos_sh=("oos_sharpe", "median"), oos_cagr=("oos_cagr", "median"), oos_mdd=("oos_mdd", "median"), win=("oos_win", "median"),
      tpy=("oos_tpy", "median"), avg_bp=("oos_avg_bp", "median"), both_pos=("oos_cagr", lambda v: 0)).assign(both_pos=hw.assign(b=(hw.is_cagr > 0) & (hw.oos_cagr > 0)).groupby(["tf", "side", "regime"]).b.mean()).round(2).to_string())
print("\n--- по типу входа (винрейт>=60%, лонг) ---")
h2 = hw[hw.side == "long"]
print(h2.groupby(["entry"]).agg(n=("oos_sharpe", "size"), is_sh=("is_sharpe", "median"), oos_sh=("oos_sharpe", "median"), oos_cagr=("oos_cagr", "median"), win=("oos_win", "median"), tpy=("oos_tpy", "median"), avg_bp=("oos_avg_bp", "median")).sort_values("oos_sh", ascending=False).round(2).to_string())
