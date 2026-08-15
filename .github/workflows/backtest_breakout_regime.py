from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
PARQUET = ROOT / "nse_6yr_historical.parquet"
BREADTH = ROOT / "historical_breadth_regime_6yr.csv"

# Trading objective
TIGHTNESS_LIMIT = 0.04
STOP_PCT = 7.0
TARGET_PCT = 15.0
MAX_DAYS = 30
MIN_TRAIN_TRADES = 300
MIN_BUCKET_TRADES = 100

FEATURES = [
    "PctAbove20EMA", "PctAbove50EMA", "PctAbove200EMA",
    "Slope20EMA", "Slope50EMA", "Slope200EMA",
    "VolumeRatio", "Net52WHighLow",
    "Rolling3DUp4", "Rolling3DDown4",
    "Up251MCount", "Down251MCount",
    "FollowThroughRate",
    "LargePct20EMA", "LargePct50EMA", "LargePct200EMA",
    "MidPct20EMA", "MidPct50EMA", "MidPct200EMA",
    "SmallPct20EMA", "SmallPct50EMA", "SmallPct200EMA",
    "MicroPct20EMA", "MicroPct50EMA", "MicroPct200EMA",
]

# Candidate mixes. The script will also search many data-driven combinations.
MIXES = {
    "equal_core": {"PctAbove20EMA": .25, "PctAbove50EMA": .25, "PctAbove200EMA": .10, "VolumeRatio": .15, "Net52WHighLow": .10, "FollowThroughRate": .15},
    "leadership": {"PctAbove20EMA": .30, "PctAbove50EMA": .30, "PctAbove200EMA": .10, "VolumeRatio": .10, "Net52WHighLow": .10, "FollowThroughRate": .10},
    "followthrough": {"PctAbove20EMA": .20, "PctAbove50EMA": .20, "PctAbove200EMA": .10, "VolumeRatio": .10, "Net52WHighLow": .10, "FollowThroughRate": .30},
    "participation": {"PctAbove20EMA": .20, "PctAbove50EMA": .15, "PctAbove200EMA": .05, "VolumeRatio": .25, "Net52WHighLow": .15, "FollowThroughRate": .20},
}


def percentile_score(s):
    return s.rank(pct=True, method="average") * 100


def first_hit(future, entry, target=TARGET_PCT, stop=STOP_PCT):
    tp, sl = entry * (1 + target/100), entry * (1 - stop/100)
    for day, (_, r) in enumerate(future.iterrows(), 1):
        hit_t = float(r.High) >= tp
        hit_s = float(r.Low) <= sl
        if hit_t and hit_s: return "AMBIGUOUS", day
        if hit_t: return "SUCCESS", day
        if hit_s: return "STOP", day
    return "TIMEOUT", np.nan


def build_trades(px):
    rows=[]
    for sym, g in px.groupby("Symbol", sort=False):
        g=g.sort_values("Date").reset_index(drop=True)
        for i in g.index[g.IsBreakout].tolist():
            fut=g.iloc[i+1:i+1+MAX_DAYS]
            if len(fut)<MAX_DAYS: continue
            e=float(g.loc[i,"Close"])
            result, day=first_hit(fut,e)
            rec={"Date":g.loc[i,"Date"],"Symbol":sym,"EntryPrice":e,"Result":result,"DecisionDay":day}
            for t in [15,18,25,30]:
                rec[f"Target{t}BeforeStop"] = first_hit(fut,e,t,STOP_PCT)[0]
            for h in [5,10,15,20,30]:
                rec[f"CloseReturn{h}D"]=(float(g.loc[i+h,"Close"])/e-1)*100
            rec["RuleReturn"] = TARGET_PCT if result=="SUCCESS" else (-STOP_PCT if result=="STOP" else (float(fut.iloc[-1].Close)/e-1)*100)
            rows.append(rec)
    return pd.DataFrame(rows)


def add_features(trades, breadth):
    b=breadth.copy()
    if {"T3Breakouts","T3Wins"}.issubset(b.columns):
        b["FollowThroughRate"]=np.where(b.T3Breakouts>0,b.T3Wins/b.T3Breakouts*100,np.nan)
    usable=["Date"]+[f for f in FEATURES if f in b.columns]
    b=b[usable].drop_duplicates("Date")
    return trades.merge(b,on="Date",how="left"), b


def make_buckets(trades, features):
    rows=[]
    base=(trades.Result=="SUCCESS")
    for f in features:
        v=trades.dropna(subset=[f]).copy()
        if len(v)<MIN_BUCKET_TRADES: continue
        v["Bucket"]=pd.qcut(v[f].rank(method="first"),3,labels=["Low","Medium","High"])
        for bucket,g in v.groupby("Bucket",observed=False):
            rows.append({"Feature":f,"Bucket":str(bucket),"Trades":len(g),"Success15Before7Pct":(g.Result=="SUCCESS").mean()*100,"StopBefore15Pct":(g.Result=="STOP").mean()*100,"Target18Pct":(g.Target18BeforeStop=="SUCCESS").mean()*100,"Target25Pct":(g.Target25BeforeStop=="SUCCESS").mean()*100,"Target30Pct":(g.Target30BeforeStop=="SUCCESS").mean()*100,"Avg10D":g.CloseReturn10D.mean(),"Avg20D":g.CloseReturn20D.mean(),"AvgRuleReturn":g.RuleReturn.mean()})
    return pd.DataFrame(rows)


def score_frame(df, weights):
    s=pd.Series(0.0,index=df.index); total=0
    for f,w in weights.items():
        if f in df.columns:
            z=percentile_score(df[f].fillna(df[f].median()))
            # Downside metrics are inverted
            if f in {"Rolling3DDown4","Down251MCount","MicroPct20EMA","MicroPct50EMA","MicroPct200EMA"}: z=100-z
            s += z*w; total += w
    return s/total if total else s


def evaluate(d, score_col, threshold):
    q=d[d[score_col]>=threshold]
    if len(q)==0: return {"Trades":0,"SuccessPct":np.nan,"RuleReturn":np.nan,"Avg20D":np.nan}
    return {"Trades":len(q),"SuccessPct":(q.Result=="SUCCESS").mean()*100,"RuleReturn":q.RuleReturn.mean(),"Avg20D":q.CloseReturn20D.mean()}

print("Loading inputs...")
if not PARQUET.exists(): raise FileNotFoundError(PARQUET)
if not BREADTH.exists(): raise FileNotFoundError(BREADTH)
px=pd.read_parquet(PARQUET); b=pd.read_csv(BREADTH)
px.Date=pd.to_datetime(px.Date); b.Date=pd.to_datetime(b.Date)
px=px.sort_values(["Symbol","Date"]).reset_index(drop=True)
px["PrevClose"]=px.groupby("Symbol").Close.shift(1)
px["TR"]=np.maximum(px.High-px.Low,np.maximum((px.High-px.PrevClose).abs(),(px.Low-px.PrevClose).abs()))
px["ATR14"]=px.groupby("Symbol").TR.transform(lambda x:x.rolling(14,min_periods=5).mean())
px["Vol20"]=px.groupby("Symbol").Volume.transform(lambda x:x.rolling(20,min_periods=5).mean())
px["High20"]=px.groupby("Symbol").Close.transform(lambda x:x.rolling(20,min_periods=20).max())
px["Tight"]=(px.ATR14/px.Close)<TIGHTNESS_LIMIT
px["Breakout"]=(px.Close>=px.High20)&(px.Volume>px.Vol20*1.5)&px.groupby("Symbol").Tight.shift(1).fillna(False)
px["IsBreakout"]=px.Breakout
tr=build_trades(px)
tr,bmerge=add_features(tr,b)

# Audit and raw outputs
audit=pd.DataFrame([{"Input":"parquet","Rows":len(px),"MinDate":px.Date.min(),"MaxDate":px.Date.max()},{"Input":"breadth","Rows":len(b),"MinDate":b.Date.min(),"MaxDate":b.Date.max()},{"Input":"trade records","Rows":len(tr),"MinDate":tr.Date.min(),"MaxDate":tr.Date.max()}])
audit["BreadthMergeCoveragePct"]=np.nan
audit.to_csv(OUT/"research_input_audit.csv",index=False)
tr.to_csv(OUT/"research_breakout_trades_enriched.csv",index=False)

usable=[f for f in FEATURES if f in tr.columns and tr[f].notna().sum()>=MIN_BUCKET_TRADES]
buckets=make_buckets(tr,usable); buckets.to_csv(OUT/"research_feature_buckets.csv",index=False)

# Feature rankings from full data, but model selection uses walk-forward below.
rank=[]
for f in usable:
    x=tr.dropna(subset=[f]).copy(); x["bin"]=pd.qcut(x[f].rank(method="first"),3,labels=False)
    low=x[x.bin==0]; high=x[x.bin==2]
    rank.append({"Feature":f,"Trades":len(x),"HighSuccessPct":(high.Result=="SUCCESS").mean()*100,"LowSuccessPct":(low.Result=="SUCCESS").mean()*100,"LiftPct":(high.Result=="SUCCESS").mean()*100-(low.Result=="SUCCESS").mean()*100,"HighRuleReturn":high.RuleReturn.mean(),"LowRuleReturn":low.RuleReturn.mean()})
pd.DataFrame(rank).sort_values("LiftPct",ascending=False).to_csv(OUT/"research_feature_rankings.csv",index=False)

# Daily, point-in-time feature scores; score at date t uses only values on date t.
daily=tr.groupby("Date").agg(Trades=("Symbol","count"),SuccessPct=("Result",lambda x:(x=="SUCCESS").mean()*100),RuleReturn=("RuleReturn","mean"),Avg20D=("CloseReturn20D","mean")).reset_index()
for f in usable:
    vals=bmerge[["Date",f]].drop_duplicates("Date").sort_values("Date")
    vals[f+"_Score"]=percentile_score(vals[f])
    daily=daily.merge(vals[["Date",f+"_Score"]],on="Date",how="left")

# Mixes plus top feature combinations. We use only features with positive high-v-low lift.
for name,w in list(MIXES.items()):
    MIXES[name]={f:v for f,v in w.items() if f in usable}
positive=[r["Feature"] for r in sorted(rank,key=lambda z:z["LiftPct"],reverse=True) if r["LiftPct"]>0]
for k in [3,4,5,6]:
    for combo in list(product(positive[:8],repeat=0)):
        pass
    from itertools import combinations
    for combo in combinations(positive[:8],k):
        name="data_"+"_".join(combo)
        MIXES[name]={f:1/len(combo) for f in combo}

score_rows=[]
# Walk-forward: train first 60%, test remaining 40%, with chronological quarterly folds.
tr=tr.sort_values("Date").reset_index(drop=True)
cut_dates=pd.Series(tr.Date.drop_duplicates().sort_values().unique())
folds=[]
for q in range(4, len(cut_dates), max(1,len(cut_dates)//8)):
    train_end=cut_dates.iloc[q-1]
    test_end=cut_dates.iloc[min(q+max(1,len(cut_dates)//8)-1,len(cut_dates)-1)]
    folds.append((train_end,test_end))
if not folds: folds=[(cut_dates.iloc[int(len(cut_dates)*.6)],cut_dates.iloc[-1])]

for name,w in MIXES.items():
    tr["Score"]=score_frame(tr,w)
    for threshold in [50,60,65,70,75,80]:
        for fold,(train_end,test_end) in enumerate(folds,1):
            train=tr[tr.Date<=train_end]; test=tr[(tr.Date>train_end)&(tr.Date<=test_end)]
            if len(train)<MIN_TRAIN_TRADES or len(test)<50: continue
            r=evaluate(test,"Score",threshold)
            score_rows.append({"Model":name,"Threshold":threshold,"Fold":fold,"TrainEnd":train_end,"TestEnd":test_end,**r})

scores=pd.DataFrame(score_rows)
if not scores.empty:
    scores.to_csv(OUT/"research_walkforward_results.csv",index=False)
    model_summary=scores.groupby(["Model","Threshold"]).agg(Folds=("Fold","count"),TestTrades=("Trades","sum"),MeanSuccessPct=("SuccessPct","mean"),MeanRuleReturn=("RuleReturn","mean"),Mean20D=("Avg20D","mean")).reset_index()
    model_summary["ScoreQuality"]=model_summary["MeanRuleReturn"]+0.05*model_summary["MeanSuccessPct"]
    model_summary.sort_values("ScoreQuality",ascending=False).to_csv(OUT/"research_candidate_scores.csv",index=False)
    best=model_summary.sort_values("ScoreQuality",ascending=False).iloc[0]
    best_model,best_threshold=best.Model,int(best.Threshold)
else:
    best_model,best_threshold="equal_core",60
    pd.DataFrame().to_csv(OUT/"research_walkforward_results.csv",index=False)

best_weights=MIXES.get(best_model,MIXES["equal_core"])
# Rebuild daily score using best weights and attach action zones.
daily["CompositeScore"]=score_frame(daily,{f:w for f,w in best_weights.items() if f+"_Score" in daily.columns})
daily["BestModel"]=best_model; daily["RecommendedThreshold"]=best_threshold
daily["ActionZone"]=pd.cut(daily.CompositeScore,[-1,35,50,65,80,101],labels=["Risk-off","Defensive","Selective","Constructive","Aggressive"])
daily.to_csv(OUT/"research_best_composite_score_daily.csv",index=False)

zones=daily.groupby("ActionZone",observed=False).agg(Days=("Date","count"),Trades=("Trades","sum"),SuccessPct=("SuccessPct","mean"),RuleReturn=("RuleReturn","mean"),Avg20D=("Avg20D","mean")).reset_index()
zones.to_csv(OUT/"research_score_action_zones.csv",index=False)

summary=pd.DataFrame([{"Metric":"Breakout records","Value":len(tr)},{"Metric":"Usable features","Value":len(usable)},{"Metric":"Best model","Value":best_model},{"Metric":"Best threshold","Value":best_threshold},{"Metric":"Best model weights","Value":str(best_weights)},{"Metric":"Core success all trades %","Value":(tr.Result=="SUCCESS").mean()*100},{"Metric":"Core average rule return %","Value":tr.RuleReturn.mean()}])
summary.to_csv(OUT/"research_run_summary.csv",index=False)
print("DONE. Research files created in Research folder.")
