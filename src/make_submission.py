"""формирование submission.csv для kaggl"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from config import DATA_DIR,RESULTS_DIR,SUBMISSIONS_DIR

FEATURES= [
    "store_nbr","item_nbr","family","city","state","type","cluster","class","perishable",
    "onpromotion","promo_count_store","promo_ratio_store",
    "oil","is_holiday","transactions",
    "dayofweek","month","day","year","weekofyear","dayofyear",
    "dow_sin","dow_cos","month_sin","month_cos",
    "lag_1","lag_7","lag_14","lag_28",
    "roll_mean_7","roll_std_7","roll_mean_30","roll_std_30","roll_mean_90","roll_std_90",
           ]
CAT_FEATURES= ["store_nbr","item_nbr","family","city","state","type","cluster","class"]


def build_lags(feat,tgt):
    feat= feat.sort_values(["store_nbr","item_nbr","date"])
    tgt_p= tgt[["store_nbr","item_nbr"]].drop_duplicates()
    feat= feat.merge(tgt_p,on=["store_nbr","item_nbr"],how="inner")

    l90= feat.groupby(["store_nbr","item_nbr"],sort=False).tail(90).copy()
    l90["rank_end"]= l90.groupby(["store_nbr","item_nbr"],sort=False).cumcount(ascending=False)

    lags= []
    for lg,rk in [(1,1),(7,7),(14,14),(28,28)]:
        df_lg= l90[l90["rank_end"]== rk][["store_nbr","item_nbr","unit_sales"]].rename(
            columns={"unit_sales": f"lag_{lg}"}
                                                                                        )
        lags.append(df_lg)

    rl= l90.groupby(["store_nbr","item_nbr"],sort=False)["unit_sales"].agg(
        roll_mean_7=lambda x: x.tail(7).mean(),
        roll_std_7=lambda x: x.tail(7).std(),
        roll_mean_30=lambda x: x.tail(30).mean(),
        roll_std_30=lambda x: x.tail(30).std(),
        roll_mean_90= "mean",
        roll_std_90= "std",
                                                                            ).reset_index()

    return   rl,lags


def main():
    print("submission")

    ts= pd.read_csv(DATA_DIR/"test.csv",parse_dates=["date"])
    print(f"  test: {ts.shape}")
    d= ts["date"].dt
    ts["dayofweek"]= d.dayofweek.astype("int8")
    ts["month"]= d.month.astype("int8")
    ts["day"]= d.day.astype("int8")
    ts["year"]= d.year.astype("int16")
    ts["weekofyear"]= d.isocalendar().week.astype("int8")
    ts["dayofyear"]= d.dayofyear.astype("int16")
    ts["dow_sin"]= np.sin(2*np.pi*ts["dayofweek"]/7)
    ts["dow_cos"]= np.cos(2*np.pi*ts["dayofweek"]/7)
    ts["month_sin"]= np.sin(2*np.pi*ts["month"]/12)
    ts["month_cos"]= np.cos(2*np.pi*ts["month"]/12)
    st_df= pd.read_csv(DATA_DIR/"stores.csv")
    it_df= pd.read_csv(DATA_DIR/"items.csv")
    ts= ts.merge(st_df,on="store_nbr",how="left")
    ts= ts.merge(it_df,on="item_nbr",how="left")
    oil= pd.read_csv(DATA_DIR/"oil.csv",parse_dates=["date"])
    oil= oil.sort_values("date").rename(columns={"dcoilwtico": "oil"})
    oil["oil"]= oil["oil"].ffill().bfill()
    ts= ts.merge(oil,on="date",how="left")
    ts["oil"]= ts["oil"].ffill().bfill().fillna(0)
    hol= pd.read_csv(DATA_DIR/"holidays_events.csv",parse_dates=["date"])
    nat= hol[hol["locale"]== "National"]["date"].unique()
    ts["is_holiday"]= ts["date"].isin(nat).astype("int8")
    trx= pd.read_csv(DATA_DIR/"transactions.csv",parse_dates=["date"])
    ts= ts.merge(trx,on=["date","store_nbr"],how="left")
    ts["transactions"]= ts["transactions"].fillna(0).astype("float32")

    feat= pd.read_parquet(DATA_DIR/"features.parquet")
    rl,lags= build_lags(feat,ts)
    ts= ts.merge(rl,on=["store_nbr","item_nbr"],how="left")
    for df_lg in lags:
        ts= ts.merge(df_lg,on=["store_nbr","item_nbr"],how="left")
    ts["onpromotion"]= ts["onpromotion"].fillna(False).astype(int)
    ts["promo_count_store"]= ts.groupby(["date","store_nbr"])["onpromotion"].transform("sum")
    ts["promo_ratio_store"]= (
        ts["promo_count_store"]/ts.groupby(["date","store_nbr"])["onpromotion"].transform("count")
                                                                                                )

    for col in FEATURES:
        if col in ts.columns:
            ts[col]= ts[col].fillna(0)
    print("загружаем catboost")
    m= CatBoostRegressor()
    m.load_model(str(RESULTS_DIR/"catboost_model_full.cbm"))

    print("прогноз")
    Xts= ts[FEATURES].copy()
    for col in CAT_FEATURES:
        Xts[col]= Xts[col].astype(str)
    pr_log= m.predict(Xts)
    pr_sales= np.clip(np.expm1(pr_log),0,100)
    print(f"обрезаны >100: {(np.expm1(pr_log) > 100).sum()}")
    sub= pd.DataFrame({"id": ts["id"].values,"unit_sales": pr_sales})
    out= SUBMISSIONS_DIR/"submission.csv"
    sub.to_csv(out,index=False)
    print(f"\nсохраняем {out}")
    print(f"строк:  {len(sub)}")
    print(f"средние:  {sub['unit_sales'].mean():.3f}")
    print(f"максимум:  {sub['unit_sales'].max():.1f}")
    print(f"NaN:  {sub['unit_sales'].isna().sum()}")


if __name__ == "__main__":
    main()