"""бейзлайн-модели для сравнения"""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from config import DATA_DIR,RESULTS_DIR

TRAIN_END= "2016-12-31"
VAL_START= "2017-01-01"
VAL_END= "2017-08-15"


def rmsle(yt,yp):
    yt= np.clip(yt,0,None)
    yp= np.clip(yp,0,None)
    return   np.sqrt(np.mean((np.log1p(yp) -np.log1p(yt))**2))


def naive(ytr,n):
    return   np.full(n,ytr[-1])

def seasonal_naive(ytr,n,season=7):
    if len(ytr) < season:
        return   np.full(n,ytr[-1])
    return   np.array([ytr[-season +i%season] for i in range(n)])


def auto_ets(ytr,n):
    try:
        m= ExponentialSmoothing(ytr,seasonal_periods=7,trend="add",seasonal="add")
        f= m.fit(optimized=True)
        return   np.clip(np.asarray(f.forecast(n)),0,None)
    except Exception as e:
        print(f"    ets не смог: {e},откат к naive")
        return   naive(ytr,n)

def auto_theta(ytr,n):
    try:
        m= ThetaModel(ytr,period=7)
        f= m.fit()
        return   np.clip(np.asarray(f.forecast(n)),0,None)
    except Exception as e:
        print(f"    theta не смог: {e},откат к naive")
        return   naive(ytr,n)


def sn_item(tr_part,val_part,season=7):
    print("\nseasonal naive на уровне store-item")
    lw= (tr_part.sort_values("date")
                .groupby(["store_nbr","item_nbr"]).tail(season).copy())
    lw["day_rank"]= lw.groupby(["store_nbr","item_nbr"]).cumcount()

    lk= lw.set_index(["store_nbr","item_nbr","day_rank"])["unit_sales"]

    val= val_part[["date","store_nbr","item_nbr","unit_sales"]].copy()
    val["day_rank"]= (val["date"] -val["date"].min()).dt.days%season
    val["key"]= list(zip(val["store_nbr"],val["item_nbr"],val["day_rank"]))
    lk_d= lk.to_dict()
    val["pred"]= val["key"].map(lk_d).fillna(0).clip(lower=0)

    yt= np.log1p(val["unit_sales"].clip(lower=0).values)
    yp= np.log1p(val["pred"].values)
    return   np.sqrt(np.mean((yp -yt)**2))


def main():
    print("загружаем данные")
    tr= pd.read_parquet(DATA_DIR/"train_subsample.parquet")

    tr_part= tr[tr["date"] <= TRAIN_END]
    val_part= tr[(tr["date"] >= VAL_START) & (tr["date"] <= VAL_END)]
    print(f"train: {tr_part.shape},val: {val_part.shape}")

    ytr= tr_part.groupby("date")["unit_sales"].sum().sort_index().values
    yval= val_part.groupby("date")["unit_sales"].sum().sort_index().values
    print(f"train дней: {len(ytr)},val дней: {len(yval)}")

    n= len(yval)
    res= {}

    print("\nобучаем бейзлайны")
    for nm,fn in [("Naive",naive),("SeasonalNaive",seasonal_naive),
                  ("auto_ets",auto_ets),("auto_theta",auto_theta)]:
        print(f"  {nm}")
        pr= fn(ytr,n)
        sc= rmsle(yval,pr)
        res[nm]= sc
        print(f"    rmsle = {sc:.4f}")

    sn= sn_item(tr_part,val_part)
    print(f"  seasonal naive (store-item): rmsle = {sn:.4f}")
    res["SeasonalNaive_itemlevel"]= sn

    print("\nитог")
    df= pd.DataFrame(list(res.items()),columns=["model","RMSLE"]).sort_values("RMSLE")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR/"baseline_results.csv",index=False)
    print(f"\nсохраняем {RESULTS_DIR/'baseline_results.csv'}")


if __name__ == "__main__":
    main()