"""анализ ошибок catboost на валидации"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from config import DATA_DIR,RESULTS_DIR

TRAIN_END= "2016-12-31"
VAL_START= "2017-01-01"
VAL_END= "2017-08-15"

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

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"]= (12,6)


GREEN= "#2ecc71"
GREEN_D= "#27ae60"
PURPLE= "#9b59b6"
PURPLE_D= "#8e44ad"
GRAY= "#95a5a6"
RED= "#e74c3c"


def rmsle(yt,yp):
    return   np.sqrt(np.mean((yp -yt)**2))


def main():
    print("читаем features.parquet")
    df= pd.read_parquet(DATA_DIR/"features.parquet")
    vl= df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    print(f"val: {vl.shape}")
    Xvl= vl[FEATURES].copy()
    yvl= vl["target_log"].values
    for col in CAT_FEATURES:
        Xvl[col]= Xvl[col].astype(str)
    print("загружаем модель")
    m= CatBoostRegressor()
    m.load_model(str(RESULTS_DIR/"catboost_model_full.cbm"))
    print("считаем прогнозы")
    pr= m.predict(Xvl)
    vl["pred_log"]= pr
    vl["error"]= vl["pred_log"] -vl["target_log"]
    vl["sq_error"]= vl["error"]**2
    print(f"\nобщий rmsle: {rmsle(yvl,pr):.4f}")

    #  по дням недели
    fig,ax= plt.subplots()
    dw= vl.groupby("dayofweek")["sq_error"].mean().pow(0.5)
    ax.bar(dw.index,dw.values,color=GREEN,edgecolor="black")
    ax.set_title("RMSLE по дням недели")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Пн","Вт","Ср","Чт","Пт","Сб","Вс"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"05_error_by_dow.png",dpi=100)
    plt.close()
    print("  сохраняем 05_error_by_dow.png")

    #  по промо
    fig,ax= plt.subplots()
    pm= vl.groupby("onpromotion")["sq_error"].mean().pow(0.5)
    ax.bar(["без промо","промо"],pm.values,color=[GRAY,GREEN],edgecolor="black")
    ax.set_title("RMSLE: промо vs без промо")
    for i,v in enumerate(pm.values):
        ax.text(i,v +0.005,f"{v:.4f}",ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"06_error_by_promo.png",dpi=100)
    plt.close()
    print("  сохраняем 06_error_by_promo.png")

    #  остатки
    fig,ax= plt.subplots()
    ax.hist(vl["error"],bins=80,edgecolor="black",alpha=0.8,color=PURPLE)
    ax.axvline(0,color=RED,linestyle="--")
    ax.set_title("Распределение остатков (pred - true)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"07_residuals.png",dpi=100)
    plt.close()
    print("  сохраняем 07_residuals.png")

    #  по магазинам
    fig,ax= plt.subplots(figsize=(12,6))
    se= vl.groupby("store_nbr")["sq_error"].mean().pow(0.5).sort_values()
    ax.barh(range(len(se)),se.values,color=GREEN_D,edgecolor="black")
    ax.set_yticks(range(len(se)))
    ax.set_yticklabels(se.index)
    ax.set_title("RMSLE по магазинам")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"08_error_by_store.png",dpi=100)
    plt.close()
    print("  сохраняем 08_error_by_store.png")

    print("\nтоп-5 магазинов с наибольшей ошибкой:")
    print(se.tail(5))
    print("\nтоп-5 магазинов с наименьшей ошибкой:")
    print(se.head(5))

    mt= {
        "overall_rmsle": rmsle(yvl,pr),
        "rmsle_no_promo": pm.iloc[0],
        "rmsle_promo": pm.iloc[1],
        "mean_error": vl["error"].mean(),
        "std_error": vl["error"].std(),
        }
    pd.DataFrame([mt]).to_csv(RESULTS_DIR/"error_analysis_metrics.csv",index=False)
    print(f"\nметрики: {RESULTS_DIR/'error_analysis_metrics.csv'}")


if __name__ == "__main__":
    main()