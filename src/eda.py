"""разведочный анализ данных (EDA)"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR,RESULTS_DIR

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"]= (14,6)
plt.rcParams["font.size"]= 11


GREEN= "#2ecc71"
GREEN_D= "#27ae60"
GREEN_L= "#7bed9f"
PURPLE= "#9b59b6"
PURPLE_D= "#8e44ad"
PURPLE_L= "#a29bfe"
GRAY= "#95a5a6"
RED= "#e74c3c"


def load_data():
    print("читаем train_subsample.parquet")
    tr= pd.read_parquet(DATA_DIR/"train_subsample.parquet")
    print(f"  train: {tr.shape}")
    return   tr


def basic_stats(tr):
    print("\nбазовая статистика")
    print(f"период: {tr['date'].min()} -> {tr['date'].max()}")
    print(f"магазинов: {tr['store_nbr'].nunique()}")
    print(f"товаров: {tr['item_nbr'].nunique()}")
    print(f"средние: {tr['unit_sales'].mean():.3f}")
    print(f"медиана: {tr['unit_sales'].median():.3f}")
    print(f"доля нулевых: {(tr['unit_sales']== 0).mean()*100:.2f}%")
    print(f"доля промо: {tr['onpromotion'].mean()*100:.2f}%")


def plot_dist(tr):
    print("\nстроим распределение")
    fig,ax= plt.subplots(1,2,figsize=(14,5))
    ax[0].hist(tr["unit_sales"].clip(upper=50),bins=50,edgecolor="black",alpha=0.8,color=GREEN)
    ax[0].set_title("Распределение продаж (обрезано на 50)")
    ax[1].hist(np.log1p(tr["unit_sales"]),bins=50,edgecolor="black",alpha=0.8,color=PURPLE)
    ax[1].set_title("Распределение log1p(unit_sales)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"01_sales_distribution.png",dpi=100,bbox_inches="tight")
    plt.close()
    print("  сохраняем 01_sales_distribution.png")


def plot_season(tr):
    print("\nстроим сезонность")
    df= tr.copy()
    df["dayofweek"]= df["date"].dt.dayofweek
    df["month"]= df["date"].dt.month
    df["year"]= df["date"].dt.year
    fig,ax= plt.subplots(2,2,figsize=(14,10))
    dw= df.groupby("dayofweek")["unit_sales"].mean()
    ax[0,0].bar(dw.index,dw.values,color=GREEN,edgecolor="black")
    ax[0,0].set_title("Средние продажи по дням недели")
    ax[0,0].set_xticks(range(7))
    ax[0,0].set_xticklabels(["Пн","Вт","Ср","Чт","Пт","Сб","Вс"])
    mo= df.groupby("month")["unit_sales"].mean()
    ax[0,1].bar(mo.index,mo.values,color=PURPLE,edgecolor="black")
    ax[0,1].set_title("Средние продажи по месяцам")
    dy= df.groupby("date")["unit_sales"].sum()
    ax[1,0].plot(dy.index,dy.values,linewidth=0.5,color=GREEN_D)
    ax[1,0].set_title("Общий тренд продаж по дням")
    yr= df.groupby("year")["unit_sales"].mean()
    ax[1,1].bar(yr.index,yr.values,color=PURPLE_D,edgecolor="black")
    ax[1,1].set_title("Средние продажи по годам")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"02_seasonality.png",dpi=100,bbox_inches="tight")
    plt.close()
    print("  сохраняем 02_seasonality.png")


def plot_promo(tr):
    print("\nстроим эффект промо")
    fig,ax= plt.subplots(1,2,figsize=(14,5))
    ps= tr.groupby("onpromotion")["unit_sales"].agg(["mean","median","count"])
    print("\nпродажи по промо:")
    print(ps)
    ax[0].bar(["без промо","промо"],ps["mean"].values,color=[GRAY,GREEN],edgecolor="black")
    ax[0].set_title("Средние продажи: без промо vs промо")
    for i,v in enumerate(ps["mean"].values):
        ax[0].text(i,v +0.1,f"{v:.2f}",ha="center")
    df= tr.copy()
    df["dayofweek"]= df["date"].dt.dayofweek
    pd_= df.groupby("dayofweek")["onpromotion"].mean()*100
    ax[1].bar(pd_.index,pd_.values,color=PURPLE,edgecolor="black")
    ax[1].set_title("Доля промо по дням недели")
    ax[1].set_xticks(range(7))
    ax[1].set_xticklabels(["Пн","Вт","Ср","Чт","Пт","Сб","Вс"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"03_promo_effect.png",dpi=100,bbox_inches="tight")
    plt.close()
    print("  сохраняем 03_promo_effect.png")


def top_si(tr):
    print("\nтоп магазинов и товаров")
    fig,ax= plt.subplots(1,2,figsize=(14,6))
    ts= tr.groupby("store_nbr")["unit_sales"].sum().sort_values(ascending=False).head(10)
    ax[0].barh(range(len(ts)),ts.values,color=GREEN,edgecolor="black")
    ax[0].set_yticks(range(len(ts)))
    ax[0].set_yticklabels(ts.index)
    ax[0].set_title("Топ-10 магазинов")
    ax[0].invert_yaxis()
    ti= tr.groupby("item_nbr")["unit_sales"].sum().sort_values(ascending=False).head(10)
    ax[1].barh(range(len(ti)),ti.values,color=PURPLE,edgecolor="black")
    ax[1].set_yticks(range(len(ti)))
    ax[1].set_yticklabels(ti.index)
    ax[1].set_title("Топ-10 товаров")
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"04_top_stores_items.png",dpi=100,bbox_inches="tight")
    plt.close()
    print("  сохраняем 04_top_stores_items.png")


def main():
    print("eda")
    tr= load_data()
    basic_stats(tr)
    plot_dist(tr)
    plot_season(tr)
    plot_promo(tr)
    top_si(tr)


if __name__ == "__main__":
    main()