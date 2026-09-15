"""полная версия catboost: 10 млн строк,depth 8,1500 итераций"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor,Pool
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


def rmsle(yt,yp):
    return   np.sqrt(np.mean((yp -yt)**2))


def main():
    print("catboost full")
    df= pd.read_parquet(DATA_DIR/"features.parquet")
    print(f"загружено: {df.shape}")

    tr= df[df["date"] <= TRAIN_END]
    vl= df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]

    N_TRAIN= 10_000_000
    if len(tr) > N_TRAIN:
        tr= tr.sample(n=N_TRAIN,random_state=42).reset_index(drop=True)

    print(f"  train (subsample): {tr.shape}")
    print(f"  val (полный):      {vl.shape}")

    Xtr= tr[FEATURES].copy()
    ytr= tr["target_log"].values
    Xvl= vl[FEATURES].copy()
    yvl= vl["target_log"].values

    for col in CAT_FEATURES:
        Xtr[col]= Xtr[col].astype(str)
        Xvl[col]= Xvl[col].astype(str)

    tr_pool= Pool(Xtr,ytr,cat_features=CAT_FEATURES)
    vl_pool= Pool(Xvl,yvl,cat_features=CAT_FEATURES)

    print("\nобучаем catboost")
    m= CatBoostRegressor(
        iterations= 1500,
        learning_rate= 0.05,
        depth= 8,
        l2_leaf_reg= 3,
        loss_function= "RMSE",
        eval_metric= "RMSE",
        random_seed= 42,
        od_type= "Iter",
        od_wait= 50,
        verbose= 50,
        task_type= "CPU",
        thread_count= -1,
        subsample= 0.8,
        bootstrap_type= "Bernoulli",
        save_snapshot= True,
        snapshot_file= str(RESULTS_DIR/"catboost_full_snapshot.cbs"),
        snapshot_interval= 100,
        allow_writing_files= True,
                         )

    m.fit(tr_pool,eval_set=vl_pool,use_best_model=True)

    print("\nсчитаем метрики")
    pr_log= m.predict(vl_pool)
    rmsle_val= rmsle(yvl,pr_log)
    print(f"rmsle (val,log-пространство): {rmsle_val:.4f}")

    mp= RESULTS_DIR/"catboost_model_full.cbm"
    m.save_model(str(mp))
    print(f"модель: {mp}")

    imp= pd.DataFrame({
        "feature": FEATURES,
        "importance": m.get_feature_importance(tr_pool),
                      }).sort_values("importance",ascending=False)
    imp.to_csv(RESULTS_DIR/"catboost_importance_full.csv",index=False)
    print("\ntop-15 фичей:")
    print(imp.head(15).to_string(index=False))

    pd.DataFrame([{"model": "CatBoost_full","RMSLE": rmsle_val}]).to_csv(
        RESULTS_DIR/"catboost_results_full.csv",index=False
    )
    print(f"\nсохраняем {RESULTS_DIR/'catboost_results_full.csv'}")


if __name__ == "__main__":
    main()