"""lstm на лагах для прогноза продаж (store-item)"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from config import DATA_DIR,RESULTS_DIR

TRAIN_END= "2016-12-31"
VAL_START= "2017-01-01"
VAL_END= "2017-08-15"

SEQ_LEN= 30
BATCH_SIZE= 1024
EPOCHS= 10
LR= 5e-4
PATIENCE= 2
DEVICE= "cpu"

FEATURES= [
    "store_nbr","item_nbr","family","city","state","type","cluster","class","perishable",
    "onpromotion","promo_count_store","promo_ratio_store",
    "oil","is_holiday","transactions",
    "dayofweek","month","day","year","weekofyear","dayofyear",
    "dow_sin","dow_cos","month_sin","month_cos",
           ]
N_FEATURES= len(FEATURES)


class SeqDataset(Dataset):
    def __init__(self,X_seq,y):
        self.X= torch.tensor(X_seq,dtype=torch.float32)
        self.y= torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return   len(self.y)

    def __getitem__(self,i):
        return   self.X[i],self.y[i]


class LSTMModel(nn.Module):
    def __init__(self,n_features,hidden=64,layers=2,dropout=0.2):
        super().__init__()
        self.lstm= nn.LSTM(n_features,hidden,layers,batch_first=True,dropout=dropout)
        self.dropout= nn.Dropout(dropout)
        self.fc= nn.Linear(hidden,1)

    def forward(self,x):
        out,_= self.lstm(x)
        out= self.dropout(out[:,-1,:])
        return   self.fc(out).squeeze(-1)


def build_seqs(df,feat_cols,seq_len):
    df= df.sort_values(["store_nbr","item_nbr","date"]).reset_index(drop=True)
    Xl,yl= [],[]
    for (store,item),grp in df.groupby(["store_nbr","item_nbr"],sort=False):
        if len(grp) <= seq_len:
            continue
        f= grp[feat_cols].values.astype(np.float32)
        tgt= grp["target_log"].values.astype(np.float32)
        for i in range(seq_len,len(grp)):
            Xl.append(f[i -seq_len:i])
            yl.append(tgt[i])
    return   np.array(Xl),np.array(yl)


def rmsle(yt,yp):
    return   np.sqrt(np.mean((yp -yt)**2))


def main():
    print("lstm")
    df= pd.read_parquet(DATA_DIR/"features.parquet")
    print(f"загружено: {df.shape}")

    df= df.sort_values("date").reset_index(drop=True)
    df["oil"]= df["oil"].ffill().bfill().fillna(0)

    for col in FEATURES:
        if df[col].dtype== "category" or df[col].dtype== object:
            df[col]= df[col].astype("category").cat.codes.astype("float32")
    df[FEATURES]= df[FEATURES].astype("float32").fillna(0)

    #  стандартизация: z-score по train
    from sklearn.preprocessing import StandardScaler
    sc= StandardScaler()
    tr_mask= df["date"] <= TRAIN_END
    df.loc[tr_mask,FEATURES]= sc.fit_transform(df.loc[tr_mask,FEATURES])
    df.loc[~tr_mask,FEATURES]= sc.transform(df.loc[~tr_mask,FEATURES])

    tr_df= df[df["date"] <= TRAIN_END].copy()
    vl_df= df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()

    np.random.seed(42)
    n_pairs= 2000
    all_p= tr_df[["store_nbr","item_nbr"]].drop_duplicates()
    samp_p= all_p.sample(n=min(n_pairs,len(all_p)),random_state=42)

    tr_df= tr_df.merge(samp_p,on=["store_nbr","item_nbr"])
    vl_df= vl_df.merge(samp_p,on=["store_nbr","item_nbr"])
    print(f"train: {tr_df.shape},val: {vl_df.shape}")

    print("строим train-последовательности")
    Xtr,ytr= build_seqs(tr_df,FEATURES,SEQ_LEN)
    print(f"X_train: {Xtr.shape}")

    print("строим val-последовательности")
    Xvl,yvl= build_seqs(vl_df,FEATURES,SEQ_LEN)
    print(f"X_val: {Xvl.shape}")

    assert not np.isnan(Xtr).any(),"NaN в X_train!"
    assert not np.isnan(Xvl).any(),"NaN в X_val!"
    print("проверка NaN: ok")

    tr_ld= DataLoader(SeqDataset(Xtr,ytr),batch_size=BATCH_SIZE,shuffle=True)
    vl_ld= DataLoader(SeqDataset(Xvl,yvl),batch_size=BATCH_SIZE,shuffle=False)

    m= LSTMModel(N_FEATURES).to(DEVICE)
    opt= torch.optim.Adam(m.parameters(),lr=LR)
    loss_fn= nn.MSELoss()

    print("\nобучение")
    best_r= float("inf")
    best_st= None
    pat= 0

    for ep in range(EPOCHS):
        m.train()
        tr_loss= 0.0
        for xb,yb in tr_ld:
            xb,yb= xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            pr= m(xb)
            ls= loss_fn(pr,yb)
            ls.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),max_norm=1.0)
            opt.step()
            tr_loss += ls.item()*len(yb)
        tr_loss /= len(ytr)

        m.eval()
        vl_pr= []
        with torch.no_grad():
            for xb,_ in vl_ld:
                vl_pr.append(m(xb.to(DEVICE)).cpu().numpy())
        vl_pr= np.concatenate(vl_pr)
        vl_r= rmsle(yvl,vl_pr)

        mark= ""
        if vl_r < best_r:
            best_r= vl_r
            best_st= {k: v.clone() for k,v in m.state_dict().items()}
            pat= 0
            mark= " ★"
        else:
            pat += 1

        print(f"  эпоха {ep+1}/{EPOCHS} | train_mse={tr_loss:.5f} | val_rmsle={vl_r:.4f}{mark}")

        if pat >= PATIENCE:
            print(f"  early stopping (нет улучшений {PATIENCE} эпох)")
            break

    m.load_state_dict(best_st)
    m.eval()
    vl_pr= []
    with torch.no_grad():
        for xb,_ in vl_ld:
            vl_pr.append(m(xb.to(DEVICE)).cpu().numpy())
    vl_pr= np.concatenate(vl_pr)
    fin_r= rmsle(yvl,vl_pr)
    print(f"\nrmsle (val): {fin_r:.4f}")

    torch.save(m.state_dict(),RESULTS_DIR/"lstm_model.pt")
    pd.DataFrame([{"model": "LSTM","RMSLE": fin_r}]).to_csv(
        RESULTS_DIR/"lstm_results.csv",index=False
    )
    print(f"сохраняем {RESULTS_DIR/'lstm_results.csv'}")


if __name__ == "__main__":
    main()