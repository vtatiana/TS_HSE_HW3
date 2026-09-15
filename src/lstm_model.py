"""LSTM на лагах для прогноза продаж (уровень store-item)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import DATA_DIR, RESULTS_DIR

TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2017-08-15"

SEQ_LEN = 30
BATCH_SIZE = 1024
EPOCHS = 10
LR = 5e-4
PATIENCE = 2
DEVICE = "cpu"

FEATURES = [
    "store_nbr", "item_nbr", "family", "city", "state", "type", "cluster", "class", "perishable",
    "onpromotion", "promo_count_store", "promo_ratio_store",
    "oil", "is_holiday", "transactions",
    "dayofweek", "month", "day", "year", "weekofyear", "dayofyear",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]
N_FEATURES = len(FEATURES)


class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def build_sequences(df, feature_cols, seq_len):
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
    X_list, y_list = [], []
    for (store, item), group in df.groupby(["store_nbr", "item_nbr"], sort=False):
        if len(group) <= seq_len:
            continue
        feats = group[feature_cols].values.astype(np.float32)
        target = group["target_log"].values.astype(np.float32)
        for i in range(seq_len, len(group)):
            X_list.append(feats[i - seq_len:i])
            y_list.append(target[i])
    return np.array(X_list), np.array(y_list)


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def main():
    print("=" * 60)
    print("LSTM")
    print("=" * 60)

    df = pd.read_parquet(DATA_DIR / "features.parquet")
    print(f"Загружено: {df.shape}")

    df = df.sort_values("date").reset_index(drop=True)
    df["oil"] = df["oil"].ffill().bfill().fillna(0)

    for col in FEATURES:
        if df[col].dtype == "category" or df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes.astype("float32")
    df[FEATURES] = df[FEATURES].astype("float32").fillna(0)

    # Стандартизация
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_mask = df["date"] <= TRAIN_END
    df.loc[train_mask, FEATURES] = scaler.fit_transform(df.loc[train_mask, FEATURES])
    df.loc[~train_mask, FEATURES] = scaler.transform(df.loc[~train_mask, FEATURES])

    train_df = df[df["date"] <= TRAIN_END].copy()
    val_df = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()

    np.random.seed(42)
    n_pairs_train = 2000
    all_pairs = train_df[["store_nbr", "item_nbr"]].drop_duplicates()
    sample_pairs = all_pairs.sample(n=min(n_pairs_train, len(all_pairs)), random_state=42)

    train_df = train_df.merge(sample_pairs, on=["store_nbr", "item_nbr"])
    val_df = val_df.merge(sample_pairs, on=["store_nbr", "item_nbr"])
    print(f"Train: {train_df.shape}, Val: {val_df.shape}")

    print("Строим train-последовательности...")
    X_train, y_train = build_sequences(train_df, FEATURES, SEQ_LEN)
    print(f"X_train: {X_train.shape}")

    print("Строим val-последовательности...")
    X_val, y_val = build_sequences(val_df, FEATURES, SEQ_LEN)
    print(f"X_val: {X_val.shape}")

    assert not np.isnan(X_train).any(), "NaN в X_train!"
    assert not np.isnan(X_val).any(), "NaN в X_val!"
    print("Проверка NaN: OK")

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(N_FEATURES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\nОбучение...")
    best_rmsle = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(y_train)

        model.eval()
        val_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                val_preds.append(model(xb.to(DEVICE)).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_rmsle = rmsle(y_val, val_preds)

        marker = ""
        if val_rmsle < best_rmsle:
            best_rmsle = val_rmsle
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " ★"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{EPOCHS} | train_mse={train_loss:.5f} | val_rmsle={val_rmsle:.4f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping (нет улучшений {PATIENCE} эпох)")
            break

    model.load_state_dict(best_state)
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            val_preds.append(model(xb.to(DEVICE)).cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmsle = rmsle(y_val, val_preds)
    print(f"\nRMSLE (val): {final_rmsle:.4f}")

    torch.save(model.state_dict(), RESULTS_DIR / "lstm_model.pt")
    pd.DataFrame([{"model": "LSTM", "RMSLE": final_rmsle}]).to_csv(
        RESULTS_DIR / "lstm_results.csv", index=False
    )
    print(f"Сохранено: {RESULTS_DIR / 'lstm_results.csv'}")


if __name__ == "__main__":
    main()