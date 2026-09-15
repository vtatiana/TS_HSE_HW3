"""Главный скрипт пайплайна."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_script(name, script_path):
    print("\n" + "=" * 70)
    print(f"  ЗАПУСК: {name}")
    print("=" * 70)
    t0 = time.time()
    result = subprocess.run([PYTHON, str(script_path)], cwd=SRC_DIR)
    print(f"\n[{name}] {'OK' if result.returncode == 0 else 'FAILED'} за {time.time()-t0:.1f} сек")
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-lstm", action="store_true")
    parser.add_argument("--only-prepare", action="store_true")
    args = parser.parse_args()

    from config import DATA_DIR
    if not (DATA_DIR / "train_subsample.parquet").exists():
        run_script("prepare_data", SRC_DIR / "prepare_data.py")
    if not (DATA_DIR / "features.parquet").exists():
        run_script("features", SRC_DIR / "features.py")

    if args.only_prepare:
        return

    run_script("EDA", SRC_DIR / "eda.py")
    run_script("Бейзлайны", SRC_DIR / "baselines.py")
    run_script("CatBoost", SRC_DIR / "catboost_full.py")
    if args.with_lstm:
        run_script("LSTM", SRC_DIR / "lstm_model.py")

    print("\n=== ПАЙПЛАЙН ЗАВЕРШЁН ===")


if __name__ == "__main__":
    main()