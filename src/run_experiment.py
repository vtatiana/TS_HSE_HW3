"""главный скрипт пайплайна"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR= Path(__file__).parent
PYTHON= sys.executable


def run_script(nm,script_path):
    print(f"\n  запуск: {nm}")
    t0= time.time()
    res= subprocess.run([PYTHON,str(script_path)],cwd=SRC_DIR)
    print(f"\n[{nm}] {'ok' if res.returncode== 0 else 'FAILED'} за {time.time()-t0:.1f} сек")
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--with-lstm",action="store_true")
    parser.add_argument("--only-prepare",action="store_true")
    args= parser.parse_args()

    from config import DATA_DIR
    if not (DATA_DIR/"train_subsample.parquet").exists():
        run_script("prepare_data",SRC_DIR/"prepare_data.py")
    if not (DATA_DIR/"features.parquet").exists():
        run_script("features",SRC_DIR/"features.py")
    if args.only_prepare:
        return
    run_script("eda",SRC_DIR/"eda.py")
    run_script("бейзлайны",SRC_DIR/"baselines.py")
    run_script("catboost",SRC_DIR/"catboost_full.py")
    if args.with_lstm:
        run_script("lstm",SRC_DIR/"lstm_model.py")

    print("\nпайплайн завершён")


if __name__ == "__main__":
    main()