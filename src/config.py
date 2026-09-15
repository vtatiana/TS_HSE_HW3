"""Конфигурация проекта."""
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Колонки
DATE_COL = "date"
TARGET = "unit_sales"
STORE_COL = "store_nbr"
ITEM_COL = "item_nbr"

# Периоды для валидации
TRAIN_START = "2013-01-01"
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2017-08-15"
TEST_START = "2017-08-16"
TEST_END = "2017-08-31"

# Подвыборка (чтобы поместилось в 16 ГБ RAM)
N_STORES = 10
N_ITEMS = 2000

# Случайное зерно
RANDOM_STATE = 42