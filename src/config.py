"""Общая конфигурация проекта."""
from pathlib import Path

#  базовые пути, считаем от корня проекта
BASE_DIR= Path(__file__).parent.parent
DATA_DIR= BASE_DIR/"data"
RESULTS_DIR= BASE_DIR/"results"
SUBMISSIONS_DIR= BASE_DIR/"submissions"

#  названия колонок в Favorita
DATE_COL= "date"
STORE_COL= "store_nbr"
ITEM_COL= "item_nbr"
TARGET= "unit_sales"

#  разбивка по времени
TRAIN_END= "2016-12-31"
VAL_START= "2017-01-01"
VAL_END= "2017-08-15"

#  полный датасет не влезает в оперативку — берём кусок
N_STORES= 10    #  из 54
N_ITEMS= 2000   #  из 4100

RANDOM_STATE= 42