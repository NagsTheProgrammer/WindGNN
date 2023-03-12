# 230227 - ENEL 645 Final Project - DatasetAggregator.py - Group 27

import numpy as np
import pyomo
from pyomo.environ import*
from pyomo.opt import SolverFactory
import pandas as pd
import math

def main():
    df_2020_1 = pd.read_csv(r'ACISHourlyData-20200101-20200630-PID181534737.csv', encoding='latin1')
    df_2020_2 = pd.read_csv(r'ACISHourlyData-20200701-20201231-PID181622739.csv', encoding='latin1')
    df_2021_1 = pd.read_csv(r'ACISHourlyData-20210101-20210630-PID181703141.csv', encoding='latin1')
    df_2021_2 = pd.read_csv(r'ACISHourlyData-20210701-20211231-PID181746823.csv', encoding='latin1')
    df_2022_1 = pd.read_csv(r'ACISHourlyData-20220101-20220630-PID181912069.csv', encoding='latin1')
    df_2022_2 = pd.read_csv(r'ACISHourlyData-20220701-20221231-PID181944162.csv', encoding='latin1')

    df_2020_2022 = pd.concat([df_2020_1, df_2020_2, df_2021_1, df_2021_2, df_2022_1, df_2022_2])
    df_2020_2022.to_csv(r'ACISHourlyData-20200101-20221231.csv')

if __name__ == "__main__":
    main()