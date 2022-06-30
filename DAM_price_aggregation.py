
# Day ahead market houry price aggregation to monthly price 
import os
import numpy as np
import pandas as pd

working_dir = 'C:/2021/Sloan Project/1. Data/ERCOT/2 Day Ahead Market/'
os.chdir(working_dir)
years = np.arange(2011,2017,1)

file = "rpt.00013060.0000000000000000.DAMLZHBSPP_2011.xlsx"
xl = pd.ExcelFile(file)
months = xl.sheet_names  # sheet names: Jan, Feb, Mar,..., Dec

M = ['Jan', 'Feb',  'Mar',  'Apr', 'May',  'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] # month labels
df_sum = pd.DataFrame(columns= M)

for y in years:
    file = "rpt.00013060.0000000000000000.DAMLZHBSPP_" + str(y) + ".xlsx"
    for m in range(len(months)):
        df = pd.read_excel(open(file, 'rb'), sheet_name= months[m])  # read excel speadsheet:m
        avg_p = df.groupby(["Settlement Point"]).mean()              # calculatet the average price
        df_sum[M[m]] = avg_p                                            # save the average price
    df_sum.to_csv("price_" + str(y) + ".csv")             # unit:$/MWh
        
        
# the sheet names of the excel files changed so the code is slightly different
years = np.arange(2017,2022,1)
for y in years:
    file = "rpt.00013060.0000000000000000.DAMLZHBSPP_" + str(y) + ".xlsx"
    for m in range(len(months)):
        df = pd.read_excel(open(file, 'rb'), sheet_name= M[m])  # read excel speadsheet:m
        avg_p = df.groupby(["Settlement Point"]).mean()              # calculatet the average price
        df_sum[M[m]] = avg_p                                            # save the average price
    df_sum.to_csv("price_" + str(y) + ".csv")             # unit:$/MWh
        