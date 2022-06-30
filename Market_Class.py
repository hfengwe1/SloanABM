import os
import numpy as np
import pandas as pd
from scipy import interpolate
import numpy_financial as npf
import random as rd


class Market(object):

    def __init__(self, file):
        self.file = file
        self.data_dir = "C:\\2021\\Sloan Project\\ABM_Model\\Data" # data directory path; a class attribute
        self.file_path = os.path.join(self.data_dir, self.file) # the data file path 

    def read_data(self):  # read data in dataframe format
        df = pd.read_csv(self.file_path, header= 0, delimiter = ",") # read text file.
        return df
    def price_predict(self, year, mu, std):  # predcit price of a future year
        price = 10*(0.1212*(year-1990+1) + 5.9033 + np.random.normal(mu,std)) 
        # a linear model derived from regression of the observed data and a random error term
        return price

class Read_Future_Data(object):

    def __init__(self, file):
        self.file = file
        self.data_dir = "C:\\2021\\Sloan Project\\ABM_Model\\Data\\Future_Data" # data directory path; a class attribute
        self.file_path = os.path.join(self.data_dir, self.file) # the data file path 

    def read_data(self):  # read data in dataframe format
        df = pd.read_csv(self.file_path, header= 0, delimiter = ",") # read text file.
        return df
    def price_predict(self, year, mu, std):  # predcit price of a future year
        price = 10*(0.1212*(year-1990+1) + 5.9033 + np.random.normal(mu,std)) 
        # a linear model derived from regression of the observed data and a random error term
        return price

class generator():

    def __init__(self, name, cost, ls):
        self.name = name  # name of the generator
        self.cost = cost    # the invest. amount
        self.life_span = ls  # the life span of the generator


# Functions 
def pay_back_period(G_cash_flow, G_ls):
    a = np.arange(G_ls+1)
    b = np.cumsum(G_cash_flow)
    if max(b)> 0:
        f = interpolate.interp1d(b,a)
        PBP = float(f(0))
    else:
        PBP = 999 # bad investment

    return PBP

def NPV(G_IRR, G_cash_flow, G_ls):  # Gtype:0 Wind, 1 Solar, 2 NG
    npv = 0         # Net Present Worth
    for i in np.arange(1,G_ls+1):
        npv = npv + G_cash_flow[i]/((1+np.array(G_IRR))**int(i))
    return npv



#%% Technology Evaluation 

def Evaluation(p, cf, cost, G_ls, G_tech): 
    G = generator(G_tech, cost, int(G_ls))  # assined to a generator class (the cost ($/MW), the life span) of a generator
    # G_NPV = []
    # G_PbP = []
    # G_IRR = []
    x_invest = float(cost)  # assume  1MW investment for evaluation; the results can be easily scaled. 
    G.cf = cf  # capacity factor
    G.capacity = 1
    if G_tech == 'NG':
        variable_cost = 2.56 # $/MW EIA 2020 cost data; assume fixed 
        cash_flow = list(G.capacity*(p - variable_cost)*8640*G.cf*0.01*np.ones(G.life_span)) # has fuel and O&M costs
        cash_flow.insert(0,-x_invest) 
    else:      
        cash_flow = list(G.capacity*p*8640*G.cf*0.01*np.ones(G.life_span)) # no fuel cost and O&M cost
        cash_flow.insert(0,-x_invest)

    G.cash_flow = cash_flow   #save cash flow data in the object
    G.IRR =  round(npf.irr(cash_flow),4)  # internal rate of return

    npv = NPV(G.IRR, G.cash_flow, G.life_span)  # calculate Net Present Value (NPV)
    G.NPV = npv # save the data to object 
    # G.PBP = pay_back_period(G.cash_flow, G.life_span)   # save the data to object 
    # G_NPV.append(npv)
    # G_PbP.append(G.PbP*1)
    # G_IRR.append(G.IRR) 
    return G


#  Economy of Scale 
def f_cost_w(x,cost):  # cost function that consider economic of scale
    if x <= 1000:
        total_cost = x*cost*(1-np.exp(-6 + 3.7*x/1000))   # exponential function for economic of scale 
    elif x > 1000:
        total_cost = x*cost*(1-0.1) # 10% discount for investments higher than 1000 MW
    return total_cost


# Python code to merge dict using update() method
def Merge(dict1, dict2):
    return(dict2.update(dict1))
     

