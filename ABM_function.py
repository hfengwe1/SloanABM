# ABM Functions
import os
# from sqlite3 import DatabaseError
import numpy as np
import pandas as pd
# from scipy import interpolate
# import matplotlib.pyplot as plt
from Market_Class import Evaluation
# from Price_predict import future_price
import scipy.stats
cwd = 'c:/School/SloanProject/ABM_Model/SloanABM'  # current folder: 'c:\\2021\\Sloan Project\\ABM_Model\\SloanABM'

# create a directory if not exist
def create_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:  
    # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory is created!")
    return

# Generate samples for agnets' perception about cost, REC, and risk (hesitation)
def generating_ABM_samples(renewable, n_agt, date_now, k):
    # create a directory if not exist
    new_dir = os.path.join(cwd,"00 Results","Calibration", "Calibration_" + date_now )
    # create_dir(new_dir) 

    # generate random distribution for renewable costs, risks, company sizes
    wind_cost_dist = list(np.random.normal(0.84,0.15,renewable)) + list(np.ones([1,n_agt-renewable])[0])  # the cost adjustment for renewable companies
    solar_cost_dist = list(np.random.normal(0.4,0.1,renewable)) + list(np.ones([1,n_agt-renewable])[0]) # the cost adjustment for renewable companies
    solar_cost_dist = [abs(item) for item in solar_cost_dist]  # make negative values positive
    ng_cost_dist = list(np.ones([1,renewable])[0]) + list(np.random.normal(1,0.0,n_agt-renewable)) # the cost adjustment for renewable companies
    mu, sigma = 1, 0.15 # mean and standard deviation
    s = np.random.normal(mu, sigma, n_agt)  # small company recieved more financial incentive
    s = [abs(item) for item in s]  # make negative values positive

    rec_dist = np.sort(s)[::-1] 
    agt_risk_f = np.random.normal(0.8,0.2,n_agt)  # agents are risk-adverse
    sample_agt = np.random.gamma(shape=1.3, scale= 3, size = n_agt)  
    agt_size_dist = np.sort(sample_agt/sum(sample_agt))  # agent's ranked captial distribution in the market       
# save the samples
    random_cost_dict = {"NG": ng_cost_dist , "wind": wind_cost_dist, "solar": solar_cost_dist} # create a dictionary
    agt_size_risk_dict = {"risk": agt_risk_f, "size": agt_size_dist, "Rec_dist": rec_dist}  # agents' risk attitudes and sizes

    df1 = pd.DataFrame.from_dict(random_cost_dict)   
    file1 = os.path.join(new_dir, "Cost_dist_" + date_now + "_"+ str(k) +".csv")
    df1.to_csv (file1, index = False, header=True)

    df2 = pd.DataFrame.from_dict(agt_size_risk_dict)   
    file2 = os.path.join(new_dir, "agt_size_risk_dict_" + date_now + "_"+ str(k) +".csv")
    df2.to_csv (file2, index = False, header=True)
    return random_cost_dict, agt_size_risk_dict

# Utilize an agent's supply curve to estimate future market price
# a random error follows normal distribution
def Supply(demand_t, c1,c2,c3): # Agnet's supply function
    # tot_ca: total capacity in the market
    
    agt_err = np.random.normal(0,1)*c3 # agent's own projection error
    p = c1*(demand_t) + c2 + agt_err
    return p 

# The function returns a technology type for investment (with the highest IRR) and the corresponding IRR
def ZoneInvest(new_P, rec_p, df_G):
    # d_percent = [0.04,0.06,0.27,0.33,0.13,0.12]    # the percentage to total demand based on 2020 data
    # D_zone_percent = pd.DataFrame(d_percent, index = LZ, columns = ['D_perc'])  # the zone demand as a percentage to the total demand
    # row_list = [] # empty row
    G_name = df_G.columns  # the names of the generation techologies. 
    G_IRR = []   # internal rate of return
    G_NPV = []   # net present value
    G_PBP = []   # payback period
    # New_Capacity = {'NG':0,'Wind':0,'Solar':0} # new capacity [NG, Wind, Solar]
    # New_Capacity = [0,0,0]  # new capacity; ng, wind, solar
    for G_index in range(len(G_name)):
        G_tech = G_name[G_index]
        if G_index == 0: 
            p = new_P # price prediction
        else:
            p = (new_P+rec_p) # price $ with carbon credit price
        cf = df_G[G_tech]['CF'] # capacity factor
        cost = df_G[G_tech]['Cost']  # installation cost
        G_ls = df_G[G_tech]['LS']  # life span

        G = Evaluation(p, cf, cost, G_ls, G_tech)  # evaluation results: IRR, Pay-back-period (PBP), NPV
        # investment need to be decided here. 

        G_IRR.append(G.IRR) # create a list of IRR, PBP, and NPV
        # G_PBP.append(G.PBP) 
        G_NPV.append(G.NPV)
    # print(G_IRR)
    max_IRR = max(G_IRR) # maximum IRR rate of the generation technologies   
    G_i = G_IRR.index(max_IRR)  # the index of the technology invested
    G_tech = G_name[G_i]        # the technology invested
    return G_tech, max_IRR

# This function determines the technology and amount of investment in each load zone (LZ).
# Based on the Max_IRR determined from "ZoneInvest", if there are multiple LZs have IRR> threshold
# Agents will invest in all of them proportional to the IRRs. 
def Agent_Investment(IRR_t_array,capacity_deficit, IRR_threshold):
    LZ = ['LZ_AEN', 'LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST'] 
    lz_index = list(np.where(IRR_t_array > IRR_threshold)[0]) 
    # only invest in LZs having IRR >= than the IRR_threshold

    lz_irr_perc = np.zeros([1,len(LZ)])[0]

    if len(lz_index) > 0:  # if there are multiple Load Zones that exceed the IRR threshold
        lz_irr_perc[lz_index] = IRR_t_array[lz_index]/sum(IRR_t_array[lz_index])  
        # the agent invests in them proportional to the IRRs
    else:
        lz_index = list(np.where(IRR_t_array == IRR_t_array.max())[0])
        lz_irr_perc[lz_index] = 1
    lz_invest = {}  # a dictionary
    for i in range(len(LZ)):  # create an investment dictionary 
        if i in lz_index:  
            lz_invest[LZ[i]] = lz_irr_perc[i]*capacity_deficit # make the investment
        else:
            lz_invest[LZ[i]] = 0.0  # no investment
    return lz_invest # a dataframe

# %% Aggregate the investment by fuel, agent and year (Calibration) 
def aggregation_fuel(n_agt, result):
    G_name =['NG','Wind','Solar']             # generation types
    row_list = []  
    for agt_i in range(n_agt):
        subset = result[result["Agt_I"] == str(agt_i)]
        subset_invest = subset.iloc[:,0:6]  # the investment in Load Zones
        subset_fuel = subset.iloc[:,6:]     # the year, agent id, fuel invested in Load Zones
        
        for y in np.arange(len(subset_fuel.index)):  # loop through year:2012- 2020
            invest_fuel = {"Year": y+2012, "Agt_I": agt_i, 'NG': 0.0,'Wind': 0.0,'Solar': 0.0}  # initial value: [NG, Wind, Solar]            
    
            for lz in range(6):    # there are five load zones
                if subset_fuel.iloc[y, lz+2] == "NG": # the tech is NG
                    invest_fuel["NG"] +=  subset_invest.iloc[y, lz] # initial value     
                elif subset_fuel.iloc[y, lz+2] == "Wind": # the tech is Solar       
                    invest_fuel["Wind"] +=  subset_invest.iloc[y, lz] # initial value     
                elif subset_fuel.iloc[y, lz+2] == "Solar": # the tech is Solar       
                    invest_fuel["Solar"] +=  subset_invest.iloc[y, lz] # initial value 
                else: 
                    print("something went wrong")
            row_list.append(invest_fuel)                   

    fuel_agt = pd.DataFrame(row_list)  # create a dataframe for investment agrregated by fuel types
    fuel_yr = fuel_agt.groupby("Year").sum()   # sum the investment by fuel types for each year
    del fuel_yr["Agt_I"]                      # delete the Agt ID column
    fuel_yr["Total"] = fuel_yr.sum(axis=1) # add a column for the total invesetment

    return fuel_agt, fuel_yr

# aggregate the simulation results by fuel type, agent, and years (future simulation) - this can be combined with the function above. 
# Fore convenicence, I created this one only because of the "year" column. 
def aggregation_fuel_future(n_agt, result, Retire):
    G_name =['NG','Wind','Solar']             # generation types
    row_list = []  
    for agt_i in range(n_agt):
        subset = result[result["Agt_I"] == str(agt_i)]
        subset_invest = subset.iloc[:,0:6]  # the investment in Load Zones
        subset_fuel = subset.iloc[:,6:]     # the year, agent id, fuel invested in Load Zones
        
        for i in np.arange(len(subset_fuel.index)):  # loop through year:2012- 2020
            y = i + 2021
            invest_fuel = {"Year": y, "Agt_I": agt_i, 'NG': 0.0,'Wind': 0.0,'Solar': 0.0}  # initial value: [NG, Wind, Solar]            
    
            for lz in range(6):    # there are five load zones
                if subset_fuel.iloc[i, lz+2] == "NG":                 # the tech is NG
                    invest_fuel["NG"] += subset_invest.iloc[i, lz] # initial value     
                elif subset_fuel.iloc[i, lz+2] == "Wind":             # the tech is Solar       
                    invest_fuel["Wind"] +=  subset_invest.iloc[i, lz] # initial value     
                elif subset_fuel.iloc[i, lz+2] == "Solar":            # the tech is Solar       
                    invest_fuel["Solar"] +=  subset_invest.iloc[i, lz] # initial value 
                else: 
                    print("something went wrong")
            row_list.append(invest_fuel)                   

    fuel_agt = pd.DataFrame(row_list)  # create a dataframe for investment agrregated by fuel types
    fuel_yr = fuel_agt.groupby("Year").sum()  # sum the investment by fuel types for each year
    del fuel_yr["Agt_I"]                      # delete the Agt ID column
    fuel_yr["Total"] = fuel_yr.sum(axis=1)    # add a column for the total invesetment
    # investment aggregated by Load Zones
    subset = result.iloc[:,0:7]               # the investment in Load Zones
    subset_yr = subset.groupby("Year").sum()  # sum the investment by fuel types for each year

    return fuel_agt, fuel_yr, subset_yr

# this function calculate the Pearson Correlation and P-values of the simulated and observed data
def Pearson_stat(fuel_yr, tot_new_ca, cap_hist):

    r_w, p_w = scipy.stats.pearsonr(fuel_yr['Wind'], cap_hist['Wind'])
    r_s, p_s = scipy.stats.pearsonr(fuel_yr['Solar'], cap_hist['Solar'])
    r_t, p_t = scipy.stats.pearsonr(tot_new_ca, cap_hist['Total'])
    r = {'Wind':r_w, 'Solar':r_s, 'Total':r_t}  # the Peason's Coefficient of Correlation
    p = {'Wind':p_w, 'Solar':p_s, 'Total':p_t}  # the p-value for hypothesis testing
    return r, p 

# this function calculates the KGE value of the simulated and observed data.
def KGE_stat(fuel_yr, tot_new_ca, cap_hist):

    wind_cumsum_hist = np.cumsum(cap_hist['Wind'])    # calculate the cumulative sum of Wind installation
    solar_cumsum_hist = np.cumsum(cap_hist['Solar'])  # calculate the cumulative sum of Wind installation
    wind_cumsum = np.cumsum(fuel_yr['Wind'])    # calculate the cumulative sum of simulated Wind installation
    solar_cumsum = np.cumsum(fuel_yr['Solar'])  # calculate the cumulative sum of simulated Solar installation
    ## Pearson Correlation Coefficient
    r_w, p_w = scipy.stats.pearsonr(wind_cumsum, wind_cumsum_hist)
    r_s, p_s = scipy.stats.pearsonr(solar_cumsum, solar_cumsum_hist)
    r_tot, p_tot = scipy.stats.pearsonr(tot_new_ca, cap_hist['Total'])
    ## Mean Ratios
    alpha_w = np.mean(wind_cumsum)/np.mean(wind_cumsum_hist)    # wind mean ratio
    alpha_s = np.mean(solar_cumsum)/np.mean(solar_cumsum_hist)  # solar mean ratio
    alpha_tot = np.mean(tot_new_ca)/np.mean(cap_hist['Total'])  # total generation capacity mean ratio
    ## Standard Deviation Ratios
    beta_w = np.std(wind_cumsum)/np.std(wind_cumsum_hist)      # wind standard deviation ratio
    beta_s = np.std(solar_cumsum)/np.std(solar_cumsum_hist)      # solar standard deviation ratio
    beta_tot = np.std(tot_new_ca)/np.std(cap_hist['Total'])      # total generation standard deviation ratio
    ## Kling-Gupta efficiency scores (KGE)
    KGE_w = 1 - np.sqrt((1-r_w)**2+(1-alpha_w)**2+(1-beta_w)**2)  # KGE for simulated Wind
    KGE_s = 1 - np.sqrt((1-r_s)**2+(1-alpha_s)**2+(1-beta_s)**2)  # KGE for simulated Wind
    KGE_tot = 1 - np.sqrt((1-r_tot)**2+(1-alpha_tot)**2+(1-beta_tot)**2)  # KGE for simulated Wind

    KGE = {'KGE_Wind':KGE_w, 'KGE_Solar':KGE_s, 'KGE_Total':KGE_tot}  # the Peason's Coefficient of Correlation
    p = {'p_value_Wind':p_w, 'p_value_Solar':p_s, 'p_value_Total':p_tot}  # the p-value for hypothesis testing

    return KGE, p 

# Get rid of brackets

def remove_brackets(dict):
    keys = dict.keys()  # get keys of the dictionary
    for i in keys:
        dict[i] = dict[i][0]
    return dict
    
