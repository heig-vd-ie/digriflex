"""Revised on: 30.10.2020, @author: MYI"""
### To Do lists:
## - Code of lab_test
##     + Make function of RT optimziation consistent for using in lab_test
##     + Make function of DA optimization consistent for using in lab_test
## - Parameters of devices
## - Add Forecasting code
## - Qin measurements validation

# %% Cleaning console and variables
try:
    from IPython import get_ipython

    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass;

# %% Checking the Pyhton version
import os
import sys
import warnings

if sys.version_info[0] != 3 and sys.version_info[1] != 6:
    warnings.warn("Install Python version 3.6.")
# %% Management of libraries
try:
    import numpy as np
    import pandas as pd
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import pydgrid as grid
    import gurobipy as gp
    import scipy.io as sio
except:
    warnings.warn("Install required packages.")
import Functions.AuxiliaryFunctions as af
import Functions.OptimizationFunctions as opfunc

# %% Simulation program
if Mode == "Simulation":
    # Parameters of simulation
    Scen_num = 10
    Network_Name = "6BusesLaChappelle"
    Data_Name = "Data_Case0_6bus_Async_Unbalanced"
    Cases_Name = ['Robust, $1-\epsilon=0.97$']
    # ['Stochastic, $K=100$', 'Robust, $1-\epsilon=0.75$', 'Robust, $1-\epsilon=0.85$',
    #               'Robust, $1-\epsilon=0.90$', 'Robust, $1-\epsilon=0.95$', 'Robust, $1-\epsilon=0.97$', 
    #               'Robust, $1-\epsilon=1.00$']
    # Initialization of simulation
    grid_inp = af.grid_topology_sim(Network_Name, [])
    meas_inp = af.meas_load_time_series(Data_Name, grid_inp["grid_formers"])
    case_inp = af.cases_load_sim(Cases_Name, Scen_num)
    output_DF = pd.DataFrame()
    montec_res_DF = pd.DataFrame()
    meas = af.SE_time_series(grid_inp, meas_inp, "Without Control")
    # Run simulation
    for l in range(len(Cases_Name)):
        fore_inp = af.forecast_defining(meas_inp["value"]["P"][1], meas_inp["value"]["P"], meas_inp["value"]["Q"])
        if case_inp[l]['Robust_Programming']:
            DA_result, output_DF = opfunc.DA_Optimization_Robust(Cases_Name[l], case_inp[l], grid_inp, meas_inp,
                                                                 fore_inp, output_DF)
        else:
            DA_result, output_DF = opfunc.DA_Optimization(Cases_Name[l], case_inp[l], grid_inp, meas_inp, fore_inp,
                                                          output_DF)
        for om in range(case_inp[l]["MonteCarlo_Scen"]):
            res = 0
            for t in range(meas_inp["Nt"]):
                rt_meas_inp = af.rt_simulation(grid_inp, meas_inp, fore_inp, DA_result, t)
                DA_result, dres = opfunc.RT_Optimization(rt_meas_inp, meas_inp, grid_inp, DA_result)
                res = res + dres
            montec_res_DF.loc[om, Cases_Name[l]] = 100 * res / meas_inp["Nt"]
    # %% figuring
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = (6, 8)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    fig, ax = plt.subplots(1, 1)
    output_DF.loc['obj'].plot.bar()
    ax.set_ylabel('objective (\$)', fontsize=18)
    plt.yticks(rotation='vertical')
    fig.savefig('./Figures/Objective.pdf', bbox_inches='tight')
    fig, ax = plt.subplots(1, 1)
    montec_res_DF.plot.box(ax=ax, whis=1, showfliers=False, rot=90)
    plt.yticks(rotation='vertical')
    ax.set_ylabel('Deviation probability (%)', fontsize=18)
    fig.savefig('./Figures/res2.pdf', bbox_inches='tight')
# %% Lab program functions
dir_name = os.getcwd()
Vec_Inp = open(dir_name + "/Data/test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
Vec_Out = opfunc.pf_test3(Vec_Inp)
print(Vec_Out)
