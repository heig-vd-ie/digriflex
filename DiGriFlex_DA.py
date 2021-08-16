"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
#### Importing packages
import os
import sys
import warnings
import pandas as pd
import numpy as np
import pickle
import rpy2.robjects as ro
import mysql.connector
import Functions_P.AuxiliaryFunctions as af
from datetime import datetime, timedelta
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from DiGriFlex_RT import access_data_rt

print(sys.version)

#### Defining meta parameters
dir_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/Desktop/DiGriFlex_Code"
python64_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/AppData/Local/Programs/Python/Python39/python.exe"
network_name = "Case_4bus_DiGriFlex"  # Defining the network
# network_name = "Case_LabVIEW"


def dayahead_digriflex(robust_par):
    data_rt = access_data_rt()
    t_end = data_rt.index[-1].floor('1d') - timedelta(hours=1)
    t_from = t_end - timedelta(days=2) + timedelta(minutes=10)
    irra_pred_da = data_rt['irra'][t_from:t_end].to_numpy().tolist()
    Pdem_pred_da = data_rt['Pdem'][t_from:t_end].to_numpy().tolist()
    Qdem_pred_da = data_rt['Qdem'][t_from:t_end].to_numpy().tolist()
    result_p_pv, result_irr = forecasting_pv_da(irra_pred_da)
    result_p_pv = np.maximum(np.zeros((3, 144)), result_p_pv)
    result_p_dm = forecasting_active_power_da(Pdem_pred_da)
    result_q_dm = forecasting_reactive_power_da(Qdem_pred_da)
    result_Vmag = 0.03 * np.ones((2, 144))
    result_SOC = [50, 0.2, 0.2]
    result_price = np.ones((6, 144))
    result_price[0][:] = 10 * result_price[0][:]
    result_price[1][:] = 0 * result_price[1][:]
    result_price[2][:] = 1 * result_price[2][:]
    result_price[3][:] = 1 * result_price[3][:]
    result_price[4][:] = 0.5 * result_price[4][:]
    result_price[5][:] = 0.5 * result_price[5][:]
    grid_inp = af.grid_topology_sim(network_name, [])
    file_to_store = open(dir_path + r"/Result/tmp_da.pickle", "wb")
    pickle.dump(grid_inp, file_to_store)
    pickle.dump(result_Vmag, file_to_store)
    pickle.dump(result_p_pv, file_to_store)
    pickle.dump(result_p_dm, file_to_store)
    pickle.dump(result_q_dm, file_to_store)
    pickle.dump(result_SOC, file_to_store)
    pickle.dump(result_price, file_to_store)
    pickle.dump(robust_par, file_to_store)
    file_to_store.close()
    now = datetime.now()
    tomorrow = str(now.year) + str(now.month) + str(now.day + 1)
    file_to_store = open(dir_path + r"/Result/for" + tomorrow + ".pickle", "wb")
    pickle.dump(grid_inp, file_to_store)
    pickle.dump(result_Vmag, file_to_store)
    pickle.dump(result_p_pv, file_to_store)
    pickle.dump(result_p_dm, file_to_store)
    pickle.dump(result_q_dm, file_to_store)
    pickle.dump(result_SOC, file_to_store)
    pickle.dump(result_price, file_to_store)
    pickle.dump(robust_par, file_to_store)
    file_to_store.close()
    os.system(python64_path + ' -c ' +
              '\"import sys;' +
              'from datetime import datetime;' +
              'print(sys.version);' +
              'sys.path.insert(0, r\'' + dir_path + '\');'
                                                    'import pickle;' +
              'file_to_read = open(r\'' + dir_path + '\' + r\'/Result/tmp_da.pickle\', \'rb\');' +
              'grid_inp = pickle.load(file_to_read);' +
              'V_mag = pickle.load(file_to_read);' +
              'result_p_pv = pickle.load(file_to_read);' +
              'result_p_dm = pickle.load(file_to_read);' +
              'result_q_dm = pickle.load(file_to_read);' +
              'result_SOC = pickle.load(file_to_read);' +
              'result_price = pickle.load(file_to_read);' +
              'robust_par = pickle.load(file_to_read);' +
              'file_to_read.close();' +
              'import Functions_P.OptimizationFunctions as of;' +
              'P_SC, Q_SC, RPP_SC, RPN_SC, RQP_SC, RQN_SC, SOC_dersired, prices_vec2 = ' +
              'of.da_opt_digriflex(grid_inp, V_mag, result_p_pv, result_p_dm, result_q_dm, result_SOC, result_price, robust_par);' +
              'now = datetime.now();' + 'tomorrow = str(now.year) + str(now.month) + str(now.day + 1);' +
              'file_to_store = open(r\'' + dir_path + '\' + r\'/Result/res\' + tomorrow + \'.pickle\', \'wb\');' +
              'pickle.dump(P_SC, file_to_store);' +
              'pickle.dump(Q_SC, file_to_store);' +
              'pickle.dump(RPP_SC, file_to_store);' +
              'pickle.dump(RPN_SC, file_to_store);' +
              'pickle.dump(RQP_SC, file_to_store);' +
              'pickle.dump(RQN_SC, file_to_store);' +
              'pickle.dump(SOC_dersired, file_to_store);' +
              'pickle.dump(prices_vec2, file_to_store);' +
              'file_to_store.close()\"'
              )
    # now = datetime.now()
    # tomorrow = str(now.year) + str(now.month) + str(now.day + 1)
    # file_to_read = open(dir_path + r"/Result/res" + tomorrow + ".pickle", "rb")
    # P_SC = pickle.load(file_to_read)
    # Q_SC = pickle.load(file_to_read)
    # RPP_SC = pickle.load(file_to_read)
    # RPN_SC = pickle.load(file_to_read)
    # RQP_SC = pickle.load(file_to_read)
    # RQN_SC = pickle.load(file_to_read)
    # SOC_dersired = pickle.load(file_to_read)
    # prices_vec = pickle.load(file_to_read)
    # file_to_read.close()
    return True


def forecasting_pv_da(pred_for):
    """" Not Completed:
    This function is for running the day-ahead forecasting R code written by Pasquale
    Inputs:
        - pred_for: a dataframe containing predictors irra for last two days
    Output:
        - result_p: forecasted power in kW
        - result_irra: forecasted irradiance in W/m2
    """
    ## Calling R function
    r = ro.r
    r['source'](dir_path + r'\Functions_R\Function_DayAhead_Bayesboot_irra.R')
    DayAhead_Bayesboot = ro.globalenv['DayAhead_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_for_r = ro.conversion.py2rpy(pred_for)
    result_irr_r = DayAhead_Bayesboot(pred_for_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_irra = ro.conversion.rpy2py(result_irr_r)
    result_irra = np.transpose(result_irra.values)
    result_po = (result_irra * 6.2 + 26) / 1000
    result_po[1:3][:] = (result_irra[1:3][:] * 6.2) / 1000
    return result_po, result_irra


def forecasting_active_power_da(pred_for):
    """" Not Completed:
    This function is for running the day-ahead forecasting R code written by Pasquale
    Inputs:
        - pred_for: a dataframe containing predictors P for the last two days
    Output: result_Pdem: forecasted active power in kW
    """
    ## Calling R function
    r = ro.r
    r['source'](dir_path + r'\Functions_R\Function_DayAhead_Bayesboot_P.R')
    DayAhead_Bayesboot = ro.globalenv['DayAhead_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_for_r = ro.conversion.py2rpy(pred_for)
    result_Pdem_r = DayAhead_Bayesboot(pred_for_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_Pdem = ro.conversion.rpy2py(result_Pdem_r)
    result_Pdem = np.transpose(result_Pdem.values) * 0.1
    return result_Pdem


def forecasting_reactive_power_da(pred_for):
    """" Not Completed:
    This function is for running the day-ahead forecasting R code written by Pasquale
    Inputs:
        - pred_for: a dataframe containing predictors Q for the last two days
    Output: result_Qdem: forecasted active power in kW
    """
    ## Calling R function
    r = ro.r
    r['source'](dir_path + r'\Functions_R\Function_DayAhead_Bayesboot_Q.R')
    DayAhead_Bayesboot = ro.globalenv['DayAhead_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_for_r = ro.conversion.py2rpy(pred_for)
    result_Qdem_r = DayAhead_Bayesboot(pred_for_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_Qdem = ro.conversion.rpy2py(result_Qdem_r)
    result_Qdem = np.transpose(result_Qdem.values) * 0.1
    return result_Qdem


#### TESTING
# dayahead_digriflex(1)
