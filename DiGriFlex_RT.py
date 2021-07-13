"""Revised on: 01.08.2021, @author: MYI, #Python version: 3.6.8 [32 bit]"""
#### To Do lists:
## - CM#0: @DiGriFlex_DA, DiGriFlex_sim, Main_file
## - CM#0: @AuxiliaryFunctions: R and X of lines, efficiencies data of battery, X of PVs (function "reine_parameters()")
## - CM#0: @AuxiliaryFunctions: Integrating with the dayahead forecasting code of Pasquale ("forecast_defining")
## - CM#0: @AuxiliaryFunctions: reamined reviewing from functions "rt_simulation"
## - CM#0: @OptimizationFunctions: reamined reviewing from functions "DA_Optimization_Robust"
## - CM#0: Connection of realtime code with day_ahead optimization and modifiying the optimization functions
## - CM#1: Day-ahead forecasting and opt -> First solution: without weather prediction; Second solution: MeteoSwiss API
## - CM#2: Real-time forecasting of loads
## - CM#3: Finding power output from the irradiance and temperature
## - CM#4: running the test with labview
## - CM#5: testing the gurobipy licence

#### Importing packages
import os
import sys
import warnings
import pandas as pd
import math
import pickle
import random
import rpy2.robjects as ro
import mysql.connector
import Functions_P.AuxiliaryFunctions as af
from datetime import datetime, timedelta
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

print(sys.version)

#### Defining meta parameters
dir_path = r"C:\Users\mohammad.rayati\Desktop\DiGriFlex_Code"  # Defining directory path of the code
# dir_path = r"C:\Users\labo-reine-iese\Desktop\DiGriFlex_Code"
network_name = "Case_4bus_DiGriFlex"  # Defining the network
# network_name = "Case_LabVIEW"
python64_path = r"C:\Users\mohammad.rayati\AppData\Local\Programs\Python\Python39\python.exe"
# python64_path = r"C:\Users\labo-reine-iese\AppData\Local\Programs\Python\Python39\python.exe"


#### Functions
def interface_control_digriflex(Vec_Inp):
    """" Not Completed: CM#3 and CM#4 This function is for intefacing with LabVIEW in DiGriFlex project Input:
    Vec_Inp as ordered by Douglas in "07/10/2020 14:06" (See \Data\Ordre_informations_controle_2020.10.07.xlsx)
    Output: Vec_Out = [P Batt, Q Batt, P SolarMax, Q SolarMax, P_step ABB, Cos_phi ABB, P KACO, Q KACO, P Cinergia,
    Q Cinergia, P1 SOP, Q1 SOP, P2 SOP, Q2 SOP, P GM, Q GM]
    """
    ## Reading inputs
    file_to_read = open(dir_path + r"/Data/tmp_da.pickle", "rb")
    P_SC = pickle.load(file_to_read)
    Q_SC = pickle.load(file_to_read)
    RPP_SC = pickle.load(file_to_read)
    RPN_SC = pickle.load(file_to_read)
    RQP_SC = pickle.load(file_to_read)
    RQN_SC = pickle.load(file_to_read)
    SOC_dersired = pickle.load(file_to_read)
    prices_vec = pickle.load(file_to_read)
    file_to_read.close()
    _, Ligne_U, _, _, _, Charge_P, Charge_Q, Ond_P, Ond_Q, _, _, SOC = af.interface_meas(Vec_Inp)
    # Cinergia_P_m, Cinergia_Q_m = Charge_P[11], Charge_Q[11]
    # ABB_P_m, ABB_Q_m = Ond_P[1], Ond_Q[1]
    ## Algorithm
    data_rt = access_data_rt()  # Instead, the following lines can be called for testing
    pred_for = data_rt[
        ["Irralag144_for", "Irralag2_for", "Irralag3_for", "Irralag4_for", "Irralag5_for", "Irralag6_for"]]
    pred_for = pred_for[-1:]
    # pred_for = pd.read_csv(dir_path + r"\Data\pred_for.csv")
    # pred_for = pd.DataFrame(pred_for).set_index('Unnamed: 0')
    now = datetime.now()
    timestep = (now.hour - 1) * 6 + math.floor(now.minute / 10) + 2
    result_p_pv, result_irra = forecasting_pv_rt(pred_for, timestep)
    result_p_dm, result_q_dm = 0, 0
    ## For testing and saving data
    ####################################################################################
    print(data_rt[["P", "irra", "Plag2_for", "Irralag2_for"]][-1:])
    print("forecast_P, forecast_irra:" + str(round(result_p_pv, 5)) + ', ' + str(round(result_irra, 2)))
    df = pd.DataFrame([[now, pred_for.index[-1], data_rt.iloc[-1]['Plag2_for'], data_rt.iloc[-1]['Irralag2_for'],
                        round(result_p_pv, 5), round(result_irra, 2)]])
    df.to_csv(dir_path + r'\Data\recorded_forecast_pv.csv', mode='a', header=False)
    ####################################################################################
    grid_inp = af.grid_topology_sim(network_name, Vec_Inp)
    P_net = P_SC[timestep] + random.uniform(-RPN_SC[timestep], RPP_SC[timestep])  # Uniform dist. of reserves activation
    Q_net = Q_SC[timestep] + random.uniform(-RQN_SC[timestep], RQP_SC[timestep])  # Uniform dist. of reserves activation
    file_to_store = open(dir_path + r"/Data/tmp.pickle", "wb")
    pickle.dump(grid_inp, file_to_store)
    pickle.dump(400, file_to_store)  # Ligne_U[0]
    pickle.dump(P_net, file_to_store)
    pickle.dump(Q_net, file_to_store)
    pickle.dump(result_p_pv, file_to_store)
    pickle.dump([result_p_dm, result_q_dm], file_to_store)
    pickle.dump(SOC, file_to_store)
    pickle.dump(SOC_dersired[timestep], file_to_store)
    pickle.dump(prices_vec, file_to_store)
    file_to_store.close()
    os.system(python64_path + ' -c ' +
              '\"import sys;' +
              'print(sys.version);' +
              'sys.path.insert(0, r\'' + dir_path + '\');'
                                                    'import pickle;' +
              'file_to_read = open(r\'' + dir_path + '\' + r\'/Data/tmp.pickle\', \'rb\');' +
              'grid_inp = pickle.load(file_to_read);' +
              'V_mag = pickle.load(file_to_read);' +
              'P_net = pickle.load(file_to_read);' +
              'Q_net = pickle.load(file_to_read);' +
              'forecast_pv = pickle.load(file_to_read);' +
              'forecast_dm = pickle.load(file_to_read);' +
              'SOC = pickle.load(file_to_read);' +
              'SOC_desired = pickle.load(file_to_read);' +
              'prices_vec = pickle.load(file_to_read);' +
              'file_to_read.close();'
              'import Functions_P.OptimizationFunctions as of;'
              'ABB_P_sp, ABB_c_sp, Battery_P_sp, Battery_Q_sp = ' +
              'of.rt_opt_digriflex(grid_inp, V_mag, P_net, Q_net, forecast_pv, forecast_dm, SOC, SOC_desired, prices_vec);'
              # 'of.rt_following_digriflex(grid_inp, P_net, Q_net, forecast_pv, forecast_dm, SOC);' +  # this function will be changed
              'file_to_store = open(r\'' + dir_path + '\' + r\'/Data/tmp.pickle\', \'wb\');' +
              'pickle.dump(ABB_P_sp, file_to_store);' +
              'pickle.dump(ABB_c_sp, file_to_store);' +
              'pickle.dump(Battery_P_sp, file_to_store);' +
              'pickle.dump(Battery_Q_sp, file_to_store);' +
              'file_to_store.close()\"'
              )
    file_to_read = open(dir_path + r"/Data/tmp.pickle", "rb")
    ABB_P_sp = pickle.load(file_to_read)
    ABB_c_sp = pickle.load(file_to_read)
    Battery_P_sp = pickle.load(file_to_read)
    Battery_Q_sp = pickle.load(file_to_read)
    file_to_read.close()
    ## Defining outputs
    Vec_Out = [0] * 16
    Vec_Out[0] = Battery_P_sp
    Vec_Out[1] = Battery_Q_sp
    Vec_Out[4] = ABB_P_sp
    Vec_Out[5] = ABB_c_sp
    return Vec_Out


def access_data_rt():
    """" Completed:
    This function is for reading real-time data from the database of school
    Output:
        - myresult: a dataframe with rows number equal to the real-time time step and the following columns
            -- [Irralag144_for] = irradiation with 144 lags
            -- [Irralag2_for] = irradiation with 2 lags
            -- [Irralag3_for] = irradiation with 3 lags
            -- [Irralag4_for] = irradiation with 4 lags
            -- [Irralag5_for] = irradiation with 5 lags
            -- [Irralag6_for] = irradiation with 6 lags
            -- [Plag2_for] = P_ABB with 2 lags
    """
    mydb = mysql.connector.connect(host="10.192.48.47",
                                   user="heigvd_meteo_ro",
                                   port="3306",
                                   password="at5KUPusS9",
                                   database="heigvdch_meteo")  # Database for reading real-time data
    mycursor = mydb.cursor()
    mycursor.execute(
        "SELECT `Date and Time`, `Labo Reine.ABB.Power AC [W]`, `Meteo.Irradiance.Rayonnement Global moyen [W/m2]` "
        "FROM db_iese_2021")
    myresult = mycursor.fetchall()
    myresult = pd.DataFrame.from_records(myresult[-60 * 24 * 2:], columns=['Date and Time', 'P', 'irra'],
                                         index=['Date and Time'])
    myresult = myresult.resample('10min').mean()
    myresult['Irralag2_for'] = myresult['irra'].shift(1)
    myresult['Irralag3_for'] = myresult['irra'].shift(2)
    myresult['Irralag4_for'] = myresult['irra'].shift(3)
    myresult['Irralag5_for'] = myresult['irra'].shift(4)
    myresult['Irralag6_for'] = myresult['irra'].shift(5)
    myresult['Irralag144_for'] = myresult['irra'].shift(143)
    myresult['Plag2_for'] = myresult['P'].shift(1)
    now = datetime.now()
    # timestep = (now.hour-1) * 6 + math.floor(now.minute / 10)
    if datetime.strftime(myresult.index[-1], '%Y-%m-%d %H:%M') \
            < datetime.strftime(now - timedelta(hours=2), '%Y-%m-%d %H:%M'):
        warnings.warn("The recorded data is older than 2 hours.")
    return myresult


def forecasting_pv_rt(pred_for, time_step):
    """" Completed:
    This function is for running the real-time forecasting R code written by Pasquale
    Inputs:
        - pred_for: a dataframe containing predictors irra(t-144),irra(t-2),irra(t-3),irra(t-4),irra(t-5),irra(t-6)
        - time_step: number of the target 10-min interval of the day.
        - N_boot: desired number of bootstrap samples.
    Output: result_irra: forecasted irradiance in W/m2
    """
    N_boot = 300  # Desired number of bootstrap samples.
    ## Training dataset
    Irra_tra = pd.read_csv(dir_path + r"\Data\Irr_tra.csv")
    pred_tra_irr = pd.read_csv(dir_path + r"\Data\pred_tra_irr.csv")
    Irra_tra = Irra_tra['x']
    pred_tra_irr = pd.DataFrame(pred_tra_irr).set_index('Unnamed: 0')
    ## Calling R function
    r = ro.r
    r['source'](dir_path + r'\Functions_R\Function_LQR_Bayesboot_irra.R')
    LQR_Bayesboot = ro.globalenv['LQR_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        Irra_tra_r = ro.conversion.py2rpy(Irra_tra)
        pred_for_r = ro.conversion.py2rpy(pred_for)
        pred_tra_irr_r = ro.conversion.py2rpy(pred_tra_irr)
    result_irr_r = LQR_Bayesboot(Irra_tra_r, pred_tra_irr_r, pred_for_r, time_step, N_boot)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_irra = ro.conversion.rpy2py(result_irr_r)
    result_irra = result_irra[0]
    result_p = (result_irra * 6.2 + 26) / 1000
    return result_p, result_irra


#### TESTING
Vec_Inp0 = open(dir_path + r"\Data\test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
Vec_Out0 = interface_control_digriflex(Vec_Inp0)
print(Vec_Out0)
