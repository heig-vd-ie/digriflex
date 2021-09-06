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
from datetime import datetime, timedelta, date
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from DiGriFlex_RT import access_data_rt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import quantecon as qe
import scipy.stats as ss
from statistics import mean

print(sys.version)

#### Defining meta parameters
dir_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/Desktop/DiGriFlex_Code"
python64_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/AppData/Local/Programs/Python/Python39/python.exe"
network_name = "Case_4bus_DiGriFlex"  # Defining the network


# network_name = "Case_LabVIEW"


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
    r['source'](
        dir_path + r'\Functions_R\Codes_for_day-ahead_irradiance_forecasting\Function_LQR_Bayesboot_irra_24h_v2.R')
    DayAhead_Bayesboot = ro.globalenv['LQR_Bayesboot']
    result_irra, result_po = np.zeros((3, 144)), np.zeros((3, 144))
    for h in range(1, 145):
        with localconverter(ro.default_converter + pandas2ri.converter):
            pred_for_r = ro.conversion.py2rpy(pred_for[h - 1][:])
        result_irr_r = DayAhead_Bayesboot(pred_for_r, h, 10)
        with localconverter(ro.default_converter + pandas2ri.converter):
            result_irra_0 = ro.conversion.rpy2py(result_irr_r)
        temp = np.transpose(result_irra_0)
        result_irra[0][h - 1] = temp[1][0]
        result_irra[1][h - 1] = temp[2][0] - temp[1][0]
        result_irra[2][h - 1] = temp[1][0] - temp[0][0]
        result_po[0][h - 1] = (result_irra[0][h - 1] * 6.21) / 1000
        result_po[1][h - 1] = (result_irra[1][h - 1] * 6.21) / 1000
        result_po[2][h - 1] = (result_irra[2][h - 1] * 6.21) / 1000
        print('pv:' + str(h))
    return result_po, result_irra


def forecasting_active_power_da(pred_for, fac_P):
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
    result_Pdem = np.transpose(result_Pdem.values) * fac_P
    return result_Pdem


def forecasting_reactive_power_da(pred_for, fac_Q):
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
    result_Qdem = np.transpose(result_Qdem.values) * fac_Q
    return result_Qdem


def transition_matrix(tran, ndig):
    n = 1 + ndig
    M1 = [[0] * n for _ in range(n)]
    for (i, j) in zip(tran, tran[1:]):
        M1[i][j] += 1
    k = 0
    for row in M1:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
        else:
            row[k] = 1
        k = k + 1
    return M1


def forecasting1(data0, name):
    date_sp = datetime.now()
    date_sp = date_sp.replace(day=date_sp.day + 1)
    t = pd.date_range(pd.Timestamp(date_sp).floor(freq='D'), periods=144, freq='10min')
    model = KMeans(n_clusters=3, random_state=0).fit(data0)
    output = model.cluster_centers_
    avg = np.average(output, axis=1)
    rank = ss.rankdata(avg)
    f_i = np.where(rank == 2)
    forecast = output[f_i[0], :]
    err_p = np.max(output, axis=0) - forecast
    err_n = forecast - np.min(output, axis=0)
    vec_out = np.zeros((3, 144))
    vec_out[0, :] = forecast
    vec_out[1, :] = err_p
    vec_out[2, :] = err_n
    return vec_out


def forecasting2(data0, name):
    dd = 30
    date_sp = datetime.now()
    date_sp = date_sp.replace(day=date_sp.day + 1)
    t = pd.date_range(pd.Timestamp(date_sp).floor(freq='D'), periods=144, freq='10min')
    data0 = np.resize(data0, (dd * 144, 1))
    M = data0[288:]
    F = np.transpose(np.resize(np.array([data0[144:dd * 144 - 144].tolist(),
                                         data0[:dd * 144 - 288].tolist()]), (2, (dd - 2) * 144)))
    Theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(F), F)), np.matmul(np.transpose(F), M))
    error = np.resize(M - np.matmul(F, Theta), (dd - 2, 144))
    varm = np.zeros((144, 1))
    for tt in range(144):
        varm[tt] = np.std(error[:, tt])
    F = np.transpose(np.resize(np.array([data0[dd * 144 - 144:].tolist(),
                                         data0[dd * 144 - 288:dd * 144 - 144].tolist()]), (2, 144)))
    y_pred = np.matmul(F, Theta)
    y_scen = np.zeros((100, 144))
    for s in range(100):
        for tt in range(144):
            y_scen[s, tt] = y_pred[tt] + np.random.normal(0, varm[tt])
    model = KMeans(n_clusters=3, random_state=0).fit(y_scen)
    output = model.cluster_centers_
    output = np.delete(output, model.predict(np.transpose(y_pred)), 0)
    output = np.append(output, np.transpose(y_pred), axis=0)
    err_p = np.max(output, axis=0) - np.transpose(y_pred)
    err_n = np.transpose(y_pred) - np.min(output, axis=0)
    vec_out = np.zeros((3, 144))
    vec_out[0, :] = np.transpose(y_pred)
    vec_out[1, :] = err_p
    vec_out[2, :] = err_n
    return vec_out


def forecasting3(data0, name):
    dd = 30
    date_sp = datetime.now()
    date_sp = date_sp.replace(day=date_sp.day + 1)
    t = pd.date_range(pd.Timestamp(date_sp).floor(freq='D'), periods=144, freq='10min')
    data1 = np.resize(data0, (dd * 144, 1))
    data = pd.DataFrame(data1, columns=['output'])
    for tt in range(144, 184):
        data[str(tt) + 'h_delay'] = data['output'].shift(periods=tt)
    std = np.std(data1) / data['output'].std()
    mean = np.mean(data1) - data['output'].mean()
    y = data.pop('output')
    train_x, train_y = data, y
    train_x = train_x.fillna(train_x.mean())
    features = list(train_x.columns)
    RandomForestRegModel = RandomForestRegressor()
    RandomForestRegModel.fit(train_x, train_y)
    y_pred = RandomForestRegModel.predict(train_x)
    y_pred = y_pred * std + mean
    y_pred0 = y_pred[-144:]
    model = KMeans(n_clusters=3, random_state=0).fit(data0)
    ind = model.predict(np.resize(y_pred0, (1, 144)))
    ind2 = model.predict(data0)
    y = np.transpose(data0[ind2 == ind])
    b = np.min(y)
    l = np.max(y) - np.min(y)
    y2 = np.divide(y - b, l)
    y2 = np.resize(np.transpose(y2), (y2.shape[1] * 144, 1))
    y2 = y2[y2 > 0.01]
    digit = 20
    bins = np.arange(0.05, 1, 1 / digit)
    transitions = np.digitize(y2, bins, right=True)
    TM = transition_matrix(transitions, digit)
    mc = qe.MarkovChain(TM)
    y_scen = np.zeros((100, 144))
    for s in range(100):
        if name == 'PV power production (kW)':
            l = np.max(y, axis=1) - np.min(y, axis=1)
            b = 0
            initial_state = 0
        else:
            l = np.max(y) - np.min(y)
            b = np.min(y)
            initial_state = digit / 2
        X = mc.simulate(init=int(initial_state), ts_length=144) / (np.sqrt(np.size(TM)) - 1)

        y_scen[s, :] = np.multiply(X, l) + b
    model = KMeans(n_clusters=3, random_state=0).fit(y_scen)
    output = model.cluster_centers_
    delind = model.predict(np.resize(y_pred0, (1, 144)))
    output = np.delete(output, delind, 0)
    output = np.append(output, np.resize(y_pred0, (1, 144)), axis=0)
    err_p = np.max(output, axis=0) - y_pred0
    err_n = y_pred0 - np.min(output, axis=0)
    vec_out = np.zeros((3, 144))
    vec_out[0, :] = np.transpose(y_pred0)
    vec_out[1, :] = np.transpose(err_p)
    vec_out[2, :] = np.transpose(err_n)
    return vec_out


def dayahead_digriflex(robust_par):
    robust_par = 1
    fac_P, fac_Q = 0.1, 0.1
    mode_forec = 'r'  # 'r', 'b1', 'b2', 'mc'
    data_rt = access_data_rt()
    t_now = data_rt.index[-1]
    t_end = data_rt.index[-1].floor('1d') - timedelta(hours=1)
    t_end_y = t_end - timedelta(minutes=10)
    t_now_y = data_rt.index[-1] - timedelta(days=1) + timedelta(minutes=10)
    t_from = t_end - timedelta(days=2) + timedelta(minutes=10)
    # irra_pred_da = data_rt['irra'][t_from:t_end].to_numpy().tolist()
    irra_pred_da = [data_rt['irra'][t_end:t_now].to_numpy().tolist() + data_rt['irra'][t_now_y:t_end_y].to_numpy().tolist(),
                    data_rt['pres'][t_end:t_now].to_numpy().tolist() + data_rt['pres'][t_now_y:t_end_y].to_numpy().tolist(),
                    data_rt['relh'][t_end:t_now].to_numpy().tolist() + data_rt['relh'][t_now_y:t_end_y].to_numpy().tolist(),
                    data_rt['temp'][t_end:t_now].to_numpy().tolist() + data_rt['temp'][t_now_y:t_end_y].to_numpy().tolist(),
                    data_rt['wind'][t_end:t_now].to_numpy().tolist() + data_rt['wind'][t_now_y:t_end_y].to_numpy().tolist(),
                    (np.average(data_rt['irra'][t_end:t_now].to_numpy())*np.ones(144)).tolist(),
                    (np.average(data_rt['pres'][t_end:t_now].to_numpy())*np.ones(144)).tolist(),
                    (np.average(data_rt['relh'][t_end:t_now].to_numpy())*np.ones(144)).tolist(),
                    (np.average(data_rt['temp'][t_end:t_now].to_numpy())*np.ones(144)).tolist(),
                    (np.average(data_rt['wind'][t_end:t_now].to_numpy())*np.ones(144)).tolist()]
    irra_pred_da = np.transpose(np.array(irra_pred_da)).tolist()
    Pdem_pred_da = data_rt['Pdem'][t_from:t_end].to_numpy().tolist()
    Qdem_pred_da = data_rt['Qdem'][t_from:t_end].to_numpy().tolist()
    if mode_forec == 'r':
        result_p_pv, result_irr = forecasting_pv_da(irra_pred_da)
        result_p_dm = forecasting_active_power_da(Pdem_pred_da, fac_P)
        result_q_dm = forecasting_reactive_power_da(Qdem_pred_da, fac_Q)
    elif mode_forec == 'b1':
        dd = 30
        t_end = data_rt.index[-1].floor('1d') - timedelta(hours=1)
        t_from = t_end - timedelta(days=dd) + timedelta(minutes=10)
        pv_pred_da = np.resize(data_rt['P'][t_from:t_end].to_numpy(), (dd, 144)) / 1000
        Pdem_pred_da = np.resize(data_rt['Pdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
        Qdem_pred_da = np.resize(data_rt['Qdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
        result_p_pv = forecasting1(pv_pred_da, 'PV power production (kW)')
        result_p_dm = forecasting1(Pdem_pred_da, 'Demand active power (kW)')
        result_q_dm = forecasting1(Qdem_pred_da, 'Demand reactive power (kVar)')
    elif mode_forec == 'b2':
        dd = 30
        t_end = data_rt.index[-1].floor('1d') - timedelta(hours=1)
        t_from = t_end - timedelta(days=dd) + timedelta(minutes=10)
        pv_pred_da = np.resize(data_rt['P'][t_from:t_end].to_numpy(), (dd, 144)) / 1000
        Pdem_pred_da = np.resize(data_rt['Pdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
        Qdem_pred_da = np.resize(data_rt['Qdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
        result_p_pv = forecasting2(pv_pred_da, 'PV power production (kW)')
        result_p_dm = forecasting2(Pdem_pred_da, 'Demand active power (kW)')
        result_q_dm = forecasting2(Qdem_pred_da, 'Demand reactive power (kVar)')
    elif mode_forec == 'mc':
        dd = 30
        t_end = data_rt.index[-1].floor('1d') - timedelta(hours=1)
        t_from = t_end - timedelta(days=dd) + timedelta(minutes=10)
        pv_pred_da = np.resize(data_rt['P'][t_from:t_end].to_numpy(), (dd, 144)) / 1000
        Pdem_pred_da = np.resize(data_rt['Pdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
        Qdem_pred_da = np.resize(data_rt['Qdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
        result_p_pv = forecasting3(pv_pred_da, 'PV power production (kW)')
        result_p_dm = forecasting3(Pdem_pred_da, 'Demand active power (kW)')
        result_q_dm = forecasting3(Qdem_pred_da, 'Demand reactive power (kVar)')
    result_p_pv = np.maximum(np.zeros((3, 144)), result_p_pv)
    result_Vmag = 0.03 * np.ones((2, 144))
    result_SOC = [5, 0.2, 0.2]
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
              'P_SC, Q_SC, RPP_SC, RPN_SC, RQP_SC, RQN_SC, SOC_dersired, prices_vec2, Obj = ' +
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
              'pickle.dump(Obj, file_to_store);' +
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


#### TESTING
dayahead_digriflex(0.75)
