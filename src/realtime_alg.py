"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
from auxiliary import interface_meas, grid_topology_sim
from datetime import timedelta, datetime
from dotenv import dotenv_values
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import coloredlogs
import logging
import math
import mysql.connector
import os.path
import pandas as pd
import pickle
import rpy2.robjects as ro
import tqdm
import warnings


# Global variables
# ----------------------------------------------------------------------------------------------------------------------
python64_path = os.getcwd() + r"/.venv/Scripts/python.exe"
network_name = "Case_4bus_DiGriFlex"
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
# ----------------------------------------------------------------------------------------------------------------------


# Functions
# ----------------------------------------------------------------------------------------------------------------------
def interface_control_digriflex(vec_inp: list, date: datetime, forecasting=False):
    """
    @param vec_inp: list of the input values
    @param date: the date of the input values
    @param forecasting: boolean value for the forecasting
    vec_inp as ordered by Douglas in "07/10/2020 14:06" (See "data\ordre_informations_controle_2020.10.07.xlsx")
    output: vec_out = [p batt, q batt, p solar_max, q solar_max, p_step abb, cos_phi abb, p kaco, q kaco, p cinergia,
    q cinergia, p1 sop, q1 sop, p2 sop, q2 sop, p gm, q gm]
    """
    # Read inputs
    fac_p, fac_q = 0.1, 0.1
    today = date
    timestep = today.hour * 6 + math.floor(today.minute / 10)
    filepath = None
    for i in range(30):
        date = (today - timedelta(days=i)).strftime("%Y_%m_%d")
        if os.path.exists(r".cache/outputs/res" + date + ".pickle"):
            filepath = r".cache/outputs/res" + date + ".pickle"
    if filepath is None:
        raise FileNotFoundError("The file res.pickle does not exist for the last 30 days")
    with open(filepath, "rb") as file_to_read:
        (p_sc, q_sc, rpp_sc, rpn_sc, rqp_sc, rqn_sc, soc_desired, prices, obj) = pickle.load(file_to_read)
    _, ligne_u, _, _, _, charge_p, charge_q, ond_p, ond_q, _, _, soc, f_p_real, f_q_real = interface_meas(vec_inp)
    ond_p = float(ond_p[1])
    ond_q = float(ond_q[1])
    if soc == 0:
        soc = soc_desired[timestep] * 100 / 64
    else:
        soc = soc
    data_rt = access_data_rt(year=today.year, month=today.month, day=today.day, hour=today.hour,
                             minute=today.minute)
    if forecasting:
        with tqdm.tqdm(total=3, desc="Interface control") as pbar:
            data_rt = data_rt[-1:]
            pred_for = data_rt[["Irralag144_for", "Irralag2_for", "Irralag3_for", "Irralag4_for", "Irralag5_for",
                                "Irralag6_for"]]
            result_p_pv, result_irra = forecasting_pv_rt(pred_for=pred_for, time_step=timestep + 1)
            pbar.update()
            pred_for = data_rt[["Pdemlag144_for", "Pdemlag2_for", "Pdemlag3_for", "Pdemlag4_for", "Pdemlag5_for",
                                "Pdemlag6_for"]]
            result_p_dm = forecasting_active_power_rt(pred_for=pred_for, time_step=timestep + 1, fac_p=fac_p)
            pbar.update()
            pred_for = data_rt[["Qdemlag144_for", "Qdemlag2_for", "Qdemlag3_for", "Qdemlag4_for", "Qdemlag5_for",
                                "Qdemlag6_for"]]
            result_q_dm = forecasting_reactive_power_rt(pred_for=pred_for, time_step=timestep + 1, fac_q=fac_q)
            pbar.update()
    else:
        date_rt = data_rt[-1:]
        pred_for = date_rt
        result_p_pv, result_irra, result_p_dm, result_q_dm = data_rt["P"].iloc[-1], data_rt["irra"].iloc[-1], \
            data_rt["Pdem"].iloc[-1], data_rt["Qdem"].iloc[-1]
    grid_inp = grid_topology_sim(network_name, vec_inp)
    p_net = p_sc[timestep] + 0.5 * rpn_sc[timestep] - 0 * rpp_sc[timestep]
    q_net = q_sc[timestep] + 0 * rqn_sc[timestep] - 0 * rqp_sc[timestep]
    with open(r".cache/outputs/tmp_rt.pickle", "wb") as file_to_store:
        pickle.dump((grid_inp, 400, p_net, q_net, result_p_pv, [result_p_dm, result_q_dm], soc, soc_desired[timestep],
                     prices), file_to_store)
    os.system(python64_path +
              ' -c ' +
              '\"import sys;' +
              'import pickle;' +
              'file_to_read = open(r\'.cache/outputs/tmp_rt.pickle\', \'rb\');' +
              '(grid_inp, v_mag, p_net, q_net, forecast_pv, forecast_dm, soc, soc_desired, prices) = ' +
              'pickle.load(file_to_read);' +
              'file_to_read.close();'
              'from src.optimization_prob import *;'
              'abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp, abb_p_exp, f_p, f_q, rt_res = ' +
              'rt_opt_digriflex(grid_inp, v_mag, p_net, q_net, forecast_pv, forecast_dm, soc, soc_desired, prices);'
              'file_to_store = open(r\'.cache/outputs/tmp_rt.pickle\', \'wb\');' +
              'pickle.dump((abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp, abb_p_exp, f_p, f_q, rt_res), ' +
              'file_to_store);' +
              'file_to_store.close()\"'
              )
    with open(r".cache/outputs/tmp_rt.pickle", "rb") as file_to_read:
        (abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp, abb_p_exp, f_p, f_q, rt_res) = pickle.load(file_to_read)
    # Test and save data
    df = pd.DataFrame([[today, pred_for.index[-1], data_rt.iloc[-1]['Plag2_for'] / 1000,
                        data_rt.iloc[-1]['Irralag2_for'],
                        data_rt.iloc[-1]['Pdemlag2_for'] * fac_p, data_rt.iloc[-1]['Qdemlag2_for'] * fac_q,
                        round(result_p_pv, 5), round(result_irra, 2),
                        round(result_p_dm, 5), round(result_q_dm, 5),
                        round(abb_p_exp, 5), round(f_p, 5), round(f_q, 5),
                        ond_p, ond_q, soc, f_p_real, f_q_real, -p_net, -q_net,
                        - battery_p_sp, battery_q_sp, abb_p_sp, abb_c_sp
                        ]])
    df.to_csv(r'.cache/outputs/realtime_data.csv', mode='a', header=False)
    # Define outputs
    vec_out = [0] * 16
    vec_out[0] = battery_p_sp  # output in kw // sum of phase
    vec_out[1] = battery_q_sp  # output in kVar // sum of phase
    vec_out[4] = abb_p_sp
    vec_out[5] = abb_c_sp
    return vec_out, rt_res


def access_data_rt(year: int, month: int, day: int, hour: int = 23, minute: int = 50):
    """
    @description: This function is used to access data from the database
    @param year: year
    @param month: month
    @param day: day
    @param hour: hour
    @param minute: minute
    @return: data_rt
    This function is for reading real-time data from the database of school
    Output:
        - data_rt: a dataframe with rows number equal to the real-time time step and the following columns
            -- [Irralag144_for] = irradiation with 144 lags
            -- [Irralag2_for] = irradiation with 2 lags
            -- [Irralag3_for] = irradiation with 3 lags
            -- [Irralag4_for] = irradiation with 4 lags
            -- [Irralag5_for] = irradiation with 5 lags
            -- [Irralag6_for] = irradiation with 6 lags
            -- [Plag2_for] = P_ABB with 2 lags
            -- [Pdemlag144_for] = Rouge active power with 144 lags
            -- [Pdemlag2_for] = Rouge active power with 2 lags
            -- [Pdemlag3_for] = Rouge active power with 3 lags
            -- [Pdemlag4_for] = Rouge active power with 4 lags
            -- [Pdemlag5_for] = Rouge active power with 5 lags
            -- [Pdemlag6_for] = Rouge active power with 6 lags
            -- [Qdemlag144_for] = Rouge reactive power with 144 lags
            -- [Qdemlag2_for] = Rouge reactive power with 2 lags
            -- [Qdemlag3_for] = Rouge reactive power with 3 lags
            -- [Qdemlag4_for] = Rouge reactive power with 4 lags
            -- [Qdemlag5_for] = Rouge reactive power with 5 lags
            -- [Qdemlag6_for] = Rouge reactive power with 6 lags
    """
    date_now = datetime.strptime(str(year) + '-' + str(month) + '-' + str(day) + " " + str(hour) + ":" + str(minute),
                                 "%Y-%m-%d %H:%M")
    date_1month = date_now - timedelta(days=30)
    env_vars = dotenv_values()
    mydb = mysql.connector.connect(host=env_vars["HOST"],
                                   user=env_vars["USER"],
                                   port=env_vars["PORT"],
                                   password=env_vars["PASSWORD"],
                                   database=env_vars["DATABASE"])  # Database for reading real-time data
    my_cursor1 = mydb.cursor()
    my_cursor1.execute(
        "SELECT `Date and Time`, `Labo Reine.ABB.Power AC [W]`, `Meteo.Irradiance.Rayonnement Global moyen [W/m2]`,"
        "`Meteo.Air.Pression Brute [hPa]`, `Meteo.Air.Humidite [pourcent]`,"
        "`Meteo.Air.Temperature exterieure [degre C]`, `Meteo.Air.Vent.v moyen [m/s]` "
        "FROM db_iese_" + str(year)
    )
    my_data1 = my_cursor1.fetchall()
    my_result = pd.DataFrame.from_records(my_data1,
                                          columns=['Date and Time', 'P', 'irra', 'pres', 'relh', 'temp', 'wind'],
                                          index=['Date and Time'])
    my_result = my_result[my_result.index < date_now.strftime("%Y-%m-%d %H:%M")]
    my_result = my_result[my_result.index > date_1month.strftime("%Y-%m-%d %H:%M")]
    my_result = my_result.resample('10min').mean()
    my_result['Irralag2_for'] = my_result['irra'].shift(1)
    my_result['Irralag3_for'] = my_result['irra'].shift(2)
    my_result['Irralag4_for'] = my_result['irra'].shift(3)
    my_result['Irralag5_for'] = my_result['irra'].shift(4)
    my_result['Irralag6_for'] = my_result['irra'].shift(5)
    my_result['Irralag144_for'] = my_result['irra'].shift(143)
    my_result['Plag2_for'] = my_result['P'].shift(1)
    my_cursor2 = mydb.cursor()
    my_cursor2.execute(
        "SELECT `Date et Heure`, `Rouge.P.L1.Moy`, `Rouge.P.L2.Moy`, `Rouge.P.L2.Moy`, "
        "`Rouge.Q.L1.Moy`, `Rouge.Q.L2.Moy`, `Rouge.Q.L3.Moy` "
        "FROM Reseau_HEIG"
    )
    my_data2 = my_cursor2.fetchall()
    my_data2 = pd.DataFrame.from_records(my_data2, columns=['Date and Time', 'PL1', 'PL2', 'PL3', 'QL1', 'QL2', 'QL3'],
                                         index=['Date and Time'])
    my_data2 = my_data2[my_data2.index < date_now.strftime("%Y-%m-%d %H:%M")]
    my_data2 = my_data2[my_data2.index > date_1month.strftime("%Y-%m-%d %H:%M")]
    my_data2.index = my_data2.index.floor('10min')
    my_data2 = my_data2[~my_data2.index.duplicated()]
    my_data2 = my_data2.resample('10min').interpolate(method='linear')
    new_row = [[pd.to_datetime(my_data2.index[-1] + timedelta(minutes=10)), 0, 0, 0, 0, 0, 0]]
    new_row = pd.DataFrame(new_row, columns=['Date and Time', 'PL1', 'PL2', 'PL3', 'QL1', 'QL2', 'QL3'])
    new_row = new_row.set_index('Date and Time')
    my_data2 = pd.concat([my_data2, new_row], ignore_index=False)
    my_data2 = my_data2.astype(float)
    my_result['Pdem'] = (my_data2['PL1'] + my_data2['PL2'] + my_data2['PL3']) / 3
    my_result['Pdemlag2_for'] = (my_data2['PL1'].shift(1) + my_data2['PL2'].shift(1) + my_data2['PL3'].shift(1)) / 3
    my_result['Pdemlag3_for'] = (my_data2['PL1'].shift(2) + my_data2['PL2'].shift(2) + my_data2['PL3'].shift(2)) / 3
    my_result['Pdemlag4_for'] = (my_data2['PL1'].shift(3) + my_data2['PL2'].shift(3) + my_data2['PL3'].shift(3)) / 3
    my_result['Pdemlag5_for'] = (my_data2['PL1'].shift(4) + my_data2['PL2'].shift(4) + my_data2['PL3'].shift(4)) / 3
    my_result['Pdemlag6_for'] = (my_data2['PL1'].shift(5) + my_data2['PL2'].shift(5) + my_data2['PL3'].shift(5)) / 3
    my_result['Pdemlag144_for'] = \
        (my_data2['PL1'].shift(143) + my_data2['PL2'].shift(143) + my_data2['PL3'].shift(143)) / 3
    my_result['Qdem'] = (my_data2['QL1'] + my_data2['QL2'] + my_data2['QL3']) / 3
    my_result['Qdemlag2_for'] = (my_data2['QL1'].shift(1) + my_data2['QL2'].shift(1) + my_data2['QL3'].shift(1)) / 3
    my_result['Qdemlag3_for'] = (my_data2['QL1'].shift(2) + my_data2['QL2'].shift(2) + my_data2['QL3'].shift(2)) / 3
    my_result['Qdemlag4_for'] = (my_data2['QL1'].shift(3) + my_data2['QL2'].shift(3) + my_data2['QL3'].shift(3)) / 3
    my_result['Qdemlag5_for'] = (my_data2['QL1'].shift(4) + my_data2['QL2'].shift(4) + my_data2['QL3'].shift(4)) / 3
    my_result['Qdemlag6_for'] = (my_data2['QL1'].shift(5) + my_data2['QL2'].shift(5) + my_data2['QL3'].shift(5)) / 3
    my_result['Qdemlag144_for'] = \
        (my_data2['QL1'].shift(143) + my_data2['QL2'].shift(143) + my_data2['QL3'].shift(143)) / 3
    date_now = datetime.strptime(str(year) + '-' + str(month) + '-' + str(day) + " " + str(hour) + ":" + str(minute),
                                 '%Y-%m-%d %H:%M')
    if datetime.strftime(my_result.index[-1], '%Y-%m-%d %H:%M') \
            < datetime.strftime(date_now - timedelta(hours=48), '%Y-%m-%d %H:%M'):
        warnings.warn("The recorded data is older than 24 hours.")
    return my_result


def forecasting_pv_rt(pred_for: list, time_step: int):
    """
    @description: This function is for running the real-time forecasting R code written by Pasquale
    @param pred_for: the list of the predicted values, a dataframe containing predictors irra(t-144),irra(t-2),
    irra(t-3),irra(t-4),irra(t-5),irra(t-6)
    @param time_step: number of the target 10-min interval of the day.
    @return: the list of the predicted values, a dataframe containing predictors irra(t-144),irra(t-2),
        - result_p: forecasted power in kW
        - result_irra: forecasted irradiance in W/m2
    """
    n_boot = 300  # Desired number of bootstrap samples.
    # Call R function
    ro.r['source'](r'src/forecast_lqr_bayesboot_irra.R')
    func_lqr_bayesboot = ro.globalenv['LQR_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_for_r = ro.conversion.py2rpy(pred_for)
    result_irr_r = func_lqr_bayesboot(pred_for_r, time_step, n_boot)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_irra = ro.conversion.rpy2py(result_irr_r)
    result_irra = result_irra[0]
    result_p = (result_irra * 6.21) / 1000
    return result_p, result_irra


def forecasting_active_power_rt(pred_for: list, time_step: int, fac_p: float):
    """
    @description: This function is for running the real-time forecasting R code written by Pasquale
    @param pred_for: a dataframe containing predictors P(t-144),P(t-2),P(t-3),P(t-4),P(t-5),P(t-6)
    @param time_step: number of the target 10-min interval of the day.
    @param fac_p: the factor of the active power
    @return result_Pdem: forecasted active power in kW
    """
    n_boot = 200  # Desired number of bootstrap samples.
    ro.r['source'](r'src/forecast_lqr_bayesboot_p.R')
    func_lqr_bayesboot = ro.globalenv['LQR_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_for_r = ro.conversion.py2rpy(pred_for)
    result_pdem_r = func_lqr_bayesboot(pred_for_r, time_step, n_boot)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_pdem = ro.conversion.rpy2py(result_pdem_r)
    result_pdem = result_pdem[0] * fac_p
    return result_pdem


def forecasting_reactive_power_rt(pred_for: list, time_step: int, fac_q: float):
    """
    @description: This function is for running the real-time forecasting R code written by Pasquale
    @param pred_for: a dataframe containing predictors Q(t-144),Q(t-2),Q(t-3),Q(t-4),Q(t-5),Q(t-6)
    @param time_step: number of the target 10-min interval of the day.
    @param fac_q: the factor of the reactive power
    @return result_qdem: forecasted active power in kW
    """
    n_boot = 200  # Desired number of bootstrap samples.
    # Call R function
    ro.r['source'](r'src/forecast_lqr_bayesboot_q.R')
    func_lqr_bayesboot = ro.globalenv['LQR_Bayesboot']
    with localconverter(ro.default_converter + pandas2ri.converter):
        pred_for_r = ro.conversion.py2rpy(pred_for)
    result_qdem_r = func_lqr_bayesboot(pred_for_r, time_step, n_boot)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_qdem = ro.conversion.rpy2py(result_qdem_r)
    result_qdem = result_qdem[0] * fac_q
    return result_qdem


if __name__ == "__main__":
    DATE = datetime.now()
    VEC_INP = open(r"data/test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
    VEC_OUT, _ = interface_control_digriflex(vec_inp=VEC_INP, date=DATE)
    log.info("The output of the interface is: " + str(VEC_OUT))
