"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
from realtime_alg import *
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import quantecon as qe
import questionary
import scipy.stats as ss
import tqdm


# Global variables
# ----------------------------------------------------------------------------------------------------------------------
dt = datetime.strptime
python64_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/AppData/Local/Programs/Python/Python39/python.exe"
network_name = "Case_4bus_DiGriFlex"
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
# ----------------------------------------------------------------------------------------------------------------------


# Functions
# ----------------------------------------------------------------------------------------------------------------------
def forecasting_pv_da(pred_for: list, n_boot: int):
    """
    @description: This function is for running the day-ahead forecasting R code written by Pasquale
    @param pred_for: forecasted PV power in kW  
    @param n_boot: number of bootstrap samples
    @return: result_PV: forecasted PV power in kW
    """
    ro.r['source'](r'src/forecast_lqr_bayesboot_irra_24h.R')
    func_day_ahead_bayesboot = ro.globalenv['LQR_Bayesboot']
    result_irra, result_po = np.zeros((3, 144)), np.zeros((3, 144))
    for h in tqdm.tqdm(range(1, 145), desc="Forecasting PV power", ncols=100):
        with localconverter(ro.default_converter + pandas2ri.converter):
            pred_for_r = ro.conversion.py2rpy(pred_for[h - 1][:])
        result_irr_r = func_day_ahead_bayesboot(pred_for_r, h, n_boot)
        with localconverter(ro.default_converter + pandas2ri.converter):
            result_irra_0 = ro.conversion.rpy2py(result_irr_r)
        temp = np.transpose(result_irra_0)
        result_irra[0][h - 1] = temp[1][0]
        result_irra[1][h - 1] = temp[2][0] - temp[1][0]
        result_irra[2][h - 1] = temp[1][0] - temp[0][0]
        result_po[0][h - 1] = (result_irra[0][h - 1] * 6.21) / 1000
        result_po[1][h - 1] = (result_irra[1][h - 1] * 6.21) / 1000
        result_po[2][h - 1] = (result_irra[2][h - 1] * 6.21) / 1000
    return result_po, result_irra


def forecasting_active_power_da(pred_for: list, fac_p: float, n_boot: int):
    """
    @description: This function is for running the day-ahead forecasting R code written by Pasquale
    @param pred_for: forecasted PV power in kW
    @param fac_p: factor for active power
    @param n_boot: number of bootstrap samples
    @return: result_pdem: forecasted PV power in kW
    """
    ro.r['source'](r'src/forecast_lqr_bayesboot_p_24h.R')
    func_day_ahead_bayesboot = ro.globalenv['LQR_Bayesboot']
    result_pdem = np.zeros((3, 144))
    for h in tqdm.tqdm(range(1, 145), desc="Forecasting active power", ncols=100):
        with localconverter(ro.default_converter + pandas2ri.converter):
            pred_for_r = ro.conversion.py2rpy(pred_for[h - 1][:])
        result_pdem_r = func_day_ahead_bayesboot(pred_for_r, h, n_boot)
        with localconverter(ro.default_converter + pandas2ri.converter):
            result_pdem_0 = ro.conversion.rpy2py(result_pdem_r)
        temp = np.transpose(result_pdem_0) * fac_p
        result_pdem[0][h - 1] = temp[1][0]
        result_pdem[1][h - 1] = temp[2][0] - temp[1][0]
        result_pdem[2][h - 1] = temp[1][0] - temp[0][0]
    return result_pdem


def forecasting_reactive_power_da(pred_for: list, fac_q: float, n_boot: int):
    """ 
    @description: This function is for running the day-ahead forecasting R code written by Pasquale
    @param pred_for: forecasted PV power in kW
    @param fac_q: factor for reactive power
    @param n_boot: number of bootstrap samples
    @return: result_qdem: forecasted PV power in kW
    """
    ro.r['source'](r'src/forecast_lqr_bayesboot_q_24h.R')
    func_day_ahead_bayesboot = ro.globalenv['LQR_Bayesboot']
    result_qdem = np.zeros((3, 144))
    for h in tqdm.tqdm(range(1, 145), desc="Forecasting reactive power", ncols=100):
        with localconverter(ro.default_converter + pandas2ri.converter):
            pred_for_r = ro.conversion.py2rpy(pred_for[h - 1][:])
        result_qdem_r = func_day_ahead_bayesboot(pred_for_r, h, n_boot)
        with localconverter(ro.default_converter + pandas2ri.converter):
            result_qdem_0 = ro.conversion.rpy2py(result_qdem_r)
        temp = np.transpose(result_qdem_0) * fac_q
        result_qdem[0][h - 1] = temp[1][0]
        result_qdem[1][h - 1] = temp[2][0] - temp[1][0]
        result_qdem[2][h - 1] = temp[1][0] - temp[0][0]
    return result_qdem


def transition_matrix(tran: list, n_dig: int):
    """
    @description: This function is for calculating the transition matrix
    @param tran: time series of the transition matrix
    @param n_dig: number of digits
    @return: m1: transition matrix
    """
    n = 1 + n_dig
    m1 = [[0] * n for _ in range(n)]
    for (i, j) in zip(tran, tran[1:]):
        m1[i][j] += 1
    k = 0
    for row in m1:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
        else:
            row[k] = 1
        k = k + 1
    return m1


def forecasting1(data0: list, name: str):
    """
    @description: This function is for forecasting the PV power
    @param data0: time series of the PV power
    @param name: name of the data
    @return vec_out: forecasted PV power
    """
    data0 = np.nan_to_num(data0, nan=0, posinf=0, neginf=0)
    model = KMeans(n_clusters=3, random_state=0).fit(data0)
    output = model.cluster_centers_
    plt.plot(np.transpose(output))
    plt.legend(['Scenario 1', 'Scenario 2', 'Scenario 3'], fontsize=18)
    plt.ylabel(name, fontsize=18)
    plt.savefig('.cache/figures/f1_' + name + '.pdf', bbox_inches='tight')
    plt.close()
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


def forecasting2(data0: list, name: str, previous_days: int):
    """
    @description: This function is for forecasting the PV power
    @param data0: time series of the PV power
    @param name: name of the data
    @param previous_days: number of previous days
    @return vec_out: forecasted PV power
    """
    data0 = np.nan_to_num(data0, nan=0, posinf=0, neginf=0)
    dd = previous_days
    data0 = np.resize(data0, (dd * 144, 1))
    m_list = data0[288:]
    f_arr = np.transpose(np.resize(np.array([data0[144:dd * 144 - 144].tolist(), data0[:dd * 144 - 288].tolist()]),
                                   (2, (dd - 2) * 144)))
    theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(f_arr), f_arr)), np.matmul(np.transpose(f_arr), m_list))
    error = np.resize(m_list - np.matmul(f_arr, theta), (dd - 2, 144))
    var_m = np.zeros((144, 1))
    for tt in range(144):
        var_m[tt] = np.std(error[:, tt])
    f_arr = np.transpose(np.resize(np.array([data0[dd * 144 - 144:].tolist(), data0[dd * 144 - 288:dd * 144 - 144]
                                            .tolist()]), (2, 144)))
    y_pred = np.matmul(f_arr, theta)
    y_scenario = np.zeros((100, 144))
    for s in range(100):
        for tt in range(144):
            y_scenario[s, tt] = y_pred[tt] + np.random.normal(0, var_m[tt])
    model = KMeans(n_clusters=3, random_state=0).fit(y_scenario)
    output = model.cluster_centers_
    output = np.delete(output, model.predict(np.transpose(y_pred)), 0)
    output = np.append(output, np.transpose(y_pred), axis=0)
    plt.plot(np.transpose(output))
    plt.legend(['Scenario 1', 'Scenario 2', 'Scenario 3'], fontsize=18)
    plt.ylabel(name, fontsize=18)
    plt.savefig('.cache/figures/f2_' + name + '.pdf', bbox_inches='tight')
    plt.close()
    err_p = np.max(output, axis=0) - np.transpose(y_pred)
    err_n = np.transpose(y_pred) - np.min(output, axis=0)
    vec_out = np.zeros((3, 144))
    vec_out[0, :] = np.transpose(y_pred)
    vec_out[1, :] = err_p
    vec_out[2, :] = err_n
    return vec_out


def forecasting3(data0: np.array, name: str, previous_days: int):
    """
    @description: This function is for forecasting the PV power
    @param data0: time series of the PV power
    @param name: name of the forecast method
    @param previous_days: number of previous days
    @return vec_out: forecasted PV power
    """
    data0 = np.nan_to_num(data0, nan=0, posinf=0, neginf=0)
    plt.plot(np.transpose(data0))
    plt.savefig(r'.cache/figures/f3' + name + '_scen.pdf', bbox_inches='tight')
    plt.close()
    data1 = np.resize(data0, (previous_days * 144, 1))
    data = pd.DataFrame(data1, columns=['output'])
    for tt in range(144, 184):
        data[str(tt) + 'h_delay'] = data['output'].shift(periods=tt)
    std = np.std(data1) / data['output'].std()
    mean = np.mean(data1) - data['output'].mean()
    y = data.pop('output')
    train_x, train_y = data, y
    train_x = train_x.fillna(train_x.mean())
    random_forest_reg_model = RandomForestRegressor()
    random_forest_reg_model.fit(train_x, train_y)
    y_pred = random_forest_reg_model.predict(train_x)
    y_pred = y_pred * std + mean
    y_pred0 = y_pred[-144:]
    model = KMeans(n_clusters=3, random_state=0).fit(data0)
    ind = model.predict(np.resize(y_pred0, (1, 144)))
    ind2 = model.predict(data0)
    y = np.transpose(data0[ind2 == ind])
    b = np.min(y)
    l_ind = np.max(y) - np.min(y)
    y2 = np.divide(y - b, l_ind)
    y2 = np.resize(np.transpose(y2), (y2.shape[1]*144, 1))
    y2 = y2[y2 > 0.01]
    digit = 20
    bins = np.arange(0.05, 1, 1/digit)
    transitions = np.digitize(y2, bins, right=True)
    tm = transition_matrix(transitions, digit)
    fig, ax = plt.subplots()
    ax.imshow(tm)
    plt.savefig(r'.cache/figures/im' + name + '.pdf', bbox_inches='tight')
    plt.close()
    mc = qe.MarkovChain(tm)
    y_scen = np.zeros((30, 144))
    for s in range(30):
        if name == 'PV power production (kW)':
            l_ind = np.max(y, axis=1) - np.min(y, axis=1)
            b = 0
            initial_state = 0
        else:
            l_ind = np.max(y) - np.min(y)
            b = np.min(y)
            initial_state = digit / 2
        x = mc.simulate(init=int(initial_state), ts_length=144) / (np.sqrt(np.size(tm))-1)
        y_scen[s, :] = np.multiply(x, l_ind) + b
    model = KMeans(n_clusters=3, random_state=0).fit(y_scen)
    output = model.cluster_centers_
    del_ind = model.predict(np.resize(y_pred0, (1, 144)))
    output = np.delete(output, del_ind, 0)
    output = np.append(output, np.resize(y_pred0, (1, 144)), axis=0)
    plt.plot(np.transpose(output))
    plt.legend(['Scenario 1', 'Scenario 2', 'Scenario 3'], fontsize=18)
    plt.ylabel(name, fontsize=18)
    plt.savefig(r'.cache/figures/f3_' + name + '.pdf', bbox_inches='tight')
    plt.close()
    err_p = np.max(output, axis=0) - y_pred0
    err_n = y_pred0 - np.min(output, axis=0)
    vec_out = np.zeros((3, 144))
    vec_out[0, :] = np.transpose(y_pred0)
    vec_out[1, :] = np.transpose(err_p)
    vec_out[2, :] = np.transpose(err_n)
    return vec_out


def day_ahead_alg(robust_par: float, mode_forecast: str, date: datetime, previous_days: int = 10):
    """
    @description: This function is used to calculate the day-ahead power production of DiGriFlex.
    @param robust_par: The parameters of the robust model.
    @param mode_forecast: The mode of the forecast.
    @param date: The date of the forecast.
    @param previous_days: The number of previous days to be considered.
    @return: The day-ahead power production of DiGriFlex.
    """
    fac_p, fac_q = 0.1, 0.1
    n_boot = 10
    mode_for = "r" if mode_forecast == "BayesBoot" else "b1" if mode_forecast == "Clustering" else "b2" \
        if mode_forecast == "ARIMA" else "mc" if mode_forecast == "MarkovChain" else None
    data_rt = access_data_rt(year=date.year, month=date.month, day=date.day)
    t_now = data_rt.index[-1]
    t_end = data_rt.index[-1].floor('1d') - timedelta(hours=1)
    t_end_y = t_end - timedelta(minutes=10)
    t_now_y = data_rt.index[-1] - timedelta(days=1) + timedelta(minutes=10)
    t_from_1week = t_end - timedelta(days=6) + timedelta(minutes=10)
    t_end_1week = t_end - timedelta(days=5)
    irra_pred_da = [
        list(data_rt['irra'][t_end:t_now]) + list(data_rt['irra'][t_now_y:t_end_y]),
        list(data_rt['pres'][t_end:t_now]) + list(data_rt['pres'][t_now_y:t_end_y]),
        list(data_rt['relh'][t_end:t_now]) + list(data_rt['relh'][t_now_y:t_end_y]),
        list(data_rt['temp'][t_end:t_now]) + list(data_rt['temp'][t_now_y:t_end_y]),
        list(data_rt['wind'][t_end:t_now]) + list(data_rt['wind'][t_now_y:t_end_y]),
        (np.average(data_rt['irra'][t_end:t_now].to_numpy()) * np.ones(144)).tolist(),
        (np.average(data_rt['pres'][t_end:t_now].to_numpy()) * np.ones(144)).tolist(),
        (np.average(data_rt['relh'][t_end:t_now].to_numpy()) * np.ones(144)).tolist(),
        (np.average(data_rt['temp'][t_end:t_now].to_numpy()) * np.ones(144)).tolist(),
        (np.average(data_rt['wind'][t_end:t_now].to_numpy()) * np.ones(144)).tolist()
    ]
    irra_pred_da = list(map(list, zip(*irra_pred_da)))
    pdem_pred_da = [
        list(data_rt['Pdem'][t_end:t_now]) + list(data_rt['Pdem'][t_now_y:t_end_y]),
        list(data_rt['Qdem'][t_end:t_now]) + list(data_rt['Qdem'][t_now_y:t_end_y]),
        list(data_rt['Pdemlag2_for'][t_end:t_now]) + list(data_rt['Pdemlag2_for'][t_now_y:t_end_y]),
        list(data_rt['Qdemlag2_for'][t_end:t_now]) + list(data_rt['Qdemlag2_for'][t_now_y:t_end_y]),
        list(data_rt['Pdem'][t_from_1week:t_end_1week]),
        list(data_rt['Qdem'][t_from_1week:t_end_1week]),
        (np.average(data_rt['Pdem'][t_end:t_now].to_numpy()) * np.ones(144)).tolist(),
        (np.average(data_rt['Qdem'][t_end:t_now].to_numpy()) * np.ones(144)).tolist()
    ]
    pdem_pred_da = list(map(list, zip(*pdem_pred_da)))
    qdem_pred_da = pdem_pred_da
    log.info("Start to calculate the day-ahead power production of DiGriFlex.")
    if mode_for == 'r':
        result_p_pv, result_irr = forecasting_pv_da(pred_for=irra_pred_da, n_boot=n_boot)
        result_p_dm = forecasting_active_power_da(pred_for=pdem_pred_da, fac_p=fac_p, n_boot=n_boot)
        result_q_dm = forecasting_reactive_power_da(pred_for=qdem_pred_da, fac_q=fac_q, n_boot=n_boot)
    elif mode_for == 'b1':
        with tqdm.tqdm(total=5, desc="Forecasting") as pbar:
            dd = previous_days
            t_end = data_rt.index[-1].floor('1d')
            t_from = t_end - timedelta(days=dd) + timedelta(minutes=10)
            pbar.update()
            pv_pred_da = np.resize(data_rt['P'][t_from:t_end].to_numpy(), (dd, 144)) / 1000
            pdem_pred_da = np.resize(data_rt['Pdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
            qdem_pred_da = np.resize(data_rt['Qdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
            pbar.update()
            pbar.set_description("Forecasting PV")
            result_p_pv = forecasting1(data0=pv_pred_da, name='PV power production (kW)')
            pbar.update()
            pbar.set_description("Forecasting Pdem")
            result_p_dm = forecasting1(data0=pdem_pred_da, name='Demand active power (kW)')
            pbar.update()
            pbar.set_description("Forecasting Qdem")
            result_q_dm = forecasting1(data0=qdem_pred_da, name='Demand reactive power (kVar)')
            pbar.update()
            pbar.set_description("Forecasting")
    elif mode_for == 'b2':
        with tqdm.tqdm(total=5, desc="Forecasting") as pbar:
            dd = previous_days
            t_end = data_rt.index[-1].floor('1d')
            t_from = t_end - timedelta(days=dd) + timedelta(minutes=10)
            pv_pred_da = np.resize(data_rt['P'][t_from:t_end].to_numpy(), (dd, 144)) / 1000
            pbar.update()
            pdem_pred_da = np.resize(data_rt['Pdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
            qdem_pred_da = np.resize(data_rt['Qdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
            pbar.update()
            pbar.set_description("Forecasting PV")
            result_p_pv = forecasting2(data0=pv_pred_da, name='PV power production (kW)', previous_days=previous_days)
            pbar.update()
            pbar.set_description("Forecasting Pdem")
            result_p_dm = forecasting2(data0=pdem_pred_da, name='Demand active power (kW)', previous_days=previous_days)
            pbar.update()
            pbar.set_description("Forecasting Qdem")
            result_q_dm = forecasting2(data0=qdem_pred_da, name='Demand reactive power (kVar)',
                                       previous_days=previous_days)
            pbar.update()
            pbar.set_description("Forecasting")
    elif mode_for == 'mc':
        with tqdm.tqdm(total=5, desc="Forecasting") as pbar:
            dd = previous_days
            t_end = data_rt.index[-1].floor('1d')
            t_from = t_end - timedelta(days=dd) + timedelta(minutes=10)
            pbar.update()
            pv_pred_da = np.resize(data_rt['P'][t_from:t_end].to_numpy(), (dd, 144)) / 1000
            pdem_pred_da = np.resize(data_rt['Pdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
            qdem_pred_da = np.resize(data_rt['Qdem'][t_from:t_end].to_numpy(), (dd, 144)) / 10
            pbar.update()
            pbar.set_description("Forecasting PV")
            result_p_pv = forecasting3(data0=pv_pred_da, name='PV power production (kW)', previous_days=previous_days)
            pbar.update()
            pbar.set_description("Forecasting Pdem")
            result_p_dm = forecasting3(data0=pdem_pred_da, name='Demand active power (kW)', previous_days=previous_days)
            pbar.update()
            pbar.set_description("Forecasting Qdem")
            result_q_dm = forecasting3(data0=qdem_pred_da, name='Demand reactive power (kVar)', previous_days=previous_days)
            pbar.update()
            pbar.set_description("Forecasting")
    else:
        result_p_pv = None
        result_p_dm = None
        result_q_dm = None
    result_p_pv = np.maximum(np.zeros((3, 144)), result_p_pv)
    result_v_mag = 0.03 * np.ones((2, 144))
    result_soc = [50, 0.75, 0.75]
    result_price = np.ones((6, 144))
    result_price[0][:] = 10 * result_price[0][:]
    result_price[1][:] = 0 * result_price[1][:]
    result_price[2][:] = 1 * result_price[2][:]
    result_price[3][:] = 1 * result_price[3][:]
    result_price[4][:] = 0.5 * result_price[4][:]
    result_price[5][:] = 0.5 * result_price[5][:]
    grid_inp = grid_topology_sim(network_name, [])
    with open(r".cache/output/tmp_da.pickle", "wb") as file_to_store:
        pickle.dump((grid_inp, result_v_mag, result_p_pv, result_p_dm, result_q_dm, result_soc, result_price,
                     robust_par), file_to_store)
    tomorrow = datetime.now() + timedelta(days=1)
    with open(r".cache/output/for" + tomorrow.strftime("%Y_%m_%d") + ".pickle", "wb") as file_to_store:
        pickle.dump((grid_inp, result_v_mag, result_p_pv, result_p_dm, result_q_dm, result_soc, result_price,
                     robust_par), file_to_store)
    os.system(python64_path +
              ' -c ' +
              '\"import sys;' +
              'sys.path.insert(0, r\'' + os.getcwd() + '\');' +
              'import os;' +
              'import pickle;' +
              'from datetime import datetime;' +
              'from src.optimization_prob import *;' +
              'file_to_read = open(r\'.cache/output/tmp_da.pickle\', \'rb\');' +
              '(grid_inp, v_mag, result_p_pv, result_p_dm, result_q_dm, result_soc, result_price, robust_par) = ' +
              'pickle.load(file_to_read);' +
              'file_to_read.close();' +
              'p_sc, q_sc, rpp_sc, rpn_sc, rqp_sc, rqn_sc, soc_desired, prices, obj = ' +
              'da_opt_digriflex(grid_inp, v_mag, result_p_pv, result_p_dm, result_q_dm, result_soc, result_price, ' +
              'robust_par);' +
              'file_to_store = open(r\'.cache/output/tmp_da.pickle\', \'wb\');' +
              'pickle.dump((p_sc, q_sc, rpp_sc, rpn_sc, rqp_sc, rqn_sc, soc_desired, prices, obj), file_to_store);' +
              'file_to_store.close()\"'
              )
    with open(r".cache/output/tmp_da.pickle", "rb") as file_to_read:
        p_sc, q_sc, rpp_sc, rpn_sc, rqp_sc, rqn_sc, soc_desired, prices, obj = pickle.load(file_to_read)
    if (not os.path.isfile(r".cache/output/res" + tomorrow.strftime("%Y_%m_%d") + ".pickle")) or (obj != 0):
        with open(r".cache/output/res" + tomorrow.strftime("%Y_%m_%d") + ".pickle", "wb") as file_to_store:
            pickle.dump((p_sc, q_sc, rpp_sc, rpn_sc, rqp_sc, rqn_sc, soc_desired, prices, obj), file_to_store)
    return True


# Main
if __name__ == '__main__':
    MODE_FORECAST = questionary.select("Forecasting method?", choices=["BayesBoot", "Clustering", "ARIMA",
                                                                       "MarkovChain"]).ask()
    if MODE_FORECAST in ["Clustering", "ARIMA", "MarkovChain"]:
        NUM_DAYS = int(questionary.text("Number of historical days to use?", default="30").ask())
    else:
        NUM_DAYS = 20
    MODE_OPT = questionary.select("Optimization method?", choices=["Stochastic", "Robust"]).ask()
    if MODE_OPT == "Stochastic":
        ROBUST_PAR = 1.0
    else:
        ROBUST_PAR = float(questionary.text("Confidence level of robust optimization problem?", default="0.8").ask())
    DATE = questionary.text("Date", default=datetime.now().strftime("%Y-%m-%d")).ask()
    flag = day_ahead_alg(robust_par=ROBUST_PAR, mode_forecast=MODE_FORECAST, date=dt(DATE, "%Y-%m-%d"),
                         previous_days=NUM_DAYS)
    log.info("Day_ahead optimization finished: {}".format(flag))
