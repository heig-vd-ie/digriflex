"""@author: MYI, #Python version: 3.9.6 [64 bit]"""
from auxiliary import find_nearest, find_n, figuring
from gurobipy import Model, GRB
from scipy.stats import norm
import coloredlogs
import itertools
import logging
import numpy as np
import pandas as pd


# Global variables
# ----------------------------------------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


# Functions
# ----------------------------------------------------------------------------------------------------------------------
def rt_following_digriflex(grid_inp: dict, p_net: float, q_net: float, forecast_pv: float, forecast_dm: dict,
                           soc: float):
    """
    @description: This function is used to calculate the reactive power following the DiGriFlex model.
    A simple function for testing the control commands
    @param grid_inp: list of lists, the first list is the list of the grid input, the second list is the list of the
    grid output
    @param p_net: float, the net power of the grid
    @param q_net: float, the net reactive power of the grid
    @param forecast_pv: float, the forecast PV power
    @param forecast_dm: float, the forecast demand power
    @param soc: float, the state of charge of the battery
    @return: float, the reactive power following the DiGriFlex model
    Outputs: abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp
    """
    forecast_p = forecast_pv - forecast_dm[0]
    forecast_q = - forecast_dm[1]
    abb_p_cap = grid_inp["PV_elements"][0]["cap_kVA_perPhase"] * 3
    abb_steps = [0, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    battery_p_cap_pos = min([grid_inp["storage_elements"][0]["P_max_kW"],
                             (grid_inp["storage_elements"][0]["SOC_max_kWh"]
                              - soc * grid_inp["storage_elements"][0]["SOC_max_kWh"] / 100) * 6])
    battery_p_cap_neg = max([-grid_inp["storage_elements"][0]["P_max_kW"],
                             (grid_inp["storage_elements"][0]["SOC_min_kWh"]
                              - soc * grid_inp["storage_elements"][0]["SOC_max_kWh"] / 100) * 6])
    battery_q_cap = np.sqrt(grid_inp["storage_elements"][0]["S_max_kVA"] ** 2 -
                            max([battery_p_cap_pos, - battery_p_cap_neg]) ** 2)
    if forecast_p - p_net <= battery_p_cap_pos:
        battery_p_sp = max([- forecast_p + p_net, battery_p_cap_neg])
        abb_p_sp = 100
    else:
        battery_p_sp = battery_p_cap_pos
        abb_p_sp, _ = find_nearest(abb_steps, (battery_p_sp + p_net + forecast_dm[0]) * 100 / abb_p_cap)
    abb_p_exp = min([forecast_pv, abb_p_sp * abb_p_cap / 100])
    battery_q_sp = max([min([forecast_q - q_net, battery_q_cap]), - battery_q_cap])
    abb_q_max = abb_p_exp * np.tan(np.arccos(0.89))
    if - abb_q_max <= q_net + battery_q_sp - forecast_q <= abb_q_max:
        abb_c_sp = - np.cos(np.arctan(q_net + battery_q_sp - forecast_q)) \
                   * np.sign(q_net + battery_q_sp - forecast_q - np.finfo(float).eps)
    else:
        abb_c_sp, _ = find_nearest([-0.89, 0.89], np.sign(q_net + battery_q_sp - forecast_q - np.finfo(float).eps))
        abb_c_sp = - abb_c_sp
    _, abb_p_sp = find_nearest(abb_steps, abb_p_sp)
    f_p = abb_p_exp + battery_p_sp - forecast_dm[0]
    f_q = - np.sin(np.arctan(abb_c_sp)) * abb_p_exp + battery_q_sp + forecast_q
    return abb_p_sp, round(abb_c_sp, 3), round(battery_p_sp, 3), round(battery_q_sp, 3), abb_p_exp, - f_p, - f_q


def rt_opt_digriflex(grid_inp: dict, v_mag: float, p_net: float, q_net: float, forecast_pv: float, forecast_dm: dict,
                     soc_battery: float, soc_desired: float, prices_vec: list):
    """
    @description: this function is used to calculate the reactive power following the digriflex model.
    @param grid_inp: list of lists, the first list is the list of the grid input, the second list is the list of the
    grid output
    @param v_mag: float, the voltage magnitude
    @param p_net: float, the net power of the grid
    @param q_net: float, the net reactive power of the grid
    @param forecast_pv: float, the forecast PV power
    @param forecast_dm: float, the forecast demand power
    @param soc_battery: float, the SOC of the battery
    @param soc_desired: float, the desired SOC of the battery
    @param prices_vec: list, the list of the prices
    @return: float, the reactive power following the digriflex model
    outputs: abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp
    """
    p_net, q_net = - p_net, - q_net
    rt_meas_inp = {}
    meas_inp = dict()
    rt_meas_inp["delta"] = 0.001
    rt_meas_inp["Loss_Coeff"] = prices_vec[0]
    rt_meas_inp["ST_Coeff"] = prices_vec[1]
    rt_meas_inp["PV_Coeff"] = prices_vec[2]
    rt_meas_inp["dev_Coeff"] = prices_vec[3]
    meas_inp["DeltaT"] = 10 / 60
    abb_steps = [0, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    meas_inp["meas_location"] = [{"from": "1", "to": "3", "tran/line": "line"}]
    rt_meas_inp["DM_P"] = {0: forecast_dm[0]}
    rt_meas_inp["DM_Q"] = {0: forecast_dm[1]}
    rt_meas_inp["P_PV"] = forecast_pv / grid_inp["PV_elements"][0]["cap_kVA_perPhase"]
    rt_meas_inp["ST_SOC_t_1"] = {0: soc_battery}
    rt_meas_inp["ST_SOC_des"] = {0: soc_desired}
    rt_meas_inp["Vmag"] = {0: v_mag / 400}
    rt_meas_inp["fac_P_pos"] = {0: 0}
    rt_meas_inp["fac_P_neg"] = {0: 0}
    rt_meas_inp["fac_Q_pos"] = {0: 0}
    rt_meas_inp["fac_Q_neg"] = {0: 0}
    rt_meas_inp["ConPoint_P_DA_EN"] = {0: - p_net}
    rt_meas_inp["ConPoint_Q_DA_EN"] = {0: - q_net}
    rt_meas_inp["ConPoint_P_DA_RS_pos"] = {0: 0}
    rt_meas_inp["ConPoint_P_DA_RS_neg"] = {0: 0}
    rt_meas_inp["ConPoint_Q_DA_RS_pos"] = {0: 0}
    rt_meas_inp["ConPoint_Q_DA_RS_neg"] = {0: 0}
    da_result: dict = {}
    da_result, rt_res = rt_optimization(rt_meas_inp, meas_inp, grid_inp, da_result)
    if not da_result["time_out"]:
        abb_p_sp = da_result["Solution_PV_P"]
        abb_q_sp = da_result["Solution_PV_Q"]
        battery_p_sp = da_result["Solution_ST_P"]
        battery_q_sp = da_result["Solution_ST_Q"]
        f_p = da_result["Solution_con_P"]
        f_q = da_result["Solution_con_Q"]
        abb_p_sp = round(abb_p_sp[0], 3)
        abb_q_sp = round(abb_q_sp[0], 3)
        battery_p_sp = round(battery_p_sp[0], 3)
        battery_q_sp = - round(battery_q_sp[0], 3)
        log.info(f'Success, {abb_p_sp}, {abb_q_sp}, {battery_p_sp}, {battery_q_sp}')
        f_p = f_p[0]
        f_q = f_q[0]
        abb_p_exp = abb_p_sp
        abb_q_max = abb_p_exp * np.tan(np.arccos(0.89))
        if - abb_q_max <= abb_q_sp <= abb_q_max:
            abb_c_sp = - np.cos(np.arctan(abb_q_sp)) * np.sign(abb_q_sp - np.finfo(float).eps)
        else:
            abb_c_sp, _ = - find_nearest([-0.89, 0.89], np.sign(abb_q_sp - np.finfo(float).eps))
        abb_p_cap = grid_inp["PV_elements"][0]["cap_kVA_perPhase"] * 3
        abb_p_sp = abb_p_sp * 100 / abb_p_cap
        if abb_p_exp > forecast_pv - 0.1:
            abb_p_sp = 100
        _, abb_p_sp = find_nearest(abb_steps, abb_p_sp)
    else:
        abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp, abb_p_exp, f_p, f_q = \
            rt_following_digriflex(grid_inp=grid_inp, p_net=p_net, q_net=q_net, forecast_pv=forecast_pv,
                                   forecast_dm=forecast_dm, soc=soc_battery)
        rt_res = 1
    return abb_p_sp, abb_c_sp, battery_p_sp, battery_q_sp, abb_p_exp, f_p, f_q, rt_res


def rt_optimization(rt_meas_inp: dict, meas_inp: dict, grid_inp: dict, da_result: dict):
    """
    @description: the function to run the optimization
    @param rt_meas_inp: list, the list of the real time measurements
    @param meas_inp: list, the list of the measurements
    @param grid_inp: list, the list of the grid information
    @param da_result: list, the list of the DA result
    @return: da_result: list, the list of the DA result
    outputs: da_result, d_res
    """
    big_m = 1e6
    l1 = rt_meas_inp["Loss_Coeff"]
    l2 = rt_meas_inp["ST_Coeff"]
    l3 = rt_meas_inp["PV_Coeff"]
    l4 = rt_meas_inp["dev_Coeff"]
    deltat = meas_inp["DeltaT"]
    node_set = range(grid_inp["Nbus"])
    v_max = []
    v_min = []
    for nn in grid_inp["buses"]:
        v_min.append(nn["Vmin"])
        v_max.append(nn["Vmax"])
    line_set = []
    line_smax = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_vbase = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_zre = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_zim = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_b = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    for nn in grid_inp["transformers"]:
        line_set.append(tuple((find_n(nn["bus_j"], grid_inp["buses"]),
                               find_n(nn["bus_k"], grid_inp["buses"]))))
        line_smax[find_n(nn["bus_j"], grid_inp["buses"]),
                  find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        line_vbase[find_n(nn["bus_j"], grid_inp["buses"]),
                   find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][find_n(nn["bus_k"], grid_inp["buses"])]["U_kV"]
        line_zre[find_n(nn["bus_j"], grid_inp["buses"]),
                 find_n(nn["bus_k"], grid_inp["buses"])] = nn["R_cc_pu"] * grid_inp["Zbase"]
        line_zim[find_n(nn["bus_j"], grid_inp["buses"]),
                 find_n(nn["bus_k"], grid_inp["buses"])] = nn["X_cc_pu"] * grid_inp["Zbase"]
    for nn in grid_inp["lines"]:
        line_set.append(tuple((find_n(nn["bus_j"], grid_inp["buses"]),
                               find_n(nn["bus_k"], grid_inp["buses"]))))
        line_smax[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        line_vbase[find_n(nn["bus_j"], grid_inp["buses"]),
                   find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][find_n(nn["bus_k"], grid_inp["buses"])]["U_kV"]
        line_zre[find_n(nn["bus_j"], grid_inp["buses"]),
                 find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["R1"] * nn["m"] / 1000
        line_zim[find_n(nn["bus_j"], grid_inp["buses"]),
                 find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["X1"] * nn["m"] / 1000
        line_b[find_n(nn["bus_j"], grid_inp["buses"]),
               find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["B_1_mu"] * nn[
            "m"] / 2000
    dm_set = range(np.size(grid_inp["load_elements"], 0))
    dm_inc_mat = np.zeros((np.size(grid_inp["load_elements"], 0), grid_inp["Nbus"]))
    dm_p = np.zeros((np.size(grid_inp["load_elements"], 0)))
    dm_q = np.zeros((np.size(grid_inp["load_elements"], 0)))
    for nn in grid_inp["load_elements"]:
        dm_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        nnn = find_n(nn["bus"], meas_inp["meas_location"])
        dm_p[nn["index"]] = rt_meas_inp["DM_P"][nnn]
        dm_q[nn["index"]] = rt_meas_inp["DM_Q"][nnn]
    log.info("Demand data is generated for optimization.")
    pv_set = range(np.size(grid_inp["PV_elements"], 0))
    pv_inc_mat = np.zeros((np.size(grid_inp["PV_elements"], 0), grid_inp["Nbus"]))
    pv_cap = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    pv_v_grid = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_v_conv = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_x = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_cos = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_forecast = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    for nn in grid_inp["PV_elements"]:
        pv_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        pv_cap[nn["index"]] = nn["cap_kVA_perPhase"]
        pv_v_grid[nn["index"]] = nn["V_grid_pu"]
        pv_v_conv[nn["index"]] = nn["V_conv_pu"]
        pv_x[nn["index"]] = nn["X_PV_pu"]
        pv_cos[nn["index"]] = nn["cos_PV"]
        pv_forecast[nn["index"]] = rt_meas_inp["P_PV"] * nn["cap_kVA_perPhase"]
    log.info("PV data is generated for optimization.")
    st_set = range(np.size(grid_inp["storage_elements"], 0))
    st_inc_mat = np.zeros((np.size(grid_inp["storage_elements"], 0), grid_inp["Nbus"]))
    st_s_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_min = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_lv = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_pos = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_neg = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_lc = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_t_1 = np.zeros((np.size(grid_inp["storage_elements"], 0)))
    st_soc_des = np.zeros((np.size(grid_inp["storage_elements"], 0)))
    for s, nn in enumerate(grid_inp["storage_elements"]):
        st_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        st_s_max[nn["index"]] = nn["S_max_kVA"]
        st_soc_max[nn["index"]] = nn["SOC_max_kWh"]
        st_soc_min[nn["index"]] = nn["SOC_min_kWh"]
        st_eff_lv[nn["index"]] = nn["Eff_LV"]
        st_eff_pos[nn["index"]] = nn["Eff_C"]
        st_eff_neg[nn["index"]] = nn["Eff_D"]
        st_eff_lc[nn["index"]] = nn["Eff_LC"]
        st_soc_t_1[nn["index"]] = rt_meas_inp["ST_SOC_t_1"][s] * 64 / 100
        st_soc_des[nn["index"]] = rt_meas_inp["ST_SOC_des"][s] * 64 / 100
    log.info("Storage data is generated for optimization.")
    conpoint_set = range(np.size(grid_inp["grid_formers"], 0))
    conpoint_inc_mat = np.zeros((np.size(grid_inp["grid_formers"], 0), grid_inp["Nbus"]))
    conpoint_fac_p_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_fac_p_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_fac_q_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_fac_q_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_vmag = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_p_da_en = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_q_da_en = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_p_da_rs_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_p_da_rs_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_q_da_rs_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    conpoint_q_da_rs_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    for nn in grid_inp["grid_formers"]:
        conpoint_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        conpoint_vmag[nn["index"]] = rt_meas_inp["Vmag"][nn["index"]]
        conpoint_fac_p_pos[nn["index"]] = rt_meas_inp["fac_P_pos"][nn["index"]]
        conpoint_fac_p_neg[nn["index"]] = rt_meas_inp["fac_P_neg"][nn["index"]]
        conpoint_fac_q_pos[nn["index"]] = rt_meas_inp["fac_Q_pos"][nn["index"]]
        conpoint_fac_q_neg[nn["index"]] = rt_meas_inp["fac_Q_neg"][nn["index"]]
        conpoint_p_da_en[nn["index"]] = rt_meas_inp["ConPoint_P_DA_EN"][nn["index"]]
        conpoint_q_da_en[nn["index"]] = rt_meas_inp["ConPoint_Q_DA_EN"][nn["index"]]
        conpoint_p_da_rs_pos[nn["index"]] = rt_meas_inp["ConPoint_P_DA_RS_pos"][nn["index"]]
        conpoint_p_da_rs_neg[nn["index"]] = rt_meas_inp["ConPoint_P_DA_RS_neg"][nn["index"]]
        conpoint_q_da_rs_pos[nn["index"]] = rt_meas_inp["ConPoint_Q_DA_RS_pos"][nn["index"]]
        conpoint_q_da_rs_neg[nn["index"]] = rt_meas_inp["ConPoint_Q_DA_RS_neg"][nn["index"]]
    log.info("Connection point data is generated for optimization.")
    prob_rt = Model()
    line_p_t = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_t = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_p_b = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_b = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_f = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    vmag_sq = prob_rt.addVars(node_set, lb=-big_m, ub=big_m)
    line_p_t_hat = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_t_hat = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_p_b_hat = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_b_hat = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_p_t_over = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_t_over = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_p_b_over = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_b_over = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_f_over = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    vmag_sq_over = prob_rt.addVars(node_set, lb=-big_m, ub=big_m)
    line_p_t_sq_max = prob_rt.addVars(line_set, lb=0, ub=big_m)
    line_q_t_sq_max = prob_rt.addVars(line_set, lb=0, ub=big_m)
    line_p_b_sq_max = prob_rt.addVars(line_set, lb=0, ub=big_m)
    line_q_b_sq_max = prob_rt.addVars(line_set, lb=0, ub=big_m)
    line_p_t_abs_max = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_t_abs_max = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_p_b_abs_max = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    line_q_b_abs_max = prob_rt.addVars(line_set, lb=-big_m, ub=big_m)
    pv_p = prob_rt.addVars(pv_set, lb=0, ub=big_m)
    pv_q = prob_rt.addVars(pv_set, lb=-big_m, ub=big_m)
    st_p = prob_rt.addVars(st_set, lb=-big_m, ub=big_m)
    st_q = prob_rt.addVars(st_set, lb=-big_m, ub=big_m)
    st_soc = prob_rt.addVars(st_set, lb=-big_m, ub=big_m)
    st_soc_dev_pos = prob_rt.addVars(st_set, lb=0, ub=big_m)
    st_soc_dev_neg = prob_rt.addVars(st_set, lb=0, ub=big_m)
    st_soc_tilde = prob_rt.addVars(st_set, lb=-big_m, ub=big_m)
    st_p_pos = prob_rt.addVars(st_set, lb=0, ub=big_m)
    st_p_neg = prob_rt.addVars(st_set, lb=-big_m, ub=0)
    net_p = prob_rt.addVars(node_set, lb=-big_m, ub=big_m)
    net_q = prob_rt.addVars(node_set, lb=-big_m, ub=big_m)
    conpoint_p = prob_rt.addVars(conpoint_set, lb=-big_m, ub=big_m)
    conpoint_q = prob_rt.addVars(conpoint_set, lb=-big_m, ub=big_m)
    conpoint_p_dev_pos = prob_rt.addVars(conpoint_set, lb=0)
    conpoint_p_dev_neg = prob_rt.addVars(conpoint_set, lb=0)
    conpoint_q_dev_pos = prob_rt.addVars(conpoint_set, lb=0)
    conpoint_q_dev_neg = prob_rt.addVars(conpoint_set, lb=0)
    obj_rt_market = prob_rt.addVars(conpoint_set, lb=-big_m, ub=big_m)
    for nn in node_set:
        # (1a)
        prob_rt.addConstr(net_p[nn] ==
                          sum(pv_p[i] * pv_inc_mat[i, nn] for i in pv_set)
                          + sum(st_p[s] * st_inc_mat[s, nn] for s in st_set)
                          - sum(dm_p[d] * dm_inc_mat[d, nn] for d in dm_set)
                          + sum(conpoint_p[f] * conpoint_inc_mat[f, nn] for f in conpoint_set))
        # (1b)
        prob_rt.addConstr(net_q[nn] ==
                          sum(pv_q[i] * pv_inc_mat[i, nn] for i in pv_set)
                          + sum(st_q[s] * st_inc_mat[s, nn] for s in st_set)
                          - sum(dm_q[d] * dm_inc_mat[d, nn] for d in dm_set)
                          + sum(conpoint_q[f] * conpoint_inc_mat[f, nn] for f in conpoint_set))
        # (12c) abd (12d) of mostafa
        prob_rt.addConstr(vmag_sq_over[nn] <= v_max[nn] * v_max[nn])
        prob_rt.addConstr(vmag_sq[nn] >= v_min[nn] * v_min[nn])
    for n1, n2 in line_set:
        # (8a) of mostafa
        prob_rt.addConstr(line_p_t[n1, n2] == - net_p[n2]
                          + sum(line_p_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                          + line_zre[n1, n2] * line_f[n1, n2] / 1000)
        prob_rt.addConstr(line_q_t[n1, n2] == - net_q[n2]
                          + sum(line_q_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                          + line_zim[n1, n2] * line_f[n1, n2] / 1000
                          - 1000 * (vmag_sq[n1] + vmag_sq[n2]) * line_b[n1, n2] * (line_vbase[n1, n2] ** 2))
        # (8b) of mostafa
        prob_rt.addConstr((vmag_sq[n2] - vmag_sq[n1]) * (line_vbase[n1, n2] ** 2) ==
                          - 2 * line_zre[n1, n2] * line_p_t[n1, n2] / 1000
                          - 2 * line_zim[n1, n2] * line_q_t[n1, n2] / 1000
                          + 2 * line_zim[n1, n2] * vmag_sq[n1] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2)
                          + (line_zre[n1, n2] * line_zre[n1, n2]
                             + line_zim[n1, n2] * line_zim[n1, n2]) * line_f[n1, n2] / 1000000)
        # (8d) of mostafa
        prob_rt.addConstr(line_p_b[n1, n2] == - net_p[n2]
                          + sum(line_p_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        prob_rt.addConstr(line_q_b[n1, n2] == - net_q[n2]
                          + sum(line_q_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        # (10) of mostafa
        prob_rt.addConstr(line_f[n1, n2] * vmag_sq[n1] * (line_vbase[n1, n2] ** 2) >= line_p_t[n1, n2]
                          * line_p_t[n1, n2]
                          + (line_q_t[n1, n2] + vmag_sq[n1] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2) * 1000)
                          * (line_q_t[n1, n2] + vmag_sq[n1] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2) * 1000))
        # (11a) of mostafa
        prob_rt.addConstr(line_p_t_hat[n1, n2] == - net_p[n2]
                          + sum(line_p_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        prob_rt.addConstr(line_q_t_hat[n1, n2] == - net_q[n2]
                          + sum(line_q_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                          - 1000 * (vmag_sq_over[n1] + vmag_sq_over[n2]) * line_b[n1, n2] * (line_vbase[n1, n2] ** 2))
        # (11b) of mostafa
        prob_rt.addConstr((vmag_sq_over[n2] - vmag_sq_over[n1]) * (line_vbase[n1, n2] ** 2) ==
                          - 2 * line_zre[n1, n2] * line_p_t_hat[n1, n2] / 1000
                          - 2 * line_zim[n1, n2] * line_q_t_hat[n1, n2] / 1000
                          + 2 * line_zim[n1, n2] * vmag_sq_over[n1] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2))
        # (11c) of mostafa
        prob_rt.addConstr(line_p_t_over[n1, n2] == - net_p[n2]
                          + sum(line_p_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                          + line_zre[n1, n2] * line_f_over[n1, n2] / 1000)
        prob_rt.addConstr(line_q_t_over[n1, n2] == - net_q[n2]
                          + sum(line_q_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                          + line_zim[n1, n2] * line_f_over[n1, n2] / 1000
                          - 1000 * (vmag_sq[n1] + vmag_sq[n2]) * line_b[n1, n2] * (line_vbase[n1, n2] ** 2))
        # (11d) of mostafa
        prob_rt.addConstr(line_f_over[n1, n2] * vmag_sq[n2] * (line_vbase[n1, n2] ** 2)
                          >= line_p_b_sq_max[n1, n2] * line_p_b_sq_max[n1, n2]
                          + line_q_b_sq_max[n1, n2] * line_q_b_sq_max[n1, n2])
        prob_rt.addConstr(line_p_b_sq_max[n1, n2] >= line_p_b_hat[n1, n2])
        prob_rt.addConstr(line_p_b_sq_max[n1, n2] >= - line_p_b_hat[n1, n2])
        prob_rt.addConstr(line_p_b_sq_max[n1, n2] >= line_p_b_over[n1, n2])
        prob_rt.addConstr(line_p_b_sq_max[n1, n2] >= - line_p_b_over[n1, n2])
        prob_rt.addConstr(line_q_b_sq_max[n1, n2] >=
                          (line_q_b_hat[n1, n2] - vmag_sq_over[n2] * line_b[n1, n2]
                           * (line_vbase[n1, n2] ** 2) * 1000))
        prob_rt.addConstr(line_q_b_sq_max[n1, n2] >= - (line_q_b_hat[n1, n2] - vmag_sq_over[n2] * line_b[n1, n2]
                                                        * (line_vbase[n1, n2] ** 2) * 1000))
        prob_rt.addConstr(line_q_b_sq_max[n1, n2] >= (line_q_b_over[n1, n2] - vmag_sq[n2] * line_b[n1, n2]
                                                      * (line_vbase[n1, n2] ** 2) * 1000))
        prob_rt.addConstr(line_q_b_sq_max[n1, n2] >= - (line_q_b_over[n1, n2] - vmag_sq[n2] * line_b[n1, n2]
                                                        * (line_vbase[n1, n2] ** 2) * 1000))
        # (11e) of mostafa
        prob_rt.addConstr(line_f_over[n1, n2] * vmag_sq[n1] * (line_vbase[n1, n2] ** 2)
                          >= line_p_t_sq_max[n1, n2] * line_p_t_sq_max[n1, n2]
                          + line_q_t_sq_max[n1, n2] * line_q_t_sq_max[n1, n2])
        prob_rt.addConstr(line_p_t_sq_max[n1, n2] >= line_p_t_hat[n1, n2])
        prob_rt.addConstr(line_p_t_sq_max[n1, n2] >= - line_p_t_hat[n1, n2])
        prob_rt.addConstr(line_p_t_sq_max[n1, n2] >= line_p_t_over[n1, n2])
        prob_rt.addConstr(line_p_t_sq_max[n1, n2] >= - line_p_t_over[n1, n2])
        prob_rt.addConstr(line_q_t_sq_max[n1, n2] >= (line_q_t_hat[n1, n2] + vmag_sq_over[n1] * line_b[n1, n2]
                                                      * (line_vbase[n1, n2] ** 2) * 1000))
        prob_rt.addConstr(line_q_t_sq_max[n1, n2] >= - (line_q_t_hat[n1, n2] + vmag_sq_over[n1] * line_b[n1, n2]
                                                        * (line_vbase[n1, n2] ** 2) * 1000))
        prob_rt.addConstr(line_q_t_sq_max[n1, n2] >= (line_q_t_over[n1, n2] + vmag_sq[n1] * line_b[n1, n2]
                                                      * (line_vbase[n1, n2] ** 2) * 1000))
        prob_rt.addConstr(line_q_t_sq_max[n1, n2] >= - (line_q_t_over[n1, n2] + vmag_sq[n1] * line_b[n1, n2]
                                                        * (line_vbase[n1, n2] ** 2) * 1000))
        # (11f) of mostafa
        prob_rt.addConstr(line_p_b_over[n1, n2] == - net_p[n2]
                          + sum(line_p_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        prob_rt.addConstr(line_q_b_over[n1, n2] == - net_q[n2]
                          + sum(line_q_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        # (11g) of mostafa
        prob_rt.addConstr(line_p_b_hat[n1, n2] == - net_p[n2]
                          + sum(line_p_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        prob_rt.addConstr(line_q_b_hat[n1, n2] == - net_q[n2]
                          + sum(line_q_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
        # (12e) of mostafa
        prob_rt.addConstr(line_p_b_abs_max[n1, n2] * line_p_b_abs_max[n1, n2]
                          + line_q_b_abs_max[n1, n2] * line_q_b_abs_max[n1, n2]
                          <= vmag_sq[n2] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
        prob_rt.addConstr(line_p_b_abs_max[n1, n2] >= line_p_b_hat[n1, n2])
        prob_rt.addConstr(line_p_b_abs_max[n1, n2] >= -line_p_b_hat[n1, n2])
        prob_rt.addConstr(line_p_b_abs_max[n1, n2] >= line_p_b_over[n1, n2])
        prob_rt.addConstr(line_p_b_abs_max[n1, n2] >= -line_p_b_over[n1, n2])
        prob_rt.addConstr(line_q_b_abs_max[n1, n2] >= line_q_b_hat[n1, n2])
        prob_rt.addConstr(line_q_b_abs_max[n1, n2] >= -line_q_b_hat[n1, n2])
        prob_rt.addConstr(line_q_b_abs_max[n1, n2] >= line_q_b_over[n1, n2])
        prob_rt.addConstr(line_q_b_abs_max[n1, n2] >= -line_q_b_over[n1, n2])
        # (12f) of mostafa
        prob_rt.addConstr(line_p_t_abs_max[n1, n2] * line_p_t_abs_max[n1, n2]
                          + line_q_t_abs_max[n1, n2] * line_q_t_abs_max[n1, n2]
                          <= vmag_sq[n1] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
        prob_rt.addConstr(line_p_t_abs_max[n1, n2] >= line_p_t_hat[n1, n2])
        prob_rt.addConstr(line_p_t_abs_max[n1, n2] >= -line_p_t_hat[n1, n2])
        prob_rt.addConstr(line_p_t_abs_max[n1, n2] >= line_p_t_over[n1, n2])
        prob_rt.addConstr(line_p_t_abs_max[n1, n2] >= -line_p_t_over[n1, n2])
        prob_rt.addConstr(line_q_t_abs_max[n1, n2] >= line_q_t_hat[n1, n2])
        prob_rt.addConstr(line_q_t_abs_max[n1, n2] >= -line_q_t_hat[n1, n2])
        prob_rt.addConstr(line_q_t_abs_max[n1, n2] >= line_q_t_over[n1, n2])
        prob_rt.addConstr(line_q_t_abs_max[n1, n2] >= -line_q_t_over[n1, n2])
        # (12g) of mostafa
        prob_rt.addConstr(line_p_t[n1, n2] <= line_p_t_over[n1, n2])
        prob_rt.addConstr(line_q_t[n1, n2] <= line_q_t_over[n1, n2])
    for s in st_set:
        # (21a)
        prob_rt.addConstr(st_soc[s] == st_eff_lv[s] * st_soc_t_1[s]
                          - st_p_neg[s] * deltat * st_eff_neg[s]
                          - st_p_pos[s] * deltat / st_eff_pos[s])
        prob_rt.addConstr(st_soc_tilde[s] == st_eff_lv[s] * st_soc_t_1[s]
                          - st_p[s] * deltat)
        # (22d)
        prob_rt.addConstr(st_p[s] == st_p_pos[s] + st_p_neg[s])
        # (22e)
        prob_rt.addConstr(st_q[s] * st_q[s] + st_p[s] * st_p[s] <= st_s_max[s] * st_s_max[s])
        # (22b)
        prob_rt.addConstr(st_p_pos[s] <= st_s_max[s])
        # (22c)
        prob_rt.addConstr(st_p_neg[s] >= -st_s_max[s])
        # (22f)
        prob_rt.addConstr(st_soc[s] <= st_soc_max[s])
        prob_rt.addConstr(st_soc[s] >= st_soc_min[s])
        # (22h)
        prob_rt.addConstr(st_soc_tilde[s] <= st_soc_max[s])
        prob_rt.addConstr(st_soc_tilde[s] >= st_soc_min[s])
        prob_rt.addConstr(st_soc[s] - st_soc_des[s] == st_soc_dev_pos[s] - st_soc_dev_neg[s])
    for i in pv_set:
        # (28)
        prob_rt.addConstr(pv_p[i] <= pv_forecast[i])
        # (26)
        prob_rt.addConstr(pv_q[i] <= pv_p[i] * np.tan(np.arccos(pv_cos[i])))
        prob_rt.addConstr(pv_q[i] >= - pv_p[i] * np.tan(np.arccos(pv_cos[i])))
    for f in conpoint_set:
        prob_rt.addConstr(conpoint_p[f] == sum(line_p_t[n1, n2] * conpoint_inc_mat[f, n1] for n1, n2 in line_set))
        prob_rt.addConstr(conpoint_q[f] == sum(line_q_t[n1, n2] * conpoint_inc_mat[f, n1] for n1, n2 in line_set))
        # first line of (34a)
        prob_rt.addConstr(obj_rt_market[f] >= l1 * sum(line_zre[n1, n2] * line_f[n1, n2] for n1, n2 in line_set)
                          + l2 * sum(st_soc_dev_pos[s] + st_soc_dev_neg[s] for s in st_set)
                          + l3 * sum(pv_forecast[p] - pv_p[p] for p in pv_set)
                          + l4 * (conpoint_p_dev_pos[f] + conpoint_p_dev_neg[f]
                                  + conpoint_q_dev_pos[f] + conpoint_q_dev_neg[f]))
        prob_rt.addConstr(conpoint_p[f] == conpoint_p_da_en[f]
                          - conpoint_fac_p_pos[f] * conpoint_p_da_rs_pos[f]
                          + conpoint_fac_p_neg[f] * conpoint_p_da_rs_neg[f]
                          - conpoint_p_dev_pos[f] + conpoint_p_dev_neg[f])
        prob_rt.addConstr(conpoint_q[f] == conpoint_q_da_en[f]
                          - conpoint_fac_q_pos[f] * conpoint_q_da_rs_pos[f]
                          + conpoint_fac_q_neg[f] * conpoint_q_da_rs_neg[f]
                          - conpoint_q_dev_pos[f] + conpoint_q_dev_neg[f])
        prob_rt.addConstr(sum(vmag_sq[n] * conpoint_inc_mat[f, n] for n in node_set)
                          == conpoint_vmag[f] * conpoint_vmag[f])
    prob_rt.setObjective(1000 * deltat * obj_rt_market.sum(), GRB.MAXIMIZE)
    prob_rt.Params.BarHomogeneous = 1
    prob_rt.Params.OutputFlag = 0
    prob_rt.optimize()
    d_res = 0
    try:
        solution_con_p = prob_rt.getAttr('x', conpoint_p)
        solution_con_q = prob_rt.getAttr('x', conpoint_q)
        solution_pv_p = prob_rt.getAttr('x', pv_p)
        solution_pv_q = prob_rt.getAttr('x', pv_q)
        solution_st_p = prob_rt.getAttr('x', st_p)
        solution_st_q = prob_rt.getAttr('x', st_q)
        solution_st_soc = prob_rt.getAttr('x', st_soc)
        solution_st_soc0 = [solution_st_soc[s] for s in st_set]
        solution_con_p_dev_pos = prob_rt.getAttr('x', conpoint_p_dev_pos)
        solution_con_p_dev_neg = prob_rt.getAttr('x', conpoint_p_dev_neg)
        solution_con_q_dev_neg = prob_rt.getAttr('x', conpoint_q_dev_neg)
        solution_con_q_dev_pos = prob_rt.getAttr('x', conpoint_q_dev_pos)
        da_result["Solution_dev"] = solution_con_p_dev_neg.sum() > 0.01 + solution_con_q_dev_neg.sum() > 0.01 + \
            solution_con_p_dev_pos.sum() > 0.01 + solution_con_q_dev_pos.sum() > 0.01
        da_result["Solution_ST_SOC_RT"] = solution_st_soc0
        da_result["Solution_PV_P"] = solution_pv_p
        da_result["Solution_PV_Q"] = solution_pv_q
        da_result["Solution_ST_P"] = solution_st_p
        da_result["Solution_ST_Q"] = solution_st_q
        da_result["Solution_con_P"] = solution_con_p
        da_result["Solution_con_Q"] = solution_con_q
        da_result["time_out"] = False
        log.info("Real-time problem is solved.")
    except (Exception,):
        log.warning("Real-time problem is not converged.")
        da_result["Solution_ST_SOC_RT"] = st_soc_t_1
        da_result["time_out"] = True
        d_res = 1
    return da_result, d_res


def da_opt_digriflex(grid_inp: dict, v_mag: list, forecast_pv: list, forecast_p_dm: list, forecast_q_dm: list,
                     forecast_soc: list, prices_vec: list, robust_par: float):
    """
    @description: this function is used to solve the dynamic optimization problem using gurobi
    @param grid_inp: the input grid data
    @param v_mag: the voltage magnitude of the grid
    @param forecast_pv: the forecast PV power
    @param forecast_p_dm: the forecast demand power
    @param forecast_q_dm: the forecast demand reactive power
    @param forecast_soc: the forecast state of charge
    @param prices_vec: the price vector
    @param robust_par: the robust parameter
    @return: the result of the dynamic optimization problem
    """
    case_name = 'digriflex'
    case_inp = {}
    meas_inp = {}
    fore_inp = {}
    output_df = pd.DataFrame()
    case_inp['LAMBDA_P_DA_EN'] = prices_vec[0][:]
    case_inp['LAMBDA_Q_DA_EN'] = prices_vec[1][:]
    case_inp['LAMBDA_P_DA_RS_pos'] = prices_vec[2][:]
    case_inp['LAMBDA_P_DA_RS_neg'] = prices_vec[3][:]
    case_inp['LAMBDA_Q_DA_RS_pos'] = prices_vec[4][:]
    case_inp['LAMBDA_Q_DA_RS_neg'] = prices_vec[5][:]
    case_inp['Robust_prob'] = 1 - robust_par
    case_inp['Omega_Number'] = 3
    case_inp['loss_consideration'] = 0
    meas_inp['Nt'] = 144
    meas_inp['DeltaT'] = 10 / 60
    meas_inp["meas_location"] = [{"from": "1", "to": "3", "tran/line": "line"}]
    fore_inp["Dem_P"] = [forecast_p_dm[0][:].tolist()]
    fore_inp["Dem_Q"] = [forecast_q_dm[1][:].tolist()]
    fore_inp["Dem_P_zeta+"] = [forecast_p_dm[1][:].tolist()]
    fore_inp["Dem_P_zeta-"] = [forecast_p_dm[2][:].tolist()]
    fore_inp["Dem_Q_zeta+"] = [forecast_q_dm[1][:].tolist()]
    fore_inp["Dem_Q_zeta-"] = [forecast_q_dm[2][:].tolist()]
    fore_inp["P_PV"] = forecast_pv[0][:] / grid_inp["PV_elements"][0]["cap_kVA_perPhase"]
    fore_inp["P_PV_zeta+"] = forecast_pv[1][:] / grid_inp["PV_elements"][0]["cap_kVA_perPhase"]
    fore_inp["P_PV_zeta-"] = forecast_pv[2][:] / grid_inp["PV_elements"][0]["cap_kVA_perPhase"]
    fore_inp["P_PV"] = fore_inp["P_PV"].tolist()
    fore_inp["P_PV_zeta+"] = fore_inp["P_PV_zeta+"].tolist()
    fore_inp["P_PV_zeta-"] = fore_inp["P_PV_zeta-"].tolist()
    fore_inp["ST_SOC_0"] = forecast_soc[0] / 100
    fore_inp["ST_SOC_zeta+"] = forecast_soc[1] / 100
    fore_inp["ST_SOC_zeta-"] = forecast_soc[2] / 100
    fore_inp["Vmag_zeta+"] = v_mag[0][:]
    fore_inp["Vmag_zeta-"] = v_mag[1][:]
    fore_inp["confidence"] = 0.99
    if robust_par == 1:
        da_result, _ = da_optimization(case_name, case_inp, grid_inp, meas_inp, fore_inp, output_df)
    else:
        da_result, _ = da_optimization_robust(case_name, case_inp, grid_inp, meas_inp, fore_inp, output_df)
    if not da_result["time_out"]:
        p_sc = da_result["DA_P"]
        q_sc = da_result["DA_Q"]
        rpp_sc = da_result["DA_RP_pos"]
        rpn_sc = da_result["DA_RP_neg"]
        rqp_sc = da_result["DA_RQ_pos"]
        rqn_sc = da_result["DA_RQ_neg"]
        soc_desired = da_result["Solution_ST_SOC"][0]
        obj = da_result["obj"]
    else:
        p_sc, q_sc = [-9] * 144, [2] * 144
        rpp_sc, rpn_sc, rqp_sc, rqn_sc = [0.4] * 144, [0.4] * 144, [0.2] * 144, [0.2] * 144
        soc_desired = [10] * 144
        obj = 0
    prices = [1, 1, 100, 1000]
    return p_sc, q_sc, rpp_sc, rpn_sc, rqp_sc, rqn_sc, soc_desired, prices, obj


def da_optimization_robust(case_name: str, case_inp: dict, grid_inp: dict, meas_inp: dict, fore_inp: dict,
                           output_df: pd.DataFrame):
    """
    @description: this function is used to solve the dynamic optimization problem using gurobi
    @param case_name: the name of the case
    @param case_inp: the input case data
    @param grid_inp: the input grid data
    @param meas_inp: the input measurement data
    @param fore_inp: the input forecast data
    @param output_df: the output dataframe
    @return: the result of the dynamic optimization problem
    @return: d_res
    """
    big_m = 1e6
    lambda_p_da_en = case_inp['LAMBDA_P_DA_EN']
    lambda_q_da_en = case_inp['LAMBDA_Q_DA_EN']
    lambda_p_da_rs_pos = case_inp['LAMBDA_P_DA_RS_pos']
    lambda_p_da_rs_neg = case_inp['LAMBDA_P_DA_RS_neg']
    lambda_q_da_rs_pos = case_inp['LAMBDA_Q_DA_RS_pos']
    lambda_q_da_rs_neg = case_inp['LAMBDA_Q_DA_RS_neg']
    prob = case_inp['Robust_prob']
    conf_multiply = 0
    if prob != 0:
        conf_multiply = np.sqrt((1 - prob) / prob)
    time_set = range(meas_inp["Nt"])
    deltat = meas_inp["DeltaT"]
    node_set = range(grid_inp["Nbus"])
    v_max = []
    v_min = []
    for n in grid_inp["buses"]:
        v_min.append(n["Vmin"])
        v_max.append(n["Vmax"])
    line_set = []
    line_smax = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_vbase = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_zre = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_zim = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_b = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    for n in grid_inp["transformers"]:
        line_set.append(tuple((find_n(n["bus_j"], grid_inp["buses"]),
                               find_n(n["bus_k"], grid_inp["buses"]))))
        line_smax[find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"])] = n["Cap"]
        line_vbase[find_n(n["bus_j"], grid_inp["buses"]),
                   find_n(n["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][find_n(n["bus_k"], grid_inp["buses"])]["U_kV"]
        line_zre[find_n(n["bus_j"], grid_inp["buses"]),
                 find_n(n["bus_k"], grid_inp["buses"])] = n["R_cc_pu"] * grid_inp["Zbase"]
        line_zim[find_n(n["bus_j"], grid_inp["buses"]),
                 find_n(n["bus_k"], grid_inp["buses"])] = n["X_cc_pu"] * grid_inp["Zbase"]
    for n in grid_inp["lines"]:
        line_set.append(tuple((find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"]))))
        line_smax[find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"])] = n["Cap"]
        line_vbase[find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][find_n(n["bus_k"], grid_inp["buses"])]["U_kV"]
        line_zre[find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"])] = \
            grid_inp["line_codes"][n["code"]]["R1"] * n["m"] / 1000
        line_zim[find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"])] = \
            grid_inp["line_codes"][n["code"]]["X1"] * n["m"] / 1000
        line_b[find_n(n["bus_j"], grid_inp["buses"]), find_n(n["bus_k"], grid_inp["buses"])] = \
            grid_inp["line_codes"][n["code"]]["B_1_mu"] * n["m"] / 2000
    dm_set = range(np.size(grid_inp["load_elements"], 0))
    dm_inc_mat = np.zeros((np.size(grid_inp["load_elements"], 0), grid_inp["Nbus"]))
    dm_p = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    dm_q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_p_dm_p = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_n_dm_p = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_p_dm_q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_n_dm_q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_p_dm_p = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_n_dm_p = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_p_dm_q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_n_dm_q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    for n in grid_inp["load_elements"]:
        dm_inc_mat[n["index"], find_n(n["bus"], grid_inp["buses"])] = 1
        for t in time_set:
            nn = find_n(n["bus"], meas_inp["meas_location"])
            dm_p[n["index"]][t] = fore_inp["Dem_P"][nn][t]
            dm_q[n["index"]][t] = fore_inp["Dem_Q"][nn][t]
            zeta_p_dm_p[n["index"]][t] = fore_inp["Dem_P_zeta+"][nn][t]
            zeta_n_dm_p[n["index"]][t] = fore_inp["Dem_P_zeta-"][nn][t]
            zeta_p_dm_q[n["index"]][t] = fore_inp["Dem_Q_zeta+"][nn][t]
            zeta_n_dm_q[n["index"]][t] = fore_inp["Dem_Q_zeta-"][nn][t]
            sigma_p_dm_p[n["index"]][t] = zeta_p_dm_p[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_dm_p[n["index"]][t] = zeta_n_dm_p[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_p_dm_q[n["index"]][t] = zeta_p_dm_q[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_dm_q[n["index"]][t] = zeta_n_dm_q[n["index"]][t] / norm.ppf(fore_inp["confidence"])
    log.info("Demand data is generated for optimization.")
    pv_set = range(np.size(grid_inp["PV_elements"], 0))
    pv_inc_mat = np.zeros((np.size(grid_inp["PV_elements"], 0), grid_inp["Nbus"]))
    pv_cap = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    pv_v_grid = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_v_conv = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_x = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_cos = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_forecast = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    zeta_p_pv = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    zeta_n_pv = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    sigma_p_pv = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    sigma_n_pv = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    for n in grid_inp["PV_elements"]:
        pv_inc_mat[n["index"], find_n(n["bus"], grid_inp["buses"])] = 1
        pv_cap[n["index"]] = n["cap_kVA_perPhase"]
        pv_v_grid[n["index"]] = n["V_grid_pu"]
        pv_v_conv[n["index"]] = n["V_conv_pu"]
        pv_x[n["index"]] = n["X_PV_pu"]
        pv_cos[n["index"]] = n["cos_PV"]
        for t in time_set:
            pv_forecast[n["index"]][t] = fore_inp["P_PV"][t] * n["cap_kVA_perPhase"]
            zeta_p_pv[n["index"]][t] = fore_inp["P_PV_zeta+"][t] * n["cap_kVA_perPhase"]
            zeta_n_pv[n["index"]][t] = fore_inp["P_PV_zeta-"][t] * n["cap_kVA_perPhase"]
            sigma_p_pv[n["index"]][t] = zeta_p_pv[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_pv[n["index"]][t] = zeta_n_pv[n["index"]][t] / norm.ppf(fore_inp["confidence"])
    log.info("PV data is generated for optimization.")
    st_set = range(np.size(grid_inp["storage_elements"], 0))
    st_inc_mat = np.zeros((np.size(grid_inp["storage_elements"], 0), grid_inp["Nbus"]))
    st_s_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_min = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_lv = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_pos = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_neg = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_lc = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_0 = np.zeros((np.size(grid_inp["storage_elements"], 0)))
    zeta_p_st = np.zeros(np.size(grid_inp["storage_elements"], 0))
    zeta_n_st = np.zeros(np.size(grid_inp["storage_elements"], 0))
    for n in grid_inp["storage_elements"]:
        st_inc_mat[n["index"], find_n(n["bus"], grid_inp["buses"])] = 1
        st_s_max[n["index"]] = n["S_max_kVA"]
        st_soc_max[n["index"]] = n["SOC_max_kWh"]
        st_soc_min[n["index"]] = n["SOC_min_kWh"]
        st_eff_lv[n["index"]] = n["Eff_LV"]
        st_eff_pos[n["index"]] = n["Eff_C"]
        st_eff_neg[n["index"]] = n["Eff_D"]
        st_eff_lc[n["index"]] = n["Eff_LC"]
        st_soc_0[n["index"]] = n["SOC_max_kWh"] * fore_inp["ST_SOC_0"]
        zeta_p_st[n["index"]] = fore_inp["ST_SOC_zeta+"] * st_soc_0[n["index"]]
        zeta_n_st[n["index"]] = fore_inp["ST_SOC_zeta-"] * st_soc_0[n["index"]]
    log.info("Storage data is generated for optimization.")
    conpoint_set = range(np.size(grid_inp["grid_formers"], 0))
    conpoint_inc_mat = np.zeros((np.size(grid_inp["grid_formers"], 0), grid_inp["Nbus"]))
    conpoint_vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_p_conpoint_fac_p = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_n_conpoint_fac_p = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_p_conpoint_fac_q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_n_conpoint_fac_q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_p_conpoint_vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_n_conpoint_vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_p_conpoint_fac_p = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_n_conpoint_fac_p = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_p_conpoint_fac_q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_n_conpoint_fac_q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    for n in grid_inp["grid_formers"]:
        conpoint_inc_mat[n["index"], find_n(n["bus"], grid_inp["buses"])] = 1
        for t in time_set:
            conpoint_vmag[n["index"]][t] = 1
            zeta_p_conpoint_fac_p[n["index"]][t] = 1
            zeta_n_conpoint_fac_p[n["index"]][t] = 1
            zeta_p_conpoint_fac_q[n["index"]][t] = 1
            zeta_n_conpoint_fac_q[n["index"]][t] = 1
            zeta_p_conpoint_vmag[n["index"]][t] = fore_inp["Vmag_zeta+"][t]
            zeta_n_conpoint_vmag[n["index"]][t] = fore_inp["Vmag_zeta-"][t]
            sigma_p_conpoint_fac_p[n["index"]][t] = \
                zeta_p_conpoint_fac_p[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_conpoint_fac_p[n["index"]][t] = \
                zeta_n_conpoint_fac_p[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_p_conpoint_fac_q[n["index"]][t] = \
                zeta_p_conpoint_fac_q[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_conpoint_fac_q[n["index"]][t] = \
                zeta_n_conpoint_fac_q[n["index"]][t] / norm.ppf(fore_inp["confidence"])
    log.info("Connection point data is generated for optimization.")
    prob_da = Model()
    line_p_t = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_t = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_p_b = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_b = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_f = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    vmag_sq = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    max_line_p_t = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    max_line_q_t = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    min_line_p_t = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    min_line_q_t = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    max_vmag_sq = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    min_vmag_sq = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    line_p_t_hat = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_t_hat = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_p_b_hat = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_b_hat = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_p_t_over = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_t_over = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_p_b_over = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_b_over = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_f_over = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    vmag_sq_over = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    line_p_t_sq_max = prob_da.addVars(line_set, time_set, lb=0, ub=big_m)
    line_q_t_sq_max = prob_da.addVars(line_set, time_set, lb=0, ub=big_m)
    line_p_b_sq_max = prob_da.addVars(line_set, time_set, lb=0, ub=big_m)
    line_q_b_sq_max = prob_da.addVars(line_set, time_set, lb=0, ub=big_m)
    line_p_t_abs_max = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_t_abs_max = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_p_b_abs_max = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    line_q_b_abs_max = prob_da.addVars(line_set, time_set, lb=-big_m, ub=big_m)
    pv_p = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    pv_q = prob_da.addVars(pv_set, time_set, lb=-big_m, ub=big_m)
    alfa_pv_p_p = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    alfa_pv_p_n = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    alfa_pv_q_p = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    alfa_pv_q_n = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    max_pv_p = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    max_pv_q = prob_da.addVars(pv_set, time_set, lb=-big_m, ub=big_m)
    min_pv_p = prob_da.addVars(pv_set, time_set, lb=0, ub=big_m)
    min_pv_q = prob_da.addVars(pv_set, time_set, lb=-big_m, ub=big_m)
    st_p = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    st_q = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    st_soc = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    st_soc_tilde = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    st_p_pos = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    st_p_neg = prob_da.addVars(st_set, time_set, lb=-big_m, ub=0)
    alfa_st_p_p = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    alfa_st_p_n = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    alfa_st_q_p = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    alfa_st_q_n = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    max_st_p = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    max_st_q = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    max_st_p_pos = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    max_st_p_neg = prob_da.addVars(st_set, time_set, lb=-big_m, ub=0)
    max_st_soc = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    max_st_soc_tilde = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    min_st_p = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    min_st_q = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    min_st_p_pos = prob_da.addVars(st_set, time_set, lb=0, ub=big_m)
    min_st_p_neg = prob_da.addVars(st_set, time_set, lb=-big_m, ub=0)
    min_st_soc = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    min_st_soc_tilde = prob_da.addVars(st_set, time_set, lb=-big_m, ub=big_m)
    net_p = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    net_q = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    zeta_p_net_p = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    zeta_p_net_q = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    zeta_n_net_p = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    zeta_n_net_q = prob_da.addVars(node_set, time_set, lb=-big_m, ub=big_m)
    zeta_p_p = prob_da.addVars(time_set, lb=-big_m, ub=big_m)
    zeta_p_q = prob_da.addVars(time_set, lb=-big_m, ub=big_m)
    zeta_n_p = prob_da.addVars(time_set, lb=-big_m, ub=big_m)
    zeta_n_q = prob_da.addVars(time_set, lb=-big_m, ub=big_m)
    sigma_p_p = prob_da.addVars(time_set, lb=0, ub=big_m)
    sigma_p_q = prob_da.addVars(time_set, lb=0, ub=big_m)
    sigma_n_p = prob_da.addVars(time_set, lb=0, ub=big_m)
    sigma_n_q = prob_da.addVars(time_set, lb=0, ub=big_m)
    # mccormick envelopes vars
    theta_p_p = prob_da.addVars(time_set, lb=0, ub=big_m)
    theta_p_q = prob_da.addVars(time_set, lb=0, ub=big_m)
    theta_n_p = prob_da.addVars(time_set, lb=0, ub=big_m)
    theta_n_q = prob_da.addVars(time_set, lb=0, ub=big_m)
    conpoint_p_da_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    conpoint_q_da_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    alfa_conpoint_p_p = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    alfa_conpoint_p_n = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    alfa_conpoint_q_p = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    alfa_conpoint_q_n = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    max_conpoint_p_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    max_conpoint_q_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    min_conpoint_p_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    min_conpoint_q_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    conpoint_p_da_rs_pos = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_p_da_rs_neg = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_q_da_rs_pos = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_q_da_rs_neg = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    obj_da_market = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    for t in time_set:
        for n in node_set:
            # (1a)
            prob_da.addConstr(net_p[n, t] ==
                              sum(pv_p[i, t] * pv_inc_mat[i, n] for i in pv_set)
                              + sum(st_p[s, t] * st_inc_mat[s, n] for s in st_set)
                              - sum(dm_p[d, t] * dm_inc_mat[d, n] for d in dm_set)
                              + sum(conpoint_p_da_en[f, t] * conpoint_inc_mat[f, n] for f in conpoint_set))
            # (1b)
            prob_da.addConstr(net_q[n, t] ==
                              sum(pv_q[i, t] * pv_inc_mat[i, n] for i in pv_set)
                              + sum(st_q[s, t] * st_inc_mat[s, n] for s in st_set)
                              - sum(dm_q[d, t] * dm_inc_mat[d, n] for d in dm_set)
                              + sum(conpoint_q_da_en[f, t] * conpoint_inc_mat[f, n] for f in conpoint_set))
            # (12c) abd (12d) of mostafa
            prob_da.addConstr(vmag_sq_over[n, t] <= v_max[n] * v_max[n])
            prob_da.addConstr(vmag_sq[n, t] >= v_min[n] * v_min[n])
        for n1, n2 in line_set:
            # (8a) of mostafa
            prob_da.addConstr(line_p_t[n1, n2, t] == - net_p[n2, t]
                              + sum(line_p_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zre[n1, n2] * line_f[n1, n2, t] / 1000)
            prob_da.addConstr(line_q_t[n1, n2, t] == - net_q[n2, t]
                              + sum(line_q_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zim[n1, n2] * line_f[n1, n2, t] / 1000
                              - 1000 * (vmag_sq[n1, t] + vmag_sq[n2, t]) * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (8b) of mostafa
            prob_da.addConstr((vmag_sq[n2, t] - vmag_sq[n1, t]) * (line_vbase[n1, n2] ** 2) ==
                              - 2 * line_zre[n1, n2] * line_p_t[n1, n2, t] / 1000
                              - 2 * line_zim[n1, n2] * line_q_t[n1, n2, t] / 1000
                              + 2 * line_zim[n1, n2] * vmag_sq[n1, t] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2)
                              + (line_zre[n1, n2] * line_zre[n1, n2]
                                 + line_zim[n1, n2] * line_zim[n1, n2]) * line_f[n1, n2, t] / 1000000)
            # (8d) of mostafa
            prob_da.addConstr(line_p_b[n1, n2, t] == - net_p[n2, t]
                              + sum(line_p_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_b[n1, n2, t] == - net_q[n2, t]
                              + sum(line_q_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            # (10) of mostafa
            prob_da.addConstr(
                line_f[n1, n2, t] * vmag_sq[n1, t] * (line_vbase[n1, n2] ** 2) >= line_p_t[n1, n2, t]
                * line_p_t[n1, n2, t]
                + (line_q_t[n1, n2, t] + vmag_sq[n1, t] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2) * 1000)
                * (line_q_t[n1, n2, t] + vmag_sq[n1, t] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2) * 1000))
            # (11a) of mostafa
            prob_da.addConstr(line_p_t_hat[n1, n2, t] == - net_p[n2, t]
                              + sum(line_p_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_t_hat[n1, n2, t] == - net_q[n2, t]
                              + sum(line_q_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              - 1000 * (vmag_sq_over[n1, t] + vmag_sq_over[n2, t]) * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (11b) of mostafa
            prob_da.addConstr((vmag_sq_over[n2, t] - vmag_sq_over[n1, t]) * (line_vbase[n1, n2] ** 2) ==
                              - 2 * line_zre[n1, n2] * line_p_t_hat[n1, n2, t] / 1000
                              - 2 * line_zim[n1, n2] * line_q_t_hat[n1, n2, t] / 1000
                              + 2 * line_zim[n1, n2] * vmag_sq_over[n1, t] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2))
            # (11c) of mostafa
            prob_da.addConstr(line_p_t_over[n1, n2, t] == - net_p[n2, t]
                              + sum(line_p_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zre[n1, n2] * line_f_over[n1, n2, t] / 1000)
            prob_da.addConstr(line_q_t_over[n1, n2, t] == - net_q[n2, t]
                              + sum(line_q_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zim[n1, n2] * line_f_over[n1, n2, t] / 1000
                              - 1000 * (vmag_sq[n1, t] + vmag_sq[n2, t]) * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (11d) of mostafa
            prob_da.addConstr(line_f_over[n1, n2, t] * vmag_sq[n2, t] * (line_vbase[n1, n2] ** 2)
                              >= line_p_b_sq_max[n1, n2, t] * line_p_b_sq_max[n1, n2, t]
                              + line_q_b_sq_max[n1, n2, t] * line_q_b_sq_max[n1, n2, t])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t] >= line_p_b_hat[n1, n2, t])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t] >= - line_p_b_hat[n1, n2, t])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t] >= line_p_b_over[n1, n2, t])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t] >= - line_p_b_over[n1, n2, t])
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t] >=
                              (line_q_b_hat[n1, n2, t] - vmag_sq_over[n2, t] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t] >=
                              - (line_q_b_hat[n1, n2, t] - vmag_sq_over[n2, t] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t] >=
                              (line_q_b_over[n1, n2, t] - vmag_sq[n2, t] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t] >=
                              - (line_q_b_over[n1, n2, t] - vmag_sq[n2, t] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            # (11e) of mostafa
            prob_da.addConstr(line_f_over[n1, n2, t] * vmag_sq[n1, t] * (line_vbase[n1, n2] ** 2)
                              >= line_p_t_sq_max[n1, n2, t] * line_p_t_sq_max[n1, n2, t]
                              + line_q_t_sq_max[n1, n2, t] * line_q_t_sq_max[n1, n2, t])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t] >= line_p_t_hat[n1, n2, t])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t] >= - line_p_t_hat[n1, n2, t])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t] >= line_p_t_over[n1, n2, t])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t] >= - line_p_t_over[n1, n2, t])
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t] >=
                              (line_q_t_hat[n1, n2, t] + vmag_sq_over[n1, t] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t] >=
                              - (line_q_t_hat[n1, n2, t] + vmag_sq_over[n1, t] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t] >=
                              (line_q_t_over[n1, n2, t] + vmag_sq[n1, t] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t] >=
                              - (line_q_t_over[n1, n2, t] + vmag_sq[n1, t] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            # (11f) of mostafa
            prob_da.addConstr(line_p_b_over[n1, n2, t] == - net_p[n2, t]
                              + sum(line_p_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_b_over[n1, n2, t] == - net_q[n2, t]
                              + sum(line_q_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            # (11g) of mostafa
            prob_da.addConstr(line_p_b_hat[n1, n2, t] == - net_p[n2, t]
                              + sum(line_p_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_b_hat[n1, n2, t] == - net_q[n2, t]
                              + sum(line_q_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            # (12e) of mostafa
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t] * line_p_b_abs_max[n1, n2, t]
                              + line_q_b_abs_max[n1, n2, t] * line_q_b_abs_max[n1, n2, t]
                              <= vmag_sq[n2, t] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t] >= line_p_b_hat[n1, n2, t])
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t] >= -line_p_b_hat[n1, n2, t])
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t] >= line_p_b_over[n1, n2, t])
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t] >= -line_p_b_over[n1, n2, t])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t] >= line_q_b_hat[n1, n2, t])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t] >= -line_q_b_hat[n1, n2, t])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t] >= line_q_b_over[n1, n2, t])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t] >= -line_q_b_over[n1, n2, t])
            # (12f) of mostafa
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t] * line_p_t_abs_max[n1, n2, t]
                              + line_q_t_abs_max[n1, n2, t] * line_q_t_abs_max[n1, n2, t]
                              <= vmag_sq[n1, t] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t] >= line_p_t_hat[n1, n2, t])
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t] >= -line_p_t_hat[n1, n2, t])
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t] >= line_p_t_over[n1, n2, t])
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t] >= -line_p_t_over[n1, n2, t])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t] >= line_q_t_hat[n1, n2, t])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t] >= -line_q_t_hat[n1, n2, t])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t] >= line_q_t_over[n1, n2, t])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t] >= -line_q_t_over[n1, n2, t])
            # (12g) of mostafa
            prob_da.addConstr(line_p_t[n1, n2, t] <= line_p_t_over[n1, n2, t])
            prob_da.addConstr(line_q_t[n1, n2, t] <= line_q_t_over[n1, n2, t])
            # (current constraints) of mostafa
            prob_da.addConstr(line_p_t[n1, n2, t] * line_p_t[n1, n2, t]
                              + line_q_t[n1, n2, t] * line_q_t[n1, n2, t]
                              <= vmag_sq[n1, t] * ((line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2)))
            prob_da.addConstr(line_p_b[n1, n2, t] * line_p_b[n1, n2, t]
                              + line_q_b[n1, n2, t] * line_q_b[n1, n2, t]
                              <= vmag_sq[n2, t] * ((line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2)))
        for s in st_set:
            # (22a) and (22g)
            if t == 0:
                prob_da.addConstr(st_soc[s, t] == st_eff_lv[s] * st_soc_0[s]
                                  - st_p_neg[s, t] * deltat * st_eff_neg[s]
                                  - st_p_pos[s, t] * deltat / st_eff_pos[s])
                prob_da.addConstr(st_soc_tilde[s, t] == st_eff_lv[s] * st_soc_0[s]
                                  - st_p[s, t] * deltat)
            else:
                prob_da.addConstr(st_soc[s, t] == st_eff_lv[s] * st_soc[s, t - 1]
                                  - st_p_neg[s, t] * deltat * st_eff_neg[s]
                                  - st_p_pos[s, t] * deltat / st_eff_pos[s])
                prob_da.addConstr(st_soc_tilde[s, t] == st_eff_lv[s] * st_soc[s, t - 1]
                                  - st_p[s, t] * deltat)
            if t == 143:
                prob_da.addConstr(st_soc[s, t] <= st_soc_0[s] + zeta_p_st[s])
                prob_da.addConstr(st_soc[s, t] >= st_soc_0[s] - zeta_n_st[s])
            # (22d)
            prob_da.addConstr(st_p[s, t] == st_p_pos[s, t] + st_p_neg[s, t])
            # (22e)
            prob_da.addConstr(st_q[s, t] * st_q[s, t] + st_p[s, t] * st_p[s, t] <= st_s_max[s] * st_s_max[s])
            # (22b)
            prob_da.addConstr(st_p_pos[s, t] <= st_s_max[s])
            # (22c)
            prob_da.addConstr(st_p_neg[s, t] >= -st_s_max[s])
            # (22f)
            prob_da.addConstr(st_soc[s, t] <= st_soc_max[s])
            prob_da.addConstr(st_soc[s, t] >= st_soc_min[s])
            # (22h)
            prob_da.addConstr(st_soc_tilde[s, t] <= st_soc_max[s])
            prob_da.addConstr(st_soc_tilde[s, t] >= st_soc_min[s])
        for i in pv_set:
            # (26)
            prob_da.addConstr(pv_p[i, t] <= pv_forecast[i, t])
            # (24)
            prob_da.addConstr(pv_q[i, t] <= pv_p[i, t] * np.tan(np.arccos(pv_cos[i])))
            prob_da.addConstr(pv_q[i, t] >= - pv_p[i, t] * np.tan(np.arccos(pv_cos[i])))
        for f in conpoint_set:
            prob_da.addConstr(
                conpoint_p_da_en[f, t] == sum(line_p_t[n1, n2, t] * conpoint_inc_mat[f, n1] for n1, n2 in line_set))
            prob_da.addConstr(
                conpoint_q_da_en[f, t] == sum(line_q_t[n1, n2, t] * conpoint_inc_mat[f, n1] for n1, n2 in line_set))
            # first line of (34a)
            prob_da.addConstr(obj_da_market[f, t] == -lambda_p_da_en[t] * conpoint_p_da_en[f, t]
                              - lambda_q_da_en[t] * conpoint_q_da_en[f, t]
                              + lambda_p_da_rs_pos[t] * conpoint_p_da_rs_pos[f, t]
                              + lambda_p_da_rs_neg[t] * conpoint_p_da_rs_neg[f, t]
                              + lambda_q_da_rs_pos[t] * conpoint_q_da_rs_pos[f, t]
                              + lambda_q_da_rs_neg[t] * conpoint_q_da_rs_neg[f, t])
            # slack bus definition
            prob_da.addConstr(sum(vmag_sq[n, t] * conpoint_inc_mat[f, n] for n in node_set)
                              == conpoint_vmag[f, t] * conpoint_vmag[f, t])
            prob_da.addConstr(conpoint_p_da_rs_pos[f, t] <= line_smax[0, 1] / 2)
            prob_da.addConstr(conpoint_p_da_rs_neg[f, t] <= line_smax[0, 1] / 2)
            prob_da.addConstr(conpoint_q_da_rs_pos[f, t] <= line_smax[0, 1] / 2)
            prob_da.addConstr(conpoint_q_da_rs_neg[f, t] <= line_smax[0, 1] / 2)
        # adjustable robust constraints (max of the net)
        prob_da.addConstr(sigma_p_p[t] * sigma_p_p[t] >= sum(sigma_p_conpoint_fac_p[f, t] * conpoint_p_da_rs_pos[f, t]
                                                             * sigma_p_conpoint_fac_p[f, t] * conpoint_p_da_rs_pos[f, t]
                                                             for f in conpoint_set)
                          + sum(sigma_n_dm_p[d, t] * sigma_n_dm_p[d, t] for d in dm_set)
                          + sum(sigma_n_pv[i, t] * sigma_n_pv[i, t] for i in pv_set))
        prob_da.addConstr(sigma_n_p[t] * sigma_n_p[t] >= sum(sigma_n_conpoint_fac_p[f, t] * conpoint_p_da_rs_neg[f, t]
                                                             * sigma_n_conpoint_fac_p[f, t] * conpoint_p_da_rs_neg[f, t]
                                                             for f in conpoint_set)
                          + sum(sigma_p_dm_p[d, t] * sigma_p_dm_p[d, t] for d in dm_set)
                          + sum(sigma_p_pv[i, t] * sigma_p_pv[i, t] for i in pv_set))
        prob_da.addConstr(sigma_p_q[t] * sigma_p_q[t] >= sum(sigma_p_conpoint_fac_q[f, t] * conpoint_q_da_rs_pos[f, t]
                                                             * sigma_p_conpoint_fac_q[f, t] * conpoint_q_da_rs_pos[f, t]
                                                             for f in conpoint_set)
                          + sum(sigma_n_dm_q[d, t] * sigma_n_dm_q[d, t] for d in dm_set))
        prob_da.addConstr(sigma_n_q[t] * sigma_n_q[t] >= sum(sigma_n_conpoint_fac_q[f, t] * conpoint_q_da_rs_neg[f, t]
                                                             * sigma_n_conpoint_fac_q[f, t] * conpoint_q_da_rs_neg[f, t]
                                                             for f in conpoint_set)
                          + sum(sigma_p_dm_q[d, t] * sigma_p_dm_q[d, t] for d in dm_set))
        prob_da.addConstr(
            zeta_p_p[t] == sum(zeta_p_conpoint_fac_p[f, t] * conpoint_p_da_rs_pos[f, t] for f in conpoint_set)
            + sum(zeta_n_dm_p[d, t] for d in dm_set) + sum(zeta_n_pv[i, t] for i in pv_set))
        prob_da.addConstr(
            zeta_n_p[t] == sum(zeta_n_conpoint_fac_p[f, t] * conpoint_p_da_rs_neg[f, t] for f in conpoint_set)
            + sum(zeta_p_dm_p[d, t] for d in dm_set) + sum(zeta_p_pv[i, t] for i in pv_set))
        prob_da.addConstr(
            zeta_p_q[t] == sum(zeta_p_conpoint_fac_q[f, t] * conpoint_q_da_rs_pos[f, t] for f in conpoint_set)
            + sum(zeta_n_dm_q[d, t] for d in dm_set))
        prob_da.addConstr(
            zeta_n_q[t] == sum(zeta_n_conpoint_fac_q[f, t] * conpoint_q_da_rs_neg[f, t] for f in conpoint_set)
            + sum(zeta_p_dm_q[d, t] for d in dm_set))
        # mccormick envelopes
        prob_da.addConstr(sum(alfa_conpoint_p_p[f, t] for f in conpoint_set) == zeta_p_p[t] - theta_p_p[t])
        prob_da.addConstr(sum(alfa_conpoint_p_n[f, t] for f in conpoint_set) == zeta_n_p[t] - theta_n_p[t])
        prob_da.addConstr(sum(alfa_conpoint_q_p[f, t] for f in conpoint_set) == zeta_p_q[t] - theta_p_q[t])
        prob_da.addConstr(sum(alfa_conpoint_q_n[f, t] for f in conpoint_set) == zeta_n_q[t] - theta_n_q[t])
        if prob == 0:
            prob_da.addConstr(theta_p_p[t] >= zeta_p_p[t])
            prob_da.addConstr(theta_n_p[t] >= zeta_n_p[t])
            prob_da.addConstr(theta_p_q[t] >= zeta_p_q[t])
            prob_da.addConstr(theta_n_q[t] >= zeta_n_q[t])
        else:
            prob_da.addConstr(theta_p_p[t] >= sigma_p_p[t] * conf_multiply)
            prob_da.addConstr(theta_n_p[t] >= sigma_n_p[t] * conf_multiply)
            prob_da.addConstr(theta_p_q[t] >= sigma_p_q[t] * conf_multiply)
            prob_da.addConstr(theta_n_q[t] >= sigma_n_q[t] * conf_multiply)
        prob_da.addConstr(zeta_p_p[t] == sum(alfa_st_p_p[s, t] for s in st_set) + sum(alfa_pv_p_p[i, t] for i in pv_set)
                          + sum(alfa_conpoint_p_n[f, t] for f in conpoint_set))
        prob_da.addConstr(zeta_n_p[t] == sum(alfa_st_p_n[s, t] for s in st_set) + sum(alfa_pv_p_n[i, t] for i in pv_set)
                          + sum(alfa_conpoint_p_p[f, t] for f in conpoint_set))
        prob_da.addConstr(zeta_p_q[t] == sum(alfa_st_q_p[s, t] for s in st_set) + sum(alfa_pv_q_p[i, t] for i in pv_set)
                          + sum(alfa_conpoint_q_n[f, t] for f in conpoint_set))
        prob_da.addConstr(zeta_n_q[t] == sum(alfa_st_q_n[s, t] for s in st_set) + sum(alfa_pv_q_n[i, t] for i in pv_set)
                          + sum(alfa_conpoint_q_p[f, t] for f in conpoint_set))
        # adjustable robust constraints (storages)
        for s in st_set:
            prob_da.addConstr(max_st_p[s, t] == st_p[s, t] + alfa_st_p_p[s, t])
            prob_da.addConstr(min_st_p[s, t] == st_p[s, t] - alfa_st_p_n[s, t])
            prob_da.addConstr(max_st_q[s, t] == st_q[s, t] + alfa_st_q_p[s, t])
            prob_da.addConstr(min_st_q[s, t] == st_q[s, t] - alfa_st_q_n[s, t])
            prob_da.addConstr(max_st_p[s, t] == max_st_p_pos[s, t] + max_st_p_neg[s, t])
            prob_da.addConstr(min_st_p[s, t] == min_st_p_pos[s, t] + min_st_p_neg[s, t])
            if t == 0:
                prob_da.addConstr(max_st_soc[s, t] == st_eff_lv[s] * st_soc_0[s] + st_eff_lv[s] * zeta_p_st[s]
                                  - min_st_p_neg[s, t] * deltat * st_eff_neg[s]
                                  - min_st_p_pos[s, t] * deltat / st_eff_pos[s])
                prob_da.addConstr(max_st_soc_tilde[s, t] == st_eff_lv[s] * st_soc_0[s] + st_eff_lv[s] * zeta_p_st[s]
                                  - min_st_p_neg[s, t] * deltat
                                  - min_st_p_pos[s, t] * deltat)
                prob_da.addConstr(min_st_soc[s, t] == st_eff_lv[s] * st_soc_0[s] - st_eff_lv[s] * zeta_n_st[s]
                                  - max_st_p_neg[s, t] * deltat * st_eff_neg[s]
                                  - max_st_p_pos[s, t] * deltat / st_eff_pos[s])
                prob_da.addConstr(min_st_soc_tilde[s, t] == st_eff_lv[s] * st_soc_0[s] - st_eff_lv[s] * zeta_n_st[s]
                                  - max_st_p_neg[s, t] * deltat
                                  - max_st_p_pos[s, t] * deltat)
            else:
                prob_da.addConstr(max_st_soc[s, t] == st_eff_lv[s] * max_st_soc[s, t - 1]
                                  - min_st_p_neg[s, t] * deltat * st_eff_neg[s]
                                  - min_st_p_pos[s, t] * deltat / st_eff_pos[s])
                prob_da.addConstr(max_st_soc_tilde[s, t] == st_eff_lv[s] * max_st_soc[s, t - 1]
                                  - min_st_p_neg[s, t] * deltat
                                  - min_st_p_pos[s, t] * deltat)
                prob_da.addConstr(min_st_soc[s, t] == st_eff_lv[s] * min_st_soc[s, t - 1]
                                  - max_st_p_neg[s, t] * deltat * st_eff_neg[s]
                                  - max_st_p_pos[s, t] * deltat / st_eff_pos[s])
                prob_da.addConstr(min_st_soc_tilde[s, t] == st_eff_lv[s] * min_st_soc[s, t - 1]
                                  - max_st_p_neg[s, t] * deltat
                                  - max_st_p_pos[s, t] * deltat)
            prob_da.addConstr(max_st_q[s, t] * max_st_q[s, t] + max_st_p[s, t] * max_st_p[s, t]
                              <= st_s_max[s] * st_s_max[s])
            prob_da.addConstr(min_st_q[s, t] * min_st_q[s, t] + max_st_p[s, t] * max_st_p[s, t]
                              <= st_s_max[s] * st_s_max[s])
            prob_da.addConstr(max_st_q[s, t] * max_st_q[s, t] + min_st_p[s, t] * min_st_p[s, t]
                              <= st_s_max[s] * st_s_max[s])
            prob_da.addConstr(min_st_q[s, t] * min_st_q[s, t] + min_st_p[s, t] * min_st_p[s, t]
                              <= st_s_max[s] * st_s_max[s])
            prob_da.addConstr(max_st_p_pos[s, t] <= st_s_max[s])
            prob_da.addConstr(min_st_p_neg[s, t] >= -st_s_max[s])
            prob_da.addConstr(max_st_soc[s, t] <= st_soc_max[s])
            prob_da.addConstr(min_st_soc[s, t] >= st_soc_min[s])
            prob_da.addConstr(max_st_soc_tilde[s, t] <= st_soc_max[s])
            prob_da.addConstr(min_st_soc_tilde[s, t] >= st_soc_min[s])
        # adjustable robust constraints (pv)
        for i in pv_set:
            prob_da.addConstr(max_pv_p[i, t] == pv_p[i, t] + zeta_p_pv[i, t] + alfa_pv_p_p[i, t])
            prob_da.addConstr(min_pv_p[i, t] == pv_p[i, t] - zeta_n_pv[i, t] - alfa_pv_p_n[i, t])
            prob_da.addConstr(max_pv_q[i, t] == pv_q[i, t] + alfa_pv_q_p[i, t])
            prob_da.addConstr(min_pv_q[i, t] == pv_q[i, t] - alfa_pv_q_n[i, t])
            prob_da.addConstr(max_pv_p[i, t] <= pv_forecast[i, t] + zeta_p_pv[i, t])
            prob_da.addConstr(max_pv_q[i, t] <= max_pv_p[i, t] * np.tan(np.arccos(pv_cos[i])))
            prob_da.addConstr(max_pv_q[i, t] >= - max_pv_p[i, t] * np.tan(np.arccos(pv_cos[i])))
            prob_da.addConstr(min_pv_q[i, t] <= max_pv_p[i, t] * np.tan(np.arccos(pv_cos[i])))
            prob_da.addConstr(min_pv_q[i, t] >= - max_pv_p[i, t] * np.tan(np.arccos(pv_cos[i])))
        # adjustable robust constraints (connection point)
        for f in conpoint_set:
            prob_da.addConstr(max_conpoint_p_en[f, t] == conpoint_p_da_en[f, t] + alfa_conpoint_p_n[f, t])
            prob_da.addConstr(min_conpoint_p_en[f, t] == conpoint_p_da_en[f, t] - alfa_conpoint_p_p[f, t])
            prob_da.addConstr(max_conpoint_q_en[f, t] == conpoint_q_da_en[f, t] + alfa_conpoint_q_n[f, t])
            prob_da.addConstr(min_conpoint_q_en[f, t] == conpoint_q_da_en[f, t] - alfa_conpoint_q_p[f, t])
            # adjustable robust constraints (slack def)
            prob_da.addConstr(sum((max_vmag_sq[nn, t] - vmag_sq[nn, t]) * conpoint_inc_mat[f, nn] for nn in node_set)
                              == 2 * vmag_sq[0, t] * zeta_p_conpoint_vmag[f, t])
            prob_da.addConstr(sum((vmag_sq[nn, t] - min_vmag_sq[nn, t]) * conpoint_inc_mat[f, nn] for nn in node_set)
                              == 2 * zeta_n_conpoint_vmag[f, t] * vmag_sq[0, t])
            # adjustable robust constraints (nodes and branches)
        for n in node_set:
            prob_da.addConstr(
                zeta_p_net_p[n, t] == sum((zeta_p_pv[i, t] + alfa_pv_p_p[i, t]) * pv_inc_mat[i, n] for i in pv_set)
                + sum(alfa_st_p_p[s, t] * st_inc_mat[s, n] for s in st_set)
                + sum(zeta_n_dm_p[d, t] * dm_inc_mat[d, n] for d in dm_set)
                + sum(alfa_conpoint_p_n[f, t] * conpoint_inc_mat[f, n] for f in conpoint_set))
            prob_da.addConstr(
                zeta_n_net_p[n, t] == sum((zeta_n_pv[i, t] + alfa_pv_p_n[i, t]) * pv_inc_mat[i, n] for i in pv_set)
                + sum(alfa_st_p_n[s, t] * st_inc_mat[s, n] for s in st_set)
                + sum(zeta_p_dm_p[d, t] * dm_inc_mat[d, n] for d in dm_set)
                + sum(alfa_conpoint_p_p[f, t] * conpoint_inc_mat[f, n] for f in conpoint_set))
            prob_da.addConstr(zeta_p_net_q[n, t] == sum(alfa_pv_q_p[i, t] * pv_inc_mat[i, n] for i in pv_set)
                              + sum(alfa_st_q_p[s, t] * st_inc_mat[s, n] for s in st_set)
                              + sum(zeta_n_dm_q[d, t] * dm_inc_mat[d, n] for d in dm_set)
                              + sum(alfa_conpoint_q_n[f, t] * conpoint_inc_mat[f, n] for f in conpoint_set))
            prob_da.addConstr(zeta_n_net_q[n, t] == sum(alfa_pv_q_n[i, t] * pv_inc_mat[i, n] for i in pv_set)
                              + sum(alfa_st_q_n[s, t] * st_inc_mat[s, n] for s in st_set)
                              + sum(zeta_p_dm_q[d, t] * dm_inc_mat[d, n] for d in dm_set)
                              + sum(alfa_conpoint_q_p[f, t] * conpoint_inc_mat[f, n] for f in conpoint_set))
            prob_da.addConstr(max_vmag_sq[n, t] <= v_max[n] * v_max[n])
            prob_da.addConstr(min_vmag_sq[n, t] >= v_min[n] * v_min[n])
        for n1, n2 in line_set:
            prob_da.addConstr(max_line_p_t[n1, n2, t] == line_p_t[n1, n2, t] + zeta_n_net_p[n2, t]
                              + sum((max_line_p_t[n3, n4, t] - line_p_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in line_set))
            prob_da.addConstr(min_line_p_t[n1, n2, t] == line_p_t[n1, n2, t] - zeta_p_net_p[n2, t]
                              - sum((line_p_t[n3, n4, t] - min_line_p_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in line_set))
            prob_da.addConstr(max_line_q_t[n1, n2, t] == line_q_t[n1, n2, t] + zeta_n_net_q[n2, t]
                              + sum((max_line_q_t[n3, n4, t] - line_q_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in line_set))
            prob_da.addConstr(min_line_q_t[n1, n2, t] == line_q_t[n1, n2, t] - zeta_p_net_q[n2, t]
                              - sum((line_q_t[n3, n4, t] - min_line_q_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in line_set))
            prob_da.addConstr((max_vmag_sq[n2, t] - max_vmag_sq[n1, t]) * (line_vbase[n1, n2] ** 2) ==
                              - 2 * line_zre[n1, n2] * max_line_p_t[n1, n2, t] / 1000
                              - 2 * line_zim[n1, n2] * max_line_q_t[n1, n2, t] / 1000)
            prob_da.addConstr((min_vmag_sq[n2, t] - min_vmag_sq[n1, t]) * (line_vbase[n1, n2] ** 2) ==
                              - 2 * line_zre[n1, n2] * min_line_p_t[n1, n2, t] / 1000
                              - 2 * line_zim[n1, n2] * min_line_q_t[n1, n2, t] / 1000)
            prob_da.addConstr(max_line_p_t[n1, n2, t] * max_line_p_t[n1, n2, t]
                              + max_line_q_t[n1, n2, t] * max_line_q_t[n1, n2, t]
                              <= min_vmag_sq[n1, t] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
            prob_da.addConstr(min_line_p_t[n1, n2, t] * min_line_p_t[n1, n2, t]
                              + min_line_q_t[n1, n2, t] * min_line_q_t[n1, n2, t]
                              <= min_vmag_sq[n1, t] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
    prob_da.setObjective(1 * deltat * obj_da_market.sum(), GRB.MAXIMIZE)
    prob_da.Params.OutputFlag = 0
    prob_da.optimize()
    da_result = {}
    try:
        solution_pv_p = prob_da.getAttr('x', pv_p)
        solution_pv_q = prob_da.getAttr('x', pv_q)
        solution_st_p = prob_da.getAttr('x', st_p)
        solution_st_q = prob_da.getAttr('x', st_q)
        solution_st_soc = prob_da.getAttr('x', st_soc)
        da_p = prob_da.getAttr('x', conpoint_p_da_en)
        da_q = prob_da.getAttr('x', conpoint_q_da_en)
        da_rp_pos = prob_da.getAttr('x', conpoint_p_da_rs_pos)
        da_rp_neg = prob_da.getAttr('x', conpoint_p_da_rs_neg)
        da_rq_pos = prob_da.getAttr('x', conpoint_q_da_rs_pos)
        da_rq_neg = prob_da.getAttr('x', conpoint_q_da_rs_neg)
        solution_st_soc1 = [[solution_st_soc[s, t] for t in time_set] for s in st_set]
        solution_st_pp = [[solution_st_p[s, t] for t in time_set] for s in st_set]
        solution_st_qq = [[solution_st_q[s, t] for t in time_set] for s in st_set]
        solution_pv_pp = [[solution_pv_p[i, t] for t in time_set] for i in pv_set]
        solution_pv_qq = [[solution_pv_q[i, t] for t in time_set] for i in pv_set]
        da_pp = [da_p[0, t] for t in time_set]
        da_qq = [da_q[0, t] for t in time_set]
        da_rpp_pos = [da_rp_pos[0, t] for t in time_set]
        da_rpp_neg = [da_rp_neg[0, t] for t in time_set]
        da_rqq_pos = [da_rq_pos[0, t] for t in time_set]
        da_rqq_neg = [da_rq_neg[0, t] for t in time_set]
        meas = dict()
        meas["DA_PP"] = -np.array(da_pp)
        meas["DA_QQ"] = -np.array(da_qq)
        meas["DA_P+"] = -np.array(da_pp) + np.array(da_rpp_pos)
        meas["DA_Q+"] = -np.array(da_qq) + np.array(da_rqq_pos)
        meas["DA_P-"] = -np.array(da_pp) - np.array(da_rpp_neg)
        meas["DA_Q-"] = -np.array(da_qq) - np.array(da_rqq_neg)
        figuring(grid_inp=grid_inp, meas_inp=meas_inp, meas=meas, fig_type="DA_Offers",
                 title="robust" + str(1 - round(prob, 3)))
        da_result["Solution_PV_P"] = solution_pv_pp
        da_result["Solution_PV_Q"] = solution_pv_qq
        da_result["Solution_ST_P"] = solution_st_pp
        da_result["Solution_ST_Q"] = solution_st_qq
        da_result["Solution_ST_SOC"] = solution_st_soc1
        da_result["DA_P"] = da_pp
        da_result["DA_Q"] = da_qq
        da_result["DA_RP_pos"] = da_rpp_pos
        da_result["DA_RP_neg"] = da_rpp_neg
        da_result["DA_RQ_pos"] = da_rqq_pos
        da_result["DA_RQ_neg"] = da_rqq_neg
        da_result["delta"] = 0.001
        obj = prob_da.getObjective()
        da_result["obj"] = obj.getValue()
        output_df.loc['obj', case_name] = obj.getValue()
        output_df.loc['DA_P_avg', case_name] = sum(da_pp) / len(da_pp)
        output_df.loc['DA_Q_avg', case_name] = sum(da_qq) / len(da_qq)
        output_df.loc['DA_RP_pos_avg', case_name] = sum(da_rpp_pos) / len(da_rpp_pos)
        output_df.loc['DA_RP_neg_avg', case_name] = sum(da_rpp_neg) / len(da_rpp_neg)
        output_df.loc['DA_RQ_pos_avg', case_name] = sum(da_rqq_pos) / len(da_rqq_pos)
        output_df.loc['DA_RQ_neg_avg', case_name] = sum(da_rqq_neg) / len(da_rqq_neg)
        da_result["time_out"] = False
        log.info("Dayahead problem is solved with final objective " + str(obj.getValue()) + ".")
    except (Exception,):
        log.warning("Dayahead problem is not converged.")
        da_result["time_out"] = True
    return da_result, output_df


def da_optimization(case_name: str, case_inp: dict, grid_inp: dict, meas_inp: dict, fore_inp: dict,
                    output_df: pd.DataFrame):
    """
    @description: DA optimization function for the given case and grid input
    @param case_name: name of the case
    @param case_inp: input of the case
    @param grid_inp: input of the grid
    @param meas_inp: input of the measurements
    @param fore_inp: input of the forecast
    @param output_df: output dataframe
    @return da_result: DA optimization result
    @return d_res: DA optimization result
    """
    big_m = 1e6
    omega_number = case_inp['Omega_Number']
    lambda_p_da_en = case_inp['LAMBDA_P_DA_EN']
    lambda_q_da_en = case_inp['LAMBDA_Q_DA_EN']
    lambda_p_da_rs_pos = case_inp['LAMBDA_P_DA_RS_pos']
    lambda_p_da_rs_neg = case_inp['LAMBDA_P_DA_RS_neg']
    lambda_q_da_rs_pos = case_inp['LAMBDA_Q_DA_RS_pos']
    lambda_q_da_rs_neg = case_inp['LAMBDA_Q_DA_RS_neg']
    loss_consideration = case_inp['loss_consideration']
    scen_omega_set = range(omega_number)
    time_set = range(meas_inp["Nt"])
    deltat = meas_inp["DeltaT"]
    node_set = range(grid_inp["Nbus"])
    v_max = []
    v_min = []
    for nn in grid_inp["buses"]:
        v_min.append(nn["Vmin"])
        v_max.append(nn["Vmax"])
    line_set = []
    line_smax = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_vbase = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_zre = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_zim = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    line_b = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    for nn in grid_inp["transformers"]:
        line_set.append(tuple((find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"]))))
        line_smax[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        line_vbase[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][find_n(nn["bus_k"], grid_inp["buses"])]["U_kV"]
        line_zre[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            nn["R_cc_pu"] * grid_inp["Zbase"]
        line_zim[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            nn["X_cc_pu"] * grid_inp["Zbase"]
    for nn in grid_inp["lines"]:
        line_set.append(tuple((find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"]))))
        line_smax[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        line_vbase[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][find_n(nn["bus_k"], grid_inp["buses"])]["U_kV"]
        line_zre[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["line_codes"][nn["code"]]["R1"] * nn["m"] / 1000
        line_zim[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["line_codes"][nn["code"]]["X1"] * nn["m"] / 1000
        line_b[find_n(nn["bus_j"], grid_inp["buses"]), find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["line_codes"][nn["code"]]["B_1_mu"] * nn[
            "m"] / 2000
    dm_set = range(np.size(grid_inp["load_elements"], 0))
    dm_inc_mat = np.zeros((np.size(grid_inp["load_elements"], 0), grid_inp["Nbus"]))
    dm_p = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"], omega_number))
    dm_q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"], omega_number))
    for nn in grid_inp["load_elements"]:
        dm_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        for t in time_set:
            for om in scen_omega_set:
                nnn = find_n(nn["bus"], meas_inp["meas_location"])
                if om == 0:
                    dm_p[nn["index"]][t][om] = fore_inp["Dem_P"][nnn][t]
                    dm_q[nn["index"]][t][om] = fore_inp["Dem_Q"][nnn][t]
                elif om == 1:
                    dm_p[nn["index"]][t][om] = fore_inp["Dem_P"][nnn][t] + fore_inp["Dem_P_zeta+"][nnn][t]
                    dm_q[nn["index"]][t][om] = fore_inp["Dem_Q"][nnn][t] + fore_inp["Dem_Q_zeta+"][nnn][t]
                elif om == 2:
                    dm_p[nn["index"]][t][om] = fore_inp["Dem_P"][nnn][t] - fore_inp["Dem_P_zeta-"][nnn][t]
                    dm_q[nn["index"]][t][om] = fore_inp["Dem_Q"][nnn][t] - fore_inp["Dem_Q_zeta-"][nnn][t]
                elif om > 2:
                    dm_p[nn["index"]][t][om] = fore_inp["Dem_P"][nnn][t] + fore_inp["Dem_P_zeta+"][nnn][t] \
                                               - (fore_inp["Dem_P_zeta+"][nnn][t] + fore_inp["Dem_P_zeta-"][nnn][t]) \
                                               * np.random.randint(0, 2)
                    dm_q[nn["index"]][t][om] = fore_inp["Dem_Q"][nnn][t] + fore_inp["Dem_Q_zeta+"][nnn][t] \
                        - (fore_inp["Dem_Q_zeta+"][nnn][t] + fore_inp["Dem_Q_zeta-"][nnn][t]) * np.random.randint(0, 2)
    log.info("Demand data is generated for optimization.")
    pv_set = range(np.size(grid_inp["PV_elements"], 0))
    pv_inc_mat = np.zeros((np.size(grid_inp["PV_elements"], 0), grid_inp["Nbus"]))
    pv_cap = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    pv_v_grid = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_v_conv = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_x = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_cos = np.zeros(np.size(grid_inp["PV_elements"], 0))
    pv_forecast = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"], omega_number))
    for nn in grid_inp["PV_elements"]:
        pv_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        pv_cap[nn["index"]] = nn["cap_kVA_perPhase"]
        pv_v_grid[nn["index"]] = nn["V_grid_pu"]
        pv_v_conv[nn["index"]] = nn["V_conv_pu"]
        pv_x[nn["index"]] = nn["X_PV_pu"]
        pv_cos[nn["index"]] = nn["cos_PV"]
        for t in time_set:
            for om in scen_omega_set:
                if om == 0:
                    pv_forecast[nn["index"]][t][om] = fore_inp["P_PV"][t] * nn["cap_kVA_perPhase"]
                elif om == 1:
                    pv_forecast[nn["index"]][t][om] = (max([0, fore_inp["P_PV"][t] - fore_inp["P_PV_zeta-"][t]])) \
                                                      * nn["cap_kVA_perPhase"]
                elif om == 2:
                    pv_forecast[nn["index"]][t][om] = (fore_inp["P_PV"][t] + fore_inp["P_PV_zeta+"][t]) \
                                                      * nn["cap_kVA_perPhase"]
                elif om > 2:
                    pv_forecast[nn["index"]][t][om] = (fore_inp["P_PV"][t] + fore_inp["P_PV_zeta+"][t]
                                                       - (fore_inp["P_PV_zeta+"][t] + fore_inp["P_PV_zeta-"][t])
                                                       * np.random.randint(0, 2)) * nn["cap_kVA_perPhase"]
    log.info("PV data is generated for optimization.")
    st_set = range(np.size(grid_inp["storage_elements"], 0))
    st_inc_mat = np.zeros((np.size(grid_inp["storage_elements"], 0), grid_inp["Nbus"]))
    st_s_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_min = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_lv = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_pos = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_neg = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_eff_lc = np.zeros(np.size(grid_inp["storage_elements"], 0))
    st_soc_0 = np.zeros((np.size(grid_inp["storage_elements"], 0), omega_number))
    for nn in grid_inp["storage_elements"]:
        st_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        st_s_max[nn["index"]] = nn["S_max_kVA"]
        st_soc_max[nn["index"]] = nn["SOC_max_kWh"]
        st_soc_min[nn["index"]] = nn["SOC_min_kWh"]
        st_eff_lv[nn["index"]] = nn["Eff_LV"]
        st_eff_pos[nn["index"]] = nn["Eff_C"]
        st_eff_neg[nn["index"]] = nn["Eff_D"]
        st_eff_lc[nn["index"]] = nn["Eff_LC"]
        for om in scen_omega_set:
            if om == 0:
                st_soc_0[nn["index"]][om] = nn["SOC_max_kWh"] * fore_inp["ST_SOC_0"]
            elif om == 1:
                st_soc_0[nn["index"]][om] = nn["SOC_max_kWh"] * fore_inp["ST_SOC_0"] * (1 - fore_inp["ST_SOC_zeta-"])
            elif om == 2:
                st_soc_0[nn["index"]][om] = nn["SOC_max_kWh"] * fore_inp["ST_SOC_0"] * (1 + fore_inp["ST_SOC_zeta+"])
            elif om > 2:
                st_soc_0[nn["index"]][om] = st_soc_0[nn["index"]][om] * (1 + fore_inp["ST_SOC_zeta+"]
                                                                         - (fore_inp["ST_SOC_zeta+"]
                                                                            + fore_inp["ST_SOC_zeta-"])
                                                                         * np.random.randint(0, 2))
    log.info("Storage data is generated for optimization.")
    conpoint_set = range(np.size(grid_inp["grid_formers"], 0))
    conpoint_inc_mat = np.zeros((np.size(grid_inp["grid_formers"], 0), grid_inp["Nbus"]))
    conpoint_fac_p_pos = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], omega_number))
    conpoint_fac_p_neg = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], omega_number))
    conpoint_fac_q_pos = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], omega_number))
    conpoint_fac_q_neg = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], omega_number))
    conpoint_vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], omega_number))
    for nn in grid_inp["grid_formers"]:
        conpoint_inc_mat[nn["index"], find_n(nn["bus"], grid_inp["buses"])] = 1
        for t in time_set:
            for om in scen_omega_set:
                if om == 0:
                    conpoint_vmag[nn["index"]][t][om] = 1.0
                    conpoint_fac_p_pos[nn["index"]][t][om] = 0
                    conpoint_fac_p_neg[nn["index"]][t][om] = 0
                    conpoint_fac_q_pos[nn["index"]][t][om] = 0
                    conpoint_fac_q_neg[nn["index"]][t][om] = 0
                elif om == 1:
                    conpoint_vmag[nn["index"]][t][om] = 1.0 - fore_inp["Vmag_zeta-"][t]
                    conpoint_fac_p_pos[nn["index"]][t][om] = 1
                    conpoint_fac_p_neg[nn["index"]][t][om] = 0
                    conpoint_fac_q_pos[nn["index"]][t][om] = 1
                    conpoint_fac_q_neg[nn["index"]][t][om] = 0
                elif om == 2:
                    conpoint_vmag[nn["index"]][t][om] = 1.0 + fore_inp["Vmag_zeta+"][t]
                    conpoint_fac_p_pos[nn["index"]][t][om] = 0
                    conpoint_fac_p_neg[nn["index"]][t][om] = 1
                    conpoint_fac_q_pos[nn["index"]][t][om] = 0
                    conpoint_fac_q_neg[nn["index"]][t][om] = 1
                elif om > 2:
                    conpoint_fac_p_pos[nn["index"]][t][om] = np.random.randint(0, 2)
                    conpoint_fac_p_neg[nn["index"]][t][om] = 1 - conpoint_fac_p_pos[nn["index"]][t][om]
                    conpoint_fac_q_pos[nn["index"]][t][om] = np.random.randint(0, 2)
                    conpoint_fac_q_neg[nn["index"]][t][om] = 1 - conpoint_fac_q_pos[nn["index"]][t][om]
                    conpoint_vmag[nn["index"]][t][om] = 1 + fore_inp["Vmag_zeta+"][t] \
                        - (fore_inp["Vmag_zeta-"][t] + fore_inp["Vmag_zeta+"][t]) * np.random.randint(0, 2)
    log.info("Connection point data is generated for optimization.")
    prob_da = Model()
    line_p_t = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_t = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_b = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_b = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_f = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    vmag_sq = prob_da.addVars(node_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_t_hat = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_t_hat = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_b_hat = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_b_hat = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_t_over = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_t_over = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_b_over = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_b_over = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_f_over = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    vmag_sq_over = prob_da.addVars(node_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_t_sq_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=0, ub=big_m)
    line_q_t_sq_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=0, ub=big_m)
    line_p_b_sq_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=0, ub=big_m)
    line_q_b_sq_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=0, ub=big_m)
    line_p_t_abs_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_t_abs_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_p_b_abs_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_q_b_abs_max = prob_da.addVars(line_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    line_rel1 = prob_da.addVars(line_set, time_set, scen_omega_set, lb=0, ub=big_m)
    line_rel2 = prob_da.addVars(line_set, time_set, scen_omega_set, lb=0, ub=big_m)
    pv_p = prob_da.addVars(pv_set, time_set, scen_omega_set, lb=0, ub=big_m)
    pv_q = prob_da.addVars(pv_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    st_p = prob_da.addVars(st_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    st_q = prob_da.addVars(st_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    st_soc = prob_da.addVars(st_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    st_soc_tilde = prob_da.addVars(st_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    st_p_pos = prob_da.addVars(st_set, time_set, scen_omega_set, lb=0, ub=big_m)
    st_p_neg = prob_da.addVars(st_set, time_set, scen_omega_set, lb=-big_m, ub=0)
    net_p = prob_da.addVars(node_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    net_q = prob_da.addVars(node_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    conpoint_p_da_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    conpoint_q_da_en = prob_da.addVars(conpoint_set, time_set, lb=-big_m, ub=big_m)
    conpoint_p_da_rs_pos = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_p_da_rs_neg = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_q_da_rs_pos = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_q_da_rs_neg = prob_da.addVars(conpoint_set, time_set, lb=0, ub=big_m)
    conpoint_p = prob_da.addVars(conpoint_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    conpoint_q = prob_da.addVars(conpoint_set, time_set, scen_omega_set, lb=-big_m, ub=big_m)
    conpoint_p_dev_pos = prob_da.addVars(conpoint_set, time_set, scen_omega_set, lb=0, ub=big_m)
    conpoint_p_dev_neg = prob_da.addVars(conpoint_set, time_set, scen_omega_set, lb=0, ub=big_m)
    conpoint_q_dev_pos = prob_da.addVars(conpoint_set, time_set, scen_omega_set, lb=0, ub=big_m)
    conpoint_q_dev_neg = prob_da.addVars(conpoint_set, time_set, scen_omega_set, lb=0, ub=big_m)
    obj_da_market = prob_da.addVars(conpoint_set, time_set)
    obj_loss = prob_da.addVars(line_set, time_set, scen_omega_set)
    for t, om in itertools.product(time_set, scen_omega_set):
        for nn in node_set:
            # (1a)
            prob_da.addConstr(net_p[nn, t, om] ==
                              sum(pv_p[i, t, om] * pv_inc_mat[i, nn] for i in pv_set)
                              + sum(st_p[s, t, om] * st_inc_mat[s, nn] for s in st_set)
                              - sum(dm_p[d, t, om] * dm_inc_mat[d, nn] for d in dm_set)
                              + sum(conpoint_p[f, t, om] * conpoint_inc_mat[f, nn] for f in conpoint_set))
            # (1b)
            prob_da.addConstr(net_q[nn, t, om] ==
                              sum(pv_q[i, t, om] * pv_inc_mat[i, nn] for i in pv_set)
                              + sum(st_q[s, t, om] * st_inc_mat[s, nn] for s in st_set)
                              - sum(dm_q[d, t, om] * dm_inc_mat[d, nn] for d in dm_set)
                              + sum(conpoint_q[f, t, om] * conpoint_inc_mat[f, nn] for f in conpoint_set))
            # (12c) abd (12d) of mostafa
            prob_da.addConstr(vmag_sq_over[nn, t, om] <= v_max[nn] * v_max[nn])
            prob_da.addConstr(vmag_sq[nn, t, om] >= v_min[nn] * v_min[nn])
        for n1, n2 in line_set:
            # (8a) of mostafa
            prob_da.addConstr(line_p_t[n1, n2, t, om] == - net_p[n2, t, om]
                              + sum(line_p_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zre[n1, n2] * line_f[n1, n2, t, om] / 1000)
            prob_da.addConstr(line_q_t[n1, n2, t, om] == - net_q[n2, t, om]
                              + sum(line_q_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zim[n1, n2] * line_f[n1, n2, t, om] / 1000
                              - 1000 * (vmag_sq[n1, t, om] + vmag_sq[n2, t, om]) * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (8b) of mostafa
            prob_da.addConstr((vmag_sq[n2, t, om] - vmag_sq[n1, t, om]) * (line_vbase[n1, n2] ** 2) ==
                              - 2 * line_zre[n1, n2] * line_p_t[n1, n2, t, om] / 1000
                              - 2 * line_zim[n1, n2] * line_q_t[n1, n2, t, om] / 1000
                              + 2 * line_zim[n1, n2] * vmag_sq[n1, t, om] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2)
                              + (line_zre[n1, n2] * line_zre[n1, n2]
                                 + line_zim[n1, n2] * line_zim[n1, n2]) * line_f[n1, n2, t, om] / 1000000)
            # (8d) of mostafa
            prob_da.addConstr(line_p_b[n1, n2, t, om] == - net_p[n2, t, om]
                              + sum(line_p_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_b[n1, n2, t, om] == - net_q[n2, t, om]
                              + sum(line_q_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            # (10) of mostafa
            prob_da.addConstr(
                line_f[n1, n2, t, om] * vmag_sq[n1, t, om] * (line_vbase[n1, n2] ** 2) >= line_p_t[n1, n2, t, om]
                * line_p_t[n1, n2, t, om]
                + (line_q_t[n1, n2, t, om] + vmag_sq[n1, t, om] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2) * 1000)
                * (line_q_t[n1, n2, t, om] + vmag_sq[n1, t, om] * line_b[n1, n2] * (line_vbase[n1, n2] ** 2) * 1000))
            # (11a) of mostafa
            prob_da.addConstr(line_p_t_hat[n1, n2, t, om] == - net_p[n2, t, om]
                              + sum(line_p_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_t_hat[n1, n2, t, om] == - net_q[n2, t, om]
                              + sum(line_q_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              - 1000 * (vmag_sq_over[n1, t, om] + vmag_sq_over[n2, t, om]) * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (11b) of mostafa
            prob_da.addConstr((vmag_sq_over[n2, t, om] - vmag_sq_over[n1, t, om]) * (line_vbase[n1, n2] ** 2) ==
                              - 2 * line_zre[n1, n2] * line_p_t_hat[n1, n2, t, om] / 1000
                              - 2 * line_zim[n1, n2] * line_q_t_hat[n1, n2, t, om] / 1000
                              + 2 * line_zim[n1, n2] * vmag_sq_over[n1, t, om] * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (11c) of mostafa
            prob_da.addConstr(line_p_t_over[n1, n2, t, om] == - net_p[n2, t, om]
                              + sum(line_p_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zre[n1, n2] * line_f_over[n1, n2, t, om] / 1000)
            prob_da.addConstr(line_q_t_over[n1, n2, t, om] == - net_q[n2, t, om]
                              + sum(line_q_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set)
                              + line_zim[n1, n2] * line_f_over[n1, n2, t, om] / 1000
                              - 1000 * (vmag_sq[n1, t, om] + vmag_sq[n2, t, om]) * line_b[n1, n2] * (
                                      line_vbase[n1, n2] ** 2))
            # (11d) of mostafa
            prob_da.addConstr(line_f_over[n1, n2, t, om] * vmag_sq[n2, t, om] * (line_vbase[n1, n2] ** 2)
                              >= line_p_b_sq_max[n1, n2, t, om] * line_p_b_sq_max[n1, n2, t, om]
                              + line_q_b_sq_max[n1, n2, t, om] * line_q_b_sq_max[n1, n2, t, om])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t, om] >= line_p_b_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t, om] >= - line_p_b_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t, om] >= line_p_b_over[n1, n2, t, om])
            prob_da.addConstr(line_p_b_sq_max[n1, n2, t, om] >= - line_p_b_over[n1, n2, t, om])
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t, om] >=
                              (line_q_b_hat[n1, n2, t, om] - vmag_sq_over[n2, t, om] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t, om] >=
                              - (line_q_b_hat[n1, n2, t, om] - vmag_sq_over[n2, t, om] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t, om] >=
                              (line_q_b_over[n1, n2, t, om] - vmag_sq[n2, t, om] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_b_sq_max[n1, n2, t, om] >=
                              - (line_q_b_over[n1, n2, t, om] - vmag_sq[n2, t, om] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            # (11e) of mostafa
            prob_da.addConstr(line_f_over[n1, n2, t, om] * vmag_sq[n1, t, om] * (line_vbase[n1, n2] ** 2)
                              >= line_p_t_sq_max[n1, n2, t, om] * line_p_t_sq_max[n1, n2, t, om]
                              + line_q_t_sq_max[n1, n2, t, om] * line_q_t_sq_max[n1, n2, t, om])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t, om] >= line_p_t_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t, om] >= - line_p_t_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t, om] >= line_p_t_over[n1, n2, t, om])
            prob_da.addConstr(line_p_t_sq_max[n1, n2, t, om] >= - line_p_t_over[n1, n2, t, om])
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t, om] >=
                              (line_q_t_hat[n1, n2, t, om] + vmag_sq_over[n1, t, om] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t, om] >=
                              - (line_q_t_hat[n1, n2, t, om] + vmag_sq_over[n1, t, om] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t, om] >=
                              (line_q_t_over[n1, n2, t, om] + vmag_sq[n1, t, om] * line_b[n1, n2]
                               * (line_vbase[n1, n2] ** 2) * 1000))
            prob_da.addConstr(line_q_t_sq_max[n1, n2, t, om] >=
                              - (line_q_t_over[n1, n2, t, om] + vmag_sq[n1, t, om] * line_b[n1, n2]
                                 * (line_vbase[n1, n2] ** 2) * 1000))
            # (11f) of mostafa
            prob_da.addConstr(line_p_b_over[n1, n2, t, om] == - net_p[n2, t, om]
                              + sum(line_p_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_b_over[n1, n2, t, om] == - net_q[n2, t, om]
                              + sum(line_q_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            # (11g) of mostafa
            prob_da.addConstr(line_p_b_hat[n1, n2, t, om] == - net_p[n2, t, om]
                              + sum(line_p_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            prob_da.addConstr(line_q_b_hat[n1, n2, t, om] == - net_q[n2, t, om]
                              + sum(line_q_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in line_set))
            # (12e) of mostafa
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t, om] * line_p_b_abs_max[n1, n2, t, om]
                              + line_q_b_abs_max[n1, n2, t, om] * line_q_b_abs_max[n1, n2, t, om]
                              - line_rel1[n1, n2, t, om]
                              <= vmag_sq[n2, t, om] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t, om] >= line_p_b_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t, om] >= -line_p_b_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t, om] >= line_p_b_over[n1, n2, t, om])
            prob_da.addConstr(line_p_b_abs_max[n1, n2, t, om] >= -line_p_b_over[n1, n2, t, om])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t, om] >= line_q_b_hat[n1, n2, t, om])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t, om] >= -line_q_b_hat[n1, n2, t, om])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t, om] >= line_q_b_over[n1, n2, t, om])
            prob_da.addConstr(line_q_b_abs_max[n1, n2, t, om] >= -line_q_b_over[n1, n2, t, om])
            # (12f) of mostafa
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t, om] * line_p_t_abs_max[n1, n2, t, om]
                              + line_q_t_abs_max[n1, n2, t, om] * line_q_t_abs_max[n1, n2, t, om]
                              - line_rel2[n1, n2, t, om]
                              <= vmag_sq[n1, t, om] * (line_smax[n1, n2] ** 2) * 9 / (1000 * line_vbase[n1, n2] ** 2))
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t, om] >= line_p_t_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t, om] >= -line_p_t_hat[n1, n2, t, om])
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t, om] >= line_p_t_over[n1, n2, t, om])
            prob_da.addConstr(line_p_t_abs_max[n1, n2, t, om] >= -line_p_t_over[n1, n2, t, om])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t, om] >= line_q_t_hat[n1, n2, t, om])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t, om] >= -line_q_t_hat[n1, n2, t, om])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t, om] >= line_q_t_over[n1, n2, t, om])
            prob_da.addConstr(line_q_t_abs_max[n1, n2, t, om] >= -line_q_t_over[n1, n2, t, om])
            # (12g) of mostafa
            prob_da.addConstr(line_p_t[n1, n2, t, om] <= line_p_t_over[n1, n2, t, om])
            prob_da.addConstr(line_q_t[n1, n2, t, om] <= line_q_t_over[n1, n2, t, om])
        for s in st_set:
            # (21a)
            if t == 0:
                prob_da.addConstr(st_soc[s, t, om] == st_eff_lv[s] * st_soc_0[s, om]
                                  - st_p_neg[s, t, om] * deltat * st_eff_neg[s]
                                  - st_p_pos[s, t, om] * deltat / st_eff_pos[s])
                prob_da.addConstr(st_soc_tilde[s, t, om] == st_eff_lv[s] * st_soc_0[s, om]
                                  - st_p[s, t, om] * deltat)
            else:
                prob_da.addConstr(st_soc[s, t, om] == st_eff_lv[s] * st_soc[s, t - 1, om]
                                  - st_p_neg[s, t, om] * deltat * st_eff_neg[s]
                                  - st_p_pos[s, t, om] * deltat / st_eff_pos[s])
                prob_da.addConstr(st_soc_tilde[s, t, om] == st_eff_lv[s] * st_soc[s, t - 1, om]
                                  - st_p[s, t, om] * deltat)
            # (22d)
            prob_da.addConstr(st_p[s, t, om] == st_p_pos[s, t, om] + st_p_neg[s, t, om])
            # (22e)
            prob_da.addConstr(
                st_q[s, t, om] * st_q[s, t, om] + st_p[s, t, om] * st_p[s, t, om] <= st_s_max[s] * st_s_max[s])
            # (22b)
            prob_da.addConstr(st_p_pos[s, t, om] <= st_s_max[s])
            # (22c)
            prob_da.addConstr(st_p_neg[s, t, om] >= -st_s_max[s])
            # (22f)
            prob_da.addConstr(st_soc[s, t, om] <= st_soc_max[s])
            prob_da.addConstr(st_soc[s, t, om] >= st_soc_min[s])
            # (22h)
            prob_da.addConstr(st_soc_tilde[s, t, om] <= st_soc_max[s])
            prob_da.addConstr(st_soc_tilde[s, t, om] >= st_soc_min[s])
        for i in pv_set:
            # (28)
            prob_da.addConstr(pv_p[i, t, om] <= pv_forecast[i, t, om])
            # (26)
            prob_da.addConstr(pv_q[i, t, om] <= pv_p[i, t, om] * np.tan(np.arccos(pv_cos[i])))
            prob_da.addConstr(pv_q[i, t, om] >= - pv_p[i, t, om] * np.tan(np.arccos(pv_cos[i])))
        for f in conpoint_set:
            prob_da.addConstr(
                conpoint_p[f, t, om] == sum(line_p_t[n1, n2, t, om] * conpoint_inc_mat[f, n1] for n1, n2 in line_set))
            prob_da.addConstr(
                conpoint_q[f, t, om] == sum(line_q_t[n1, n2, t, om] * conpoint_inc_mat[f, n1] for n1, n2 in line_set))
            # first line of (34a)
            prob_da.addConstr(obj_da_market[f, t] == - lambda_p_da_en[t] * conpoint_p_da_en[f, t]
                              - lambda_q_da_en[t] * conpoint_q_da_en[f, t]
                              + lambda_p_da_rs_pos[t] * conpoint_p_da_rs_pos[f, t]
                              + lambda_p_da_rs_neg[t] * conpoint_p_da_rs_neg[f, t]
                              + lambda_q_da_rs_pos[t] * conpoint_q_da_rs_pos[f, t]
                              + lambda_q_da_rs_neg[t] * conpoint_q_da_rs_neg[f, t]
                              - max([lambda_p_da_en[t], lambda_q_da_en[t], lambda_p_da_rs_pos[t], lambda_p_da_rs_neg[t],
                                     lambda_q_da_rs_pos[t], lambda_q_da_rs_neg[t]]) * 10
                              * sum(conpoint_p_dev_pos[f, t, om] + conpoint_p_dev_neg[f, t, om]
                                    + conpoint_q_dev_pos[f, t, om] + conpoint_q_dev_neg[f, t, om]
                                    for om in scen_omega_set))
            # constraints for defining deployed reserves
            prob_da.addConstr(conpoint_p[f, t, om] == conpoint_p_da_en[f, t]
                              - conpoint_fac_p_pos[f, t, om] * conpoint_p_da_rs_pos[f, t]
                              + conpoint_fac_p_neg[f, t, om] * conpoint_p_da_rs_neg[f, t]
                              - conpoint_p_dev_pos[f, t, om] + conpoint_p_dev_neg[f, t, om])
            prob_da.addConstr(conpoint_q[f, t, om] == conpoint_q_da_en[f, t]
                              - conpoint_fac_q_pos[f, t, om] * conpoint_q_da_rs_pos[f, t]
                              + conpoint_fac_q_neg[f, t, om] * conpoint_q_da_rs_neg[f, t]
                              - conpoint_q_dev_pos[f, t, om] + conpoint_q_dev_neg[f, t, om])
            # # slack bus definition
            prob_da.addConstr(sum(vmag_sq[nn, t, om] * conpoint_inc_mat[f, nn] for nn in node_set)
                              == conpoint_vmag[f, t, om] * conpoint_vmag[f, t, om])
            prob_da.addConstr(conpoint_p_da_rs_pos[f, t] <= line_smax[0, 1] / 2)
            prob_da.addConstr(conpoint_p_da_rs_neg[f, t] <= line_smax[0, 1] / 2)
            prob_da.addConstr(conpoint_q_da_rs_pos[f, t] <= line_smax[0, 1] / 2)
            prob_da.addConstr(conpoint_q_da_rs_neg[f, t] <= line_smax[0, 1] / 2)
        for n1, n2 in line_set:
            prob_da.addConstr(
                obj_loss[n1, n2, t, om] >= sum(line_zre[n1, n2] * line_f[n1, n2, t, om] for n1, n2 in line_set))
    if loss_consideration == 0:
        prob_da.setObjective(deltat * obj_da_market.sum() - 0.1 * (line_rel1.sum() + line_rel2.sum()), GRB.MAXIMIZE)
    else:
        prob_da.setObjective(obj_loss.sum(), GRB.MINIMIZE)
    prob_da.Params.BarHomogeneous = 1
    prob_da.Params.OutputFlag = 0
    prob_da.optimize()
    da_result = {}
    try:
        solution_pv_p = prob_da.getAttr('x', pv_p)
        solution_pv_q = prob_da.getAttr('x', pv_q)
        solution_st_p = prob_da.getAttr('x', st_p)
        solution_st_q = prob_da.getAttr('x', st_q)
        solution_st_soc = prob_da.getAttr('x', st_soc)
        da_p = prob_da.getAttr('x', conpoint_p_da_en)
        da_q = prob_da.getAttr('x', conpoint_q_da_en)
        da_rp_pos = prob_da.getAttr('x', conpoint_p_da_rs_pos)
        da_rp_neg = prob_da.getAttr('x', conpoint_p_da_rs_neg)
        da_rq_pos = prob_da.getAttr('x', conpoint_q_da_rs_pos)
        da_rq_neg = prob_da.getAttr('x', conpoint_q_da_rs_neg)
        solution_st_soc1 = [[solution_st_soc[s, t, 0] for t in time_set] for s in st_set]
        solution_st_pp = [[solution_st_p[s, t, 0] for t in time_set] for s in st_set]
        solution_st_qq = [[solution_st_q[s, t, 0] for t in time_set] for s in st_set]
        solution_pv_pp = [[solution_pv_p[i, t, 0] for t in time_set] for i in pv_set]
        solution_pv_qq = [[solution_pv_q[i, t, 0] for t in time_set] for i in pv_set]
        da_pp = [da_p[0, t] for t in time_set]
        da_qq = [da_q[0, t] for t in time_set]
        da_rpp_pos = [da_rp_pos[0, t] for t in time_set]
        da_rpp_neg = [da_rp_neg[0, t] for t in time_set]
        da_rqq_pos = [da_rq_pos[0, t] for t in time_set]
        da_rqq_neg = [da_rq_neg[0, t] for t in time_set]
        meas = dict()
        meas["DA_PP"] = -np.array(da_pp)
        meas["DA_QQ"] = -np.array(da_qq)
        meas["DA_P+"] = -np.array(da_pp) + np.array(da_rpp_pos)
        meas["DA_Q+"] = -np.array(da_qq) + np.array(da_rqq_pos)
        meas["DA_P-"] = -np.array(da_pp) - np.array(da_rpp_neg)
        meas["DA_Q-"] = -np.array(da_qq) - np.array(da_rqq_neg)
        figuring(grid_inp=grid_inp, meas_inp=meas_inp, meas=meas, fig_type="DA_Offers", title="stochastic")
        da_result["Solution_PV_P"] = solution_pv_pp
        da_result["Solution_PV_Q"] = solution_pv_qq
        da_result["Solution_ST_P"] = solution_st_pp
        da_result["Solution_ST_Q"] = solution_st_qq
        da_result["Solution_ST_SOC"] = solution_st_soc1
        da_result["DA_P"] = da_pp
        da_result["DA_Q"] = da_qq
        da_result["DA_RP_pos"] = da_rpp_pos
        da_result["DA_RP_neg"] = da_rpp_neg
        da_result["DA_RQ_pos"] = da_rqq_pos
        da_result["DA_RQ_neg"] = da_rqq_neg
        da_result["delta"] = 0.001
        obj = prob_da.getAttr('x', obj_da_market)
        obj = [obj[0, t] for t in time_set]
        da_result["obj"] = sum(obj) * deltat * 0.1
        output_df.loc['obj', case_name] = sum(obj)
        output_df.loc['DA_P_avg', case_name] = sum(da_pp) / len(da_pp)
        output_df.loc['DA_Q_avg', case_name] = sum(da_qq) / len(da_qq)
        output_df.loc['DA_RP_pos_avg', case_name] = sum(da_rpp_pos) / len(da_rpp_pos)
        output_df.loc['DA_RP_neg_avg', case_name] = sum(da_rpp_neg) / len(da_rpp_neg)
        output_df.loc['DA_RQ_pos_avg', case_name] = sum(da_rqq_pos) / len(da_rqq_pos)
        output_df.loc['DA_RQ_neg_avg', case_name] = sum(da_rqq_neg) / len(da_rqq_neg)
        da_result["time_out"] = False
        log.info("Dayahead problem is solved with final objective " + str(da_result["obj"]) + ".")
    except (Exception,):
        log.warning("Dayahead problem is not converged.")
        da_result["time_out"] = True
    return da_result, output_df
