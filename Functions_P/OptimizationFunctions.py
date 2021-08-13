"""@author: MYI, #Python version: 3.9.6 [64 bit]"""

import numpy as np
import gurobipy as gp
import itertools
import Functions_P.AuxiliaryFunctions as af
from scipy.stats import norm
from gurobipy import *


# from pydgrid import grid


def rt_following_digriflex(grid_inp, P_net, Q_net, forecast_pv, forecast_dm, SOC):
    """" Completed:
    A simple function for testing the control commands
    Outputs: ABB_P_sp, ABB_c_sp, battery_P_sp, battery_Q_sp
    """
    forecast_P = forecast_pv - forecast_dm[0]
    forecast_Q = - forecast_dm[1]
    ABB_P_cap = grid_inp["PV_elements"][0]["cap_kVA_perPhase"] * 3
    ABB_steps = [0, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    battery_P_cap_pos = min([grid_inp["storage_elements"][0]["P_max_kW"],
                             (grid_inp["storage_elements"][0]["SOC_max_kWh"]
                             - SOC * grid_inp["storage_elements"][0]["SOC_max_kWh"] / 100) * 6])
    battery_P_cap_neg = max([-grid_inp["storage_elements"][0]["P_max_kW"],
                             (grid_inp["storage_elements"][0]["SOC_min_kWh"]
                             - SOC * grid_inp["storage_elements"][0]["SOC_max_kWh"] / 100) * 6])
    battery_Q_cap = np.sqrt(grid_inp["storage_elements"][0]["S_max_kVA"] ** 2 -
                            max([battery_P_cap_pos, - battery_P_cap_neg]) ** 2)
    if forecast_P - P_net <= battery_P_cap_pos:
        battery_P_sp = max([forecast_P - P_net, battery_P_cap_neg])
        ABB_P_sp = 100
    else:
        battery_P_sp = battery_P_cap_pos
        ABB_P_sp, _ = af.find_nearest(ABB_steps, (battery_P_sp + P_net + forecast_dm[0]) * 100 / ABB_P_cap)
    ABB_P_exp = min([forecast_pv, ABB_P_sp * ABB_P_cap / 100])
    battery_Q_sp = max([min([forecast_Q - Q_net, battery_Q_cap]), - battery_Q_cap])
    ABB_Q_max = ABB_P_exp * np.tan(np.arccos(0.89))
    if - ABB_Q_max <= Q_net + battery_Q_sp - forecast_Q <= ABB_Q_max:
        ABB_c_sp = - np.cos(np.arctan(Q_net + battery_Q_sp - forecast_Q)) \
                   * np.sign(Q_net + battery_Q_sp - forecast_Q - np.finfo(float).eps)
    else:
        ABB_c_sp, _ = af.find_nearest([-0.89, 0.89], np.sign(Q_net + battery_Q_sp - forecast_Q - np.finfo(float).eps))
        ABB_c_sp = - ABB_c_sp  # Because cos<0 = capacitive based on datasheet
    _, ABB_P_sp = af.find_nearest(ABB_steps, ABB_P_sp)
    return ABB_P_sp, round(ABB_c_sp, 3), round(battery_P_sp, 3), round(battery_Q_sp, 3)


def rt_opt_digriflex(grid_inp, V_mag, P_net, Q_net, forecast_pv, forecast_dm, SOC_battery, SOC_desired, prices_vec):
    """" Completed:
    ...
    Inputs: prices_vec = [loss_coef, battery_coef, pv_coef, dev_coef]
    Outputs: ABB_P_sp, ABB_c_sp, battery_P_sp, battery_Q_sp
    """
    rt_meas_inp = {}
    meas_inp = {}
    rt_meas_inp["delta"] = 0.001
    rt_meas_inp["Loss_Coeff"] = prices_vec[0]
    rt_meas_inp["ST_Coeff"] = prices_vec[1]
    rt_meas_inp["PV_Coeff"] = prices_vec[2]
    rt_meas_inp["dev_Coeff"] = prices_vec[3]
    meas_inp["DeltaT"] = 10 / 60
    ABB_steps = [0, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    meas_inp["meas_location"] = [{"from": "1", "to": "3", "tran/line": "line"}]
    rt_meas_inp["DM_P"] = {0: forecast_dm[0]}
    rt_meas_inp["DM_Q"] = {0: forecast_dm[1]}
    rt_meas_inp["P_PV"] = forecast_pv / grid_inp["PV_elements"][0]["cap_kVA_perPhase"]
    rt_meas_inp["ST_SOC_t_1"] = {0: SOC_battery}
    rt_meas_inp["ST_SOC_des"] = {0: SOC_desired}
    rt_meas_inp["Vmag"] = {0: V_mag / 400}
    rt_meas_inp["fac_P_pos"] = {0: 0}
    rt_meas_inp["fac_P_neg"] = {0: 0}
    rt_meas_inp["fac_Q_pos"] = {0: 0}
    rt_meas_inp["fac_Q_neg"] = {0: 0}
    rt_meas_inp["ConPoint_P_DA_EN"] = {0: - P_net}
    rt_meas_inp["ConPoint_Q_DA_EN"] = {0: - Q_net}
    rt_meas_inp["ConPoint_P_DA_RS_pos"] = {0: 0}
    rt_meas_inp["ConPoint_P_DA_RS_neg"] = {0: 0}
    rt_meas_inp["ConPoint_Q_DA_RS_pos"] = {0: 0}
    rt_meas_inp["ConPoint_Q_DA_RS_neg"] = {0: 0}
    DA_result = {}
    DA_result, _ = RT_Optimization(rt_meas_inp, meas_inp, grid_inp, DA_result)
    if not DA_result["time_out"]:
        ABB_P_sp = DA_result["Solution_PV_P"]
        ABB_Q_sp = DA_result["Solution_PV_Q"]
        Battery_P_sp = DA_result["Solution_ST_P"]
        Battery_Q_sp = DA_result["Solution_ST_Q"]
        ABB_P_sp = round(ABB_P_sp[0], 3)
        ABB_Q_sp = round(ABB_Q_sp[0], 3)
        Battery_P_sp = - round(Battery_P_sp[0], 3)
        Battery_Q_sp = - round(Battery_Q_sp[0], 3)
        print(f'Success, {ABB_P_sp}, {ABB_Q_sp}, {Battery_P_sp}, {Battery_Q_sp}')
        ABB_P_exp = ABB_P_sp
        ABB_Q_max = ABB_P_exp * np.tan(np.arccos(0.89))
        if - ABB_Q_max <= ABB_Q_sp <= ABB_Q_max:
            ABB_c_sp = - np.cos(np.arctan(ABB_Q_sp)) * np.sign(ABB_Q_sp - np.finfo(float).eps)
        else:
            ABB_c_sp, _ = - af.find_nearest([-0.89, 0.89], np.sign(ABB_Q_sp - np.finfo(float).eps))
        ABB_P_cap = grid_inp["PV_elements"][0]["cap_kVA_perPhase"] * 3
        ABB_P_sp = ABB_P_sp * 100 / ABB_P_cap
        if ABB_P_exp > forecast_pv - 0.1:
            ABB_P_sp = 100
        _, ABB_P_sp = af.find_nearest(ABB_steps, ABB_P_sp)
    else:
        ABB_P_sp, ABB_c_sp, Battery_P_sp, Battery_Q_sp = rt_following_digriflex(grid_inp, P_net, Q_net, forecast_pv, forecast_dm, SOC_battery)
    return ABB_P_sp, ABB_c_sp, Battery_P_sp, Battery_Q_sp


def RT_Optimization(rt_meas_inp, meas_inp, grid_inp, DA_result):
    """" Completed:
    ...
    Inputs: rt_meas_inp, meas_inp, grid_inp, DA_result
    Outputs: DA_result, dres
    """
    Big_M = 1e6
    l1 = rt_meas_inp["Loss_Coeff"]
    l2 = rt_meas_inp["ST_Coeff"]
    l3 = rt_meas_inp["PV_Coeff"]
    l4 = rt_meas_inp["dev_Coeff"]
    DeltaT = meas_inp["DeltaT"]
    Node_Set = range(grid_inp["Nbus"])
    #### Min and Max of Voltage (Hard Constraints)
    V_max = []
    V_min = []
    for nn in grid_inp["buses"]:
        V_min.append(nn["Vmin"])
        V_max.append(nn["Vmax"])
    #### Trans_Set
    Line_Set = []
    Line_Smax = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Vbase = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Zre = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Zim = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_b = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    for nn in grid_inp["transformers"]:
        Line_Set.append(tuple((af.find_n(nn["bus_j"], grid_inp["buses"]),
                               af.find_n(nn["bus_k"], grid_inp["buses"]))))
        Line_Smax[af.find_n(nn["bus_j"], grid_inp["buses"]),
                  af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        Line_Vbase[af.find_n(nn["bus_j"], grid_inp["buses"]),
                   af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["buses"][af.find_n(nn["bus_k"],
                                                                                            grid_inp["buses"])]["U_kV"]
        Line_Zre[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["R_cc_pu"] * grid_inp["Zbase"]
        Line_Zim[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["X_cc_pu"] * grid_inp["Zbase"]
    #### Line_Set
    for nn in grid_inp["lines"]:
        Line_Set.append(tuple((af.find_n(nn["bus_j"], grid_inp["buses"]),
                               af.find_n(nn["bus_k"], grid_inp["buses"]))))
        Line_Smax[af.find_n(nn["bus_j"], grid_inp["buses"]),
                  af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        Line_Vbase[af.find_n(nn["bus_j"], grid_inp["buses"]),
                   af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["buses"][af.find_n(nn["bus_k"],
                                                                                            grid_inp["buses"])]["U_kV"]
        Line_Zre[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["R1"] * nn["m"] / 1000
        Line_Zim[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["X1"] * nn["m"] / 1000
        Line_b[af.find_n(nn["bus_j"], grid_inp["buses"]),
               af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["B_1_mu"] * nn[
            "m"] / 2000
    ### DEMANDs Parameters
    DM_Set = range(np.size(grid_inp["load_elements"], 0))
    DM_Inc_Mat = np.zeros((np.size(grid_inp["load_elements"], 0), grid_inp["Nbus"]))
    DM_P = np.zeros((np.size(grid_inp["load_elements"], 0)))
    DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0)))
    for nn in grid_inp["load_elements"]:
        DM_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        nnn = af.find_n(nn["bus"], meas_inp["meas_location"])
        DM_P[nn["index"]] = rt_meas_inp["DM_P"][nnn]
        DM_Q[nn["index"]] = rt_meas_inp["DM_Q"][nnn]
    print("- Demand data is generated for optimization.")
    ### PV Systems Parameters
    PV_Set = range(np.size(grid_inp["PV_elements"], 0))
    PV_Inc_Mat = np.zeros((np.size(grid_inp["PV_elements"], 0), grid_inp["Nbus"]))
    PV_cap = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    PV_V_grid = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_V_conv = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_X = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_cos = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_Forecast = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    for nn in grid_inp["PV_elements"]:
        PV_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        PV_cap[nn["index"]] = nn["cap_kVA_perPhase"]
        PV_V_grid[nn["index"]] = nn["V_grid_pu"]
        PV_V_conv[nn["index"]] = nn["V_conv_pu"]
        PV_X[nn["index"]] = nn["X_PV_pu"]
        PV_cos[nn["index"]] = nn["cos_PV"]
        PV_Forecast[nn["index"]] = rt_meas_inp["P_PV"] * nn["cap_kVA_perPhase"]
    print("- PV data is generated for optimization.")
    ### Storages Parameters
    ST_Set = range(np.size(grid_inp["storage_elements"], 0))
    ST_Inc_Mat = np.zeros((np.size(grid_inp["storage_elements"], 0), grid_inp["Nbus"]))
    ST_S_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_min = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_LV = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_pos = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_neg = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_LC = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_t_1 = np.zeros((np.size(grid_inp["storage_elements"], 0)))
    ST_SOC_des = np.zeros((np.size(grid_inp["storage_elements"], 0)))
    for s, nn in enumerate(grid_inp["storage_elements"]):
        ST_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        ST_S_max[nn["index"]] = nn["S_max_kVA"]
        ST_SOC_max[nn["index"]] = nn["SOC_max_kWh"]
        ST_SOC_min[nn["index"]] = nn["SOC_min_kWh"]
        ST_Eff_LV[nn["index"]] = nn["Eff_LV"]
        ST_Eff_pos[nn["index"]] = nn["Eff_C"]
        ST_Eff_neg[nn["index"]] = nn["Eff_D"]
        ST_Eff_LC[nn["index"]] = nn["Eff_LC"]
        ST_SOC_t_1[nn["index"]] = rt_meas_inp["ST_SOC_t_1"][s]
        ST_SOC_des[nn["index"]] = rt_meas_inp["ST_SOC_des"][s]
    print("- Storage data is generated for optimization.")
    ### Transmission System and Cost Function Parameters
    ConPoint_Set = range(np.size(grid_inp["grid_formers"], 0))
    ConPoint_Inc_Mat = np.zeros((np.size(grid_inp["grid_formers"], 0), grid_inp["Nbus"]))
    ConPoint_fac_P_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_fac_P_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_fac_Q_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_fac_Q_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_Vmag = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_P_DA_EN = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_Q_DA_EN = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_P_DA_RS_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_P_DA_RS_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_Q_DA_RS_pos = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    ConPoint_Q_DA_RS_neg = np.zeros((np.size(grid_inp["grid_formers"], 0)))
    for nn in grid_inp["grid_formers"]:
        ConPoint_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        ConPoint_Vmag[nn["index"]] = rt_meas_inp["Vmag"][nn["index"]]
        ConPoint_fac_P_pos[nn["index"]] = rt_meas_inp["fac_P_pos"][nn["index"]]
        ConPoint_fac_P_neg[nn["index"]] = rt_meas_inp["fac_P_neg"][nn["index"]]
        ConPoint_fac_Q_pos[nn["index"]] = rt_meas_inp["fac_Q_pos"][nn["index"]]
        ConPoint_fac_Q_neg[nn["index"]] = rt_meas_inp["fac_Q_neg"][nn["index"]]
        ConPoint_P_DA_EN[nn["index"]] = rt_meas_inp["ConPoint_P_DA_EN"][nn["index"]]
        ConPoint_Q_DA_EN[nn["index"]] = rt_meas_inp["ConPoint_Q_DA_EN"][nn["index"]]
        ConPoint_P_DA_RS_pos[nn["index"]] = rt_meas_inp["ConPoint_P_DA_RS_pos"][nn["index"]]
        ConPoint_P_DA_RS_neg[nn["index"]] = rt_meas_inp["ConPoint_P_DA_RS_neg"][nn["index"]]
        ConPoint_Q_DA_RS_pos[nn["index"]] = rt_meas_inp["ConPoint_Q_DA_RS_pos"][nn["index"]]
        ConPoint_Q_DA_RS_neg[nn["index"]] = rt_meas_inp["ConPoint_Q_DA_RS_neg"][nn["index"]]
    print("- Connection point data is generated for optimization.")
    ### Defining Variables
    PROB_RT = Model("PROB_RT")
    Line_P_t = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_P_b = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_f = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Vmag_sq = PROB_RT.addVars(Node_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_hat = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_hat = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_hat = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_hat = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    # Line_f_hat = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_over = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_over = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_over = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_over = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_f_over = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Vmag_sq_over = PROB_RT.addVars(Node_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_sq_max = PROB_RT.addVars(Line_Set, lb=0, ub=Big_M)
    Line_Q_t_sq_max = PROB_RT.addVars(Line_Set, lb=0, ub=Big_M)
    Line_P_b_sq_max = PROB_RT.addVars(Line_Set, lb=0, ub=Big_M)
    Line_Q_b_sq_max = PROB_RT.addVars(Line_Set, lb=0, ub=Big_M)
    Line_P_t_abs_max = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_abs_max = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_abs_max = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_abs_max = PROB_RT.addVars(Line_Set, lb=-Big_M, ub=Big_M)
    PV_P = PROB_RT.addVars(PV_Set, lb=0, ub=Big_M)
    PV_Q = PROB_RT.addVars(PV_Set, lb=-Big_M, ub=Big_M)
    ST_P = PROB_RT.addVars(ST_Set, lb=-Big_M, ub=Big_M)
    ST_Q = PROB_RT.addVars(ST_Set, lb=-Big_M, ub=Big_M)
    ST_SOC = PROB_RT.addVars(ST_Set, lb=-Big_M, ub=Big_M)
    ST_SOC_dev_pos = PROB_RT.addVars(ST_Set, lb=0, ub=Big_M)
    ST_SOC_dev_neg = PROB_RT.addVars(ST_Set, lb=0, ub=Big_M)
    ST_SOC_tilde = PROB_RT.addVars(ST_Set, lb=-Big_M, ub=Big_M)
    ST_P_pos = PROB_RT.addVars(ST_Set, lb=0, ub=Big_M)
    ST_P_neg = PROB_RT.addVars(ST_Set, lb=-Big_M, ub=0)
    # ST_U_pos = PROB_RT.addVars(ST_Set, vtype=GRB.BINARY)
    # ST_U_neg = PROB_RT.addVars(ST_Set, vtype=GRB.BINARY)
    Net_P = PROB_RT.addVars(Node_Set, lb=-Big_M, ub=Big_M)
    Net_Q = PROB_RT.addVars(Node_Set, lb=-Big_M, ub=Big_M)
    ConPoint_P = PROB_RT.addVars(ConPoint_Set, lb=-Big_M, ub=Big_M)
    ConPoint_Q = PROB_RT.addVars(ConPoint_Set, lb=-Big_M, ub=Big_M)
    ConPoint_P_dev_pos = PROB_RT.addVars(ConPoint_Set, lb=0)
    ConPoint_P_dev_neg = PROB_RT.addVars(ConPoint_Set, lb=0)
    ConPoint_Q_dev_pos = PROB_RT.addVars(ConPoint_Set, lb=0)
    ConPoint_Q_dev_neg = PROB_RT.addVars(ConPoint_Set, lb=0)
    OBJ_RT_MARKET = PROB_RT.addVars(ConPoint_Set, lb=-Big_M, ub=Big_M)
    ### Defining Constraints
    for nn in Node_Set:
        # (1a)
        PROB_RT.addConstr(Net_P[nn] ==
                          sum(PV_P[i] * PV_Inc_Mat[i, nn] for i in PV_Set)
                          + sum(ST_P[s] * ST_Inc_Mat[s, nn] for s in ST_Set)
                          - sum(DM_P[d] * DM_Inc_Mat[d, nn] for d in DM_Set)
                          + sum(ConPoint_P[f] * ConPoint_Inc_Mat[f, nn] for f in ConPoint_Set))
        # (1b)
        PROB_RT.addConstr(Net_Q[nn] ==
                          sum(PV_Q[i] * PV_Inc_Mat[i, nn] for i in PV_Set)
                          + sum(ST_Q[s] * ST_Inc_Mat[s, nn] for s in ST_Set)
                          - sum(DM_Q[d] * DM_Inc_Mat[d, nn] for d in DM_Set)
                          + sum(ConPoint_Q[f] * ConPoint_Inc_Mat[f, nn] for f in ConPoint_Set))
        # (12c) abd (12d) of Mostafa
        PROB_RT.addConstr(Vmag_sq_over[nn] <= V_max[nn] * V_max[nn])
        PROB_RT.addConstr(Vmag_sq[nn] >= V_min[nn] * V_min[nn])
    for n1, n2 in Line_Set:
        # (8a) of Mostafa
        PROB_RT.addConstr(Line_P_t[n1, n2] == - Net_P[n2]
                          + sum(Line_P_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                          + Line_Zre[n1, n2] * Line_f[n1, n2] / 1000)
        PROB_RT.addConstr(Line_Q_t[n1, n2] == - Net_Q[n2]
                          + sum(Line_Q_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                          + Line_Zim[n1, n2] * Line_f[n1, n2] / 1000
                          - 1000 * (Vmag_sq[n1] + Vmag_sq[n2]) * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2))
        # (8b) of Mostafa
        PROB_RT.addConstr((Vmag_sq[n2] - Vmag_sq[n1]) * (Line_Vbase[n1, n2] ** 2) ==
                          - 2 * Line_Zre[n1, n2] * Line_P_t[n1, n2] / 1000
                          - 2 * Line_Zim[n1, n2] * Line_Q_t[n1, n2] / 1000
                          + 2 * Line_Zim[n1, n2] * Vmag_sq[n1] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2)
                          + (Line_Zre[n1, n2] * Line_Zre[n1, n2]
                             + Line_Zim[n1, n2] * Line_Zim[n1, n2]) * Line_f[n1, n2] / 1000000)
        # (8d) of Mostafa
        PROB_RT.addConstr(Line_P_b[n1, n2] == - Net_P[n2]
                          + sum(Line_P_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        PROB_RT.addConstr(Line_Q_b[n1, n2] == - Net_Q[n2]
                          + sum(Line_Q_t[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        # (10) of Mostafa
        PROB_RT.addConstr(Line_f[n1, n2] * Vmag_sq[n1] * (Line_Vbase[n1, n2] ** 2) >= Line_P_t[n1, n2]
                          * Line_P_t[n1, n2]
                          + (Line_Q_t[n1, n2] + Vmag_sq[n1] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2) * 1000)
                          * (Line_Q_t[n1, n2] + Vmag_sq[n1] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2) * 1000))
        # (11a) of Mostafa
        PROB_RT.addConstr(Line_P_t_hat[n1, n2] == - Net_P[n2]
                          + sum(Line_P_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        PROB_RT.addConstr(Line_Q_t_hat[n1, n2] == - Net_Q[n2]
                          + sum(Line_Q_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                          - 1000 * (Vmag_sq_over[n1] + Vmag_sq_over[n2]) * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2))
        # (11b) of Mostafa
        PROB_RT.addConstr((Vmag_sq_over[n2] - Vmag_sq_over[n1]) * (Line_Vbase[n1, n2] ** 2) ==
                          - 2 * Line_Zre[n1, n2] * Line_P_t_hat[n1, n2] / 1000
                          - 2 * Line_Zim[n1, n2] * Line_Q_t_hat[n1, n2] / 1000
                          + 2 * Line_Zim[n1, n2] * Vmag_sq_over[n1] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2))
        # (11c) of Mostafa
        PROB_RT.addConstr(Line_P_t_over[n1, n2] == - Net_P[n2]
                          + sum(Line_P_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                          + Line_Zre[n1, n2] * Line_f_over[n1, n2] / 1000)
        PROB_RT.addConstr(Line_Q_t_over[n1, n2] == - Net_Q[n2]
                          + sum(Line_Q_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                          + Line_Zim[n1, n2] * Line_f_over[n1, n2] / 1000
                          - 1000 * (Vmag_sq[n1] + Vmag_sq[n2]) * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2))
        # (11d) of Mostafa
        PROB_RT.addConstr(Line_f_over[n1, n2] * Vmag_sq[n2] * (Line_Vbase[n1, n2] ** 2)
                          >= Line_P_b_sq_max[n1, n2] * Line_P_b_sq_max[n1, n2]
                          + Line_Q_b_sq_max[n1, n2] * Line_Q_b_sq_max[n1, n2])
        PROB_RT.addConstr(Line_P_b_sq_max[n1, n2] >= Line_P_b_hat[n1, n2])
        PROB_RT.addConstr(Line_P_b_sq_max[n1, n2] >= - Line_P_b_hat[n1, n2])
        PROB_RT.addConstr(Line_P_b_sq_max[n1, n2] >= Line_P_b_over[n1, n2])
        PROB_RT.addConstr(Line_P_b_sq_max[n1, n2] >= - Line_P_b_over[n1, n2])
        PROB_RT.addConstr(Line_Q_b_sq_max[n1, n2] >=
                          (Line_Q_b_hat[n1, n2] - Vmag_sq_over[n2] * Line_b[n1, n2]
                           * (Line_Vbase[n1, n2] ** 2) * 1000))
        PROB_RT.addConstr(Line_Q_b_sq_max[n1, n2] >= - (Line_Q_b_hat[n1, n2] - Vmag_sq_over[n2] * Line_b[n1, n2]
                                                        * (Line_Vbase[n1, n2] ** 2) * 1000))
        PROB_RT.addConstr(Line_Q_b_sq_max[n1, n2] >= (Line_Q_b_over[n1, n2] - Vmag_sq[n2] * Line_b[n1, n2]
                                                      * (Line_Vbase[n1, n2] ** 2) * 1000))
        PROB_RT.addConstr(Line_Q_b_sq_max[n1, n2] >= - (Line_Q_b_over[n1, n2] - Vmag_sq[n2] * Line_b[n1, n2]
                                                        * (Line_Vbase[n1, n2] ** 2) * 1000))
        # (11e) of Mostafa
        PROB_RT.addConstr(Line_f_over[n1, n2] * Vmag_sq[n1] * (Line_Vbase[n1, n2] ** 2)
                          >= Line_P_t_sq_max[n1, n2] * Line_P_t_sq_max[n1, n2]
                          + Line_Q_t_sq_max[n1, n2] * Line_Q_t_sq_max[n1, n2])
        PROB_RT.addConstr(Line_P_t_sq_max[n1, n2] >= Line_P_t_hat[n1, n2])
        PROB_RT.addConstr(Line_P_t_sq_max[n1, n2] >= - Line_P_t_hat[n1, n2])
        PROB_RT.addConstr(Line_P_t_sq_max[n1, n2] >= Line_P_t_over[n1, n2])
        PROB_RT.addConstr(Line_P_t_sq_max[n1, n2] >= - Line_P_t_over[n1, n2])
        PROB_RT.addConstr(Line_Q_t_sq_max[n1, n2] >= (Line_Q_t_hat[n1, n2] + Vmag_sq_over[n1] * Line_b[n1, n2]
                                                      * (Line_Vbase[n1, n2] ** 2) * 1000))
        PROB_RT.addConstr(Line_Q_t_sq_max[n1, n2] >= - (Line_Q_t_hat[n1, n2] + Vmag_sq_over[n1] * Line_b[n1, n2]
                                                        * (Line_Vbase[n1, n2] ** 2) * 1000))
        PROB_RT.addConstr(Line_Q_t_sq_max[n1, n2] >= (Line_Q_t_over[n1, n2] + Vmag_sq[n1] * Line_b[n1, n2]
                                                      * (Line_Vbase[n1, n2] ** 2) * 1000))
        PROB_RT.addConstr(Line_Q_t_sq_max[n1, n2] >= - (Line_Q_t_over[n1, n2] + Vmag_sq[n1] * Line_b[n1, n2]
                                                        * (Line_Vbase[n1, n2] ** 2) * 1000))
        # (11f) of Mostafa
        PROB_RT.addConstr(Line_P_b_over[n1, n2] == - Net_P[n2]
                          + sum(Line_P_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        PROB_RT.addConstr(Line_Q_b_over[n1, n2] == - Net_Q[n2]
                          + sum(Line_Q_t_over[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        # (11g) of Mostafa
        PROB_RT.addConstr(Line_P_b_hat[n1, n2] == - Net_P[n2]
                          + sum(Line_P_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        PROB_RT.addConstr(Line_Q_b_hat[n1, n2] == - Net_Q[n2]
                          + sum(Line_Q_t_hat[n3, n4] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
        # (12e) of Mostafa
        PROB_RT.addConstr(Line_P_b_abs_max[n1, n2] * Line_P_b_abs_max[n1, n2]
                          + Line_Q_b_abs_max[n1, n2] * Line_Q_b_abs_max[n1, n2]
                          <= Vmag_sq[n2] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
        PROB_RT.addConstr(Line_P_b_abs_max[n1, n2] >= Line_P_b_hat[n1, n2])
        PROB_RT.addConstr(Line_P_b_abs_max[n1, n2] >= -Line_P_b_hat[n1, n2])
        PROB_RT.addConstr(Line_P_b_abs_max[n1, n2] >= Line_P_b_over[n1, n2])
        PROB_RT.addConstr(Line_P_b_abs_max[n1, n2] >= -Line_P_b_over[n1, n2])
        PROB_RT.addConstr(Line_Q_b_abs_max[n1, n2] >= Line_Q_b_hat[n1, n2])
        PROB_RT.addConstr(Line_Q_b_abs_max[n1, n2] >= -Line_Q_b_hat[n1, n2])
        PROB_RT.addConstr(Line_Q_b_abs_max[n1, n2] >= Line_Q_b_over[n1, n2])
        PROB_RT.addConstr(Line_Q_b_abs_max[n1, n2] >= -Line_Q_b_over[n1, n2])
        # (12f) of Mostafa
        PROB_RT.addConstr(Line_P_t_abs_max[n1, n2] * Line_P_t_abs_max[n1, n2]
                          + Line_Q_t_abs_max[n1, n2] * Line_Q_t_abs_max[n1, n2]
                          <= Vmag_sq[n1] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
        PROB_RT.addConstr(Line_P_t_abs_max[n1, n2] >= Line_P_t_hat[n1, n2])
        PROB_RT.addConstr(Line_P_t_abs_max[n1, n2] >= -Line_P_t_hat[n1, n2])
        PROB_RT.addConstr(Line_P_t_abs_max[n1, n2] >= Line_P_t_over[n1, n2])
        PROB_RT.addConstr(Line_P_t_abs_max[n1, n2] >= -Line_P_t_over[n1, n2])
        PROB_RT.addConstr(Line_Q_t_abs_max[n1, n2] >= Line_Q_t_hat[n1, n2])
        PROB_RT.addConstr(Line_Q_t_abs_max[n1, n2] >= -Line_Q_t_hat[n1, n2])
        PROB_RT.addConstr(Line_Q_t_abs_max[n1, n2] >= Line_Q_t_over[n1, n2])
        PROB_RT.addConstr(Line_Q_t_abs_max[n1, n2] >= -Line_Q_t_over[n1, n2])
        # (12g) of Mostafa
        PROB_RT.addConstr(Line_P_t[n1, n2] <= Line_P_t_over[n1, n2])
        PROB_RT.addConstr(Line_Q_t[n1, n2] <= Line_Q_t_over[n1, n2])
    for s in ST_Set:
        # (21a)
        PROB_RT.addConstr(ST_SOC[s] == ST_Eff_LV[s] * ST_SOC_t_1[s]
                          - ST_P_neg[s] * DeltaT * ST_Eff_neg[s]
                          - ST_P_pos[s] * DeltaT / ST_Eff_pos[s])
        PROB_RT.addConstr(ST_SOC_tilde[s] == ST_Eff_LV[s] * ST_SOC_t_1[s]
                          - ST_P[s] * DeltaT)
        # (22d)
        PROB_RT.addConstr(ST_P[s] == ST_P_pos[s] + ST_P_neg[s])
        # (22e)
        PROB_RT.addConstr(ST_Q[s] * ST_Q[s] + ST_P[s] * ST_P[s] <= ST_S_max[s] * ST_S_max[s])
        # (22b)
        PROB_RT.addConstr(ST_P_pos[s] <= ST_S_max[s])
        # (22c)
        PROB_RT.addConstr(ST_P_neg[s] >= -ST_S_max[s])
        # (22f)
        PROB_RT.addConstr(ST_SOC[s] <= ST_SOC_max[s])
        PROB_RT.addConstr(ST_SOC[s] >= ST_SOC_min[s])
        # (22h)
        PROB_RT.addConstr(ST_SOC_tilde[s] <= ST_SOC_max[s])
        PROB_RT.addConstr(ST_SOC_tilde[s] >= ST_SOC_min[s])
        PROB_RT.addConstr(ST_SOC[s] - ST_SOC_des[s] == ST_SOC_dev_pos[s] - ST_SOC_dev_neg[s])
    for i in PV_Set:
        # (28)
        PROB_RT.addConstr(PV_P[i] <= PV_Forecast[i])
        # (26)
        PROB_RT.addConstr(PV_Q[i] <= PV_P[i] * np.tan(np.arccos(PV_cos[i])))
        PROB_RT.addConstr(PV_Q[i] >= - PV_P[i] * np.tan(np.arccos(PV_cos[i])))
        # PROB_RT.addConstr(PV_P[i] * PV_P[i]
        #                   + (PV_Q[i] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
        #                   * (PV_Q[i] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
        #                   <= (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i])
        #                   * (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i]))
        # (27)
        PROB_RT.addConstr(PV_P[i] * PV_P[i] + PV_Q[i] * PV_Q[i] <= PV_cap[i] * PV_cap[i])
        # PROB_RT.addConstr(PV_Q[i] == 0)
    for f in ConPoint_Set:
        PROB_RT.addConstr(ConPoint_P[f] == sum(Line_P_t[n1, n2] * ConPoint_Inc_Mat[f, n1] for n1, n2 in Line_Set))
        PROB_RT.addConstr(ConPoint_Q[f] == sum(Line_Q_t[n1, n2] * ConPoint_Inc_Mat[f, n1] for n1, n2 in Line_Set))
        # first line of (34a)
        PROB_RT.addConstr(OBJ_RT_MARKET[f] >= l1 * sum(Line_Zre[n1, n2] * Line_f[n1, n2] for n1, n2 in Line_Set)
                          + l2 * sum(ST_SOC_dev_pos[s] + ST_SOC_dev_neg[s] for s in ST_Set)
                          + l3 * sum(PV_Forecast[p] - PV_P[p] for p in PV_Set)
                          + l4 * (ConPoint_P_dev_pos[f] + ConPoint_P_dev_neg[f]
                                  + ConPoint_Q_dev_pos[f] + ConPoint_Q_dev_neg[f]))
        # constraints for defining deployed reserves
        PROB_RT.addConstr(ConPoint_P[f] == ConPoint_P_DA_EN[f]
                          - ConPoint_fac_P_pos[f] * ConPoint_P_DA_RS_pos[f]
                          + ConPoint_fac_P_neg[f] * ConPoint_P_DA_RS_neg[f]
                          - ConPoint_P_dev_pos[f] + ConPoint_P_dev_neg[f])
        PROB_RT.addConstr(ConPoint_Q[f] == ConPoint_Q_DA_EN[f]
                          - ConPoint_fac_Q_pos[f] * ConPoint_Q_DA_RS_pos[f]
                          + ConPoint_fac_Q_neg[f] * ConPoint_Q_DA_RS_neg[f]
                          - ConPoint_Q_dev_pos[f] + ConPoint_Q_dev_neg[f])
        # slack bus definition
        PROB_RT.addConstr(sum(Vmag_sq[n] * ConPoint_Inc_Mat[f, n] for n in Node_Set)
                          == ConPoint_Vmag[f] * ConPoint_Vmag[f])
        ### Defining Objective and Solving the Problem
    PROB_RT.setObjective(1000 * DeltaT * OBJ_RT_MARKET.sum(), GRB.MINIMIZE)
    PROB_RT.Params.BarHomogeneous = 1
    PROB_RT.optimize()
    ### Solution
    try:
        print(PROB_RT.getAttr('x', Line_P_t))
        print(PROB_RT.getAttr('x', Line_Q_t))
        print(PROB_RT.getAttr('x', Net_P))
        print(PROB_RT.getAttr('x', Net_Q))
        print(PROB_RT.getAttr('x', Line_f))
        Solution_PV_P = PROB_RT.getAttr('x', PV_P)
        Solution_PV_Q = PROB_RT.getAttr('x', PV_Q)
        Solution_ST_P = PROB_RT.getAttr('x', ST_P)
        Solution_ST_Q = PROB_RT.getAttr('x', ST_Q)
        Solution_ST_SOC = PROB_RT.getAttr('x', ST_SOC)
        # Solution_ST_SOC2 = PROB_RT.getAttr('x', ST_SOC_tilde)
        Solution_P_dev_pos = PROB_RT.getAttr('x', ConPoint_P_dev_pos)
        Solution_P_dev_neg = PROB_RT.getAttr('x', ConPoint_P_dev_neg)
        Solution_Q_dev_pos = PROB_RT.getAttr('x', ConPoint_P_dev_pos)
        Solution_Q_dev_neg = PROB_RT.getAttr('x', ConPoint_P_dev_neg)
        Solution_ST_SOCC = [Solution_ST_SOC[s] for s in ST_Set]
        DA_result["Solution_ST_SOC_RT"] = Solution_ST_SOCC
        DA_result["Solution_PV_P"] = Solution_PV_P
        DA_result["Solution_PV_Q"] = Solution_PV_Q
        DA_result["Solution_ST_P"] = Solution_ST_P
        DA_result["Solution_ST_Q"] = Solution_ST_Q
        dres = (Solution_P_dev_pos[0] + Solution_P_dev_neg[0] + Solution_Q_dev_pos[0] + Solution_Q_dev_neg[0] >
                rt_meas_inp["delta"])
        DA_result["time_out"] = False
    except:
        DA_result["Solution_ST_SOC_RT"] = ST_SOC_t_1
        DA_result["time_out"] = True
        dres = 1
    return DA_result, dres


def da_opt_digriflex(grid_inp, V_mag, forecast_pv, forecast_p_dm, forecast_q_dm, forecast_SOC, prices_vec, robust_par):
    """" Completed:
        ...
        Inputs
        Outputs: P_SC, Q_SC, RPP_SC, RPN_SC, RQP_SC, RQN_SC, SOC_dersired,
        prices_vec2 = [loss_coef, battery_coef, pv_coef, dev_coef]
    """
    case_name = 'DiGriFlex'
    case_inp = {}
    meas_inp = {}
    fore_inp = {}
    output_DF = {}
    case_inp['LAMBDA_P_DA_EN'] = prices_vec[0][:]
    case_inp['LAMBDA_Q_DA_EN'] = prices_vec[1][:]
    case_inp['LAMBDA_P_DA_RS_pos'] = prices_vec[2][:]
    case_inp['LAMBDA_P_DA_RS_neg'] = prices_vec[3][:]
    case_inp['LAMBDA_Q_DA_RS_pos'] = prices_vec[4][:]
    case_inp['LAMBDA_Q_DA_RS_neg'] = prices_vec[5][:]
    case_inp['Robust_prob'] = robust_par
    case_inp['Omega_Number'] = 20
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
    fore_inp["ST_SOC_0"] = forecast_SOC[0] / 100
    fore_inp["ST_SOC_zeta+"] = forecast_SOC[1] / 100
    fore_inp["ST_SOC_zeta-"] = forecast_SOC[2] / 100
    fore_inp["Vmag_zeta+"] = V_mag[0][:]
    fore_inp["Vmag_zeta-"] = V_mag[1][:]
    fore_inp["confidence"] = 0.99
    if robust_par == 1:
        DA_result, _ = DA_Optimization(case_name, case_inp, grid_inp, meas_inp, fore_inp, output_DF)
    else:
        DA_result, _ = DA_Optimization_Robust(case_name, case_inp, grid_inp, meas_inp, fore_inp, output_DF)
    if not DA_result["time_out"]:
        P_SC = DA_result["DA_P"]
        Q_SC = DA_result["DA_Q"]
        RPP_SC = DA_result["DA_RP_pos"]
        RPN_SC = DA_result["DA_RP_neg"]
        RQP_SC = DA_result["DA_RQ_pos"]
        RQN_SC = DA_result["DA_RQ_neg"]
        SOC_dersired = DA_result["Solution_ST_SOC"]
    else:
        P_SC, Q_SC = [-9] * 144, [2] * 144
        RPP_SC, RPN_SC, RQP_SC, RQN_SC = [0.4] * 144, [0.4] * 144, [0.2] * 144, [0.2] * 144
        SOC_dersired = [10] * 144
    prices_vec2 = [0, 1, 100, 1000]
    return P_SC, Q_SC, RPP_SC, RPN_SC, RQP_SC, RQN_SC, SOC_dersired, prices_vec2


def DA_Optimization_Robust(case_name, case_inp, grid_inp, meas_inp, fore_inp, output_DF):
    """" Completed:
    ...
    Inputs: case_name, case_inp, grid_inp, meas_inp, fore_inp, output_DF
    Outputs: DA_result, dres
    """
    Big_M = 1e6
    LAMBDA_P_DA_EN = case_inp['LAMBDA_P_DA_EN']
    LAMBDA_Q_DA_EN = case_inp['LAMBDA_Q_DA_EN']
    LAMBDA_P_DA_RS_pos = case_inp['LAMBDA_P_DA_RS_pos']
    LAMBDA_P_DA_RS_neg = case_inp['LAMBDA_P_DA_RS_neg']
    LAMBDA_Q_DA_RS_pos = case_inp['LAMBDA_Q_DA_RS_pos']
    LAMBDA_Q_DA_RS_neg = case_inp['LAMBDA_Q_DA_RS_neg']
    # loss_consideration = case_inp['loss_consideration']
    prob = case_inp['Robust_prob']
    conf_multip = 0
    if prob != 0:
        conf_multip = norm.ppf(1 - prob)  # np.sqrt((1-prob)/prob)
    #### Main Sets
    Time_Set = range(meas_inp["Nt"])
    DeltaT = meas_inp["DeltaT"]
    Node_Set = range(grid_inp["Nbus"])
    #### Min and Max of Voltage (Hard Constraints)
    V_max = []
    V_min = []
    for n in grid_inp["buses"]:
        V_min.append(n["Vmin"])
        V_max.append(n["Vmax"])
    #### Trans_Set
    Line_Set = []
    Line_Smax = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Vbase = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Zre = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Zim = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_b = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    for n in grid_inp["transformers"]:
        Line_Set.append(tuple((af.find_n(n["bus_j"], grid_inp["buses"]),
                               af.find_n(n["bus_k"], grid_inp["buses"]))))
        Line_Smax[af.find_n(n["bus_j"], grid_inp["buses"]),
                  af.find_n(n["bus_k"], grid_inp["buses"])] = n["Cap"]
        Line_Vbase[af.find_n(n["bus_j"], grid_inp["buses"]),
                   af.find_n(n["bus_k"], grid_inp["buses"])] = grid_inp["buses"][af.find_n(n["bus_k"],
                                                                                           grid_inp["buses"])]["U_kV"]
        Line_Zre[af.find_n(n["bus_j"], grid_inp["buses"]),
                 af.find_n(n["bus_k"], grid_inp["buses"])] = n["R_cc_pu"] * grid_inp["Zbase"]
        Line_Zim[af.find_n(n["bus_j"], grid_inp["buses"]),
                 af.find_n(n["bus_k"], grid_inp["buses"])] = n["X_cc_pu"] * grid_inp["Zbase"]
    #### Line_Set
    for n in grid_inp["lines"]:
        Line_Set.append(tuple((af.find_n(n["bus_j"], grid_inp["buses"]),
                               af.find_n(n["bus_k"], grid_inp["buses"]))))
        Line_Smax[af.find_n(n["bus_j"], grid_inp["buses"]),
                  af.find_n(n["bus_k"], grid_inp["buses"])] = n["Cap"]
        Line_Vbase[af.find_n(n["bus_j"], grid_inp["buses"]),
                   af.find_n(n["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][af.find_n(n["bus_k"], grid_inp["buses"])][
                "U_kV"]
        Line_Zre[af.find_n(n["bus_j"], grid_inp["buses"]),
                 af.find_n(n["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][n["code"]]["R1"] * n["m"] / 1000
        Line_Zim[af.find_n(n["bus_j"], grid_inp["buses"]),
                 af.find_n(n["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][n["code"]]["X1"] * n["m"] / 1000
        Line_b[af.find_n(n["bus_j"], grid_inp["buses"]),
               af.find_n(n["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][n["code"]]["B_1_mu"] * n["m"] / 2000
    ### DEMANDs Parameters
    DM_Set = range(np.size(grid_inp["load_elements"], 0))
    DM_Inc_Mat = np.zeros((np.size(grid_inp["load_elements"], 0), grid_inp["Nbus"]))
    DM_P = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_p_DM_P = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_n_DM_P = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_p_DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    zeta_n_DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_p_DM_P = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_n_DM_P = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_p_DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    sigma_n_DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"]))
    for n in grid_inp["load_elements"]:
        DM_Inc_Mat[n["index"], af.find_n(n["bus"], grid_inp["buses"])] = 1
        for t in Time_Set:
            nn = af.find_n(n["bus"], meas_inp["meas_location"])
            DM_P[n["index"]][t] = fore_inp["Dem_P"][nn][t]
            DM_Q[n["index"]][t] = fore_inp["Dem_Q"][nn][t]
            zeta_p_DM_P[n["index"]][t] = fore_inp["Dem_P_zeta+"][nn][t]
            zeta_n_DM_P[n["index"]][t] = fore_inp["Dem_P_zeta-"][nn][t]
            zeta_p_DM_Q[n["index"]][t] = fore_inp["Dem_Q_zeta+"][nn][t]
            zeta_n_DM_Q[n["index"]][t] = fore_inp["Dem_Q_zeta-"][nn][t]
            sigma_p_DM_P[n["index"]][t] = zeta_p_DM_P[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_DM_P[n["index"]][t] = zeta_n_DM_P[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_p_DM_Q[n["index"]][t] = zeta_p_DM_Q[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_DM_Q[n["index"]][t] = zeta_n_DM_Q[n["index"]][t] / norm.ppf(fore_inp["confidence"])
    print("- Demand data is generated for optimization.")
    ### PV Systems Parameters
    PV_Set = range(np.size(grid_inp["PV_elements"], 0))
    PV_Inc_Mat = np.zeros((np.size(grid_inp["PV_elements"], 0), grid_inp["Nbus"]))
    PV_cap = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    PV_V_grid = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_V_conv = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_X = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_cos = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_Forecast = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    zeta_p_PV = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    zeta_n_PV = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    sigma_p_PV = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    sigma_n_PV = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"]))
    for n in grid_inp["PV_elements"]:
        PV_Inc_Mat[n["index"], af.find_n(n["bus"], grid_inp["buses"])] = 1
        PV_cap[n["index"]] = n["cap_kVA_perPhase"]
        PV_V_grid[n["index"]] = n["V_grid_pu"]
        PV_V_conv[n["index"]] = n["V_conv_pu"]
        PV_X[n["index"]] = n["X_PV_pu"]
        PV_cos[n["index"]] = n["cos_PV"]
        for t in Time_Set:
            PV_Forecast[n["index"]][t] = fore_inp["P_PV"][t] * n["cap_kVA_perPhase"]
            zeta_p_PV[n["index"]][t] = fore_inp["P_PV_zeta+"][t] * n["cap_kVA_perPhase"]
            zeta_n_PV[n["index"]][t] = fore_inp["P_PV_zeta-"][t] * n["cap_kVA_perPhase"]
            sigma_p_PV[n["index"]][t] = zeta_p_PV[n["index"]][t] / norm.ppf(fore_inp["confidence"])
            sigma_n_PV[n["index"]][t] = zeta_n_PV[n["index"]][t] / norm.ppf(fore_inp["confidence"])
    print("- PV data is generated for optimization.")
    ### Storages Parameters
    ST_Set = range(np.size(grid_inp["storage_elements"], 0))
    ST_Inc_Mat = np.zeros((np.size(grid_inp["storage_elements"], 0), grid_inp["Nbus"]))
    ST_S_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_min = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_LV = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_pos = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_neg = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_LC = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_0 = np.zeros((np.size(grid_inp["storage_elements"], 0)))
    zeta_p_ST = np.zeros(np.size(grid_inp["storage_elements"], 0))
    zeta_n_ST = np.zeros(np.size(grid_inp["storage_elements"], 0))
    for n in grid_inp["storage_elements"]:
        ST_Inc_Mat[n["index"], af.find_n(n["bus"], grid_inp["buses"])] = 1
        ST_S_max[n["index"]] = n["S_max_kVA"]
        ST_SOC_max[n["index"]] = n["SOC_max_kWh"]
        ST_SOC_min[n["index"]] = n["SOC_min_kWh"]
        ST_Eff_LV[n["index"]] = n["Eff_LV"]
        ST_Eff_pos[n["index"]] = n["Eff_C"]
        ST_Eff_neg[n["index"]] = n["Eff_D"]
        ST_Eff_LC[n["index"]] = n["Eff_LC"]
        ST_SOC_0[n["index"]] = n["SOC_max_kWh"] * fore_inp["ST_SOC_0"]
        zeta_p_ST[n["index"]] = fore_inp["ST_SOC_zeta+"] * ST_SOC_0[n["index"]]
        zeta_n_ST[n["index"]] = fore_inp["ST_SOC_zeta-"] * ST_SOC_0[n["index"]]
    print("- Storage data is generated for optimization.")
    ### Transmission System and Cost Function Parameters
    ConPoint_Set = range(np.size(grid_inp["grid_formers"], 0))
    ConPoint_Inc_Mat = np.zeros((np.size(grid_inp["grid_formers"], 0), grid_inp["Nbus"]))
    ConPoint_Vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_p_ConPoint_fac_P = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_n_ConPoint_fac_P = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_p_ConPoint_fac_Q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_n_ConPoint_fac_Q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_p_ConPoint_Vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    zeta_n_ConPoint_Vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_p_ConPoint_fac_P = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_n_ConPoint_fac_P = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_p_ConPoint_fac_Q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    sigma_n_ConPoint_fac_Q = np.ones((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"]))
    for n in grid_inp["grid_formers"]:
        ConPoint_Inc_Mat[n["index"], af.find_n(n["bus"], grid_inp["buses"])] = 1
        for t in Time_Set:
            ConPoint_Vmag[n["index"]][t] = 1
            zeta_p_ConPoint_fac_P[n["index"]][t] = 1
            zeta_n_ConPoint_fac_P[n["index"]][t] = 1
            zeta_p_ConPoint_fac_Q[n["index"]][t] = 1
            zeta_n_ConPoint_fac_Q[n["index"]][t] = 1
            zeta_p_ConPoint_Vmag[n["index"]][t] = fore_inp["Vmag_zeta+"][t]
            zeta_n_ConPoint_Vmag[n["index"]][t] = fore_inp["Vmag_zeta-"][t]
            sigma_p_ConPoint_fac_P[n["index"]][t] = zeta_p_ConPoint_fac_P[n["index"]][t] / norm.ppf(
                fore_inp["confidence"])
            sigma_n_ConPoint_fac_P[n["index"]][t] = zeta_n_ConPoint_fac_P[n["index"]][t] / norm.ppf(
                fore_inp["confidence"])
            sigma_p_ConPoint_fac_Q[n["index"]][t] = zeta_p_ConPoint_fac_Q[n["index"]][t] / norm.ppf(
                fore_inp["confidence"])
            sigma_n_ConPoint_fac_Q[n["index"]][t] = zeta_n_ConPoint_fac_Q[n["index"]][t] / norm.ppf(
                fore_inp["confidence"])
    print("- Connection point data is generated for optimization.")
    ### Defining Variables
    PROB_DA = Model("PROB_DA")
    Line_P_t = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_b = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_f = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Vmag_sq = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_Line_P_t = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_Line_Q_t = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_Line_P_t = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_Line_Q_t = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_Vmag_sq = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_Vmag_sq = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_hat = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_hat = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_hat = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_hat = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    # Line_f_hat = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_over = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_over = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_over = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_over = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_f_over = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Vmag_sq_over = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_sq_max = PROB_DA.addVars(Line_Set, Time_Set, lb=0, ub=Big_M)
    Line_Q_t_sq_max = PROB_DA.addVars(Line_Set, Time_Set, lb=0, ub=Big_M)
    Line_P_b_sq_max = PROB_DA.addVars(Line_Set, Time_Set, lb=0, ub=Big_M)
    Line_Q_b_sq_max = PROB_DA.addVars(Line_Set, Time_Set, lb=0, ub=Big_M)
    Line_P_t_abs_max = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_abs_max = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_abs_max = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_abs_max = PROB_DA.addVars(Line_Set, Time_Set, lb=-Big_M, ub=Big_M)
    PV_P = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    PV_Q = PROB_DA.addVars(PV_Set, Time_Set, lb=-Big_M, ub=Big_M)
    alfa_PV_P_p = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    alfa_PV_P_n = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    alfa_PV_Q_p = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    alfa_PV_Q_n = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    max_PV_P = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    max_PV_Q = PROB_DA.addVars(PV_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_PV_P = PROB_DA.addVars(PV_Set, Time_Set, lb=0, ub=Big_M)
    min_PV_Q = PROB_DA.addVars(PV_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ST_P = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ST_Q = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ST_SOC = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ST_SOC_tilde = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ST_P_pos = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    ST_P_neg = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=0)
    alfa_ST_P_p = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    alfa_ST_P_n = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    alfa_ST_Q_p = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    alfa_ST_Q_n = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    max_ST_P = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_ST_Q = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_ST_P_pos = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    max_ST_P_neg = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=0)
    max_ST_SOC = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_ST_SOC_tilde = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_ST_P = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_ST_Q = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_ST_P_pos = PROB_DA.addVars(ST_Set, Time_Set, lb=0, ub=Big_M)
    min_ST_P_neg = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=0)
    min_ST_SOC = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_ST_SOC_tilde = PROB_DA.addVars(ST_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Net_P = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    Net_Q = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    zeta_p_Net_P = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    zeta_p_Net_Q = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    zeta_n_Net_P = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    zeta_n_Net_Q = PROB_DA.addVars(Node_Set, Time_Set, lb=-Big_M, ub=Big_M)
    zeta_p_P = PROB_DA.addVars(Time_Set, lb=-Big_M, ub=Big_M)
    zeta_p_Q = PROB_DA.addVars(Time_Set, lb=-Big_M, ub=Big_M)
    zeta_n_P = PROB_DA.addVars(Time_Set, lb=-Big_M, ub=Big_M)
    zeta_n_Q = PROB_DA.addVars(Time_Set, lb=-Big_M, ub=Big_M)
    sigma_p_P = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    sigma_p_Q = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    sigma_n_P = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    sigma_n_Q = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    # McCormick Envelopes vars
    theta_p_P = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    theta_p_Q = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    theta_n_P = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    theta_n_Q = PROB_DA.addVars(Time_Set, lb=0, ub=Big_M)
    ConPoint_P_DA_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ConPoint_Q_DA_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    alfa_ConPoint_P_p = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    alfa_ConPoint_P_n = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    alfa_ConPoint_Q_p = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    alfa_ConPoint_Q_n = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    max_ConPoint_P_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    max_ConPoint_Q_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_ConPoint_P_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    min_ConPoint_Q_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ConPoint_P_DA_RS_pos = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_P_DA_RS_neg = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_Q_DA_RS_pos = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_Q_DA_RS_neg = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    OBJ_DA_MARKET = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ### Defining Constraints
    for t in Time_Set:
        for n in Node_Set:
            # (1a)
            PROB_DA.addConstr(Net_P[n, t] ==
                              sum(PV_P[i, t] * PV_Inc_Mat[i, n] for i in PV_Set)
                              + sum(ST_P[s, t] * ST_Inc_Mat[s, n] for s in ST_Set)
                              - sum(DM_P[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                              + sum(ConPoint_P_DA_EN[f, t] * ConPoint_Inc_Mat[f, n] for f in ConPoint_Set))
            # (1b)
            PROB_DA.addConstr(Net_Q[n, t] ==
                              sum(PV_Q[i, t] * PV_Inc_Mat[i, n] for i in PV_Set)
                              + sum(ST_Q[s, t] * ST_Inc_Mat[s, n] for s in ST_Set)
                              - sum(DM_Q[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                              + sum(ConPoint_Q_DA_EN[f, t] * ConPoint_Inc_Mat[f, n] for f in ConPoint_Set))
            # (12c) abd (12d) of Mostafa
            PROB_DA.addConstr(Vmag_sq_over[n, t] <= V_max[n] * V_max[n])
            PROB_DA.addConstr(Vmag_sq[n, t] >= V_min[n] * V_min[n])
        for n1, n2 in Line_Set:
            # (8a) of Mostafa
            PROB_DA.addConstr(Line_P_t[n1, n2, t] == - Net_P[n2, t]
                              + sum(Line_P_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zre[n1, n2] * Line_f[n1, n2, t] / 1000)
            PROB_DA.addConstr(Line_Q_t[n1, n2, t] == - Net_Q[n2, t]
                              + sum(Line_Q_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zim[n1, n2] * Line_f[n1, n2, t] / 1000
                              - 1000 * (Vmag_sq[n1, t] + Vmag_sq[n2, t]) * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (8b) of Mostafa
            PROB_DA.addConstr((Vmag_sq[n2, t] - Vmag_sq[n1, t]) * (Line_Vbase[n1, n2] ** 2) ==
                              - 2 * Line_Zre[n1, n2] * Line_P_t[n1, n2, t] / 1000
                              - 2 * Line_Zim[n1, n2] * Line_Q_t[n1, n2, t] / 1000
                              + 2 * Line_Zim[n1, n2] * Vmag_sq[n1, t] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2)
                              + (Line_Zre[n1, n2] * Line_Zre[n1, n2]
                                 + Line_Zim[n1, n2] * Line_Zim[n1, n2]) * Line_f[n1, n2, t] / 1000000)
            # (8d) of Mostafa
            PROB_DA.addConstr(Line_P_b[n1, n2, t] == - Net_P[n2, t]
                              + sum(Line_P_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_b[n1, n2, t] == - Net_Q[n2, t]
                              + sum(Line_Q_t[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            # (10) of Mostafa
            PROB_DA.addConstr(
                Line_f[n1, n2, t] * Vmag_sq[n1, t] * (Line_Vbase[n1, n2] ** 2) >= Line_P_t[n1, n2, t]
                * Line_P_t[n1, n2, t]
                + (Line_Q_t[n1, n2, t] + Vmag_sq[n1, t] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2) * 1000)
                * (Line_Q_t[n1, n2, t] + Vmag_sq[n1, t] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2) * 1000))
            # (11a) of Mostafa
            PROB_DA.addConstr(Line_P_t_hat[n1, n2, t] == - Net_P[n2, t]
                              + sum(Line_P_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_t_hat[n1, n2, t] == - Net_Q[n2, t]
                              + sum(Line_Q_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              - 1000 * (Vmag_sq_over[n1, t] + Vmag_sq_over[n2, t]) * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (11b) of Mostafa
            PROB_DA.addConstr((Vmag_sq_over[n2, t] - Vmag_sq_over[n1, t]) * (Line_Vbase[n1, n2] ** 2) ==
                              - 2 * Line_Zre[n1, n2] * Line_P_t_hat[n1, n2, t] / 1000
                              - 2 * Line_Zim[n1, n2] * Line_Q_t_hat[n1, n2, t] / 1000
                              + 2 * Line_Zim[n1, n2] * Vmag_sq_over[n1, t] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2))
            # (11c) of Mostafa
            PROB_DA.addConstr(Line_P_t_over[n1, n2, t] == - Net_P[n2, t]
                              + sum(Line_P_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zre[n1, n2] * Line_f_over[n1, n2, t] / 1000)
            PROB_DA.addConstr(Line_Q_t_over[n1, n2, t] == - Net_Q[n2, t]
                              + sum(Line_Q_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zim[n1, n2] * Line_f_over[n1, n2, t] / 1000
                              - 1000 * (Vmag_sq[n1, t] + Vmag_sq[n2, t]) * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (11d) of Mostafa
            PROB_DA.addConstr(Line_f_over[n1, n2, t] * Vmag_sq[n2, t] * (Line_Vbase[n1, n2] ** 2)
                              >= Line_P_b_sq_max[n1, n2, t] * Line_P_b_sq_max[n1, n2, t]
                              + Line_Q_b_sq_max[n1, n2, t] * Line_Q_b_sq_max[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t] >= Line_P_b_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t] >= - Line_P_b_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t] >= Line_P_b_over[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t] >= - Line_P_b_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t] >=
                              (Line_Q_b_hat[n1, n2, t] - Vmag_sq_over[n2, t] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t] >=
                              - (Line_Q_b_hat[n1, n2, t] - Vmag_sq_over[n2, t] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t] >=
                              (Line_Q_b_over[n1, n2, t] - Vmag_sq[n2, t] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t] >=
                              - (Line_Q_b_over[n1, n2, t] - Vmag_sq[n2, t] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            # (11e) of Mostafa
            PROB_DA.addConstr(Line_f_over[n1, n2, t] * Vmag_sq[n1, t] * (Line_Vbase[n1, n2] ** 2)
                              >= Line_P_t_sq_max[n1, n2, t] * Line_P_t_sq_max[n1, n2, t]
                              + Line_Q_t_sq_max[n1, n2, t] * Line_Q_t_sq_max[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t] >= Line_P_t_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t] >= - Line_P_t_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t] >= Line_P_t_over[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t] >= - Line_P_t_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t] >=
                              (Line_Q_t_hat[n1, n2, t] + Vmag_sq_over[n1, t] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t] >=
                              - (Line_Q_t_hat[n1, n2, t] + Vmag_sq_over[n1, t] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t] >=
                              (Line_Q_t_over[n1, n2, t] + Vmag_sq[n1, t] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t] >=
                              - (Line_Q_t_over[n1, n2, t] + Vmag_sq[n1, t] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            # (11f) of Mostafa
            PROB_DA.addConstr(Line_P_b_over[n1, n2, t] == - Net_P[n2, t]
                              + sum(Line_P_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_b_over[n1, n2, t] == - Net_Q[n2, t]
                              + sum(Line_Q_t_over[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            # (11g) of Mostafa
            PROB_DA.addConstr(Line_P_b_hat[n1, n2, t] == - Net_P[n2, t]
                              + sum(Line_P_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_b_hat[n1, n2, t] == - Net_Q[n2, t]
                              + sum(Line_Q_t_hat[n3, n4, t] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            # (12e) of Mostafa
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t] * Line_P_b_abs_max[n1, n2, t]
                              + Line_Q_b_abs_max[n1, n2, t] * Line_Q_b_abs_max[n1, n2, t]
                              <= Vmag_sq[n2, t] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t] >= Line_P_b_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t] >= -Line_P_b_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t] >= Line_P_b_over[n1, n2, t])
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t] >= -Line_P_b_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t] >= Line_Q_b_hat[n1, n2, t])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t] >= -Line_Q_b_hat[n1, n2, t])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t] >= Line_Q_b_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t] >= -Line_Q_b_over[n1, n2, t])
            # (12f) of Mostafa
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t] * Line_P_t_abs_max[n1, n2, t]
                              + Line_Q_t_abs_max[n1, n2, t] * Line_Q_t_abs_max[n1, n2, t]
                              <= Vmag_sq[n1, t] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t] >= Line_P_t_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t] >= -Line_P_t_hat[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t] >= Line_P_t_over[n1, n2, t])
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t] >= -Line_P_t_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t] >= Line_Q_t_hat[n1, n2, t])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t] >= -Line_Q_t_hat[n1, n2, t])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t] >= Line_Q_t_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t] >= -Line_Q_t_over[n1, n2, t])
            # (12g) of Mostafa
            PROB_DA.addConstr(Line_P_t[n1, n2, t] <= Line_P_t_over[n1, n2, t])
            PROB_DA.addConstr(Line_Q_t[n1, n2, t] <= Line_Q_t_over[n1, n2, t])
            # (current constraints) of Mostafa
            PROB_DA.addConstr(Line_P_t[n1, n2, t] * Line_P_t[n1, n2, t]
                              + Line_Q_t[n1, n2, t] * Line_Q_t[n1, n2, t]
                              <= Vmag_sq[n1, t] * ((Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2)))
            PROB_DA.addConstr(Line_P_b[n1, n2, t] * Line_P_b[n1, n2, t]
                              + Line_Q_b[n1, n2, t] * Line_Q_b[n1, n2, t]
                              <= Vmag_sq[n2, t] * ((Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2)))
        for s in ST_Set:
            # (22a) and (22g)
            if t == 0:
                PROB_DA.addConstr(ST_SOC[s, t] == ST_Eff_LV[s] * ST_SOC_0[s]
                                  - ST_P_neg[s, t] * DeltaT * ST_Eff_neg[s]
                                  - ST_P_pos[s, t] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(ST_SOC_tilde[s, t] == ST_Eff_LV[s] * ST_SOC_0[s]
                                  - ST_P[s, t] * DeltaT)
            else:
                PROB_DA.addConstr(ST_SOC[s, t] == ST_Eff_LV[s] * ST_SOC[s, t - 1]
                                  - ST_P_neg[s, t] * DeltaT * ST_Eff_neg[s]
                                  - ST_P_pos[s, t] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(ST_SOC_tilde[s, t] == ST_Eff_LV[s] * ST_SOC[s, t - 1]
                                  - ST_P[s, t] * DeltaT)
            # (22d)
            PROB_DA.addConstr(ST_P[s, t] == ST_P_pos[s, t] + ST_P_neg[s, t])
            # (22e)
            PROB_DA.addConstr(ST_Q[s, t] * ST_Q[s, t] + ST_P[s, t] * ST_P[s, t] <= ST_S_max[s] * ST_S_max[s])
            # (22b)
            PROB_DA.addConstr(ST_P_pos[s, t] <= ST_S_max[s])
            # (22c)
            PROB_DA.addConstr(ST_P_neg[s, t] >= -ST_S_max[s])
            # (22f)
            PROB_DA.addConstr(ST_SOC[s, t] <= ST_SOC_max[s])
            PROB_DA.addConstr(ST_SOC[s, t] >= ST_SOC_min[s])
            # (22h)
            PROB_DA.addConstr(ST_SOC_tilde[s, t] <= ST_SOC_max[s])
            PROB_DA.addConstr(ST_SOC_tilde[s, t] >= ST_SOC_min[s])
        for i in PV_Set:
            # (26)
            PROB_DA.addConstr(PV_P[i, t] <= PV_Forecast[i, t])
            # (24)
            PROB_DA.addConstr(PV_Q[i, t] <= PV_P[i, t] * np.tan(np.arccos(PV_cos[i])))
            PROB_DA.addConstr(PV_Q[i, t] >= - PV_P[i, t] * np.tan(np.arccos(PV_cos[i])))
            # PROB_DA.addConstr(PV_P[i, t] * PV_P[i, t]
            #                   + (PV_Q[i, t] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   * (PV_Q[i, t] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   <= (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i])
            #                   * (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i]))
            # (25)
            PROB_DA.addConstr(PV_P[i, t] * PV_P[i, t] + PV_Q[i, t] * PV_Q[i, t] <= PV_cap[i] * PV_cap[i])
        for f in ConPoint_Set:
            PROB_DA.addConstr(
                ConPoint_P_DA_EN[f, t] == sum(Line_P_t[n1, n2, t] * ConPoint_Inc_Mat[f, n1] for n1, n2 in Line_Set))
            PROB_DA.addConstr(
                ConPoint_Q_DA_EN[f, t] == sum(Line_Q_t[n1, n2, t] * ConPoint_Inc_Mat[f, n1] for n1, n2 in Line_Set))
            # first line of (34a)
            PROB_DA.addConstr(OBJ_DA_MARKET[f, t] == -LAMBDA_P_DA_EN[t] * ConPoint_P_DA_EN[f, t]
                              - LAMBDA_Q_DA_EN[t] * ConPoint_Q_DA_EN[f, t]
                              + LAMBDA_P_DA_RS_pos[t] * ConPoint_P_DA_RS_pos[f, t]
                              + LAMBDA_P_DA_RS_neg[t] * ConPoint_P_DA_RS_neg[f, t]
                              + LAMBDA_Q_DA_RS_pos[t] * ConPoint_Q_DA_RS_pos[f, t]
                              + LAMBDA_Q_DA_RS_neg[t] * ConPoint_Q_DA_RS_neg[f, t])
            # slack bus definition
            PROB_DA.addConstr(sum(Vmag_sq[n, t] * ConPoint_Inc_Mat[f, n] for n in Node_Set)
                              == ConPoint_Vmag[f, t] * ConPoint_Vmag[f, t])
            PROB_DA.addConstr(ConPoint_P_DA_RS_pos[f, t] <= Line_Smax[0, 1] / 2)
            PROB_DA.addConstr(ConPoint_P_DA_RS_neg[f, t] <= Line_Smax[0, 1] / 2)
            PROB_DA.addConstr(ConPoint_Q_DA_RS_pos[f, t] <= Line_Smax[0, 1] / 2)
            PROB_DA.addConstr(ConPoint_Q_DA_RS_neg[f, t] <= Line_Smax[0, 1] / 2)
        # Adjustable robust constraints (max of the net)
        PROB_DA.addConstr(sigma_p_P[t] * sigma_p_P[t] >= sum(sigma_p_ConPoint_fac_P[f, t] * ConPoint_P_DA_RS_pos[f, t]
                                                             * sigma_p_ConPoint_fac_P[f, t] * ConPoint_P_DA_RS_pos[f, t]
                                                             for f in ConPoint_Set)
                          + sum(sigma_n_DM_P[d, t] * sigma_n_DM_P[d, t] for d in DM_Set)
                          + sum(sigma_n_PV[i, t] * sigma_n_PV[i, t] for i in PV_Set))
        PROB_DA.addConstr(sigma_n_P[t] * sigma_n_P[t] >= sum(sigma_n_ConPoint_fac_P[f, t] * ConPoint_P_DA_RS_neg[f, t]
                                                             * sigma_n_ConPoint_fac_P[f, t] * ConPoint_P_DA_RS_neg[f, t]
                                                             for f in ConPoint_Set)
                          + sum(sigma_p_DM_P[d, t] * sigma_p_DM_P[d, t] for d in DM_Set)
                          + sum(sigma_p_PV[i, t] * sigma_p_PV[i, t] for i in PV_Set))
        PROB_DA.addConstr(sigma_p_Q[t] * sigma_p_Q[t] >= sum(sigma_p_ConPoint_fac_Q[f, t] * ConPoint_Q_DA_RS_pos[f, t]
                                                             * sigma_p_ConPoint_fac_Q[f, t] * ConPoint_Q_DA_RS_pos[f, t]
                                                             for f in ConPoint_Set)
                          + sum(sigma_n_DM_Q[d, t] * sigma_n_DM_Q[d, t] for d in DM_Set))
        PROB_DA.addConstr(sigma_n_Q[t] * sigma_n_Q[t] >= sum(sigma_n_ConPoint_fac_Q[f, t] * ConPoint_Q_DA_RS_neg[f, t]
                                                             * sigma_n_ConPoint_fac_Q[f, t] * ConPoint_Q_DA_RS_neg[f, t]
                                                             for f in ConPoint_Set)
                          + sum(sigma_p_DM_Q[d, t] * sigma_p_DM_Q[d, t] for d in DM_Set))
        PROB_DA.addConstr(
            zeta_p_P[t] == sum(zeta_p_ConPoint_fac_P[f, t] * ConPoint_P_DA_RS_pos[f, t] for f in ConPoint_Set)
            + sum(zeta_n_DM_P[d, t] for d in DM_Set) + sum(zeta_n_PV[i, t] for i in PV_Set))
        PROB_DA.addConstr(
            zeta_n_P[t] == sum(zeta_n_ConPoint_fac_P[f, t] * ConPoint_P_DA_RS_neg[f, t] for f in ConPoint_Set)
            + sum(zeta_p_DM_P[d, t] for d in DM_Set) + sum(zeta_p_PV[i, t] for i in PV_Set))
        PROB_DA.addConstr(
            zeta_p_Q[t] == sum(zeta_p_ConPoint_fac_Q[f, t] * ConPoint_Q_DA_RS_pos[f, t] for f in ConPoint_Set)
            + sum(zeta_n_DM_Q[d, t] for d in DM_Set))
        PROB_DA.addConstr(
            zeta_n_Q[t] == sum(zeta_n_ConPoint_fac_Q[f, t] * ConPoint_Q_DA_RS_neg[f, t] for f in ConPoint_Set)
            + sum(zeta_p_DM_Q[d, t] for d in DM_Set))
        # McCormick Envelopes
        PROB_DA.addConstr(sum(alfa_ConPoint_P_p[f, t] for f in ConPoint_Set) == zeta_p_P[t] - theta_p_P[t])
        PROB_DA.addConstr(sum(alfa_ConPoint_P_n[f, t] for f in ConPoint_Set) == zeta_n_P[t] - theta_n_P[t])
        PROB_DA.addConstr(sum(alfa_ConPoint_Q_p[f, t] for f in ConPoint_Set) == zeta_p_Q[t] - theta_p_Q[t])
        PROB_DA.addConstr(sum(alfa_ConPoint_Q_n[f, t] for f in ConPoint_Set) == zeta_n_Q[t] - theta_n_Q[t])
        if prob == 0:
            PROB_DA.addConstr(theta_p_P[t] >= zeta_p_P[t])
            PROB_DA.addConstr(theta_n_P[t] >= zeta_n_P[t])
            PROB_DA.addConstr(theta_p_Q[t] >= zeta_p_Q[t])
            PROB_DA.addConstr(theta_n_Q[t] >= zeta_n_Q[t])
        else:
            PROB_DA.addConstr(theta_p_P[t] >= sigma_p_P[t] * conf_multip)
            PROB_DA.addConstr(theta_n_P[t] >= sigma_n_P[t] * conf_multip)
            PROB_DA.addConstr(theta_p_Q[t] >= sigma_p_Q[t] * conf_multip)
            PROB_DA.addConstr(theta_n_Q[t] >= sigma_n_Q[t] * conf_multip)
            ## sum of PV+ST+CP==zeta
        PROB_DA.addConstr(zeta_p_P[t] == sum(alfa_ST_P_p[s, t] for s in ST_Set) + sum(alfa_PV_P_p[i, t] for i in PV_Set)
                          + sum(alfa_ConPoint_P_n[f, t] for f in ConPoint_Set))
        PROB_DA.addConstr(zeta_n_P[t] == sum(alfa_ST_P_n[s, t] for s in ST_Set) + sum(alfa_PV_P_n[i, t] for i in PV_Set)
                          + sum(alfa_ConPoint_P_p[f, t] for f in ConPoint_Set))
        PROB_DA.addConstr(zeta_p_Q[t] == sum(alfa_ST_Q_p[s, t] for s in ST_Set) + sum(alfa_PV_Q_p[i, t] for i in PV_Set)
                          + sum(alfa_ConPoint_Q_n[f, t] for f in ConPoint_Set))
        PROB_DA.addConstr(zeta_n_Q[t] == sum(alfa_ST_Q_n[s, t] for s in ST_Set) + sum(alfa_PV_Q_n[i, t] for i in PV_Set)
                          + sum(alfa_ConPoint_Q_p[f, t] for f in ConPoint_Set))
        # Adjustable robust constraints (storages)
        for s in ST_Set:
            PROB_DA.addConstr(max_ST_P[s, t] == ST_P[s, t] + alfa_ST_P_p[s, t])
            PROB_DA.addConstr(min_ST_P[s, t] == ST_P[s, t] - alfa_ST_P_n[s, t])
            PROB_DA.addConstr(max_ST_Q[s, t] == ST_Q[s, t] + alfa_ST_Q_p[s, t])
            PROB_DA.addConstr(min_ST_Q[s, t] == ST_Q[s, t] - alfa_ST_Q_n[s, t])
            PROB_DA.addConstr(max_ST_P[s, t] == max_ST_P_pos[s, t] + max_ST_P_neg[s, t])
            PROB_DA.addConstr(min_ST_P[s, t] == min_ST_P_pos[s, t] + min_ST_P_neg[s, t])
            if t == 0:
                PROB_DA.addConstr(max_ST_SOC[s, t] == ST_Eff_LV[s] * ST_SOC_0[s] + ST_Eff_LV[s] * zeta_p_ST[s]
                                  - min_ST_P_neg[s, t] * DeltaT * ST_Eff_neg[s]
                                  - min_ST_P_pos[s, t] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(max_ST_SOC_tilde[s, t] == ST_Eff_LV[s] * ST_SOC_0[s] + ST_Eff_LV[s] * zeta_p_ST[s]
                                  - min_ST_P_neg[s, t] * DeltaT
                                  - min_ST_P_pos[s, t] * DeltaT)
                PROB_DA.addConstr(min_ST_SOC[s, t] == ST_Eff_LV[s] * ST_SOC_0[s] - ST_Eff_LV[s] * zeta_n_ST[s]
                                  - max_ST_P_neg[s, t] * DeltaT * ST_Eff_neg[s]
                                  - max_ST_P_pos[s, t] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(min_ST_SOC_tilde[s, t] == ST_Eff_LV[s] * ST_SOC_0[s] - ST_Eff_LV[s] * zeta_n_ST[s]
                                  - max_ST_P_neg[s, t] * DeltaT
                                  - max_ST_P_pos[s, t] * DeltaT)
            else:
                PROB_DA.addConstr(max_ST_SOC[s, t] == ST_Eff_LV[s] * max_ST_SOC[s, t - 1]
                                  - min_ST_P_neg[s, t] * DeltaT * ST_Eff_neg[s]
                                  - min_ST_P_pos[s, t] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(max_ST_SOC_tilde[s, t] == ST_Eff_LV[s] * max_ST_SOC[s, t - 1]
                                  - min_ST_P_neg[s, t] * DeltaT
                                  - min_ST_P_pos[s, t] * DeltaT)
                PROB_DA.addConstr(min_ST_SOC[s, t] == ST_Eff_LV[s] * min_ST_SOC[s, t - 1]
                                  - max_ST_P_neg[s, t] * DeltaT * ST_Eff_neg[s]
                                  - max_ST_P_pos[s, t] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(min_ST_SOC_tilde[s, t] == ST_Eff_LV[s] * min_ST_SOC[s, t - 1]
                                  - max_ST_P_neg[s, t] * DeltaT
                                  - max_ST_P_pos[s, t] * DeltaT)
            PROB_DA.addConstr(max_ST_Q[s, t] * max_ST_Q[s, t] + max_ST_P[s, t] * max_ST_P[s, t]
                              <= ST_S_max[s] * ST_S_max[s])
            PROB_DA.addConstr(min_ST_Q[s, t] * min_ST_Q[s, t] + max_ST_P[s, t] * max_ST_P[s, t]
                              <= ST_S_max[s] * ST_S_max[s])
            PROB_DA.addConstr(max_ST_Q[s, t] * max_ST_Q[s, t] + min_ST_P[s, t] * min_ST_P[s, t]
                              <= ST_S_max[s] * ST_S_max[s])
            PROB_DA.addConstr(min_ST_Q[s, t] * min_ST_Q[s, t] + min_ST_P[s, t] * min_ST_P[s, t]
                              <= ST_S_max[s] * ST_S_max[s])
            PROB_DA.addConstr(max_ST_P_pos[s, t] <= ST_S_max[s])
            PROB_DA.addConstr(min_ST_P_neg[s, t] >= -ST_S_max[s])
            PROB_DA.addConstr(max_ST_SOC[s, t] <= ST_SOC_max[s])
            PROB_DA.addConstr(min_ST_SOC[s, t] >= ST_SOC_min[s])
            PROB_DA.addConstr(max_ST_SOC_tilde[s, t] <= ST_SOC_max[s])
            PROB_DA.addConstr(min_ST_SOC_tilde[s, t] >= ST_SOC_min[s])
        # Adjustable robust constraints (PV)
        for i in PV_Set:
            PROB_DA.addConstr(max_PV_P[i, t] == PV_P[i, t] + zeta_p_PV[i, t] + alfa_PV_P_p[i, t])
            PROB_DA.addConstr(min_PV_P[i, t] == PV_P[i, t] - zeta_n_PV[i, t] - alfa_PV_P_n[i, t])
            PROB_DA.addConstr(max_PV_Q[i, t] == PV_Q[i, t] + alfa_PV_Q_p[i, t])
            PROB_DA.addConstr(min_PV_Q[i, t] == PV_Q[i, t] - alfa_PV_Q_n[i, t])
            PROB_DA.addConstr(max_PV_P[i, t] <= PV_Forecast[i, t] + zeta_p_PV[i, t])
            PROB_DA.addConstr(max_PV_Q[i, t] <= max_PV_P[i, t] * np.tan(np.arccos(PV_cos[i])))
            PROB_DA.addConstr(max_PV_Q[i, t] >= - max_PV_P[i, t] * np.tan(np.arccos(PV_cos[i])))
            PROB_DA.addConstr(min_PV_Q[i, t] <= max_PV_P[i, t] * np.tan(np.arccos(PV_cos[i])))
            PROB_DA.addConstr(min_PV_Q[i, t] >= - max_PV_P[i, t] * np.tan(np.arccos(PV_cos[i])))
            # PROB_DA.addConstr(max_PV_P[i, t] * max_PV_P[i, t]
            #                   + (max_PV_Q[i, t] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   * (max_PV_Q[i, t] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   <= (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i])
            #                   * (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i]))
            # PROB_DA.addConstr(max_PV_P[i, t] * max_PV_P[i, t]
            #                   + (min_PV_Q[i, t] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   * (min_PV_Q[i, t] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   <= (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i])
            #                   * (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i]))
            PROB_DA.addConstr(
                max_PV_P[i, t] * max_PV_P[i, t] + max_PV_Q[i, t] * max_PV_Q[i, t] <= PV_cap[i] * PV_cap[i])
            PROB_DA.addConstr(
                max_PV_P[i, t] * max_PV_P[i, t] + min_PV_Q[i, t] * min_PV_Q[i, t] <= PV_cap[i] * PV_cap[i])
        # Adjustable robust constraints (connection point)
        for f in ConPoint_Set:
            PROB_DA.addConstr(max_ConPoint_P_EN[f, t] == ConPoint_P_DA_EN[f, t] + alfa_ConPoint_P_n[f, t])
            PROB_DA.addConstr(min_ConPoint_P_EN[f, t] == ConPoint_P_DA_EN[f, t] - alfa_ConPoint_P_p[f, t])
            PROB_DA.addConstr(max_ConPoint_Q_EN[f, t] == ConPoint_Q_DA_EN[f, t] + alfa_ConPoint_Q_n[f, t])
            PROB_DA.addConstr(min_ConPoint_Q_EN[f, t] == ConPoint_Q_DA_EN[f, t] - alfa_ConPoint_Q_p[f, t])
            # Adjustable robust constriants (slack def)
            PROB_DA.addConstr(sum((max_Vmag_sq[nn, t] - Vmag_sq[nn, t]) * ConPoint_Inc_Mat[f, nn] for nn in Node_Set)
                              == 2 * Vmag_sq[0, t] * zeta_p_ConPoint_Vmag[f, t])
            PROB_DA.addConstr(sum((Vmag_sq[nn, t] - min_Vmag_sq[nn, t]) * ConPoint_Inc_Mat[f, nn] for nn in Node_Set)
                              == 2 * zeta_n_ConPoint_Vmag[f, t] * Vmag_sq[0, t])
            # Adjustable robust constraints (nodes and branches)
        for n in Node_Set:
            PROB_DA.addConstr(
                zeta_p_Net_P[n, t] == sum((zeta_p_PV[i, t] + alfa_PV_P_p[i, t]) * PV_Inc_Mat[i, n] for i in PV_Set)
                + sum(alfa_ST_P_p[s, t] * ST_Inc_Mat[s, n] for s in ST_Set)
                + sum(zeta_n_DM_P[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                + sum(alfa_ConPoint_P_n[f, t] * ConPoint_Inc_Mat[f, n] for f in ConPoint_Set))
            PROB_DA.addConstr(
                zeta_n_Net_P[n, t] == sum((zeta_n_PV[i, t] + alfa_PV_P_n[i, t]) * PV_Inc_Mat[i, n] for i in PV_Set)
                + sum(alfa_ST_P_n[s, t] * ST_Inc_Mat[s, n] for s in ST_Set)
                + sum(zeta_p_DM_P[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                + sum(alfa_ConPoint_P_p[f, t] * ConPoint_Inc_Mat[f, n] for f in ConPoint_Set))
            PROB_DA.addConstr(zeta_p_Net_Q[n, t] == sum(alfa_PV_Q_p[i, t] * PV_Inc_Mat[i, n] for i in PV_Set)
                              + sum(alfa_ST_Q_p[s, t] * ST_Inc_Mat[s, n] for s in ST_Set)
                              + sum(zeta_n_DM_Q[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                              + sum(alfa_ConPoint_Q_n[f, t] * ConPoint_Inc_Mat[f, n] for f in ConPoint_Set))
            PROB_DA.addConstr(zeta_n_Net_Q[n, t] == sum(alfa_PV_Q_n[i, t] * PV_Inc_Mat[i, n] for i in PV_Set)
                              + sum(alfa_ST_Q_n[s, t] * ST_Inc_Mat[s, n] for s in ST_Set)
                              + sum(zeta_p_DM_Q[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                              + sum(alfa_ConPoint_Q_p[f, t] * ConPoint_Inc_Mat[f, n] for f in ConPoint_Set))
            PROB_DA.addConstr(max_Vmag_sq[n, t] <= V_max[n] * V_max[n])
            PROB_DA.addConstr(min_Vmag_sq[n, t] >= V_min[n] * V_min[n])
        for n1, n2 in Line_Set:
            PROB_DA.addConstr(max_Line_P_t[n1, n2, t] == Line_P_t[n1, n2, t] + zeta_n_Net_P[n2, t]
                              + sum((max_Line_P_t[n3, n4, t] - Line_P_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in Line_Set))
            PROB_DA.addConstr(min_Line_P_t[n1, n2, t] == Line_P_t[n1, n2, t] - zeta_p_Net_P[n2, t]
                              - sum((Line_P_t[n3, n4, t] - min_Line_P_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in Line_Set))
            PROB_DA.addConstr(max_Line_Q_t[n1, n2, t] == Line_Q_t[n1, n2, t] + zeta_n_Net_Q[n2, t]
                              + sum((max_Line_Q_t[n3, n4, t] - Line_Q_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in Line_Set))
            PROB_DA.addConstr(min_Line_Q_t[n1, n2, t] == Line_Q_t[n1, n2, t] - zeta_p_Net_Q[n2, t]
                              - sum((Line_Q_t[n3, n4, t] - min_Line_Q_t[n3, n4, t]) * np.where(n3 == n2, 1, 0)
                                    for n3, n4 in Line_Set))
            PROB_DA.addConstr((max_Vmag_sq[n2, t] - max_Vmag_sq[n1, t]) * (Line_Vbase[n1, n2] ** 2) ==
                              - 2 * Line_Zre[n1, n2] * max_Line_P_t[n1, n2, t] / 1000
                              - 2 * Line_Zim[n1, n2] * max_Line_Q_t[n1, n2, t] / 1000)
            PROB_DA.addConstr((min_Vmag_sq[n2, t] - min_Vmag_sq[n1, t]) * (Line_Vbase[n1, n2] ** 2) ==
                              - 2 * Line_Zre[n1, n2] * min_Line_P_t[n1, n2, t] / 1000
                              - 2 * Line_Zim[n1, n2] * min_Line_Q_t[n1, n2, t] / 1000)
            PROB_DA.addConstr(max_Line_P_t[n1, n2, t] * max_Line_P_t[n1, n2, t]
                              + max_Line_Q_t[n1, n2, t] * max_Line_Q_t[n1, n2, t]
                              <= min_Vmag_sq[n1, t] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
            PROB_DA.addConstr(min_Line_P_t[n1, n2, t] * min_Line_P_t[n1, n2, t]
                              + min_Line_Q_t[n1, n2, t] * min_Line_Q_t[n1, n2, t]
                              <= min_Vmag_sq[n1, t] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
            ### Defining Objective and Solving Problem
    PROB_DA.setObjective(1000 * DeltaT * OBJ_DA_MARKET.sum(), GRB.MAXIMIZE)
    PROB_DA.Params.BarHomogeneous = 1
    PROB_DA.optimize()
    ### Solution
    DA_result = {}
    try:
        Solution_PV_P = PROB_DA.getAttr('x', PV_P)
        Solution_PV_Q = PROB_DA.getAttr('x', PV_Q)
        Solution_ST_P = PROB_DA.getAttr('x', ST_P)
        Solution_ST_Q = PROB_DA.getAttr('x', ST_Q)
        Solution_ST_SOC = PROB_DA.getAttr('x', ST_SOC)
        for t in Time_Set:
            for k in range(3):
                for n in Node_Set:
                    if n > 0:
                        nn = grid_inp["buses"][n]["bus"]
                        nn = af.find_n(nn, meas_inp["meas_location"])
                        meas_inp["P"][nn][k][t] = sum(Solution_PV_P[i, t] * PV_Inc_Mat[i, n] for i in PV_Set) \
                                                  + sum(Solution_ST_P[s, t] * ST_Inc_Mat[s, n] for s in ST_Set) \
                                                  - sum(DM_P[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
                        meas_inp["Q"][nn][k][t] = sum(Solution_PV_Q[i, t] * PV_Inc_Mat[i, n] for i in PV_Set) \
                                                  + sum(Solution_ST_Q[s, t] * ST_Inc_Mat[s, n] for s in ST_Set) \
                                                  - sum(DM_Q[d, t] * DM_Inc_Mat[d, n] for d in DM_Set)
        DA_P = PROB_DA.getAttr('x', ConPoint_P_DA_EN)
        DA_Q = PROB_DA.getAttr('x', ConPoint_Q_DA_EN)
        DA_RP_pos = PROB_DA.getAttr('x', ConPoint_P_DA_RS_pos)
        DA_RP_neg = PROB_DA.getAttr('x', ConPoint_P_DA_RS_neg)
        DA_RQ_pos = PROB_DA.getAttr('x', ConPoint_Q_DA_RS_pos)
        DA_RQ_neg = PROB_DA.getAttr('x', ConPoint_Q_DA_RS_neg)
        # zeta_p_PP = PROB_DA.getAttr('x', zeta_p_P)
        # theta_p_PP = PROB_DA.getAttr('x', theta_p_P)
        # zeta_n_PP = PROB_DA.getAttr('x', zeta_n_P)
        # theta_n_PP = PROB_DA.getAttr('x', theta_n_P)
        # zeta_p_QQ = PROB_DA.getAttr('x', zeta_p_Q)
        # theta_p_QQ = PROB_DA.getAttr('x', theta_p_Q)
        # zeta_n_QQ = PROB_DA.getAttr('x', zeta_n_Q)
        # theta_n_QQ = PROB_DA.getAttr('x', theta_n_Q)
        # del_p_P = [zeta_p_PP[t] - theta_p_PP[t] for t in Time_Set]
        # del_n_P = [zeta_n_PP[t] - theta_n_PP[t] for t in Time_Set]
        # del_p_Q = [zeta_p_QQ[t] - theta_p_QQ[t] for t in Time_Set]
        # del_n_Q = [zeta_n_QQ[t] - theta_n_QQ[t] for t in Time_Set]
        # delta = sum([max(del_p_P), max(del_n_P), max(del_p_Q), max(del_n_Q)])
        Solution_ST_SOCC = [[Solution_ST_SOC[s, t] for t in Time_Set] for s in ST_Set]
        Solution_ST_PP = [[Solution_ST_P[s, t] for t in Time_Set] for s in ST_Set]
        Solution_ST_QQ = [[Solution_ST_Q[s, t] for t in Time_Set] for s in ST_Set]
        Solution_PV_PP = [[Solution_PV_P[i, t] for t in Time_Set] for i in PV_Set]
        Solution_PV_QQ = [[Solution_PV_Q[i, t] for t in Time_Set] for i in PV_Set]
        DA_PP = [DA_P[0, t] for t in Time_Set]
        DA_QQ = [DA_Q[0, t] for t in Time_Set]
        DA_RPP_pos = [DA_RP_pos[0, t] for t in Time_Set]
        DA_RPP_neg = [DA_RP_neg[0, t] for t in Time_Set]
        DA_RQQ_pos = [DA_RQ_pos[0, t] for t in Time_Set]
        DA_RQQ_neg = [DA_RQ_neg[0, t] for t in Time_Set]
        meas = {}
        meas["DA_PP"] = -np.array(DA_PP)
        meas["DA_QQ"] = -np.array(DA_QQ)
        meas["DA_P+"] = -np.array(DA_PP) + np.array(DA_RPP_pos)
        meas["DA_Q+"] = -np.array(DA_QQ) + np.array(DA_RQQ_pos)
        meas["DA_P-"] = -np.array(DA_PP) - np.array(DA_RPP_neg)
        meas["DA_Q-"] = -np.array(DA_QQ) - np.array(DA_RQQ_neg)
        af.figuring(grid_inp, meas_inp, meas, "DA_Offers", "robust" + str(prob))
        # meas2 = af.SE_time_series(grid_inp, meas_inp, "robust"+str(prob))
        DA_result["Solution_PV_P"] = Solution_PV_PP
        DA_result["Solution_PV_Q"] = Solution_PV_QQ
        DA_result["Solution_ST_P"] = Solution_ST_PP
        DA_result["Solution_ST_Q"] = Solution_ST_QQ
        DA_result["Solution_ST_SOC"] = Solution_ST_SOCC
        DA_result["DA_P"] = DA_PP
        DA_result["DA_Q"] = DA_QQ
        DA_result["DA_RP_pos"] = DA_RPP_pos
        DA_result["DA_RP_neg"] = DA_RPP_neg
        DA_result["DA_RQ_pos"] = DA_RQQ_pos
        DA_result["DA_RQ_neg"] = DA_RQQ_neg
        DA_result["delta"] = 0.001
        obj = PROB_DA.getObjective()
        output_DF.loc['obj', case_name] = obj.getValue()
        output_DF.loc['DA_P_avg', case_name] = sum(DA_PP) / len(DA_PP)
        output_DF.loc['DA_Q_avg', case_name] = sum(DA_QQ) / len(DA_QQ)
        output_DF.loc['DA_RP_pos_avg', case_name] = sum(DA_RPP_pos) / len(DA_RPP_pos)
        output_DF.loc['DA_RP_neg_avg', case_name] = sum(DA_RPP_neg) / len(DA_RPP_neg)
        output_DF.loc['DA_RQ_pos_avg', case_name] = sum(DA_RQQ_pos) / len(DA_RQQ_pos)
        output_DF.loc['DA_RQ_neg_avg', case_name] = sum(DA_RQQ_neg) / len(DA_RQQ_neg)
        DA_result["time_out"] = False
    except:
        DA_result["time_out"] = True
    return DA_result, output_DF


def DA_Optimization(case_name, case_inp, grid_inp, meas_inp, fore_inp, output_DF):
    """" Completed:
    ...
    Inputs: case_name, case_inp, grid_inp, meas_inp, fore_inp, output_DF
    Outputs: DA_result, dres
    """
    Big_M = 1e6
    Omega_Number = case_inp['Omega_Number']
    LAMBDA_P_DA_EN = case_inp['LAMBDA_P_DA_EN']
    LAMBDA_Q_DA_EN = case_inp['LAMBDA_Q_DA_EN']
    LAMBDA_P_DA_RS_pos = case_inp['LAMBDA_P_DA_RS_pos']
    LAMBDA_P_DA_RS_neg = case_inp['LAMBDA_P_DA_RS_neg']
    LAMBDA_Q_DA_RS_pos = case_inp['LAMBDA_Q_DA_RS_pos']
    LAMBDA_Q_DA_RS_neg = case_inp['LAMBDA_Q_DA_RS_neg']
    loss_consideration = case_inp['loss_consideration']
    Scen_Omega_Set = range(Omega_Number)
    Time_Set = range(meas_inp["Nt"])
    DeltaT = meas_inp["DeltaT"]
    Node_Set = range(grid_inp["Nbus"])
    #### Min and Max of Voltage (Hard Constraints)
    V_max = []
    V_min = []
    for nn in grid_inp["buses"]:
        V_min.append(nn["Vmin"])
        V_max.append(nn["Vmax"])
    #### Trans_Set
    Line_Set = []
    Line_Smax = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Vbase = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Zre = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_Zim = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    Line_b = np.zeros((grid_inp["Nbus"], grid_inp["Nbus"]))
    for nn in grid_inp["transformers"]:
        Line_Set.append(tuple((af.find_n(nn["bus_j"], grid_inp["buses"]),
                               af.find_n(nn["bus_k"], grid_inp["buses"]))))
        Line_Smax[af.find_n(nn["bus_j"], grid_inp["buses"]),
                  af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        Line_Vbase[af.find_n(nn["bus_j"], grid_inp["buses"]),
                   af.find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][af.find_n(nn["bus_k"], grid_inp["buses"])]["U_kV"]
        Line_Zre[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["R_cc_pu"] * grid_inp["Zbase"]
        Line_Zim[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["X_cc_pu"] * grid_inp["Zbase"]
    #### Line_Set
    for nn in grid_inp["lines"]:
        Line_Set.append(tuple((af.find_n(nn["bus_j"], grid_inp["buses"]),
                               af.find_n(nn["bus_k"], grid_inp["buses"]))))
        Line_Smax[af.find_n(nn["bus_j"], grid_inp["buses"]),
                  af.find_n(nn["bus_k"], grid_inp["buses"])] = nn["Cap"]
        Line_Vbase[af.find_n(nn["bus_j"], grid_inp["buses"]),
                   af.find_n(nn["bus_k"], grid_inp["buses"])] = \
            grid_inp["buses"][af.find_n(nn["bus_k"], grid_inp["buses"])]["U_kV"]
        Line_Zre[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["R1"] * nn["m"] / 1000
        Line_Zim[af.find_n(nn["bus_j"], grid_inp["buses"]),
                 af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["X1"] * nn["m"] / 1000
        Line_b[af.find_n(nn["bus_j"], grid_inp["buses"]),
               af.find_n(nn["bus_k"], grid_inp["buses"])] = grid_inp["line_codes"][nn["code"]]["B_1_mu"] * nn[
            "m"] / 2000
    print(Line_Set)
    ### DEMANDs Parameters
    DM_Set = range(np.size(grid_inp["load_elements"], 0))
    DM_Inc_Mat = np.zeros((np.size(grid_inp["load_elements"], 0), grid_inp["Nbus"]))
    DM_P = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"], Omega_Number))
    DM_Q = np.zeros((np.size(grid_inp["load_elements"], 0), meas_inp["Nt"], Omega_Number))
    for nn in grid_inp["load_elements"]:
        DM_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        for t in Time_Set:
            for om in Scen_Omega_Set:
                nnn = af.find_n(nn["bus"], meas_inp["meas_location"])
                DM_P[nn["index"]][t][om] = fore_inp["Dem_P"][nnn][t]
                DM_Q[nn["index"]][t][om] = fore_inp["Dem_Q"][nnn][t]
                if om > 0:
                    DM_P[nn["index"]][t][om] = fore_inp["Dem_P"][nnn][t] + fore_inp["Dem_P_zeta+"][nnn][t] \
                                               - (fore_inp["Dem_P_zeta+"][nnn][t] + fore_inp["Dem_P_zeta-"][nnn][t]) \
                                               * np.random.randint(0, 2)
                    DM_Q[nn["index"]][t][om] = fore_inp["Dem_Q"][nnn][t] + fore_inp["Dem_Q_zeta+"][nnn][t] \
                                               - (fore_inp["Dem_Q_zeta+"][nnn][t] + fore_inp["Dem_Q_zeta-"][nnn][t]) \
                                               * np.random.randint(0, 2)
    print("- Demand data is generated for optimization.")
    ### PV Systems Parameters
    PV_Set = range(np.size(grid_inp["PV_elements"], 0))
    PV_Inc_Mat = np.zeros((np.size(grid_inp["PV_elements"], 0), grid_inp["Nbus"]))
    PV_cap = np.zeros((np.size(grid_inp["PV_elements"], 0)))
    PV_V_grid = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_V_conv = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_X = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_cos = np.zeros(np.size(grid_inp["PV_elements"], 0))
    PV_Forecast = np.zeros((np.size(grid_inp["PV_elements"], 0), meas_inp["Nt"], Omega_Number))
    for nn in grid_inp["PV_elements"]:
        PV_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        PV_cap[nn["index"]] = nn["cap_kVA_perPhase"]
        PV_V_grid[nn["index"]] = nn["V_grid_pu"]
        PV_V_conv[nn["index"]] = nn["V_conv_pu"]
        PV_X[nn["index"]] = nn["X_PV_pu"]
        PV_cos[nn["index"]] = nn["cos_PV"]
        for t in Time_Set:
            for om in Scen_Omega_Set:
                PV_Forecast[nn["index"]][t][om] = fore_inp["P_PV"][t] * nn["cap_kVA_perPhase"]
                if om > 0:
                    PV_Forecast[nn["index"]][t][om] = (fore_inp["P_PV"][t] + fore_inp["P_PV_zeta+"][t]
                                                       - (fore_inp["P_PV_zeta+"][t] + fore_inp["P_PV_zeta-"][t])
                                                       * np.random.randint(0, 2)) * nn["cap_kVA_perPhase"]
    print("- PV data is generated for optimization.")
    ### Storages Parameters
    ST_Set = range(np.size(grid_inp["storage_elements"], 0))
    ST_Inc_Mat = np.zeros((np.size(grid_inp["storage_elements"], 0), grid_inp["Nbus"]))
    ST_S_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_max = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_min = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_LV = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_pos = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_neg = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_Eff_LC = np.zeros(np.size(grid_inp["storage_elements"], 0))
    ST_SOC_0 = np.zeros((np.size(grid_inp["storage_elements"], 0), Omega_Number))
    for nn in grid_inp["storage_elements"]:
        ST_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        ST_S_max[nn["index"]] = nn["S_max_kVA"]
        ST_SOC_max[nn["index"]] = nn["SOC_max_kWh"]
        ST_SOC_min[nn["index"]] = nn["SOC_min_kWh"]
        ST_Eff_LV[nn["index"]] = nn["Eff_LV"]
        ST_Eff_pos[nn["index"]] = nn["Eff_C"]
        ST_Eff_neg[nn["index"]] = nn["Eff_D"]
        ST_Eff_LC[nn["index"]] = nn["Eff_LC"]
        for om in Scen_Omega_Set:
            ST_SOC_0[nn["index"]][om] = nn["SOC_max_kWh"] * fore_inp["ST_SOC_0"]
            if om > 0:
                ST_SOC_0[nn["index"]][om] = ST_SOC_0[nn["index"]][om] * (1 + fore_inp["ST_SOC_zeta+"]
                                                                         - (fore_inp["ST_SOC_zeta+"]
                                                                            + fore_inp["ST_SOC_zeta-"])
                                                                         * np.random.randint(0, 2))
    print("- Storage data is generated for optimization.")
    ### Transmission System and Cost Function Parameters
    ConPoint_Set = range(np.size(grid_inp["grid_formers"], 0))
    ConPoint_Inc_Mat = np.zeros((np.size(grid_inp["grid_formers"], 0), grid_inp["Nbus"]))
    ConPoint_fac_P_pos = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], Omega_Number))
    ConPoint_fac_P_neg = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], Omega_Number))
    ConPoint_fac_Q_pos = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], Omega_Number))
    ConPoint_fac_Q_neg = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], Omega_Number))
    ConPoint_Vmag = np.zeros((np.size(grid_inp["grid_formers"], 0), meas_inp["Nt"], Omega_Number))
    for nn in grid_inp["grid_formers"]:
        ConPoint_Inc_Mat[nn["index"], af.find_n(nn["bus"], grid_inp["buses"])] = 1
        for t in Time_Set:
            for om in Scen_Omega_Set:
                ConPoint_Vmag[nn["index"]][t][om] = 1.0
                if om > 0:
                    ConPoint_fac_P_pos[nn["index"]][t][om] = np.random.randint(0, 2)
                    ConPoint_fac_P_neg[nn["index"]][t][om] = 1 - ConPoint_fac_P_pos[nn["index"]][t][om]
                    ConPoint_fac_Q_pos[nn["index"]][t][om] = np.random.randint(0, 2)
                    ConPoint_fac_Q_neg[nn["index"]][t][om] = 1 - ConPoint_fac_Q_pos[nn["index"]][t][om]
                    ConPoint_Vmag[nn["index"]][t][om] = 1 + fore_inp["Vmag_zeta+"][t] \
                                                        - (fore_inp["Vmag_zeta-"][t] +
                                                           fore_inp["Vmag_zeta+"][t]) * np.random.randint(0, 2)
    print("- Connection point data is generated for optimization.")
    ### Defining Variables
    PROB_DA = Model("PROB_DA")
    Line_P_t = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_b = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_f = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Vmag_sq = PROB_DA.addVars(Node_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_hat = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_hat = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_hat = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_hat = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    # Line_f_hat = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_over = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_over = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_over = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_over = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_f_over = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Vmag_sq_over = PROB_DA.addVars(Node_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_t_sq_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    Line_Q_t_sq_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    Line_P_b_sq_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    Line_Q_b_sq_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    Line_P_t_abs_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_t_abs_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_P_b_abs_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Line_Q_b_abs_max = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    PV_P = PROB_DA.addVars(PV_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    PV_Q = PROB_DA.addVars(PV_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ST_P = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ST_Q = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ST_SOC = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ST_SOC_tilde = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ST_P_pos = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    ST_P_neg = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=0)
    # ST_U_pos = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, vtype=GRB.BINARY)
    # ST_U_neg = PROB_DA.addVars(ST_Set, Time_Set, Scen_Omega_Set, vtype=GRB.BINARY)
    Net_P = PROB_DA.addVars(Node_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    Net_Q = PROB_DA.addVars(Node_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ConPoint_P_DA_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ConPoint_Q_DA_EN = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    ConPoint_P_DA_RS_pos = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_P_DA_RS_neg = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_Q_DA_RS_pos = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_Q_DA_RS_neg = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=0, ub=Big_M)
    ConPoint_P = PROB_DA.addVars(ConPoint_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ConPoint_Q = PROB_DA.addVars(ConPoint_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ConPoint_P_dev_pos = PROB_DA.addVars(ConPoint_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    ConPoint_P_dev_neg = PROB_DA.addVars(ConPoint_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    ConPoint_Q_dev_pos = PROB_DA.addVars(ConPoint_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    ConPoint_Q_dev_neg = PROB_DA.addVars(ConPoint_Set, Time_Set, Scen_Omega_Set, lb=0, ub=Big_M)
    OBJ_DA_MARKET = PROB_DA.addVars(ConPoint_Set, Time_Set, lb=-Big_M, ub=Big_M)
    OBJ_Loss = PROB_DA.addVars(Line_Set, Time_Set, Scen_Omega_Set, lb=-Big_M, ub=Big_M)
    ### Defning Constraints
    for t, om in itertools.product(Time_Set, Scen_Omega_Set):
        for nn in Node_Set:
            # (1a)
            PROB_DA.addConstr(Net_P[nn, t, om] ==
                              sum(PV_P[i, t, om] * PV_Inc_Mat[i, nn] for i in PV_Set)
                              + sum(ST_P[s, t, om] * ST_Inc_Mat[s, nn] for s in ST_Set)
                              - sum(DM_P[d, t, om] * DM_Inc_Mat[d, nn] for d in DM_Set)
                              + sum(ConPoint_P[f, t, om] * ConPoint_Inc_Mat[f, nn] for f in ConPoint_Set))
            # (1b)
            PROB_DA.addConstr(Net_Q[nn, t, om] ==
                              sum(PV_Q[i, t, om] * PV_Inc_Mat[i, nn] for i in PV_Set)
                              + sum(ST_Q[s, t, om] * ST_Inc_Mat[s, nn] for s in ST_Set)
                              - sum(DM_Q[d, t, om] * DM_Inc_Mat[d, nn] for d in DM_Set)
                              + sum(ConPoint_Q[f, t, om] * ConPoint_Inc_Mat[f, nn] for f in ConPoint_Set))
            # (12c) abd (12d) of Mostafa
            PROB_DA.addConstr(Vmag_sq_over[nn, t, om] <= V_max[nn] * V_max[nn])
            PROB_DA.addConstr(Vmag_sq[nn, t, om] >= V_min[nn] * V_min[nn])
        for n1, n2 in Line_Set:
            # (8a) of Mostafa
            PROB_DA.addConstr(Line_P_t[n1, n2, t, om] == - Net_P[n2, t, om]
                              + sum(Line_P_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zre[n1, n2] * Line_f[n1, n2, t, om] / 1000)
            PROB_DA.addConstr(Line_Q_t[n1, n2, t, om] == - Net_Q[n2, t, om]
                              + sum(Line_Q_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zim[n1, n2] * Line_f[n1, n2, t, om] / 1000
                              - 1000 * (Vmag_sq[n1, t, om] + Vmag_sq[n2, t, om]) * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (8b) of Mostafa
            PROB_DA.addConstr((Vmag_sq[n2, t, om] - Vmag_sq[n1, t, om]) * (Line_Vbase[n1, n2] ** 2) ==
                              - 2 * Line_Zre[n1, n2] * Line_P_t[n1, n2, t, om] / 1000
                              - 2 * Line_Zim[n1, n2] * Line_Q_t[n1, n2, t, om] / 1000
                              + 2 * Line_Zim[n1, n2] * Vmag_sq[n1, t, om] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2)
                              + (Line_Zre[n1, n2] * Line_Zre[n1, n2]
                                 + Line_Zim[n1, n2] * Line_Zim[n1, n2]) * Line_f[n1, n2, t, om] / 1000000)
            # (8d) of Mostafa
            PROB_DA.addConstr(Line_P_b[n1, n2, t, om] == - Net_P[n2, t, om]
                              + sum(Line_P_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_b[n1, n2, t, om] == - Net_Q[n2, t, om]
                              + sum(Line_Q_t[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            # (10) of Mostafa
            PROB_DA.addConstr(
                Line_f[n1, n2, t, om] * Vmag_sq[n1, t, om] * (Line_Vbase[n1, n2] ** 2) >= Line_P_t[n1, n2, t, om]
                * Line_P_t[n1, n2, t, om]
                + (Line_Q_t[n1, n2, t, om] + Vmag_sq[n1, t, om] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2) * 1000)
                * (Line_Q_t[n1, n2, t, om] + Vmag_sq[n1, t, om] * Line_b[n1, n2] * (Line_Vbase[n1, n2] ** 2) * 1000))
            # (11a) of Mostafa
            PROB_DA.addConstr(Line_P_t_hat[n1, n2, t, om] == - Net_P[n2, t, om]
                              + sum(Line_P_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_t_hat[n1, n2, t, om] == - Net_Q[n2, t, om]
                              + sum(Line_Q_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              - 1000 * (Vmag_sq_over[n1, t, om] + Vmag_sq_over[n2, t, om]) * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (11b) of Mostafa
            PROB_DA.addConstr((Vmag_sq_over[n2, t, om] - Vmag_sq_over[n1, t, om]) * (Line_Vbase[n1, n2] ** 2) ==
                              - 2 * Line_Zre[n1, n2] * Line_P_t_hat[n1, n2, t, om] / 1000
                              - 2 * Line_Zim[n1, n2] * Line_Q_t_hat[n1, n2, t, om] / 1000
                              + 2 * Line_Zim[n1, n2] * Vmag_sq_over[n1, t, om] * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (11c) of Mostafa
            PROB_DA.addConstr(Line_P_t_over[n1, n2, t, om] == - Net_P[n2, t, om]
                              + sum(Line_P_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zre[n1, n2] * Line_f_over[n1, n2, t, om] / 1000)
            PROB_DA.addConstr(Line_Q_t_over[n1, n2, t, om] == - Net_Q[n2, t, om]
                              + sum(Line_Q_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set)
                              + Line_Zim[n1, n2] * Line_f_over[n1, n2, t, om] / 1000
                              - 1000 * (Vmag_sq[n1, t, om] + Vmag_sq[n2, t, om]) * Line_b[n1, n2] * (
                                      Line_Vbase[n1, n2] ** 2))
            # (11d) of Mostafa
            PROB_DA.addConstr(Line_f_over[n1, n2, t, om] * Vmag_sq[n2, t, om] * (Line_Vbase[n1, n2] ** 2)
                              >= Line_P_b_sq_max[n1, n2, t, om] * Line_P_b_sq_max[n1, n2, t, om]
                              + Line_Q_b_sq_max[n1, n2, t, om] * Line_Q_b_sq_max[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t, om] >= Line_P_b_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t, om] >= - Line_P_b_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t, om] >= Line_P_b_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_sq_max[n1, n2, t, om] >= - Line_P_b_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t, om] >=
                              (Line_Q_b_hat[n1, n2, t, om] - Vmag_sq_over[n2, t, om] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t, om] >=
                              - (Line_Q_b_hat[n1, n2, t, om] - Vmag_sq_over[n2, t, om] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t, om] >=
                              (Line_Q_b_over[n1, n2, t, om] - Vmag_sq[n2, t, om] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_b_sq_max[n1, n2, t, om] >=
                              - (Line_Q_b_over[n1, n2, t, om] - Vmag_sq[n2, t, om] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            # (11e) of Mostafa
            PROB_DA.addConstr(Line_f_over[n1, n2, t, om] * Vmag_sq[n1, t, om] * (Line_Vbase[n1, n2] ** 2)
                              >= Line_P_t_sq_max[n1, n2, t, om] * Line_P_t_sq_max[n1, n2, t, om]
                              + Line_Q_t_sq_max[n1, n2, t, om] * Line_Q_t_sq_max[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t, om] >= Line_P_t_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t, om] >= - Line_P_t_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t, om] >= Line_P_t_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_sq_max[n1, n2, t, om] >= - Line_P_t_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t, om] >=
                              (Line_Q_t_hat[n1, n2, t, om] + Vmag_sq_over[n1, t, om] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t, om] >=
                              - (Line_Q_t_hat[n1, n2, t, om] + Vmag_sq_over[n1, t, om] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t, om] >=
                              (Line_Q_t_over[n1, n2, t, om] + Vmag_sq[n1, t, om] * Line_b[n1, n2]
                               * (Line_Vbase[n1, n2] ** 2) * 1000))
            PROB_DA.addConstr(Line_Q_t_sq_max[n1, n2, t, om] >=
                              - (Line_Q_t_over[n1, n2, t, om] + Vmag_sq[n1, t, om] * Line_b[n1, n2]
                                 * (Line_Vbase[n1, n2] ** 2) * 1000))
            # (11f) of Mostafa
            PROB_DA.addConstr(Line_P_b_over[n1, n2, t, om] == - Net_P[n2, t, om]
                              + sum(Line_P_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_b_over[n1, n2, t, om] == - Net_Q[n2, t, om]
                              + sum(Line_Q_t_over[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            # (11g) of Mostafa
            PROB_DA.addConstr(Line_P_b_hat[n1, n2, t, om] == - Net_P[n2, t, om]
                              + sum(Line_P_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            PROB_DA.addConstr(Line_Q_b_hat[n1, n2, t, om] == - Net_Q[n2, t, om]
                              + sum(Line_Q_t_hat[n3, n4, t, om] * np.where(n3 == n2, 1, 0) for n3, n4 in Line_Set))
            # (12e) of Mostafa
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t, om] * Line_P_b_abs_max[n1, n2, t, om]
                              + Line_Q_b_abs_max[n1, n2, t, om] * Line_Q_b_abs_max[n1, n2, t, om]
                              <= Vmag_sq[n2, t, om] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t, om] >= Line_P_b_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t, om] >= -Line_P_b_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t, om] >= Line_P_b_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_b_abs_max[n1, n2, t, om] >= -Line_P_b_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t, om] >= Line_Q_b_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t, om] >= -Line_Q_b_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t, om] >= Line_Q_b_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_b_abs_max[n1, n2, t, om] >= -Line_Q_b_over[n1, n2, t, om])
            # (12f) of Mostafa
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t, om] * Line_P_t_abs_max[n1, n2, t, om]
                              + Line_Q_t_abs_max[n1, n2, t, om] * Line_Q_t_abs_max[n1, n2, t, om]
                              <= Vmag_sq[n1, t, om] * (Line_Smax[n1, n2] ** 2) * 9 / (1000 * Line_Vbase[n1, n2] ** 2))
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t, om] >= Line_P_t_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t, om] >= -Line_P_t_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t, om] >= Line_P_t_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_P_t_abs_max[n1, n2, t, om] >= -Line_P_t_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t, om] >= Line_Q_t_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t, om] >= -Line_Q_t_hat[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t, om] >= Line_Q_t_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_t_abs_max[n1, n2, t, om] >= -Line_Q_t_over[n1, n2, t, om])
            # (12g) of Mostafa
            PROB_DA.addConstr(Line_P_t[n1, n2, t, om] <= Line_P_t_over[n1, n2, t, om])
            PROB_DA.addConstr(Line_Q_t[n1, n2, t, om] <= Line_Q_t_over[n1, n2, t, om])
        for s in ST_Set:
            # (21a)
            if t == 0:
                PROB_DA.addConstr(ST_SOC[s, t, om] == ST_Eff_LV[s] * ST_SOC_0[s, om]
                                  - ST_P_neg[s, t, om] * DeltaT * ST_Eff_neg[s]
                                  - ST_P_pos[s, t, om] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(ST_SOC_tilde[s, t, om] == ST_Eff_LV[s] * ST_SOC_0[s, om]
                                  - ST_P[s, t, om] * DeltaT)
            else:
                PROB_DA.addConstr(ST_SOC[s, t, om] == ST_Eff_LV[s] * ST_SOC[s, t - 1, om]
                                  - ST_P_neg[s, t, om] * DeltaT * ST_Eff_neg[s]
                                  - ST_P_pos[s, t, om] * DeltaT / ST_Eff_pos[s])
                PROB_DA.addConstr(ST_SOC_tilde[s, t, om] == ST_Eff_LV[s] * ST_SOC[s, t - 1, om]
                                  - ST_P[s, t, om] * DeltaT)
            # (22d)
            PROB_DA.addConstr(ST_P[s, t, om] == ST_P_pos[s, t, om] + ST_P_neg[s, t, om])
            # (22e)
            PROB_DA.addConstr(
                ST_Q[s, t, om] * ST_Q[s, t, om] + ST_P[s, t, om] * ST_P[s, t, om] <= ST_S_max[s] * ST_S_max[s])
            # (22b)
            PROB_DA.addConstr(ST_P_pos[s, t, om] <= ST_S_max[s])
            # (22c)
            PROB_DA.addConstr(ST_P_neg[s, t, om] >= -ST_S_max[s])
            # (22f)
            PROB_DA.addConstr(ST_SOC[s, t, om] <= ST_SOC_max[s])
            PROB_DA.addConstr(ST_SOC[s, t, om] >= ST_SOC_min[s])
            # (22h)
            PROB_DA.addConstr(ST_SOC_tilde[s, t, om] <= ST_SOC_max[s])
            PROB_DA.addConstr(ST_SOC_tilde[s, t, om] >= ST_SOC_min[s])
        for i in PV_Set:
            # (28)
            PROB_DA.addConstr(PV_P[i, t, om] <= PV_Forecast[i, t, om])
            # (26)
            PROB_DA.addConstr(PV_Q[i, t, om] <= PV_P[i, t, om] * np.tan(np.arccos(PV_cos[i])))
            PROB_DA.addConstr(PV_Q[i, t, om] >= - PV_P[i, t, om] * np.tan(np.arccos(PV_cos[i])))
            # PROB_DA.addConstr(PV_P[i, t, om] * PV_P[i, t, om]
            #                   + (PV_Q[i, t, om] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   * (PV_Q[i, t, om] + PV_V_grid[i] * PV_V_grid[i] * PV_cap[i] / PV_X[i])
            #                   <= (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i])
            #                   * (PV_cap[i] * PV_V_grid[i] * PV_V_conv[i] / PV_X[i]))
            # (27)
            PROB_DA.addConstr(
                PV_P[i, t, om] * PV_P[i, t, om] + PV_Q[i, t, om] * PV_Q[i, t, om] <= PV_cap[i] * PV_cap[i])
            # PROB_DA.addConstr(PV_Q[i, t, om] == 0)
        for f in ConPoint_Set:
            PROB_DA.addConstr(
                ConPoint_P[f, t, om] == sum(Line_P_t[n1, n2, t, om] * ConPoint_Inc_Mat[f, n1] for n1, n2 in Line_Set))
            PROB_DA.addConstr(
                ConPoint_Q[f, t, om] == sum(Line_Q_t[n1, n2, t, om] * ConPoint_Inc_Mat[f, n1] for n1, n2 in Line_Set))
            # first line of (34a)
            PROB_DA.addConstr(OBJ_DA_MARKET[f, t] == - LAMBDA_P_DA_EN[t] * ConPoint_P_DA_EN[f, t]
                              - LAMBDA_Q_DA_EN[t] * ConPoint_Q_DA_EN[f, t]
                              + LAMBDA_P_DA_RS_pos[t] * ConPoint_P_DA_RS_pos[f, t]
                              + LAMBDA_P_DA_RS_neg[t] * ConPoint_P_DA_RS_neg[f, t]
                              + LAMBDA_Q_DA_RS_pos[t] * ConPoint_Q_DA_RS_pos[f, t]
                              + LAMBDA_Q_DA_RS_neg[t] * ConPoint_Q_DA_RS_neg[f, t]
                              - Big_M * sum(ConPoint_P_dev_pos[f, t, om] + ConPoint_P_dev_neg[f, t, om]
                                            + ConPoint_Q_dev_pos[f, t, om] + ConPoint_Q_dev_neg[f, t, om]
                                            for om in Scen_Omega_Set))
            # constraints for defining deployed reserves
            PROB_DA.addConstr(ConPoint_P[f, t, om] == ConPoint_P_DA_EN[f, t]
                              - ConPoint_fac_P_pos[f, t, om] * ConPoint_P_DA_RS_pos[f, t]
                              + ConPoint_fac_P_neg[f, t, om] * ConPoint_P_DA_RS_neg[f, t]
                              - ConPoint_P_dev_pos[f, t, om] + ConPoint_P_dev_neg[f, t, om])
            PROB_DA.addConstr(ConPoint_Q[f, t, om] == ConPoint_Q_DA_EN[f, t]
                              - ConPoint_fac_Q_pos[f, t, om] * ConPoint_Q_DA_RS_pos[f, t]
                              + ConPoint_fac_Q_neg[f, t, om] * ConPoint_Q_DA_RS_neg[f, t]
                              - ConPoint_Q_dev_pos[f, t, om] + ConPoint_Q_dev_neg[f, t, om])
            # slack bus definition
            PROB_DA.addConstr(sum(Vmag_sq[nn, t, om] * ConPoint_Inc_Mat[f, nn] for nn in Node_Set)
                              == ConPoint_Vmag[f, t, om] * ConPoint_Vmag[f, t, om])
            PROB_DA.addConstr(ConPoint_P_DA_RS_pos[f, t] <= Line_Smax[0, 1] / 2)
            PROB_DA.addConstr(ConPoint_P_DA_RS_neg[f, t] <= Line_Smax[0, 1] / 2)
            PROB_DA.addConstr(ConPoint_Q_DA_RS_pos[f, t] <= Line_Smax[0, 1] / 2)
            PROB_DA.addConstr(ConPoint_Q_DA_RS_neg[f, t] <= Line_Smax[0, 1] / 2)
        for n1, n2 in Line_Set:
            PROB_DA.addConstr(
                OBJ_Loss[n1, n2, t, om] >= sum(Line_Zre[n1, n2] * Line_f[n1, n2, t, om] for n1, n2 in Line_Set))
    ### Defining Objective and Solving Problem
    if loss_consideration == 0:
        PROB_DA.setObjective(1000 * DeltaT * OBJ_DA_MARKET.sum(), GRB.MAXIMIZE)
    else:
        PROB_DA.setObjective(OBJ_Loss.sum(), GRB.MINIMIZE)
    PROB_DA.Params.BarHomogeneous = 1
    PROB_DA.optimize()
    ### Solution
    DA_result = {}
    try:
        Solution_PV_P = PROB_DA.getAttr('x', PV_P)
        Solution_PV_Q = PROB_DA.getAttr('x', PV_Q)
        Solution_ST_P = PROB_DA.getAttr('x', ST_P)
        Solution_ST_Q = PROB_DA.getAttr('x', ST_Q)
        Solution_ST_SOC = PROB_DA.getAttr('x', ST_SOC)
        for t in Time_Set:
            for k in range(3):
                for nn in Node_Set:
                    if nn > 0:
                        nnn = grid_inp["buses"][nn]["bus"]
                        nnn = af.find_n(nnn, meas_inp["meas_location"])
                        meas_inp["P"][nnn][k][t] = sum(Solution_PV_P[i, t, 0] * PV_Inc_Mat[i, nn] for i in PV_Set) \
                                                   + sum(Solution_ST_P[s, t, 0] * ST_Inc_Mat[s, nn] for s in ST_Set) \
                                                   - sum(DM_P[d, t, 0] * DM_Inc_Mat[d, nn] for d in DM_Set)
                        meas_inp["Q"][nnn][k][t] = sum(Solution_PV_Q[i, t, 0] * PV_Inc_Mat[i, nn] for i in PV_Set) \
                                                   + sum(Solution_ST_Q[s, t, 0] * ST_Inc_Mat[s, nn] for s in ST_Set) \
                                                   - sum(DM_Q[d, t, 0] * DM_Inc_Mat[d, nn] for d in DM_Set)
        DA_P = PROB_DA.getAttr('x', ConPoint_P_DA_EN)
        DA_Q = PROB_DA.getAttr('x', ConPoint_Q_DA_EN)
        DA_RP_pos = PROB_DA.getAttr('x', ConPoint_P_DA_RS_pos)
        DA_RP_neg = PROB_DA.getAttr('x', ConPoint_P_DA_RS_neg)
        DA_RQ_pos = PROB_DA.getAttr('x', ConPoint_Q_DA_RS_pos)
        DA_RQ_neg = PROB_DA.getAttr('x', ConPoint_Q_DA_RS_neg)
        Solution_ST_SOCC = [[Solution_ST_SOC[s, t, 0] for t in Time_Set] for s in ST_Set]
        Solution_ST_PP = [[Solution_ST_P[s, t, 0] for t in Time_Set] for s in ST_Set]
        Solution_ST_QQ = [[Solution_ST_Q[s, t, 0] for t in Time_Set] for s in ST_Set]
        Solution_PV_PP = [[Solution_PV_P[i, t, 0] for t in Time_Set] for i in PV_Set]
        Solution_PV_QQ = [[Solution_PV_Q[i, t, 0] for t in Time_Set] for i in PV_Set]
        DA_PP = [DA_P[0, t] for t in Time_Set]
        DA_QQ = [DA_Q[0, t] for t in Time_Set]
        DA_RPP_pos = [DA_RP_pos[0, t] for t in Time_Set]
        DA_RPP_neg = [DA_RP_neg[0, t] for t in Time_Set]
        DA_RQQ_pos = [DA_RQ_pos[0, t] for t in Time_Set]
        DA_RQQ_neg = [DA_RQ_neg[0, t] for t in Time_Set]
        meas = {}
        meas["DA_PP"] = -np.array(DA_PP)
        meas["DA_QQ"] = -np.array(DA_QQ)
        meas["DA_P+"] = -np.array(DA_PP) + np.array(DA_RPP_pos)
        meas["DA_Q+"] = -np.array(DA_QQ) + np.array(DA_RQQ_pos)
        meas["DA_P-"] = -np.array(DA_PP) - np.array(DA_RPP_neg)
        meas["DA_Q-"] = -np.array(DA_QQ) - np.array(DA_RQQ_neg)
        af.figuring(grid_inp, meas_inp, meas, "DA_Offers", case_name)
        # meas2 = af.SE_time_series(grid_inp, meas_inp, case_name)
        DA_result["Solution_PV_P"] = Solution_PV_PP
        DA_result["Solution_PV_Q"] = Solution_PV_QQ
        DA_result["Solution_ST_P"] = Solution_ST_PP
        DA_result["Solution_ST_Q"] = Solution_ST_QQ
        DA_result["Solution_ST_SOC"] = Solution_ST_SOCC
        DA_result["DA_P"] = DA_PP
        DA_result["DA_Q"] = DA_QQ
        DA_result["DA_RP_pos"] = DA_RPP_pos
        DA_result["DA_RP_neg"] = DA_RPP_neg
        DA_result["DA_RQ_pos"] = DA_RQQ_pos
        DA_result["DA_RQ_neg"] = DA_RQQ_neg
        DA_result["delta"] = 0.001
        obj = PROB_DA.getObjective()
        output_DF.loc['obj', case_name] = obj.getValue()
        output_DF.loc['DA_P_avg', case_name] = sum(DA_PP) / len(DA_PP)
        output_DF.loc['DA_Q_avg', case_name] = sum(DA_QQ) / len(DA_QQ)
        output_DF.loc['DA_RP_pos_avg', case_name] = sum(DA_RPP_pos) / len(DA_RPP_pos)
        output_DF.loc['DA_RP_neg_avg', case_name] = sum(DA_RPP_neg) / len(DA_RPP_neg)
        output_DF.loc['DA_RQ_pos_avg', case_name] = sum(DA_RQQ_pos) / len(DA_RQQ_pos)
        output_DF.loc['DA_RQ_neg_avg', case_name] = sum(DA_RQQ_neg) / len(DA_RQQ_neg)
        DA_result["time_out"] = False
    except:
        DA_result["time_out"] = True
    return DA_result, output_DF
