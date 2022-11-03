"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
from random import gauss
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Global variables
# ----------------------------------------------------------------------------------------------------------------------
os.environ['R_HOME'] = r'C:/Program Files/R/R-4.2.2'
plt.set_loglevel('WARNING')


# Functions
# ----------------------------------------------------------------------------------------------------------------------
def reine_parameters():
    """
    @description: This function is for loading the constant parameters of REINE
    @return parameters[{R_ReIne_N, R_ReIne_U, R_ReIne_V, R_ReIne_W, L_ReIne_N, L_ReIne_U, L_ReIne_V,
                        L_ReIne_W, fn, Vbase, Sbase, Zbase, Zsc_tr, Xi_tr, X_Tr, R_Tr, L_cable, L2X,
                        R_cable, X_cable, Cap, Ond_Cap, Ond_V_grid_pu, Ond_V_conv_pu, Ond_X_PV_pu,
                        Con_SOC_min_kWh, Con_SOC_max_kWh, Con_S_max_kVA, Con_P_max_kW, Con_Eff_LV,
                        Con_Eff_D, Con_Eff_C, Con_Eff_LC}]
    """
    parameters = {}
    # Common data of ReIne
    parameters['R_ReIne_N'] = [0.000189979014171097, 0.000202101052864018, 0.000197770150897533, 0.000201645981833550,
                               0.000196816884062269, 0.000204301729810993, 0.000203295359336467, 0.000205423185075427,
                               0.000190519829003150]
    parameters['R_ReIne_U'] = [0.000190003424595008, 0.000199636801763190, 0.000199864761049872, 0.000199096710029844,
                               0.000194292188081219, 0.000206087597804205, 0.000209955365455973, 0.000202863220028132,
                               0.000193121046397916]
    parameters['R_ReIne_V'] = [0.000190199480230380, 0.000202379367815811, 0.000198593986621042, 0.000199528277520922,
                               0.000196408241285373, 0.000203320659149361, 0.000205236928248479, 0.000207965037632736,
                               0.000192629220111695]
    parameters['R_ReIne_W'] = [0.000190156300418500, 0.000202047164279434, 0.000201759384596844, 0.000199871082906516,
                               0.000196836385513956, 0.000207017314097440, 0.000203131454082999, 0.000206428069784467,
                               0.000192713982701088]
    parameters['L_ReIne_N'] = [0.000189979014171097, 0.000202101052864018, 0.000197770150897533, 0.000201645981833550,
                               0.000196816884062269, 0.000204301729810993, 0.000203295359336467, 0.000205423185075427,
                               0.000190519829003150]
    parameters['L_ReIne_U'] = [0.000190003424595008, 0.000199636801763190, 0.000199864761049872, 0.000199096710029844,
                               0.000194292188081219, 0.000206087597804205, 0.000209955365455973, 0.000202863220028132,
                               0.000193121046397916]
    parameters['L_ReIne_V'] = [0.000190199480230380, 0.000202379367815811, 0.000198593986621042, 0.000199528277520922,
                               0.000196408241285373, 0.000203320659149361, 0.000205236928248479, 0.000207965037632736,
                               0.000192629220111695]
    parameters['L_ReIne_W'] = [0.000190156300418500, 0.000202047164279434, 0.000201759384596844, 0.000199871082906516,
                               0.000196836385513956, 0.000207017314097440, 0.000203131454082999, 0.000206428069784467,
                               0.000192713982701088]
    parameters['fn'] = 50
    parameters['Vbase'] = 400  # in [V]
    parameters['Sbase'] = 1e5  # in [VA]
    parameters['Zbase'] = parameters['Vbase'] ** 2 / parameters['Sbase']
    parameters['Zsc_tr'] = 0.0257
    parameters['Xi_tr'] = 0.0144
    parameters['X_Tr'] = parameters['Xi_tr']  # in per-unit
    parameters['R_Tr'] = np.sqrt(parameters['Zsc_tr'] ** 2 - parameters['Xi_tr'] ** 2)  # in per-unit
    parameters['L_cable'] = 0.11
    parameters['L2X'] = 2 * 3.14 * parameters['fn']
    parameters['R_cable'] = parameters['L_cable'] * 0.272 / parameters['Zbase']  # in [per-unit] Cable EPR-PVR 5*70mm2
    parameters['X_cable'] = parameters['L_cable'] * 0.070 / parameters['Zbase']  # in [per-unit] Cable EPR-PVR 5*70mm2
    parameters['Cap'] = 100
    # Ond = [SolarMax, ABB, KACO]
    parameters['Ond_Cap'] = [5, 8.5, 9]
    parameters['Ond_V_grid_pu'] = [1, 1, 1]
    parameters['Ond_V_conv_pu'] = [545 / 400, 620 / 400, 575 / 400]
    parameters['Ond_X_PV_pu'] = \
        [3 * parameters['Ond_V_grid_pu'][x] * (parameters['Ond_V_conv_pu'][x] - parameters['Ond_V_grid_pu'][x])
         * parameters['Sbase'] / (1000 * parameters['Ond_Cap'][x]) for x in range(3)]
    parameters['Ond_cos_PV'] = [0.89, 0.89, 0.89]
    # Con = [Battery]
    parameters['Con_SOC_min_kWh'] = [64*10/100]
    parameters['Con_SOC_max_kWh'] = [64*90/100]
    parameters['Con_S_max_kVA'] = [120]
    parameters['Con_P_max_kW'] = [100]
    parameters['Con_Eff_LV'] = [1]
    parameters['Con_Eff_D'] = [0.98]
    parameters['Con_Eff_C'] = [0.98]
    parameters['Con_Eff_LC'] = [1]
    return parameters


def grid_topology_sim(case_name: str, vec_inp: list):
    """
    @description: This function is for generating the file of grid_inp
    @param case_name: name of the case
    @param vec_inp: vector of input parameters as ordered by Douglas in "07/10/2020 14:06"
    (See "\data\Ordre_informations_controle_2020.10.07.xlsx")
    @return data: a dictionary gives the specification of network based on the standard of pydgrid package
          data = {"lines": [{"bus_j": "1", "bus_k": "4", "code": "Line4", "m": 219, "Cap": 470},],
                  "buses": [{"bus": "0", "pos_x": -1500.0, "pos_y": 200.0, "units": "m", "U_kV": 20, "Vmin": 0.9,
                  "Vmax": 1.1},],
                  "transformers": [{"bus_j": "0", "bus_k": "1", "S_n_kVA": 250.0, "U_1_kV": 20.0, "U_2_kV": 0.4,
                                  "Cap": 250.0, "R_cc_pu": np.sqrt(0.041 ** 2 / (1 + 2.628 ** 2)),
                                  "X_cc_pu": 2.628 * np.sqrt(0.041 ** 2 / (1 + 2.628 ** 2)),
                                  "connection": "Dyn11", "conductors_1": 3, "conductors_2": 4},],
                  "grid_formers": [{"index": 0, "bus": "0", "bus_nodes": [1, 2, 3], "kV": [20, 20, 20],
                                    "deg": [30, 150, 270]},],
                  "grid_feeders": [{"bus": "1", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},],
                  "shunts": [{"bus": "1", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},],
                  "line_codes": {"Line2": {"R1": 0.263, "X1": 0.078, "B_1_mu": 0 * 0.73 * (2 * 3.14 * 50)}}
                }
          data["PV_elements"] = [{"index": 0, "bus": "2", "cap_kVA_perPhase": 200 / 3, "V_grid_pu": 1, "V_conv_pu": 1,
          "X_PV_pu": 1}]
          data["load_elements"] = [{"index": 0, "bus": "1"}]
          data["storage_elements"] = [{"index": 0, "bus": "2", "SOC_min_kWh": 0, "SOC_max_kWh": 40, "S_max_kVA": 20,
                                      "P_max_kW": 10, "Eff_LV": 1, "Eff_D": 1, "Eff_C": 1, "Eff_LC": 1}]
          data["Nbus"] = 7
          data["Zbase"] = 400 ** 2 / 25e4
    """
    # Input data
    parameters = reine_parameters()
    r_reine_u = parameters['R_ReIne_U']
    r_reine_v = parameters['R_ReIne_V']
    r_reine_w = parameters['R_ReIne_W']
    l_reine_u = parameters['L_ReIne_U']
    l_reine_v = parameters['L_ReIne_V']
    l_reine_w = parameters['L_ReIne_W']
    sbase = parameters['Sbase']
    zbase = parameters['Zbase']
    x_tr = parameters['X_Tr']
    r_tr = parameters['R_Tr']
    l2x = parameters['L2X']
    r_cable = parameters['R_cable']
    x_cable = parameters['X_cable']
    cap = parameters['Cap']
    ond_cap = parameters['Ond_Cap']
    ond_v_grid_pu = parameters['Ond_V_grid_pu']
    ond_v_conv_pu = parameters['Ond_V_conv_pu']
    ond_x_pv_pu = parameters['Ond_X_PV_pu']
    ond_cos_pv = parameters['Ond_cos_PV']
    con_soc_min_kwh = parameters['Con_SOC_min_kWh']
    con_soc_max_kwh = parameters['Con_SOC_max_kWh']
    con_s_max_kva = parameters['Con_S_max_kVA']
    con_p_max_kw = parameters['Con_P_max_kW']
    con_eff_lv = parameters['Con_Eff_LV']
    con_eff_d = parameters['Con_Eff_D']
    con_eff_c = parameters['Con_Eff_C']
    con_eff_lc = parameters['Con_Eff_LC']
    charge, ond, con = [], [], []
    s_73k, s_74k, s_l9 = [], [], []
    data = {}
    # Defining switches state for different cases
    # Sample Case
    # https://pydgrid.readthedocs.io/en/latest/jupyter_notebooks/wind_farm.html
    if case_name == "9BusesPydGridExample":
        data = {
            "lines": [
                {"bus_j": "W1mv", "bus_k": "W2mv", "code": "mv_al_300", "m": 500},
                {"bus_j": "W2mv", "bus_k": "W3mv", "code": "mv_al_300", "m": 500},
                {"bus_j": "W3mv", "bus_k": "POImv", "code": "mv_al_300", "m": 500},
                {"bus_j": "POI", "bus_k": "GRID", "code": "hv_line", "m": 50.0e3},
            ],
            "buses": [
                {"bus": "W1lv", "pos_x": -1500.0, "pos_y": 200.0, "units": "m", "U_kV": 0.69},
                {"bus": "W2lv", "pos_x": -1000.0, "pos_y": 200.0, "units": "m", "U_kV": 0.69},
                {"bus": "W3lv", "pos_x": -500.0, "pos_y": 200.0, "units": "m", "U_kV": 0.69},
                {"bus": "W1mv", "pos_x": -1500.0, "pos_y": 180.0, "units": "m", "U_kV": 20.0},
                {"bus": "W2mv", "pos_x": -1000.0, "pos_y": 180.0, "units": "m", "U_kV": 20.0},
                {"bus": "W3mv", "pos_x": -500.0, "pos_y": 180.0, "units": "m", "U_kV": 20.0},
                {"bus": "POImv", "pos_x": 0.0, "pos_y": 0.0, "units": "m", "U_kV": 20.0},
                {"bus": "POI", "pos_x": 100.0, "pos_y": 0.0, "units": "m", "U_kV": 66.0},
                {"bus": "GRID", "pos_x": 500.0, "pos_y": 0.0, "units": "m", "U_kV": 66.0},
            ],
            "transformers": [
                {"bus_j": "POImv", "bus_k": "POI", "S_n_kVA": 10000.0, "U_1_kV": 20.0, "U_2_kV": 66.0,
                 "R_cc_pu": 0.01, "X_cc_pu": 0.08, "connection": "Dyg11_3w", "conductors_1": 3, "conductors_2": 3},
                {"bus_j": "W1mv", "bus_k": "W1lv", "S_n_kVA": 2500.0, "U_1_kV": 20, "U_2_kV": 0.69,
                 "R_cc_pu": 0.01, "X_cc_pu": 0.06, "connection": "Dyg11_3w", "conductors_1": 3, "conductors_2": 3},
                {"bus_j": "W2mv", "bus_k": "W2lv", "S_n_kVA": 2500.0, "U_1_kV": 20, "U_2_kV": 0.69,
                 "R_cc_pu": 0.01, "X_cc_pu": 0.06, "connection": "Dyg11_3w", "conductors_1": 3, "conductors_2": 3},
                {"bus_j": "W3mv", "bus_k": "W3lv", "S_n_kVA": 2500.0, "U_1_kV": 20, "U_2_kV": 0.69,
                 "R_cc_pu": 0.01, "X_cc_pu": 0.06, "connection": "Dyg11_3w", "conductors_1": 3, "conductors_2": 3},
            ],
            "grid_formers": [
                {"bus": "GRID", "bus_nodes": [1, 2, 3],
                 "kV": [38.105, 38.105, 38.105], "deg": [30, 150, 270.0]
                 }
            ],
            "grid_feeders": [{"bus": "W1lv", "bus_nodes": [1, 2, 3], "kW": 2000, "kvar": 0},
                             {"bus": "W2lv", "bus_nodes": [1, 2, 3], "kW": 2000, "kvar": 0},
                             {"bus": "W3lv", "bus_nodes": [1, 2, 3], "kW": 2000, "kvar": 0},
                             {"bus": "POImv", "bus_nodes": [1, 2, 3], "kW": 0, "kvar": 0}  
                             ],
            "groundings": [
                {"bus": "POImv", "R_gnd": 32.0, "X_gnd": 0.0, "conductors": 3}
            ],
            "line_codes":
                {
                    "mv_al_150": {"R1": 0.262, "X1": 0.118, "C_1_muF": 0.250},
                    "mv_al_185": {"R1": 0.209, "X1": 0.113, "C_1_muF": 0.281},
                    "mv_al_240": {"R1": 0.161, "X1": 0.109, "C_1_muF": 0.301},
                    "mv_al_300": {"R1": 0.128, "X1": 0.105, "C_1_muF": 0.340},
                    "hv_line": {"R1": 0.219, "X1": 0.365, "R0": 0.219, "X0": 0.365}
                }
        }
    elif case_name == "6BusesLaChappelle":
        data: dict = {
            "lines": [
                {"bus_j": "1", "bus_k": "4", "code": "Line4", "m": 219, "Cap": 470},
                {"bus_j": "1", "bus_k": "2", "code": "Line2", "m": 145, "Cap": 500},
                {"bus_j": "2", "bus_k": "3", "code": "Line3", "m": 128, "Cap": 380},
                {"bus_j": "1", "bus_k": "5", "code": "Line5", "m": 118, "Cap": 225},
                {"bus_j": "5", "bus_k": "6", "code": "Line6", "m": 68, "Cap": 290},
            ],
            "buses": [
                {"bus": "0", "pos_x": -1500.0, "pos_y": 200.0, "units": "m", "U_kV": 20, "Vmin": 0.9, "Vmax": 1.1},
                {"bus": "1", "pos_x": -1000.0, "pos_y": 200.0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1},
                {"bus": "2", "pos_x": -500.0, "pos_y": 200.0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1},
                {"bus": "3", "pos_x": -1500.0, "pos_y": 180.0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1},
                {"bus": "4", "pos_x": -1000.0, "pos_y": 180.0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1},
                {"bus": "5", "pos_x": -500.0, "pos_y": 180.0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1},
                {"bus": "6", "pos_x": 0.0, "pos_y": 0.0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1},
            ],
            "transformers": [
                {"bus_j": "0", "bus_k": "1", "S_n_kVA": 250.0, "U_1_kV": 20.0, "U_2_kV": 0.4, "Cap": 250.0,
                 "R_cc_pu": np.sqrt(0.041 ** 2 / (1 + 2.628 ** 2)),
                 "X_cc_pu": 2.628 * np.sqrt(0.041 ** 2 / (1 + 2.628 ** 2)),
                 "connection": "Dyn11", "conductors_1": 3, "conductors_2": 4},
            ],
            "grid_formers": [
                {"index": 0, "bus": "0", "bus_nodes": [1, 2, 3],
                 "kV": [20, 20, 20], "deg": [30, 150, 270]
                 }
            ],
            "grid_feeders": [{"bus": "1", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},
                             {"bus": "2", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},
                             {"bus": "3", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},
                             {"bus": "4", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},
                             {"bus": "5", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},
                             {"bus": "6", "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0},
                             ],
            "shunts": [{"bus": "1", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},
                       {"bus": "2", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},
                       {"bus": "3", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},
                       {"bus": "4", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},
                       {"bus": "5", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},
                       {"bus": "6", "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]},
                       ],
            "line_codes":
                {
                    "Line2": {"R1": 0.263, "X1": 0.078, "B_1_mu": 0 * 0.73 * (2 * 3.14 * 50)},
                    "Line3": {"R1": 0.020, "X1": 0.009, "B_1_mu": 0 * 31.222},
                    "Line4": {"R1": 0.096, "X1": 0.072, "B_1_mu": 0 * 0.77 * (2 * 3.14 * 50)},
                    "Line5": {"R1": 0.028, "X1": 0.009, "B_1_mu": 0 * 91.194},
                    "Line6": {"R1": 0.018, "X1": 0.005, "B_1_mu": 0 * 15.664}
                }
        }
        data["PV_elements"] = [{"index": 0, "bus": "2",
                                "cap_kVA_perPhase": 200 / 3, "V_grid_pu": 1,
                                "V_conv_pu": 1, "X_PV_pu": 1},
                               {"index": 1, "bus": "3",
                                "cap_kVA_perPhase": 10 / 3, "V_grid_pu": 1,
                                "V_conv_pu": 1, "X_PV_pu": 1},
                               {"index": 2, "bus": "5",
                                "cap_kVA_perPhase": 60 / 3, "V_grid_pu": 1,
                                "V_conv_pu": 1, "X_PV_pu": 1}]
        data["load_elements"] = [{"index": 0, "bus": "1"},
                                 {"index": 1, "bus": "2"},
                                 {"index": 2, "bus": "3"},
                                 {"index": 3, "bus": "4"},
                                 {"index": 4, "bus": "5"},
                                 {"index": 5, "bus": "6"}]
        data["storage_elements"] = [{"index": 0, "bus": "2",
                                     "SOC_min_kWh": 0,
                                     "SOC_max_kWh": 40,
                                     "S_max_kVA": 20,
                                     "Eff_LV": 1,
                                     "Eff_D": 1, "Eff_C": 1,
                                     "Eff_LC": 1},
                                    {"index": 1, "bus": "5",
                                     "SOC_min_kWh": 0,
                                     "SOC_max_kWh": 290,
                                     "S_max_kVA": 220,
                                     "Eff_LV": 1,
                                     "Eff_D": 1, "Eff_C": 1,
                                     "Eff_LC": 1}]
        data["Nbus"] = 7
        data["Zbase"] = 400 ** 2 / 250000
    # "\\eistore2\iese\institut\Projets\Projets_MRB\Validation Tests and Analysis\Analysis"
    elif case_name == "Case_4bus_SmileFA_Tr":
        s_73k = [True, False, False, False, True, False, False]
        # 73k = [14, 15, 16, 17, 18, 19, 20]
        s_74k = [False, True, False, False, True, False, False, True, True, False,
                 True, True, True, True, False]
        # 74k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        s_l9 = [0, 0]
        # s_l9 = [1-5, 6-9]
        charge = [0, 0, 0, 0, 6, 3, 0, 0]
        # ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        ond = [0, 5, 0]
        # ond = [solar_max, abb, kaco]
        con = [0, 0, 1]
        # con = [cinergia, sop, battery]
    elif case_name == "Case_4bus_Test":
        s_73k = [True, False, True, False, False, False, False]
        # 73k = [14, 15, 16, 17, 18, 19, 20]
        s_74k = [True, False, False, True, False, False, False, False, False, True,
                 True, False, False, True, False]
        # 74k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        s_l9 = [0, 0]
        # s_l9 = [1-5, 6-9]
        charge = [1, 4, 5, 8, 0, 0, 0, 0]
        # ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        ond = [0, 0, 0]
        # ond = [SolarMax, ABB, KACO]
        con = [0, 0, 0]
        # con = [Cinergia, SOP, Battery]
    elif case_name == "Case_4bus_DiGriFlex":
        s_73k = [True, False, False, False, True, False, False]
        # 73k = [14, 15, 16, 17, 18, 19, 20]
        s_74k = [False, False, False, False, False, False, False, True, True, False,
                 False, True, True, True, True]
        # 74k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        s_l9 = [0, 0]
        # s_l9 = [1-5, 6-9]
        charge = [3, 0, 0, 0, 0, 0, 0, 0]
        # ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        ond = [0, 9, 0]
        # ond = [solar_max, abb, kaco]
        con = [1]
        # Con = [Battery]
    elif case_name == "Case_LabVIEW":
        switches, _, _, _, _, _, _, _, _, _, _, _, _, _ = interface_meas(vec_inp)
        s_73k = switches["S_73K"]
        # 73k = [14, 15, 16, 17, 18, 19, 20]
        s_74k = switches["S_74K"]
        # 74k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        s_l9: list = switches["S_L9"]
        # s_l9 = [1-5, 6-9]
        charge = switches["Charge"]
        # ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        ond = switches["Ond"]
        # ond = [solar_max, abb, kaco]
        con = switches["Con"]
        # con = [battery]
    # Automatic determination of output dictionary from state of switches
    if case_name != "9BusesPydGridExample" or case_name != "6BusesLaChappelle":
        nodes = list(set().union([1] + charge + ond + con))
        nodes.pop(0)
        for i in range(3):
            for index, item in enumerate(nodes):
                if item == 9 and s_74k[2] == 1 and s_74k[5] == 1:
                    nodes[index] = 5
                elif item == 8 and s_74k[4] == 1 and s_74k[1] == 1:
                    nodes[index] = 4
                elif item == 7 and s_74k[3] == 1 and s_74k[0] == 1:
                    nodes[index] = 3
                elif item == 6 and s_73k[6] == 1:
                    nodes[index] = 2
                elif item == 5 and s_74k[2] == 1 and s_73k[5] == 1:
                    nodes[index] = 1
                elif item == 4 and s_74k[1] == 1 and s_73k[3] == 1:
                    nodes[index] = 1
                elif item == 3 and s_74k[0] == 1 and s_73k[1] == 1:
                    nodes[index] = 1
        nodes = list(set().union(nodes))
        m_big = 1e5
        y_big = 300 / complex(min(r_reine_u) + min(r_reine_v) + min(r_reine_w) + 3 * r_cable,
                              (min(l_reine_u) + min(l_reine_v) + min(l_reine_w)) * l2x + 3 * x_cable)
        y_bus_1_2 = 3 / complex(r_reine_u[0] + r_reine_v[0] + r_reine_w[0] + 3 * r_cable,
                                (l_reine_u[0] + l_reine_v[0] + l_reine_w[0]) * l2x + 3 * x_cable)
        y_bus_1_6 = 3 / complex(r_reine_u[4] + r_reine_v[4] + r_reine_w[4] + 3 * r_cable,
                                (l_reine_u[4] + l_reine_v[4] + l_reine_w[4]) * l2x + 3 * x_cable)
        y_bus_1_10 = y_bus_1_11 = y_bus_1_12 = y_big
        y_bus_2_3 = 3 / complex(r_reine_u[1] + r_reine_v[1] + r_reine_w[1] + 3 * r_cable,
                                (l_reine_u[1] + l_reine_v[1] + l_reine_w[1]) * l2x + 3 * x_cable)
        y_bus_2_6 = y_big
        y_bus_3_4 = 3 / complex(r_reine_u[2] + r_reine_v[2] + r_reine_w[2] + 3 * r_cable,
                                (l_reine_u[2] + l_reine_v[2] + l_reine_w[2]) * l2x + 3 * x_cable)
        y_bus_3_10 = y_big
        y_bus_4_5 = 3 / complex(r_reine_u[3] + r_reine_v[3] + r_reine_w[3] + 3 * r_cable,
                                (l_reine_u[3] + l_reine_v[3] + l_reine_w[3]) * l2x + 3 * x_cable)
        y_bus_4_11 = y_big
        y_bus_5_12 = y_big
        y_bus_6_7 = 3 / complex(r_reine_u[5] + r_reine_v[5] + r_reine_w[5] + 3 * r_cable,
                                (l_reine_u[5] + l_reine_v[5] + l_reine_w[5]) * l2x + 3 * x_cable)
        y_bus_7_8 = 3 / complex(r_reine_u[6] + r_reine_v[6] + r_reine_w[6] + 3 * r_cable,
                                (l_reine_u[6] + l_reine_v[6] + l_reine_w[6]) * l2x + 3 * x_cable)
        y_bus_7_10 = y_big
        y_bus_8_9 = 3 / complex(r_reine_u[7] + r_reine_v[7] + r_reine_w[7] + 3 * r_cable,
                                (l_reine_u[7] + l_reine_v[7] + l_reine_w[7]) * l2x + 3 * x_cable)
        y_bus_8_11 = y_big
        y_bus_9_12 = y_big
        y_bus_l9 = 3 / complex(r_reine_u[8] + r_reine_v[8] + r_reine_w[8] + 3 * r_cable,
                               (l_reine_u[8] + l_reine_v[8] + l_reine_w[8]) * l2x + 3 * x_cable)
        y_bus_1_2 = y_bus_1_2 * s_74k[7]
        y_bus_1_6 = y_bus_1_6 * s_74k[11]
        y_bus_1_10 = y_bus_1_10 * s_73k[2]
        y_bus_1_11 = y_bus_1_11 * s_73k[3]
        y_bus_1_12 = y_bus_1_12 * s_73k[5]
        y_bus_2_3 = y_bus_2_3 * s_74k[8]
        y_bus_2_6 = y_bus_2_6 * s_73k[6]
        y_bus_3_4 = y_bus_3_4 * s_74k[9]
        y_bus_3_10 = y_bus_3_10 * s_74k[0]
        y_bus_4_5 = y_bus_4_5 * s_74k[10]
        y_bus_4_11 = y_bus_4_11 * s_74k[1]
        y_bus_5_12 = y_bus_5_12 * s_74k[2]
        y_bus_6_7 = y_bus_6_7 * s_74k[12]
        y_bus_7_8 = y_bus_7_8 * s_74k[13]
        y_bus_7_10 = y_bus_7_10 * s_74k[3]
        y_bus_8_9 = y_bus_8_9 * s_74k[14]
        y_bus_8_11 = y_bus_8_11 * s_74k[4]
        y_bus_9_12 = y_bus_9_12 * s_74k[5]
        y_bus: list = \
            [[y_bus_1_2 + y_bus_1_6 + y_bus_1_10 + y_bus_1_11 + y_bus_1_12, -y_bus_1_2, 0, 0, 0, -y_bus_1_6,
             0, 0, 0, -y_bus_1_10, -y_bus_1_11, -y_bus_1_12],
             [-y_bus_1_2, y_bus_1_2 + y_bus_2_3 + y_bus_2_6, -y_bus_2_3, 0, 0, -y_bus_2_6,
              0, 0, 0, 0, 0, 0],
             [0, -y_bus_2_3, y_bus_2_3 + y_bus_3_4 + y_bus_3_10, -y_bus_3_4, 0, 0,
              0, 0, 0, -y_bus_3_10, 0, 0],
             [0, 0, -y_bus_3_4, y_bus_3_4 + y_bus_4_5 + y_bus_4_11, -y_bus_4_5, 0,
              0, 0, 0, 0, -y_bus_4_11, 0],
             [0, 0, 0, -y_bus_4_5, y_bus_4_5 + y_bus_5_12, 0,
              0, 0, 0, 0, 0, -y_bus_5_12],
             [-y_bus_1_6, -y_bus_2_6, 0, 0, 0, y_bus_1_6 + y_bus_2_6 + y_bus_6_7,
              -y_bus_6_7, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, -y_bus_6_7,
              y_bus_6_7 + y_bus_7_8 + y_bus_7_10, -y_bus_7_8, 0, -y_bus_7_10, 0, 0],
             [0, 0, 0, 0, 0, 0,
              -y_bus_7_8, y_bus_7_8 + y_bus_8_9 + y_bus_8_11, -y_bus_8_9, 0, -y_bus_8_11, 0],
             [0, 0, 0, 0, 0, 0,
              0, -y_bus_8_9, y_bus_8_9 + y_bus_9_12, 0, 0, -y_bus_9_12],
             [-y_bus_1_10, 0, -y_bus_3_10, 0, 0, 0,
              -y_bus_7_10, 0, 0, (y_bus_1_10 + y_bus_3_10 + y_bus_7_10), 0, 0],
             [-y_bus_1_11, 0, 0, -y_bus_4_11, 0, 0,
              0, -y_bus_8_11, 0, 0, (y_bus_1_11 + y_bus_4_11 + y_bus_8_11), 0],
             [-y_bus_1_12, 0, 0, 0, -y_bus_5_12, 0,
              0, 0, -y_bus_9_12, 0, 0, (y_bus_1_12 + y_bus_5_12 + y_bus_9_12)]]
        if s_l9[0] > 0 and s_l9[1] > 0:
            y_bus[s_l9[0] - 1][s_l9[1] - 1] = y_bus[s_l9[0] - 1][s_l9[1] - 1] - y_bus_l9
            y_bus[s_l9[1] - 1][s_l9[0] - 1] = y_bus[s_l9[1] - 1][s_l9[0] - 1] - y_bus_l9
            y_bus[s_l9[0] - 1][s_l9[0] - 1] = y_bus[s_l9[0] - 1][s_l9[1] - 1] + y_bus_l9
            y_bus[s_l9[1] - 1][s_l9[1] - 1] = y_bus[s_l9[0] - 1][s_l9[1] - 1] + y_bus_l9
        y_bus_arr = np.array(y_bus, dtype=object)
        nodes_dis = np.where(sum(abs(y_bus_arr)) < 1 / m_big)
        all_nodes = list(range(1, len(y_bus) + 1))
        for ele in nodes_dis[0]:
            all_nodes.remove(ele + 1)
        ordered_nodes = list(set().union(nodes + all_nodes))
        ordered_nodes = [x - 1 for x in ordered_nodes]
        y_bus = [[y_bus[i][j] for j in ordered_nodes] for i in ordered_nodes]
        a_loc = []
        for ele in nodes:
            a_loc.append(ordered_nodes.index(ele - 1))
        d_loc = list(set(range(len(ordered_nodes))).difference(set(a_loc)))
        y_bus_arr = np.array(y_bus, dtype=object)
        a, b, c, d = y_bus_arr[np.ix_(a_loc, a_loc)], y_bus_arr[np.ix_(a_loc, d_loc)], y_bus_arr[np.ix_(d_loc, d_loc)],\
            y_bus_arr[np.ix_(d_loc, a_loc)]
        y_bus_red = a - np.matmul(np.matmul(b, np.array(np.linalg.inv(c.tolist()))), d)
        data["buses"] = [{"bus": "0", "pos_x": 0, "pos_y": 0, "units": "m", "U_kV": 20, "Vmin": 0.9, "Vmax": 1.1}]
        data["shunts"] = []
        data["grid_feeders"] = []
        for n in nodes:
            data["buses"]\
                .append({"bus": str(n), "pos_x": 0, "pos_y": 0, "units": "m", "U_kV": 0.4, "Vmin": 0.9, "Vmax": 1.1})
            data["shunts"].append({"bus": str(n), "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]})
            data["grid_feeders"].append({"bus": str(n), "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0})
        data["lines"] = []
        data["line_codes"] = {}
        for i in nodes[:-1]:
            for j in nodes[nodes.index(i) + 1:]:
                if abs(y_bus_red[nodes.index(i)][nodes.index(j)]).real > 1:
                    ii = i
                    jj = j
                    for d in data["lines"]:
                        if str(j) == d["bus_k"]:
                            ii = j
                            jj = i
                    data["lines"].append(
                        {"bus_j": str(ii), "bus_k": str(jj), "code": str(ii) + ',' + str(jj), "m": 1000,  "Cap": cap})
                    data["line_codes"][str(ii) + ',' + str(jj)] = {
                        "R1": -(1 / (y_bus_red[nodes.index(i)][nodes.index(j)])).real,
                        "X1": -(1 / (y_bus_red[nodes.index(i)][nodes.index(j)])).imag,
                        "R0": -(1 / (y_bus_red[nodes.index(i)][nodes.index(j)])).real,
                        "X0": -(1 / (y_bus_red[nodes.index(i)][nodes.index(j)])).imag,
                        "B_1_mu": 0,
                        "B_0_mu": 0}
        data["transformers"] = [{"bus_j": "0", "bus_k": "1", "S_n_kVA": sbase / 1000, "U_1_kV": 20, "U_2_kV": 0.4,
                                 "Cap": cap,
                                 "R_cc_pu": r_tr + r_cable, "X_cc_pu": x_tr + x_cable,
                                 "connection": "Dyn11",
                                 "conductors_1": 3, "conductors_2": 4}]
        data["grid_formers"] = [{"index": 0, "bus": "0", "bus_nodes": [1, 2, 3],
                                 "kV": [20 / np.sqrt(3), 20 / np.sqrt(3), 20 / np.sqrt(3)],
                                 "deg": [30, 150, 270]}]
        data["Nbus"] = np.size(data["buses"], 0)
        data["Zbase"] = zbase
        data["PV_elements"] = []
        j = -1
        for index, p in enumerate(ond, start=0):
            if p != 0:
                j = j + 1
                data["PV_elements"].append({"index": j, "bus": str(p),
                                            "cap_kVA_perPhase": ond_cap[index] / 3, "V_grid_pu": ond_v_grid_pu[index],
                                            "V_conv_pu": ond_v_conv_pu[index], "X_PV_pu": ond_x_pv_pu[index],
                                            "cos_PV": ond_cos_pv[index]})
        data["load_elements"]: list = []
        j = -1
        for ch in charge:
            if ch != 0:
                j = j + 1
                data["load_elements"].append({"index": j, "bus": str(ch)})
        data["storage_elements"] = []
        j = -1
        for index, st in enumerate(con, start=0):
            if st != 0:
                j = j + 1
                data["storage_elements"].append({"index": j, "bus": str(st),
                                                 "SOC_min_kWh": con_soc_min_kwh[index],
                                                 "SOC_max_kWh": con_soc_max_kwh[index],
                                                 "S_max_kVA": con_s_max_kva[index],
                                                 "P_max_kW": con_p_max_kw[index],
                                                 "Eff_LV": con_eff_lv[index],
                                                 "Eff_D": con_eff_d[index], "Eff_C": con_eff_c[index],
                                                 "Eff_LC": con_eff_lc[index]})
    return data


def interface_meas(vec_inp: list):
    """
    @description: This function is for reading measurements from LabVIEW
    @param vec_inp: vector of measurements ordered by Douglas in "07/10/2020 14:06"
    (See "\data\Ordre_informations_controle_2020.10.07.xlsx")
    @return: dictionary of measurements
    - Switches = {"S_73K":[]*8, "S_74_K": []*16, "S_L9": []*1, "Charge": []*14, "Ond": []*16, "Con"@*: []*1}
    @*Con-->Battery
    - Ligne_U = []*9 @* nine nodes
    - Ligne_I = []*9 @* nine nodes
    - Charge_I = []*14 @* 14 loads [[10kW]*8, SOPS_1, SOPS_2, GreenMotion, Cinergia, 25kW, MIRI]
    - Ond_I = []*3 @* 3 PVs [SolarMax, ABB, KACO]
    - Charge_P = []*14 @* 14 loads [[10kW]*8, SOPS_1, SOPS_2, GreenMotion, Cinergia, 25kW, MIRI]
    - Charge_Q = []*14 @* 14 loads [[10kW]*8, SOPS_1, SOPS_2, GreenMotion, Cinergia, 25kW, MIRI]
    - Ond_P = []*3 @* 3 loads [SolarMax, ABB, KACO]
    - Ond_Q = []*3 @* 3 loads [SolarMax, ABB, KACO]
    - Ligne_P = []*9 @* nine lines [SolarMax, ABB, KACO]
    - Ligne_Q = []*9 @* nine lines [SolarMax, ABB, KACO]
    - SOC = []*1 @* nine battery
    """
    switches = {}
    switches["S_73K"] = [float(i) for i in vec_inp[385:392]]
    switches["S_74K"] = [float(i) for i in vec_inp[392:407]]
    switches["S_L9"] = [int(vec_inp[411]) * 1 + int(vec_inp[412]) * 2 + int(vec_inp[413]) * 3 +
                        int(vec_inp[414]) * 4 + int(vec_inp[415]) * 5,
                        int(vec_inp[407]) * 6 + int(vec_inp[408]) * 7 + int(vec_inp[409]) * 8 + int(vec_inp[410]) * 9]
    switches["Charge"] = [0] * (11 + 3)
    switches["Ond"] = [0] * 3
    switches["Con"] = [0] * 1
    for index, item in enumerate(vec_inp[416:]):
        for ch in range(1, 12):
            if item == ("Charge " + str(ch)):
                switches["Charge"][ch - 1] = int(vec_inp[416 + index + 1])
        for ch in range(12, 15):
            if item == ("Charge " + str(ch)):
                switches["Ond"][ch - 12] = int(vec_inp[416 + index + 1])
        for ch in range(15, 18):
            if item == ("Charge " + str(ch)):
                switches["Charge"][ch - 4] = int(vec_inp[416 + index + 1])
        if "Batterie" in item:
            switches["Con"][0] = int(vec_inp[416 + index + 1])
    ligne_u: list = [0] * 9
    ligne_u[0] = np.sqrt(3) * ((float(vec_inp[3]) + float(vec_inp[4]) + float(vec_inp[5])) / 3 - float(vec_inp[6]))
    ligne_u[1] = np.sqrt(3) * ((float(vec_inp[7]) + float(vec_inp[8]) + float(vec_inp[9])) / 3 - float(vec_inp[10]))
    ligne_u[2] = np.sqrt(3) * ((float(vec_inp[11]) + float(vec_inp[12]) + float(vec_inp[13])) / 3 - float(vec_inp[14]))
    ligne_u[3] = np.sqrt(3) * ((float(vec_inp[15]) + float(vec_inp[16]) + float(vec_inp[17])) / 3 - float(vec_inp[18]))
    ligne_u[4] = np.sqrt(3) * ((float(vec_inp[79]) + float(vec_inp[80]) + float(vec_inp[81])) / 3 - float(vec_inp[82]))
    ligne_u[5] = np.sqrt(3) * ((float(vec_inp[83]) + float(vec_inp[84]) + float(vec_inp[85])) / 3 - float(vec_inp[86]))
    ligne_u[6] = np.sqrt(3) * ((float(vec_inp[87]) + float(vec_inp[88]) + float(vec_inp[89])) / 3 - float(vec_inp[90]))
    ligne_u[7] = np.sqrt(3) * ((float(vec_inp[91]) + float(vec_inp[92]) + float(vec_inp[93])) / 3 - float(vec_inp[94]))
    ligne_u[8] = \
        np.sqrt(3) * ((float(vec_inp[145]) + float(vec_inp[146]) + float(vec_inp[147])) / 3 - float(vec_inp[148]))
    ligne_i: list = [0] * 9
    ligne_i[0] = (float(vec_inp[41]) + float(vec_inp[42]) + float(vec_inp[43])) - float(vec_inp[44])
    ligne_i[1] = (float(vec_inp[45]) + float(vec_inp[46]) + float(vec_inp[47])) - float(vec_inp[48])
    ligne_i[2] = (float(vec_inp[49]) + float(vec_inp[50]) + float(vec_inp[51])) - float(vec_inp[52])
    ligne_i[3] = (float(vec_inp[53]) + float(vec_inp[54]) + float(vec_inp[55])) - float(vec_inp[56])
    ligne_i[4] = (float(vec_inp[133]) + float(vec_inp[134]) + float(vec_inp[135])) - float(vec_inp[136])
    ligne_i[5] = (float(vec_inp[117]) + float(vec_inp[118]) + float(vec_inp[119])) - float(vec_inp[120])
    ligne_i[6] = (float(vec_inp[121]) + float(vec_inp[122]) + float(vec_inp[123])) - float(vec_inp[124])
    ligne_i[7] = (float(vec_inp[125]) + float(vec_inp[126]) + float(vec_inp[127])) - float(vec_inp[128])
    ligne_i[8] = (float(vec_inp[170]) + float(vec_inp[171]) + float(vec_inp[172])) - float(vec_inp[173])
    charge_i: list = [0] * (11 + 3)
    ond_i: list = [0] * 3
    charge_i[0] = (float(vec_inp[149]) + float(vec_inp[150]) + float(vec_inp[151])) - float(vec_inp[152])
    charge_i[1] = (float(vec_inp[153]) + float(vec_inp[154]) + float(vec_inp[155])) - float(vec_inp[156])
    charge_i[2] = (float(vec_inp[157]) + float(vec_inp[158]) + float(vec_inp[159])) - float(vec_inp[160])
    charge_i[3] = (float(vec_inp[72]) + float(vec_inp[73]) + float(vec_inp[74])) - float(vec_inp[75])
    charge_i[4] = (float(vec_inp[129]) + float(vec_inp[130]) + float(vec_inp[131])) - float(vec_inp[132])
    charge_i[5] = (float(vec_inp[59]) + float(vec_inp[60]) + float(vec_inp[61])) - float(vec_inp[62])
    charge_i[6] = (float(vec_inp[21]) + float(vec_inp[22]) + float(vec_inp[23])) - float(vec_inp[24])
    charge_i[7] = (float(vec_inp[25]) + float(vec_inp[26]) + float(vec_inp[27])) - float(vec_inp[28])
    charge_i[8] = (float(vec_inp[29]) + float(vec_inp[30]) + float(vec_inp[31])) - float(vec_inp[32])
    charge_i[9] = (float(vec_inp[33]) + float(vec_inp[34]) + float(vec_inp[35])) - float(vec_inp[36])
    charge_i[10] = (float(vec_inp[63]) + float(vec_inp[64]) + float(vec_inp[65])) - float(vec_inp[66])
    ond_i[0] = (float(vec_inp[96]) + float(vec_inp[97]) + float(vec_inp[98])) - float(vec_inp[99])
    ond_i[1] = (float(vec_inp[100]) + float(vec_inp[101]) + float(vec_inp[102])) - float(vec_inp[103])
    ond_i[2] = (float(vec_inp[104]) + float(vec_inp[105]) + float(vec_inp[106])) - float(vec_inp[107])
    charge_i[11] = (float(vec_inp[108]) + float(vec_inp[109]) + float(vec_inp[110])) - float(vec_inp[111])
    charge_i[12] = (float(vec_inp[112]) + float(vec_inp[113]) + float(vec_inp[114])) - float(vec_inp[115])
    charge_i[13] = (float(vec_inp[67]) + float(vec_inp[68]) + float(vec_inp[69])) - float(vec_inp[70])
    charge_p: list = [0] * (11 + 3)
    ond_p: list = [0] * 3
    charge_p[0] = (float(vec_inp[191]) + float(vec_inp[192]) + float(vec_inp[193])) / 1000
    charge_p[1] = (float(vec_inp[194]) + float(vec_inp[195]) + float(vec_inp[196])) / 1000
    charge_p[2] = (float(vec_inp[197]) + float(vec_inp[198]) + float(vec_inp[199])) / 1000
    charge_p[3] = (float(vec_inp[200]) + float(vec_inp[201]) + float(vec_inp[202])) / 1000
    charge_p[4] = (float(vec_inp[203]) + float(vec_inp[204]) + float(vec_inp[205])) / 1000
    charge_p[5] = (float(vec_inp[206]) + float(vec_inp[207]) + float(vec_inp[208])) / 1000
    charge_p[6] = (float(vec_inp[209]) + float(vec_inp[210]) + float(vec_inp[211])) / 1000
    charge_p[7] = (float(vec_inp[212]) + float(vec_inp[213]) + float(vec_inp[214])) / 1000
    charge_p[8] = (float(vec_inp[215]) + float(vec_inp[216]) + float(vec_inp[217])) / 1000
    charge_p[9] = (float(vec_inp[218]) + float(vec_inp[219]) + float(vec_inp[220])) / 1000
    charge_p[10] = (float(vec_inp[221]) + float(vec_inp[222]) + float(vec_inp[223])) / 1000
    ond_p[0] = (float(vec_inp[224]) + float(vec_inp[225]) + float(vec_inp[226])) / 1000
    ond_p[1] = (float(vec_inp[227]) + float(vec_inp[228]) + float(vec_inp[229])) / 1000
    ond_p[2] = (float(vec_inp[230]) + float(vec_inp[231]) + float(vec_inp[232])) / 1000
    charge_p[11] = (float(vec_inp[233]) + float(vec_inp[234]) + float(vec_inp[235])) / 1000
    charge_p[12] = (float(vec_inp[236]) + float(vec_inp[237]) + float(vec_inp[238])) / 1000
    charge_p[13] = (float(vec_inp[239]) + float(vec_inp[240]) + float(vec_inp[241])) / 1000
    charge_q: list = [0] * (11 + 3)
    ond_q: list = [0] * 3
    charge_q[0] = (float(vec_inp[242]) + float(vec_inp[243]) + float(vec_inp[244])) / 1000
    charge_q[1] = (float(vec_inp[245]) + float(vec_inp[246]) + float(vec_inp[247])) / 1000
    charge_q[2] = (float(vec_inp[248]) + float(vec_inp[249]) + float(vec_inp[250])) / 1000
    charge_q[3] = (float(vec_inp[251]) + float(vec_inp[252]) + float(vec_inp[253])) / 1000
    charge_q[4] = (float(vec_inp[254]) + float(vec_inp[255]) + float(vec_inp[256])) / 1000
    charge_q[5] = (float(vec_inp[257]) + float(vec_inp[258]) + float(vec_inp[259])) / 1000
    charge_q[6] = (float(vec_inp[260]) + float(vec_inp[261]) + float(vec_inp[262])) / 1000
    charge_q[7] = (float(vec_inp[263]) + float(vec_inp[264]) + float(vec_inp[265])) / 1000
    charge_q[8] = (float(vec_inp[266]) + float(vec_inp[267]) + float(vec_inp[268])) / 1000
    charge_q[9] = (float(vec_inp[269]) + float(vec_inp[270]) + float(vec_inp[271])) / 1000
    charge_q[10] = (float(vec_inp[272]) + float(vec_inp[273]) + float(vec_inp[274])) / 1000
    ond_q[0] = (float(vec_inp[275]) + float(vec_inp[276]) + float(vec_inp[277])) / 1000
    ond_q[1] = (float(vec_inp[278]) + float(vec_inp[279]) + float(vec_inp[280])) / 1000
    ond_q[2] = (float(vec_inp[281]) + float(vec_inp[282]) + float(vec_inp[283])) / 1000
    charge_q[11] = (float(vec_inp[284]) + float(vec_inp[285]) + float(vec_inp[286])) / 1000
    charge_q[12] = (float(vec_inp[287]) + float(vec_inp[288]) + float(vec_inp[289])) / 1000
    charge_q[13] = (float(vec_inp[290]) + float(vec_inp[291]) + float(vec_inp[292])) / 1000
    ligne_p: list = [0] * 9
    ligne_p[0] = (float(vec_inp[293]) + float(vec_inp[294]) + float(vec_inp[295])) / 1000
    ligne_p[1] = (float(vec_inp[296]) + float(vec_inp[297]) + float(vec_inp[298])) / 1000
    ligne_p[2] = (float(vec_inp[299]) + float(vec_inp[300]) + float(vec_inp[301])) / 1000
    ligne_p[3] = (float(vec_inp[302]) + float(vec_inp[303]) + float(vec_inp[304])) / 1000
    ligne_p[4] = (float(vec_inp[305]) + float(vec_inp[306]) + float(vec_inp[307])) / 1000
    ligne_p[5] = (float(vec_inp[308]) + float(vec_inp[309]) + float(vec_inp[310])) / 1000
    ligne_p[6] = (float(vec_inp[311]) + float(vec_inp[312]) + float(vec_inp[313])) / 1000
    ligne_p[7] = (float(vec_inp[314]) + float(vec_inp[315]) + float(vec_inp[316])) / 1000
    ligne_p[8] = (float(vec_inp[317]) + float(vec_inp[318]) + float(vec_inp[319])) / 1000
    ligne_q: list = [0] * 9
    ligne_q[0] = (float(vec_inp[320]) + float(vec_inp[321]) + float(vec_inp[322])) / 1000
    ligne_q[1] = (float(vec_inp[323]) + float(vec_inp[324]) + float(vec_inp[325])) / 1000
    ligne_q[2] = (float(vec_inp[326]) + float(vec_inp[327]) + float(vec_inp[328])) / 1000
    ligne_q[3] = (float(vec_inp[329]) + float(vec_inp[330]) + float(vec_inp[331])) / 1000
    ligne_q[4] = (float(vec_inp[332]) + float(vec_inp[333]) + float(vec_inp[334])) / 1000
    ligne_q[5] = (float(vec_inp[335]) + float(vec_inp[336]) + float(vec_inp[337])) / 1000
    ligne_q[6] = (float(vec_inp[338]) + float(vec_inp[339]) + float(vec_inp[340])) / 1000
    ligne_q[7] = (float(vec_inp[341]) + float(vec_inp[342]) + float(vec_inp[343])) / 1000
    ligne_q[8] = (float(vec_inp[344]) + float(vec_inp[345]) + float(vec_inp[346])) / 1000
    f_p = (float(vec_inp[347]) + float(vec_inp[348]) + float(vec_inp[349])) / 1000
    f_q = (float(vec_inp[350]) + float(vec_inp[351]) + float(vec_inp[352])) / 1000
    soc = float(vec_inp[383])
    return switches, ligne_u, ligne_i, charge_i, ond_i, charge_p, charge_q, ond_p, ond_q, ligne_p, ligne_q, soc, f_p, \
        f_q


def figuring(grid_inp: plt.plot, meas_inp: dict, meas: dict, fig_type: str, title: str):
    """
    @description: This function is used to plot the results of the simulation.
    @param grid_inp: grid input file
    @param meas_inp: measurement input file
    @param meas: measurement file
    @param fig_type: type of figure to plot
    @param title: title of the figure
    @return: figure
    """
    w, h = 10, 4
    plt.style.use('bmh')
    plt.set_loglevel('WARNING')
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    times = pd.date_range('01-01-2018', periods=meas_inp["Nt"], freq='10MIN')
    x_lab = times.strftime('%H:%M').to_list()
    times = pd.date_range('01-01-2018 02:00', periods=6, freq='4H')
    x_tik = times.strftime('%H:%M').to_list()
    if fig_type == "Power":
        fig_n = 1
        figure = {"fig_" + str(fig_n): plt.figure()}
        figure["fig_" + str(fig_n)].set_figheight(h * 2)
        figure["fig_" + str(fig_n)].set_figwidth(w)
        figure["ax_" + str(fig_n) + "_1"] = figure["fig_" + str(fig_n)].add_subplot(2, 1, 1)
        figure["ax_" + str(fig_n) + "_2"] = figure["fig_" + str(fig_n)].add_subplot(2, 1, 2)
        figure["ax_" + str(fig_n) + "_1"].set_title(title, fontsize=18)
        figure["ax_" + str(fig_n) + "_1"].plot(x_lab, - meas["P_in"][0:(meas_inp["Nt"])] / 1000,
                                               label='with control', linewidth=2, linestyle='-')
        figure["ax_" + str(fig_n) + "_1"].set_ylabel("$P_{CP}$ (kW)", fontsize=18)
        figure["ax_" + str(fig_n) + "_2"].plot(x_lab, - meas["Q_in"][0:(meas_inp["Nt"])] / 1000,
                                               label='with control', linewidth=2, linestyle='-')
        figure["ax_" + str(fig_n) + "_2"].set_xlabel("time", fontsize=18)
        figure["ax_" + str(fig_n) + "_2"].set_ylabel("$Q_{CP}$ (kVar)", fontsize=18)
        if meas_inp["meas"] is True:
            figure["ax_" + str(fig_n) + "_1"].plot(x_lab, np.abs(meas_inp["P_CP"][0:(meas_inp["Nt"])]),
                                                   label='without control', linewidth=2, linestyle='--')
            figure["ax_" + str(fig_n) + "_2"].plot(x_lab, np.abs(meas_inp["Q_CP"][0:(meas_inp["Nt"])]),
                                                   label='without control', linewidth=2, linestyle='--')
        figure["ax_" + str(fig_n) + "_1"].legend(fontsize=18)
        figure["ax_" + str(fig_n) + "_2"].legend(fontsize=18)
        plt.sca(figure["ax_" + str(fig_n) + "_1"])
        plt.xticks(x_tik, x_tik)
        plt.sca(figure["ax_" + str(fig_n) + "_2"])
        plt.xticks(x_tik, x_tik)
    elif fig_type == "Voltages":
        fig_n = 2
        figure = {"fig_" + str(fig_n): plt.figure()}
        figure["fig_" + str(fig_n)].set_figheight(h * np.size(grid_inp["buses"], 0))
        figure["fig_" + str(fig_n)].set_figwidth(w)
        for nn in range((np.size(grid_inp["buses"], 0))):
            figure["ax_" + str(fig_n) + "_" + str(nn + 1)] = figure["fig_" + str(fig_n)].add_subplot(
                np.size(grid_inp["buses"], 0), 1, nn + 1)
            figure["ax_" + str(fig_n) + "_" + str(nn + 1)].plot(x_lab,
                                                                (meas["V"][nn][0][0:(meas_inp["Nt"])]
                                                                 + meas["V"][nn][1][0:(meas_inp["Nt"])]
                                                                 + meas["V"][nn][2][0:(meas_inp["Nt"])]) / 3,
                                                                label='with control', linewidth=2, linestyle='-')
            figure["ax_" + str(fig_n) + "_" + str(nn + 1)].set_ylabel("$V_{}$".format(nn) + "(V)", fontsize=18)
            plt.sca(figure["ax_" + str(fig_n) + "_" + str(nn + 1)])
            plt.xticks(x_tik, x_tik)
        figure["ax_" + str(fig_n) + "_" + str(1)].set_title(title, fontsize=18)
        figure["ax_" + str(fig_n) + "_" + str(1)].set_xlabel("time", fontsize=18)
        if meas_inp["meas"] is True:
            for n in range(meas_inp["meas_number"]):
                nn = find_n(meas_inp["meas_location"][n]["to"], grid_inp["buses"])
                figure["ax_" + str(fig_n) + "_" + str(nn + 1)].plot(x_lab,
                                                                    (meas_inp["V_CP"][n][0][0:(meas_inp["Nt"])]
                                                                     + meas_inp["V_CP"][n][1][0:(meas_inp["Nt"])]
                                                                     + meas_inp["V_CP"][n][2][0:(meas_inp["Nt"])]) / 3,
                                                                    label='without control', linewidth=2,
                                                                    linestyle='--')
                figure["ax_" + str(fig_n) + "_" + str(nn + 1)].legend(fontsize=18)
    elif fig_type == "Currents":
        fig_n = 3
        figure = {"fig_" + str(fig_n): plt.figure()}
        figure["fig_" + str(fig_n)].set_figheight(h * np.size(grid_inp["lines"], 0))
        figure["fig_" + str(fig_n)].set_figwidth(w)
        for n in range(np.size(grid_inp["lines"], 0)):
            figure["ax_" + str(fig_n) + "_" + str(n + 1)] = figure["fig_" + str(fig_n)].add_subplot(
                np.size(grid_inp["lines"], 0), 1, n + 1)
            figure["ax_" + str(fig_n) + "_" + str(n + 1)].plot(x_lab,
                                                               meas["I"][n][0][0:(meas_inp["Nt"])]
                                                               + meas["I"][n][1][0:(meas_inp["Nt"])]
                                                               + meas["I"][n][2][0:(meas_inp["Nt"])],
                                                               label='with control', linewidth=2, linestyle='-')
            figure["ax_" + str(fig_n) + "_" + str(n + 1)].set_ylabel("$I_{}$".format(n + 1) + "(A)", fontsize=18)
            plt.sca(figure["ax_" + str(fig_n) + "_" + str(n + 1)])
            plt.xticks(x_tik, x_tik)
        figure["ax_" + str(fig_n) + "_" + str(1)].set_title(title, fontsize=18)
        figure["ax_" + str(fig_n) + "_" + str(1)].set_xlabel("time", fontsize=18)
        if meas_inp["meas"] is True:
            for n in range(meas_inp["meas_number"]):
                if meas_inp["meas_location"][n]["tran/line"] == "line":
                    nn = find_n(meas_inp["meas_location"][n]["to"], grid_inp["lines"])
                    figure["ax_" + str(fig_n) + "_" + str(nn + 1)].plot(x_lab,
                                                                        np.abs(
                                                                            meas_inp["I_CP"][n][0][0:(meas_inp["Nt"])]
                                                                            + meas_inp["I_CP"][n][1][0:(meas_inp["Nt"])]
                                                                            + meas_inp["I_CP"][n][2][
                                                                              0:(meas_inp["Nt"])]),
                                                                        label='without control', linewidth=2,
                                                                        linestyle='--')
                    figure["ax_" + str(fig_n) + "_" + str(nn + 1)].legend(fontsize=18)
    elif fig_type == "DA_Offers":
        fig_n = 4
        figure = {"fig_" + str(fig_n): plt.figure()}
        figure["fig_" + str(fig_n)].set_figheight(h * 2)
        figure["fig_" + str(fig_n)].set_figwidth(w)
        figure["ax_" + str(fig_n) + "_1"] = figure["fig_" + str(fig_n)].add_subplot(2, 1, 1)
        figure["ax_" + str(fig_n) + "_2"] = figure["fig_" + str(fig_n)].add_subplot(2, 1, 2)
        # Figure["ax_" + str(fig_n) + "_1"].set_title(title, fontsize=18)
        figure["ax_" + str(fig_n) + "_1"].plot(x_lab, meas["DA_PP"][0:(meas_inp["Nt"])],
                                               label='$p_{t}^{(DA)}$', linewidth=2, linestyle='-')
        figure["ax_" + str(fig_n) + "_1"].set_ylabel("Active power (kW)", fontsize=18)
        figure["ax_" + str(fig_n) + "_2"].plot(x_lab, meas["DA_QQ"][0:(meas_inp["Nt"])],
                                               label='$q_{t}^{(DA)}$', linewidth=2, linestyle='-')
        figure["ax_" + str(fig_n) + "_2"].set_xlabel("time", fontsize=18)
        figure["ax_" + str(fig_n) + "_2"].set_ylabel("Reactive power (kVAR)", fontsize=18)
        figure["ax_" + str(fig_n) + "_1"].plot(x_lab, meas["DA_P+"][0:(meas_inp["Nt"])],
                                               label='$p_{t}^{(DA)}+r_t^{(p\u2191)}$', linewidth=2, linestyle='--')
        figure["ax_" + str(fig_n) + "_2"].plot(x_lab, meas["DA_Q+"][0:(meas_inp["Nt"])],
                                               label='$q_{t}^{(DA)}+r_t^{(q\u2191)}$', linewidth=2, linestyle='--')
        figure["ax_" + str(fig_n) + "_1"].plot(x_lab, meas["DA_P-"][0:(meas_inp["Nt"])],
                                               label='$p_{t}^{(DA)}-r_t^{(p\u2193)}$', linewidth=2, linestyle=':')
        figure["ax_" + str(fig_n) + "_2"].plot(x_lab, meas["DA_Q-"][0:(meas_inp["Nt"])],
                                               label='$q_{t}^{(DA)}-r_t^{(q\u2193)}$', linewidth=2, linestyle=':')
        figure["ax_" + str(fig_n) + "_1"].legend(fontsize=18, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.29))
        figure["ax_" + str(fig_n) + "_2"].legend(fontsize=18, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.29))
        # Figure["ax_" + str(fig_n) + "_2"].legend().set_visible(False)
        plt.sca(figure["ax_" + str(fig_n) + "_1"])
        plt.xticks(x_tik, x_tik)
        plt.sca(figure["ax_" + str(fig_n) + "_2"])
        plt.xticks(x_tik, x_tik)
        plt.subplots_adjust(hspace=0.4)
    if True:
        plt.savefig(os.path.join('.cache/figures',
                                 title.replace('\\', '').replace('$', '').replace('.', '') + '.pdf'),
                    bbox_inches='tight')


def forecast_defining(pv_hist: np.array, dem_p_hist: np.array, dem_q_hist: np.array):
    """
    @description: This function is used to make the forecast inputs
    @param pv_hist: historical PV power
    @param dem_p_hist: historical demand power
    @param dem_q_hist: historical demand reactive power
    @return fore_inp: dictionary gives the forecast values of the PV and demands
        fore_inp = {"P_PV" = np.abs(pv_hist[0] + pv_hist[1] + pv_hist[2]) / (3 * 200),  ## in [per unit]
                    "P_PV_zeta+" = 0.2 * fore_inp["P_PV"],
                    "P_PV_zeta-" = 0.2 * fore_inp["P_PV"],
                    "ST_SOC_0" = 0.5,
                    "ST_SOC_zeta+" = 0.2,
                    "ST_SOC_zeta-" = 0.2,
                    "Vmag_zeta+" = 0.02,
                    "Vmag_zeta-" = 0.02,
                    "confidence" = 0.99,
                    "Dem_P" = [-(dem_P_hist[n][0] + dem_P_hist[n][1] + dem_P_hist[n][2]) / (3)],  ## in [kW]
                    "Dem_Q" = [(dem_Q_hist[n][0] + dem_Q_hist[n][1] + dem_Q_hist[n][2]) / (3)], ## in [kW]
                    "Dem_P_zeta+" = [0 * fore_inp["Dem_P"][n]],
                    "Dem_P_zeta-" = [0 * fore_inp["Dem_P"][n]],
                    "Dem_Q_zeta+" = [0 * fore_inp["Dem_Q"][n]],
                    "Dem_Q_zeta-" = [0 * fore_inp["Dem_Q"][n]],
        }
    """
    fore_inp = {}
    fore_inp["P_PV"] = np.abs(pv_hist[0] + pv_hist[1] + pv_hist[2]) / (3 * 200)  # in [per unit]
    fore_inp["P_PV_zeta+"] = 0.2 * fore_inp["P_PV"]
    fore_inp["P_PV_zeta-"] = 0.2 * fore_inp["P_PV"]
    fore_inp["ST_SOC_0"] = 0.5
    fore_inp["ST_SOC_zeta+"] = 0.2
    fore_inp["ST_SOC_zeta-"] = 0.2
    fore_inp["Vmag_zeta+"] = 0.02
    fore_inp["Vmag_zeta-"] = 0.02
    fore_inp["confidence"] = 0.99
    fore_inp["Dem_P"] = {}
    fore_inp["Dem_Q"] = {}
    fore_inp["Dem_P_zeta+"] = {}
    fore_inp["Dem_P_zeta-"] = {}
    fore_inp["Dem_Q_zeta+"] = {}
    fore_inp["Dem_Q_zeta-"] = {}
    for n in range(dem_p_hist.shape[0]):
        fore_inp["Dem_P"][n] = -(dem_p_hist[n][0] + dem_p_hist[n][1] + dem_p_hist[n][2]) / 3  # in [kW]
        fore_inp["Dem_Q"][n] = (dem_q_hist[n][0] + dem_q_hist[n][1] + dem_q_hist[n][2]) / 3  # in [kW]
        fore_inp["Dem_P_zeta+"][n] = 0 * fore_inp["Dem_P"][n]
        fore_inp["Dem_P_zeta-"][n] = 0 * fore_inp["Dem_P"][n]
        fore_inp["Dem_Q_zeta+"][n] = 0 * fore_inp["Dem_Q"][n]
        fore_inp["Dem_Q_zeta-"][n] = 0 * fore_inp["Dem_Q"][n]
    return fore_inp


def rt_simulation(grid_inp: dict, meas_inp: dict, fore_inp: dict, da_result: dict, t: int):
    """
    @description: This function is used to simulate the real-time system
    @param grid_inp: the input of the grid
    @param meas_inp: the input of the measurements
    @param fore_inp: the input of the forecast
    @param da_result: the result of the DA
    @param t: the time step
    @return: the output of the real-time system
    """
    _ = meas_inp
    rt_meas_inp = {}
    rt_meas_inp["delta"] = da_result["delta"]
    rt_meas_inp["Loss_Coeff"] = 0
    rt_meas_inp["ST_Coeff"] = 1
    rt_meas_inp["PV_Coeff"] = 100
    rt_meas_inp["dev_Coeff"] = 1000
    b = norm.ppf(0.999)
    a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
    rt_meas_inp["P_PV"] = fore_inp["P_PV"][t] + fore_inp["P_PV_zeta+"][t] / norm.ppf(fore_inp["confidence"]) \
        - (fore_inp["P_PV_zeta+"][t] + fore_inp["P_PV_zeta-"][t]) * a
    rt_meas_inp["DM_P"] = np.zeros(len(fore_inp["Dem_P"]))
    rt_meas_inp["DM_Q"] = np.zeros(len(fore_inp["Dem_Q"]))
    for d in range(len(fore_inp["Dem_P"])):
        a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
        rt_meas_inp["DM_P"][d] = fore_inp["Dem_P"][d][t] + fore_inp["Dem_P_zeta+"][d][t] / norm.ppf(
            fore_inp["confidence"]) - (fore_inp["Dem_P_zeta+"][d][t] + fore_inp["Dem_P_zeta-"][d][t]) * a
        a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
        rt_meas_inp["DM_Q"][d] = fore_inp["Dem_Q"][d][t] + fore_inp["Dem_Q_zeta+"][d][t] / norm.ppf(
            fore_inp["confidence"]) - (fore_inp["Dem_Q_zeta+"][d][t] + fore_inp["Dem_Q_zeta-"][d][t]) * a
    rt_meas_inp["ST_SOC_des"] = np.zeros(np.size(grid_inp["storage_elements"], 0))
    rt_meas_inp["ST_SOC_t_1"] = np.zeros(np.size(grid_inp["storage_elements"], 0))
    for s in range(np.size(grid_inp["storage_elements"], 0)):
        rt_meas_inp["ST_SOC_des"][s] = da_result["Solution_ST_SOC"][s][t]
        if t == 0:
            a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
            rt_meas_inp["ST_SOC_t_1"][s] = \
                (fore_inp["ST_SOC_0"] + fore_inp["ST_SOC_zeta+"] / norm.ppf(fore_inp["confidence"])
                 * (fore_inp["ST_SOC_zeta+"] + fore_inp["ST_SOC_zeta-"]) * a) \
                * grid_inp["storage_elements"][s]["SOC_max_kWh"]
        else:
            rt_meas_inp["ST_SOC_t_1"][s] = da_result["Solution_ST_SOC_RT"][s]
    rt_meas_inp["Vmag"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["fac_P_pos"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["fac_P_neg"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["fac_Q_pos"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["fac_Q_neg"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["ConPoint_P_DA_EN"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["ConPoint_Q_DA_EN"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["ConPoint_P_DA_RS_pos"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["ConPoint_P_DA_RS_neg"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["ConPoint_Q_DA_RS_pos"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    rt_meas_inp["ConPoint_Q_DA_RS_neg"] = np.zeros(np.size(grid_inp["grid_formers"], 0))
    for f in range(np.size(grid_inp["grid_formers"], 0)):
        a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
        rt_meas_inp["Vmag"][f] = 1 + fore_inp["Vmag_zeta+"] / norm.ppf(fore_inp["confidence"]) \
            - (fore_inp["Vmag_zeta+"] + fore_inp["Vmag_zeta-"]) * a
        a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
        rt_meas_inp["fac_P_pos"][f] = max([0, a])
        rt_meas_inp["fac_P_neg"][f] = - min([0, a])
        a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
        rt_meas_inp["fac_Q_pos"][f] = max([0, a])
        rt_meas_inp["fac_Q_neg"][f] = - min([0, a])
        rt_meas_inp["ConPoint_P_DA_EN"][f] = da_result["DA_P"][t]
        rt_meas_inp["ConPoint_Q_DA_EN"][f] = da_result["DA_Q"][t]
        rt_meas_inp["ConPoint_P_DA_RS_pos"][f] = da_result["DA_RP_pos"][t]
        rt_meas_inp["ConPoint_P_DA_RS_neg"][f] = da_result["DA_RP_neg"][t]
        rt_meas_inp["ConPoint_Q_DA_RS_pos"][f] = da_result["DA_RQ_pos"][t]
        rt_meas_inp["ConPoint_Q_DA_RS_neg"][f] = da_result["DA_RQ_neg"][t]
    return rt_meas_inp


def cases_load_sim(cases_name: str, scenario_num: int):
    """
    @description: Loads the simulation results of the case
    @param cases_name: Name of the case
    @param scenario_num: Number of the scenario
    @return: Dictionary with the simulation results
    """
    data = {}
    l_case = 0
    for case in cases_name:
        data[l_case] = {}
        data[l_case]['MonteCarlo_Scen'] = scenario_num
        if case == 'Stochastic, $K=3$':
            data[l_case]['Omega_Number'] = 3
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = False
        elif case == 'Stochastic, $K=100$':
            data[l_case]['Omega_Number'] = 20
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = False
        elif case == 'Stochastic, $K=10$':
            data[l_case]['Omega_Number'] = 100
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = False
        elif case == 'Robust, $1-\epsilon=1.00$':
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = True
            data[l_case]['Robust_prob'] = 0
        elif case == 'Robust, $1-\epsilon=0.97$':
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = True
            data[l_case]['Robust_prob'] = 0.03
        elif case == 'Robust, $1-\epsilon=0.95$':
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = True
            data[l_case]['Robust_prob'] = 0.05
        elif case == 'Robust, $1-\epsilon=0.90$':
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = True
            data[l_case]['Robust_prob'] = 0.1
        elif case == 'Robust, $1-\epsilon=0.85$':
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = True
            data[l_case]['Robust_prob'] = 0.15
        elif case == 'Robust, $1-\epsilon=0.75$':
            data[l_case]['LAMBDA_P_DA_EN'] = 100
            data[l_case]['LAMBDA_Q_DA_EN'] = 0
            data[l_case]['LAMBDA_P_DA_RS_pos'] = 10
            data[l_case]['LAMBDA_P_DA_RS_neg'] = 10
            data[l_case]['LAMBDA_Q_DA_RS_pos'] = 0
            data[l_case]['LAMBDA_Q_DA_RS_neg'] = 0
            data[l_case]['loss_consideration'] = 0
            data[l_case]['Robust_Programming'] = True
            data[l_case]['Robust_prob'] = 0.25
        l_case = l_case + 1
    return data


def find_n(bus_name: int, buses: list):
    """
    @description: Finds the index of the bus in the bus list
    @param bus_name: Name of the bus
    @param buses: List of buses
    @return: Index of the bus in the bus list
    """
    nn = 0
    for nn in range(np.size(buses, 0)):
        if "bus" in buses[nn] and bus_name == buses[nn]["bus"]:
            break
        elif "bus_j" in buses[nn] and (bus_name == buses[nn]["bus_j"] or bus_name == buses[nn]["bus_k"]):
            break
        if "from" in buses[nn] and (bus_name == buses[nn]["from"] or bus_name == buses[nn]["to"]):
            break
        elif bus_name == buses[nn]:
            break
        else:
            nn += 1
    return nn


def find_nearest(array: np.array, value: float):
    """
    @description: Finds the index of the nearest value in the array
    @param array: Array of values
    @param value: Value to find the nearest
    @return: Index of the nearest value in the array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
