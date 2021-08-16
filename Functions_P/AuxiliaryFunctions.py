"""@author: MYI, #Python version: 3.6.8 [32 bit]"""


#### Importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os


def reine_parameters():
    """" Not Completed: CM#0
    This function is for loading the constant parameters of REINE
    Output: dictionary of parameters
    parameters = {R_ReIne_N, R_ReIne_U, R_ReIne_V, R_ReIne_W, L_ReIne_N, L_ReIne_U, L_ReIne_V,
                  L_ReIne_W, fn, Vbase, Sbase, Zbase, Zsc_tr, Xi_tr, X_Tr, R_Tr, L_cable, L2X,
                  R_cable, X_cable, Cap, Ond_Cap, Ond_V_grid_pu, Ond_V_conv_pu, Ond_X_PV_pu,
                  Con_SOC_min_kWh, Con_SOC_max_kWh, Con_S_max_kVA, Con_P_max_kW, Con_Eff_LV,
                  Con_Eff_D, Con_Eff_C, Con_Eff_LC}
    """
    parameters = {}
    ## Common data of ReIne
    # IndRes = sio.loadmat(dir_name + r'\Data\' + lines_data_name)
    # PARA['R_ReIne_N'] = IndRes['R_ReIne_N']
    # PARA['R_ReIne_U'] = IndRes['R_ReIne_U']
    # PARA['R_ReIne_V'] = IndRes['R_ReIne_V']
    # PARA['R_ReIne_W'] = IndRes['R_ReIne_W']
    # PARA['L_ReIne_N'] = IndRes['L_ReIne_N']
    # PARA['L_ReIne_U'] = IndRes['L_ReIne_U']
    # PARA['L_ReIne_V'] = IndRes['L_ReIne_V']
    # PARA['L_ReIne_W'] = IndRes['L_ReIne_W']
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
    parameters['Sbase'] = 100000  # in [VA]
    parameters['Zbase'] = parameters['Vbase'] ** 2 / parameters['Sbase']
    parameters['Zsc_tr'] = 0.0257
    parameters['Xi_tr'] = 0.0144
    parameters['X_Tr'] = parameters['Xi_tr']  # in per-unit
    parameters['R_Tr'] = np.sqrt(parameters['Zsc_tr'] ** 2 - parameters['Xi_tr'] ** 2)  # in per-unit
    parameters['L_cable'] = 0.11 * 0
    parameters['L2X'] = 2 * 3.14 * parameters['fn']
    parameters['R_cable'] = parameters['L_cable'] * 0.272 / parameters['Zbase']  # in [per-unit] Cable EPR-PVR 5*70mm2
    parameters['X_cable'] = parameters['L_cable'] * 0.070 / parameters['Zbase']  # in [per-unit] Cable EPR-PVR 5*70mm2
    parameters['Cap'] = 100
    # Ond = [SolarMax, ABB, KACO]
    parameters['Ond_Cap'] = [5, 8.5, 9]
    parameters['Ond_V_grid_pu'] = [1, 1, 1]
    parameters['Ond_V_conv_pu'] = [545 / 400, 620 / 400, 575 / 400]
    parameters['Ond_X_PV_pu'] = [3 * parameters['Ond_V_grid_pu'][x] *
                                 (parameters['Ond_V_conv_pu'][x] - parameters['Ond_V_grid_pu'][x]) * parameters['Sbase']
                                 / (1000 * parameters['Ond_Cap'][x]) for x in range(3)]
    parameters['Ond_cos_PV'] = [0.89, 0.89, 0.89]
    # Con = [Battery]
    parameters['Con_SOC_min_kWh'] = [0]
    parameters['Con_SOC_max_kWh'] = [64]
    parameters['Con_S_max_kVA'] = [120]
    parameters['Con_P_max_kW'] = [100]
    parameters['Con_Eff_LV'] = [1]
    parameters['Con_Eff_D'] = [0.98]
    parameters['Con_Eff_C'] = [0.98]
    parameters['Con_Eff_LC'] = [1]
    return parameters


def grid_topology_sim(case_name, Vec_inp):
    """" Completed
    This function is for generating the file of grid_inp
    Inputs:
        - case_name: Name of the case you are working
        - Vec_inp: Vec_Inp as ordered by Douglas in "07/10/2020 14:06"
                                                            (See "\Data\Ordre_informations_controle_2020.10.07.xlsx")
    Output: a dictionary gives the specification of network based on the standard of pydgrid package
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
          data["PV_elements"] = [{"index": 0, "bus": "2", "cap_kVA_perPhase": 200 / 3, "V_grid_pu": 1,
                                  "V_conv_pu": 1, "X_PV_pu": 1}]
          data["load_elements"] = [{"index": 0, "bus": "1"}]
          data["storage_elements"] = [{"index": 0, "bus": "2", "SOC_min_kWh": 0, "SOC_max_kWh": 40, "S_max_kVA": 20,
                                      "P_max_kW": 10, "Eff_LV": 1, "Eff_D": 1, "Eff_C": 1, "Eff_LC": 1}]
          data["Nbus"] = 7
          data["Zbase"] = 400 ** 2 / 250000
    """
    # Input data
    if True:
        parameters = reine_parameters()
        R_ReIne_U = parameters['R_ReIne_U']
        R_ReIne_V = parameters['R_ReIne_V']
        R_ReIne_W = parameters['R_ReIne_W']
        L_ReIne_U = parameters['L_ReIne_U']
        L_ReIne_V = parameters['L_ReIne_V']
        L_ReIne_W = parameters['L_ReIne_W']
        Sbase = parameters['Sbase']  # in [VA]
        Zbase = parameters['Zbase']
        X_Tr = parameters['X_Tr']
        R_Tr = parameters['R_Tr']
        L2X = parameters['L2X']
        R_cable = parameters['R_cable']
        X_cable = parameters['X_cable']
        Cap = parameters['Cap']
        ##
        Ond_Cap = parameters['Ond_Cap']
        Ond_V_grid_pu = parameters['Ond_V_grid_pu']
        Ond_V_conv_pu = parameters['Ond_V_conv_pu']
        Ond_X_PV_pu = parameters['Ond_X_PV_pu']
        Ond_cos_PV = parameters['Ond_cos_PV']
        Con_SOC_min_kWh = parameters['Con_SOC_min_kWh']
        Con_SOC_max_kWh = parameters['Con_SOC_max_kWh']
        Con_S_max_kVA = parameters['Con_S_max_kVA']
        Con_P_max_kW = parameters['Con_P_max_kW']
        Con_Eff_LV = parameters['Con_Eff_LV']
        Con_Eff_D = parameters['Con_Eff_D']
        Con_Eff_C = parameters['Con_Eff_C']
        Con_Eff_LC = parameters['Con_Eff_LC']
    Charge, Ond, Con = [], [], []
    S_73K, S_74K, S_L9 = [], [], []
    data = {}
    # Defining switches state for different cases
    ## Sample Case
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
                             {"bus": "POImv", "bus_nodes": [1, 2, 3], "kW": 0, "kvar": 0}  # STATCOM
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
        data = {
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
    # \\eistore2\iese\institut\Projets\Projets_MRB\Validation Tests and Analysis\Analysis
    elif case_name == "Case_4bus_SmileFA_Tr":
        S_73K = [True, False, False, False, True, False, False]
        # 73K = [14, 15, 16, 17, 18, 19, 20]
        S_74K = [False, True, False, False, True, False, False, True, True, False,
                 True, True, True, True, False]
        # 74K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        S_L9 = [0, 0]
        # S_L9 = [1-5, 6-9]
        Charge = [0, 0, 0, 0, 6, 3, 0, 0]
        # Ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        Ond = [0, 5, 0]
        # Ond = [SolarMax, ABB, KACO]
        Con = [0, 0, 1]
        # Con = [Cinergia, SOP, Battery]
    elif case_name == "Case_4bus_Test":
        S_73K = [True, False, True, False, False, False, False]
        # 73K = [14, 15, 16, 17, 18, 19, 20]
        S_74K = [True, False, False, True, False, False, False, False, False, True,
                 True, False, False, True, False]
        # 74K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        S_L9 = [0, 0]
        # S_L9 = [1-5, 6-9]
        Charge = [1, 4, 5, 8, 0, 0, 0, 0]
        # Ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        Ond = [0, 0, 0]
        # Ond = [SolarMax, ABB, KACO]
        Con = [0, 0, 0]
        # Con = [Cinergia, SOP, Battery]
    elif case_name == "Case_4bus_DiGriFlex":
        S_73K = [True, False, False, False, True, False, False]
        # 73K = [14, 15, 16, 17, 18, 19, 20]
        S_74K = [False, False, False, False, False, False, False, True, True, False,
                 False, True, True, True, True]
        # 74K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        S_L9 = [0, 0]
        # S_L9 = [1-5, 6-9]
        Charge = [3, 0, 0, 0, 0, 0, 0, 0]
        # Ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        Ond = [0, 9, 0]
        # Ond = [SolarMax, ABB, KACO]
        Con = [1]
        # Con = [Battery]
    elif case_name == "Case_LabVIEW":
        Switches, _, _, _, _, _, _, _, _, _, _, _ = interface_meas(Vec_inp)
        S_73K = Switches["S_73K"]
        # 73K = [14, 15, 16, 17, 18, 19, 20]
        S_74K = Switches["S_74K"]
        # 74K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        S_L9 = Switches["S_L9"]
        # S_L9 = [1-5, 6-9]
        Charge = Switches["Charge"]
        # Ch_node = [1, 2, 3, 4, 5, 6, 7, 8]
        Ond = Switches["Ond"]
        # Ond = [SolarMax, ABB, KACO]
        Con = Switches["Con"]
        # Con = [Battery]
    # Automatic determination of output dictionary from state of switches
    if case_name != "9BusesPydGridExample" or case_name != "6BusesLaChappelle":
        Nodes = list(set().union([1] + Charge + Ond + Con))
        Nodes.pop(0)
        for i in range(3):
            for index, item in enumerate(Nodes):
                if item == 9 and S_74K[2] == 1 and S_74K[5] == 1:
                    Nodes[index] = 5
                elif item == 8 and S_74K[4] == 1 and S_74K[1] == 1:
                    Nodes[index] = 4
                elif item == 7 and S_74K[3] == 1 and S_74K[0] == 1:
                    Nodes[index] = 3
                elif item == 6 and S_73K[6] == 1:
                    Nodes[index] = 2
                elif item == 5 and S_74K[2] == 1 and S_73K[5] == 1:
                    Nodes[index] = 1
                elif item == 4 and S_74K[1] == 1 and S_73K[3] == 1:
                    Nodes[index] = 1
                elif item == 3 and S_74K[0] == 1 and S_73K[1] == 1:
                    Nodes[index] = 1
        Nodes = list(set().union(Nodes))
        Mbig = 1e5
        Ybig = 300 / complex(min(R_ReIne_U) + min(R_ReIne_V) + min(R_ReIne_W) + 3 * R_cable,
                             (min(L_ReIne_U) + min(L_ReIne_V) + min(L_ReIne_W)) * L2X + 3 * X_cable)
        ybus_1_2 = 3 / complex(R_ReIne_U[0] + R_ReIne_V[0] + R_ReIne_W[0] + 3 * R_cable,
                               (L_ReIne_U[0] + L_ReIne_V[0] + L_ReIne_W[0]) * L2X + 3 * X_cable)
        ybus_1_6 = 3 / complex(R_ReIne_U[4] + R_ReIne_V[4] + R_ReIne_W[4] + 3 * R_cable,
                               (L_ReIne_U[4] + L_ReIne_V[4] + L_ReIne_W[4]) * L2X + 3 * X_cable)
        ybus_1_10 = ybus_1_11 = ybus_1_12 = Ybig
        ybus_2_3 = 3 / complex(R_ReIne_U[1] + R_ReIne_V[1] + R_ReIne_W[1] + 3 * R_cable,
                               (L_ReIne_U[1] + L_ReIne_V[1] + L_ReIne_W[1]) * L2X + 3 * X_cable)
        ybus_2_6 = Ybig
        ybus_3_4 = 3 / complex(R_ReIne_U[2] + R_ReIne_V[2] + R_ReIne_W[2] + 3 * R_cable,
                               (L_ReIne_U[2] + L_ReIne_V[2] + L_ReIne_W[2]) * L2X + 3 * X_cable)
        ybus_3_10 = Ybig
        ybus_4_5 = 3 / complex(R_ReIne_U[3] + R_ReIne_V[3] + R_ReIne_W[3] + 3 * R_cable,
                               (L_ReIne_U[3] + L_ReIne_V[3] + L_ReIne_W[3]) * L2X + 3 * X_cable)
        ybus_4_11 = Ybig
        ybus_5_12 = Ybig
        ybus_6_7 = 3 / complex(R_ReIne_U[5] + R_ReIne_V[5] + R_ReIne_W[5] + 3 * R_cable,
                               (L_ReIne_U[5] + L_ReIne_V[5] + L_ReIne_W[5]) * L2X + 3 * X_cable)
        ybus_7_8 = 3 / complex(R_ReIne_U[6] + R_ReIne_V[6] + R_ReIne_W[6] + 3 * R_cable,
                               (L_ReIne_U[6] + L_ReIne_V[6] + L_ReIne_W[6]) * L2X + 3 * X_cable)
        ybus_7_10 = Ybig
        ybus_8_9 = 3 / complex(R_ReIne_U[7] + R_ReIne_V[7] + R_ReIne_W[7] + 3 * R_cable,
                               (L_ReIne_U[7] + L_ReIne_V[7] + L_ReIne_W[7]) * L2X + 3 * X_cable)
        ybus_8_11 = Ybig
        ybus_9_12 = Ybig
        ybus_L9 = 3 / complex(R_ReIne_U[8] + R_ReIne_V[8] + R_ReIne_W[8] + 3 * R_cable,
                              (L_ReIne_U[8] + L_ReIne_V[8] + L_ReIne_W[8]) * L2X + 3 * X_cable)
        ybus_1_2 = ybus_1_2 * S_74K[7]
        ybus_1_6 = ybus_1_6 * S_74K[11]
        ybus_1_10 = ybus_1_10 * S_73K[2]
        ybus_1_11 = ybus_1_11 * S_73K[3]
        ybus_1_12 = ybus_1_12 * S_73K[5]
        ybus_2_3 = ybus_2_3 * S_74K[8]
        ybus_2_6 = ybus_2_6 * S_73K[6]
        ybus_3_4 = ybus_3_4 * S_74K[9]
        ybus_3_10 = ybus_3_10 * S_74K[0]
        ybus_4_5 = ybus_4_5 * S_74K[10]
        ybus_4_11 = ybus_4_11 * S_74K[1]
        ybus_5_12 = ybus_5_12 * S_74K[2]
        ybus_6_7 = ybus_6_7 * S_74K[12]
        ybus_7_8 = ybus_7_8 * S_74K[13]
        ybus_7_10 = ybus_7_10 * S_74K[3]
        ybus_8_9 = ybus_8_9 * S_74K[14]
        ybus_8_11 = ybus_8_11 * S_74K[4]
        ybus_9_12 = ybus_9_12 * S_74K[5]
        Ybus = [[ybus_1_2 + ybus_1_6 + ybus_1_10 + ybus_1_11 + ybus_1_12, -ybus_1_2, 0, 0, 0, -ybus_1_6,
                 0, 0, 0, -ybus_1_10, -ybus_1_11, -ybus_1_12],
                [-ybus_1_2, ybus_1_2 + ybus_2_3 + ybus_2_6, -ybus_2_3, 0, 0, -ybus_2_6,
                 0, 0, 0, 0, 0, 0],
                [0, -ybus_2_3, ybus_2_3 + ybus_3_4 + ybus_3_10, -ybus_3_4, 0, 0,
                 0, 0, 0, -ybus_3_10, 0, 0],
                [0, 0, -ybus_3_4, ybus_3_4 + ybus_4_5 + ybus_4_11, -ybus_4_5, 0,
                 0, 0, 0, 0, -ybus_4_11, 0],
                [0, 0, 0, -ybus_4_5, ybus_4_5 + ybus_5_12, 0,
                 0, 0, 0, 0, 0, -ybus_5_12],
                [-ybus_1_6, -ybus_2_6, 0, 0, 0, ybus_1_6 + ybus_2_6 + ybus_6_7,
                 -ybus_6_7, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -ybus_6_7,
                 ybus_6_7 + ybus_7_8 + ybus_7_10, -ybus_7_8, 0, -ybus_7_10, 0, 0],
                [0, 0, 0, 0, 0, 0,
                 -ybus_7_8, ybus_7_8 + ybus_8_9 + ybus_8_11, -ybus_8_9, 0, -ybus_8_11, 0],
                [0, 0, 0, 0, 0, 0,
                 0, -ybus_8_9, ybus_8_9 + ybus_9_12, 0, 0, -ybus_9_12],
                [-ybus_1_10, 0, -ybus_3_10, 0, 0, 0,
                 -ybus_7_10, 0, 0, (ybus_1_10 + ybus_3_10 + ybus_7_10), 0, 0],
                [-ybus_1_11, 0, 0, -ybus_4_11, 0, 0,
                 0, -ybus_8_11, 0, 0, (ybus_1_11 + ybus_4_11 + ybus_8_11), 0],
                [-ybus_1_12, 0, 0, 0, -ybus_5_12, 0,
                 0, 0, -ybus_9_12, 0, 0, (ybus_1_12 + ybus_5_12 + ybus_9_12)]]
        if S_L9[0] > 0 and S_L9[1] > 0:
            Ybus[S_L9[0] - 1][S_L9[1] - 1] = Ybus[S_L9[0] - 1][S_L9[1] - 1] - ybus_L9
            Ybus[S_L9[1] - 1][S_L9[0] - 1] = Ybus[S_L9[1] - 1][S_L9[0] - 1] - ybus_L9
            Ybus[S_L9[0] - 1][S_L9[0] - 1] = Ybus[S_L9[0] - 1][S_L9[1] - 1] + ybus_L9
            Ybus[S_L9[1] - 1][S_L9[1] - 1] = Ybus[S_L9[0] - 1][S_L9[1] - 1] + ybus_L9
        Ybus_arr = np.array(Ybus, dtype=object)
        Nodes_dis = np.where(sum(abs(Ybus_arr)) < 1 / Mbig)
        All_Nodes = list(range(1, len(Ybus) + 1))
        for ele in Nodes_dis[0]:
            All_Nodes.remove(ele + 1)
        Ordered_Nodes = list(set().union(Nodes + All_Nodes))
        Ordered_Nodes = [x - 1 for x in Ordered_Nodes]
        Ybus = [[Ybus[i][j] for j in Ordered_Nodes] for i in Ordered_Nodes]
        Aloc = []
        for ele in Nodes:
            Aloc.append(Ordered_Nodes.index(ele - 1))
        Dloc = list(set(range(len(Ordered_Nodes))).difference(set(Aloc)))
        Ybus_arr = np.array(Ybus, dtype=object)
        A, B, C, D = Ybus_arr[np.ix_(Aloc, Aloc)], Ybus_arr[np.ix_(Aloc, Dloc)], \
                     Ybus_arr[np.ix_(Dloc, Dloc)], Ybus_arr[np.ix_(Dloc, Aloc)]
        Ybus_red = A - np.matmul(np.matmul(B, np.array(np.linalg.inv(C.tolist()))), D)
        data["buses"] = [{"bus": "0", "pos_x": 0, "pos_y": 0, "units": "m", "U_kV": 20,
                          "Vmin": 0.9, "Vmax": 1.1}]  # Slack, primary
        data["shunts"] = []
        data["grid_feeders"] = []
        for n in Nodes:
            data["buses"].append({"bus": str(n), "pos_x": 0, "pos_y": 0, "units": "m", "U_kV": 0.4,
                                  "Vmin": 0.9, "Vmax": 1.1})
            data["shunts"].append({"bus": str(n), "R": 0.0001, "X": 0.0001, "bus_nodes": [4, 0]})
            data["grid_feeders"].append({"bus": str(n), "bus_nodes": [1, 2, 3, 4], "kW": 0, "kvar": 0})
        data["lines"] = []
        data["line_codes"] = {}
        for i in Nodes[:-1]:
            for j in Nodes[Nodes.index(i) + 1:]:
                if abs(Ybus_red[Nodes.index(i)][Nodes.index(j)]).real > 1:
                    ii = i
                    jj = j
                    for d in data["lines"]:
                        if str(j) == d["bus_k"]:
                            ii = j
                            jj = i
                    data["lines"].append(
                        {"bus_j": str(ii), "bus_k": str(jj), "code": str(ii) + ',' + str(jj), "m": 1000,
                         "Cap": Cap})
                    data["line_codes"][str(ii) + ',' + str(jj)] = {
                        "R1": -(1 / (Ybus_red[Nodes.index(i)][Nodes.index(j)])).real,
                        "X1": -(1 / (Ybus_red[Nodes.index(i)][Nodes.index(j)])).imag,
                        "R0": -(1 / (Ybus_red[Nodes.index(i)][Nodes.index(j)])).real,
                        "X0": -(1 / (Ybus_red[Nodes.index(i)][Nodes.index(j)])).imag,
                        "B_1_mu": 0,
                        "B_0_mu": 0}
        data["transformers"] = [{"bus_j": "0", "bus_k": "1", "S_n_kVA": Sbase / 1000, "U_1_kV": 20, "U_2_kV": 0.4,
                                 "Cap": Cap,
                                 "R_cc_pu": R_Tr + R_cable, "X_cc_pu": X_Tr + X_cable,
                                 "connection": "Dyn11",
                                 "conductors_1": 3, "conductors_2": 4}]

        data["grid_formers"] = [{"index": 0, "bus": "0", "bus_nodes": [1, 2, 3],
                                 "kV": [20 / np.sqrt(3), 20 / np.sqrt(3), 20 / np.sqrt(3)],
                                 "deg": [30, 150, 270]}]
        data["Nbus"] = np.size(data["buses"], 0)
        data["Zbase"] = Zbase
        data["PV_elements"] = []
        j = -1
        for index, p in enumerate(Ond, start=0):
            if p != 0:
                j = j + 1
                data["PV_elements"].append({"index": j, "bus": str(p),
                                            "cap_kVA_perPhase": Ond_Cap[index] / 3, "V_grid_pu": Ond_V_grid_pu[index],
                                            "V_conv_pu": Ond_V_conv_pu[index], "X_PV_pu": Ond_X_PV_pu[index], "cos_PV": Ond_cos_PV[index]})
        data["load_elements"] = []
        j = -1
        for ch in Charge:
            if ch != 0:
                j = j + 1
                data["load_elements"].append({"index": j, "bus": str(ch)})
        data["storage_elements"] = []
        j = -1
        for index, st in enumerate(Con, start=0):
            if st != 0:
                j = j + 1
                data["storage_elements"].append({"index": j, "bus": str(st),
                                                 "SOC_min_kWh": Con_SOC_min_kWh[index],
                                                 "SOC_max_kWh": Con_SOC_max_kWh[index],
                                                 "S_max_kVA": Con_S_max_kVA[index],
                                                 "P_max_kW": Con_P_max_kW[index],
                                                 "Eff_LV": Con_Eff_LV[index],
                                                 "Eff_D": Con_Eff_D[index], "Eff_C": Con_Eff_C[index],
                                                 "Eff_LC": Con_Eff_LC[index]})
    # print(data)
    return data


def interface_meas(Vec_inp):
    """" Completed
    This function is for reading measurements from LabVIEW
    Input: Vec_Inp as ordered by Douglas in "07/10/2020 14:06" (See "\Data\Ordre_informations_controle_2020.10.07.xlsx")
    Outputs:
        - Switches = {"S_73K":[]*8, "S_74_K": []*16, "S_L9": []*1, "Charge": []*14,
                                                        "Ond": []*16, "Con"@*: []*1}   @*Con-->Battery
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
    Switches = {}
    Switches["S_73K"] = [float(i) for i in Vec_inp[385:392]]
    Switches["S_74K"] = [float(i) for i in Vec_inp[392:407]]
    Switches["S_L9"] = [int(Vec_inp[411]) * 1 + int(Vec_inp[412]) * 2 + int(Vec_inp[413]) * 3 +
                        int(Vec_inp[414]) * 4 + int(Vec_inp[415]) * 5,
                        int(Vec_inp[407]) * 6 + int(Vec_inp[408]) * 7 +
                        int(Vec_inp[409]) * 8 + int(Vec_inp[410]) * 9]
    Switches["Charge"] = [0] * (11 + 3)
    Switches["Ond"] = [0] * 3
    Switches["Con"] = [0] * 1
    for index, item in enumerate(Vec_inp[416:]):
        for ch in range(1, 12):
            if item == ("Charge " + str(ch)):
                Switches["Charge"][ch - 1] = int(Vec_inp[416 + index + 1])
        for ch in range(12, 15):
            if item == ("Charge " + str(ch)):
                Switches["Ond"][ch - 12] = int(Vec_inp[416 + index + 1])
        for ch in range(15, 18):
            if item == ("Charge " + str(ch)):
                Switches["Charge"][ch - 4] = int(Vec_inp[416 + index + 1])
        if "Batterie" in item:
            Switches["Con"][0] = int(Vec_inp[416 + index + 1])
    Ligne_U = [0] * 9
    Ligne_U[0] = np.sqrt(3) * ((float(Vec_inp[3]) + float(Vec_inp[4]) + float(Vec_inp[5])) / 3 - float(Vec_inp[6]))
    Ligne_U[1] = np.sqrt(3) * ((float(Vec_inp[7]) + float(Vec_inp[8]) + float(Vec_inp[9])) / 3 - float(Vec_inp[10]))
    Ligne_U[2] = np.sqrt(3) * ((float(Vec_inp[11]) + float(Vec_inp[12]) + float(Vec_inp[13])) / 3 - float(Vec_inp[14]))
    Ligne_U[3] = np.sqrt(3) * ((float(Vec_inp[15]) + float(Vec_inp[16]) + float(Vec_inp[17])) / 3 - float(Vec_inp[18]))
    Ligne_U[4] = np.sqrt(3) * ((float(Vec_inp[79]) + float(Vec_inp[80]) + float(Vec_inp[81])) / 3 - float(Vec_inp[82]))
    Ligne_U[5] = np.sqrt(3) * ((float(Vec_inp[83]) + float(Vec_inp[84]) + float(Vec_inp[85])) / 3 - float(Vec_inp[86]))
    Ligne_U[6] = np.sqrt(3) * ((float(Vec_inp[87]) + float(Vec_inp[88]) + float(Vec_inp[89])) / 3 - float(Vec_inp[90]))
    Ligne_U[7] = np.sqrt(3) * ((float(Vec_inp[91]) + float(Vec_inp[92]) + float(Vec_inp[93])) / 3 - float(Vec_inp[94]))
    Ligne_U[8] = np.sqrt(3) * (
            (float(Vec_inp[145]) + float(Vec_inp[146]) + float(Vec_inp[147])) / 3 - float(Vec_inp[148]))
    Ligne_I = [0] * 9
    Ligne_I[0] = (float(Vec_inp[41]) + float(Vec_inp[42]) + float(Vec_inp[43])) - float(Vec_inp[44])
    Ligne_I[1] = (float(Vec_inp[45]) + float(Vec_inp[46]) + float(Vec_inp[47])) - float(Vec_inp[48])
    Ligne_I[2] = (float(Vec_inp[49]) + float(Vec_inp[50]) + float(Vec_inp[51])) - float(Vec_inp[52])
    Ligne_I[3] = (float(Vec_inp[53]) + float(Vec_inp[54]) + float(Vec_inp[55])) - float(Vec_inp[56])
    Ligne_I[4] = (float(Vec_inp[133]) + float(Vec_inp[134]) + float(Vec_inp[135])) - float(Vec_inp[136])
    Ligne_I[5] = (float(Vec_inp[117]) + float(Vec_inp[118]) + float(Vec_inp[119])) - float(Vec_inp[120])
    Ligne_I[6] = (float(Vec_inp[121]) + float(Vec_inp[122]) + float(Vec_inp[123])) - float(Vec_inp[124])
    Ligne_I[7] = (float(Vec_inp[125]) + float(Vec_inp[126]) + float(Vec_inp[127])) - float(Vec_inp[128])
    Ligne_I[8] = (float(Vec_inp[170]) + float(Vec_inp[171]) + float(Vec_inp[172])) - float(Vec_inp[173])
    Charge_I = [0] * (11 + 3)
    Ond_I = [0] * 3
    Charge_I[0] = (float(Vec_inp[149]) + float(Vec_inp[150]) + float(Vec_inp[151])) - float(Vec_inp[152])
    Charge_I[1] = (float(Vec_inp[153]) + float(Vec_inp[154]) + float(Vec_inp[155])) - float(Vec_inp[156])
    Charge_I[2] = (float(Vec_inp[157]) + float(Vec_inp[158]) + float(Vec_inp[159])) - float(Vec_inp[160])
    Charge_I[3] = (float(Vec_inp[72]) + float(Vec_inp[73]) + float(Vec_inp[74])) - float(Vec_inp[75])
    Charge_I[4] = (float(Vec_inp[129]) + float(Vec_inp[130]) + float(Vec_inp[131])) - float(Vec_inp[132])
    Charge_I[5] = (float(Vec_inp[59]) + float(Vec_inp[60]) + float(Vec_inp[61])) - float(Vec_inp[62])
    Charge_I[6] = (float(Vec_inp[21]) + float(Vec_inp[22]) + float(Vec_inp[23])) - float(Vec_inp[24])
    Charge_I[7] = (float(Vec_inp[25]) + float(Vec_inp[26]) + float(Vec_inp[27])) - float(Vec_inp[28])
    Charge_I[8] = (float(Vec_inp[29]) + float(Vec_inp[30]) + float(Vec_inp[31])) - float(Vec_inp[32])
    Charge_I[9] = (float(Vec_inp[33]) + float(Vec_inp[34]) + float(Vec_inp[35])) - float(Vec_inp[36])
    Charge_I[10] = (float(Vec_inp[63]) + float(Vec_inp[64]) + float(Vec_inp[65])) - float(Vec_inp[66])
    Ond_I[0] = (float(Vec_inp[96]) + float(Vec_inp[97]) + float(Vec_inp[98])) - float(Vec_inp[99])
    Ond_I[1] = (float(Vec_inp[100]) + float(Vec_inp[101]) + float(Vec_inp[102])) - float(Vec_inp[103])
    Ond_I[2] = (float(Vec_inp[104]) + float(Vec_inp[105]) + float(Vec_inp[106])) - float(Vec_inp[107])
    Charge_I[11] = (float(Vec_inp[108]) + float(Vec_inp[109]) + float(Vec_inp[110])) - float(Vec_inp[111])
    Charge_I[12] = (float(Vec_inp[112]) + float(Vec_inp[113]) + float(Vec_inp[114])) - float(Vec_inp[115])
    Charge_I[13] = (float(Vec_inp[67]) + float(Vec_inp[68]) + float(Vec_inp[69])) - float(Vec_inp[70])
    Charge_P = [0] * (11 + 3)
    Ond_P = [0] * 3
    Charge_P[0] = (float(Vec_inp[191]) + float(Vec_inp[192]) + float(Vec_inp[193])) / 1000
    Charge_P[1] = (float(Vec_inp[194]) + float(Vec_inp[195]) + float(Vec_inp[196])) / 1000
    Charge_P[2] = (float(Vec_inp[197]) + float(Vec_inp[198]) + float(Vec_inp[199])) / 1000
    Charge_P[3] = (float(Vec_inp[200]) + float(Vec_inp[201]) + float(Vec_inp[202])) / 1000
    Charge_P[4] = (float(Vec_inp[203]) + float(Vec_inp[204]) + float(Vec_inp[205])) / 1000
    Charge_P[5] = (float(Vec_inp[206]) + float(Vec_inp[207]) + float(Vec_inp[208])) / 1000
    Charge_P[6] = (float(Vec_inp[209]) + float(Vec_inp[210]) + float(Vec_inp[211])) / 1000
    Charge_P[7] = (float(Vec_inp[212]) + float(Vec_inp[213]) + float(Vec_inp[214])) / 1000
    Charge_P[8] = (float(Vec_inp[215]) + float(Vec_inp[216]) + float(Vec_inp[217])) / 1000
    Charge_P[9] = (float(Vec_inp[218]) + float(Vec_inp[219]) + float(Vec_inp[220])) / 1000
    Charge_P[10] = (float(Vec_inp[221]) + float(Vec_inp[222]) + float(Vec_inp[223])) / 1000
    Ond_P[0] = (float(Vec_inp[224]) + float(Vec_inp[225]) + float(Vec_inp[226])) / 1000
    Ond_P[1] = (float(Vec_inp[227]) + float(Vec_inp[228]) + float(Vec_inp[229])) / 1000
    Ond_P[2] = (float(Vec_inp[230]) + float(Vec_inp[231]) + float(Vec_inp[232])) / 1000
    Charge_P[11] = (float(Vec_inp[233]) + float(Vec_inp[234]) + float(Vec_inp[235])) / 1000
    Charge_P[12] = (float(Vec_inp[236]) + float(Vec_inp[237]) + float(Vec_inp[238])) / 1000
    Charge_P[13] = (float(Vec_inp[239]) + float(Vec_inp[240]) + float(Vec_inp[241])) / 1000
    Charge_Q = [0] * (11 + 3)
    Ond_Q = [0] * 3
    Charge_Q[0] = (float(Vec_inp[242]) + float(Vec_inp[243]) + float(Vec_inp[244])) / 1000
    Charge_Q[1] = (float(Vec_inp[245]) + float(Vec_inp[246]) + float(Vec_inp[247])) / 1000
    Charge_Q[2] = (float(Vec_inp[248]) + float(Vec_inp[249]) + float(Vec_inp[250])) / 1000
    Charge_Q[3] = (float(Vec_inp[251]) + float(Vec_inp[252]) + float(Vec_inp[253])) / 1000
    Charge_Q[4] = (float(Vec_inp[254]) + float(Vec_inp[255]) + float(Vec_inp[256])) / 1000
    Charge_Q[5] = (float(Vec_inp[257]) + float(Vec_inp[258]) + float(Vec_inp[259])) / 1000
    Charge_Q[6] = (float(Vec_inp[260]) + float(Vec_inp[261]) + float(Vec_inp[262])) / 1000
    Charge_Q[7] = (float(Vec_inp[263]) + float(Vec_inp[264]) + float(Vec_inp[265])) / 1000
    Charge_Q[8] = (float(Vec_inp[266]) + float(Vec_inp[267]) + float(Vec_inp[268])) / 1000
    Charge_Q[9] = (float(Vec_inp[269]) + float(Vec_inp[270]) + float(Vec_inp[271])) / 1000
    Charge_Q[10] = (float(Vec_inp[272]) + float(Vec_inp[273]) + float(Vec_inp[274])) / 1000
    Ond_Q[0] = (float(Vec_inp[275]) + float(Vec_inp[276]) + float(Vec_inp[277])) / 1000
    Ond_Q[1] = (float(Vec_inp[278]) + float(Vec_inp[279]) + float(Vec_inp[280])) / 1000
    Ond_Q[2] = (float(Vec_inp[281]) + float(Vec_inp[282]) + float(Vec_inp[283])) / 1000
    Charge_Q[11] = (float(Vec_inp[284]) + float(Vec_inp[285]) + float(Vec_inp[286])) / 1000
    Charge_Q[12] = (float(Vec_inp[287]) + float(Vec_inp[288]) + float(Vec_inp[289])) / 1000
    Charge_Q[13] = (float(Vec_inp[290]) + float(Vec_inp[291]) + float(Vec_inp[292])) / 1000
    Ligne_P = [0] * 9
    Ligne_P[0] = (float(Vec_inp[293]) + float(Vec_inp[294]) + float(Vec_inp[295])) / 1000
    Ligne_P[1] = (float(Vec_inp[296]) + float(Vec_inp[297]) + float(Vec_inp[298])) / 1000
    Ligne_P[2] = (float(Vec_inp[299]) + float(Vec_inp[300]) + float(Vec_inp[301])) / 1000
    Ligne_P[3] = (float(Vec_inp[302]) + float(Vec_inp[303]) + float(Vec_inp[304])) / 1000
    Ligne_P[4] = (float(Vec_inp[305]) + float(Vec_inp[306]) + float(Vec_inp[307])) / 1000
    Ligne_P[5] = (float(Vec_inp[308]) + float(Vec_inp[309]) + float(Vec_inp[310])) / 1000
    Ligne_P[6] = (float(Vec_inp[311]) + float(Vec_inp[312]) + float(Vec_inp[313])) / 1000
    Ligne_P[7] = (float(Vec_inp[314]) + float(Vec_inp[315]) + float(Vec_inp[316])) / 1000
    Ligne_P[8] = (float(Vec_inp[317]) + float(Vec_inp[318]) + float(Vec_inp[319])) / 1000
    Ligne_Q = [0] * 9
    Ligne_Q[0] = (float(Vec_inp[320]) + float(Vec_inp[321]) + float(Vec_inp[322])) / 1000
    Ligne_Q[1] = (float(Vec_inp[323]) + float(Vec_inp[324]) + float(Vec_inp[325])) / 1000
    Ligne_Q[2] = (float(Vec_inp[326]) + float(Vec_inp[327]) + float(Vec_inp[328])) / 1000
    Ligne_Q[3] = (float(Vec_inp[329]) + float(Vec_inp[330]) + float(Vec_inp[331])) / 1000
    Ligne_Q[4] = (float(Vec_inp[332]) + float(Vec_inp[333]) + float(Vec_inp[334])) / 1000
    Ligne_Q[5] = (float(Vec_inp[335]) + float(Vec_inp[336]) + float(Vec_inp[337])) / 1000
    Ligne_Q[6] = (float(Vec_inp[338]) + float(Vec_inp[339]) + float(Vec_inp[340])) / 1000
    Ligne_Q[7] = (float(Vec_inp[341]) + float(Vec_inp[342]) + float(Vec_inp[343])) / 1000
    Ligne_Q[8] = (float(Vec_inp[344]) + float(Vec_inp[345]) + float(Vec_inp[346])) / 1000
    SOC = float(Vec_inp[383])
    return Switches, Ligne_U, Ligne_I, Charge_I, Ond_I, Charge_P, Charge_Q, Ond_P, Ond_Q, Ligne_P, Ligne_Q, SOC


def sim_meas_time_series(data_name, grid_formers):
    """" Completed
    This function is used in load flow of the recorded measurement in mat files in simulation
    Inputs: - data_name
            - grid_formers: a part of standard output dictionary of function "grid_topology_sim"
    Output: a dictionary gives the specification of recorded data
        data = {"Name": str, "value": loaded mat file, "phase_name": [list of phase names],
                "unit_conv_PQinp": , "unit_conv_Vmeas": ,  "P_name": , "Q_name": , "ind_bus": [list of measured nodes],
                "ord_bus": , "ord_ph": , "ord_time": , "meas": , "DeltaT": , "P": np.array, "Q": np.array,
                "V_CP": np.array, "I_CP": np.array, "P_CP": np.array, "Q_CP": np.array,
                "P_meas_name": , "Q_meas_name": , "V_meas_name": , "I_meas_name": ,
                "meas_location": [{"from": "0", "to": "1", "tran/line": "tran"},],
                "Nt": , "Nf": , "meas_number": ,
        }
    """
    Day = 1
    data = {}
    data["Name"] = dir_name + r"/Data/" + data_name + ".mat"
    data["value"] = sio.loadmat(data["Name"])
    data["phase_name"] = ["a", "b", "c"]
    # \\eistore2\iese\institut\Projets\Projets_MRB\Validation Tests and Analysis\Analysis
    if data_name == "Synthetized_Profiles_Smile-FA":
        Day = 1
        data["unit_conv_PQinp"] = 1
        data["P_name"] = "Pn_out"
        data["Q_name"] = "Qn_out"
        data["ind_bus"] = ["3", "4", "5"]
        data["ord_bus"] = 0
        data["ord_ph"] = 1
        data["ord_time"] = 2
        data["meas"] = False
        data["DeltaT"] = 10 / 60  # 10 minutes
        data["P"] = np.true_divide(data["value"][data["P_name"]], data["unit_conv_PQinp"])
        data["Q"] = np.true_divide(data["value"][data["Q_name"]], data["unit_conv_PQinp"])
        data["V_CP"] = np.true_divide(data["value"][data["V_meas_name"]], data["unit_conv_Vmeas"])
        data["I_CP"] = data["value"][data["I_meas_name"]]
        # Fidning index of measured data from the name, "grid_formers[0]" as we have one connecting point
        nn = find_n(grid_formers[0]["bus"], data["meas_location"])
        data["P_CP"] = (data["value"][data["P_meas_name"]][nn][0]
                        + data["value"][data["P_meas_name"]][nn][1]
                        + data["value"][data["P_meas_name"]][nn][2])
        data["Q_CP"] = -np.abs(data["value"][data["Q_meas_name"]][nn][0]
                               + data["value"][data["Q_meas_name"]][nn][1]
                               + data["value"][data["Q_meas_name"]][nn][2])
    # \\eistore2\iese\institut\Projets\Projets_MRB\Validation Tests and Analysis\Analysis
    elif data_name == "Data_Meas_10min_DASRIL_SMILE-FA2018.11.22":
        Day = 1
        data["unit_conv_PQinp"] = 1000  # standard unit is kW and kVAR but here they are measured in W and VAR
        data["unit_conv_Vmeas"] = 1  # standard unit is kV but here they are measured in V
        data["P_name"] = "Pn"
        data["Q_name"] = "Qn"
        data["ind_bus"] = ["1", "3", "6", "5"]
        data["ord_bus"] = 0
        data["ord_ph"] = 1
        data["ord_time"] = 2
        data["meas"] = True
        data["DeltaT"] = 10 / 60  # 10 minutes
        if data["meas"] == 1:
            data["P_meas_name"] = "P"
            data["Q_meas_name"] = "Q"
            data["V_meas_name"] = "V"
            data["I_meas_name"] = "I"
            data["meas_location"] = [{"from": "0", "to": "1", "tran/line": "tran"},
                                     {"from": "1", "to": "3", "tran/line": "line"},
                                     {"from": "1", "to": "6", "tran/line": "line"},
                                     {"from": "6", "to": "5", "tran/line": "line"}]
        data["P"] = np.true_divide(data["value"][data["P_name"]], data["unit_conv_PQinp"])
        data["Q"] = np.true_divide(data["value"][data["Q_name"]], data["unit_conv_PQinp"])
        data["V_CP"] = np.true_divide(data["value"][data["V_meas_name"]], data["unit_conv_Vmeas"])
        data["I_CP"] = data["value"][data["I_meas_name"]]
        # Fidning index of measured data from the name, "grid_formers[0]" as we have one connecting point
        nn = find_n(grid_formers[0]["bus"], data["meas_location"])
        data["P_CP"] = (data["value"][data["P_meas_name"]][nn][0]
                        + data["value"][data["P_meas_name"]][nn][1]
                        + data["value"][data["P_meas_name"]][nn][2])
        data["Q_CP"] = -np.abs(data["value"][data["Q_meas_name"]][nn][0]
                               + data["value"][data["Q_meas_name"]][nn][1]
                               + data["value"][data["Q_meas_name"]][nn][2])
    elif data_name == "Data_Case0_6bus_Async_Unbalanced":
        Day = 320
        data["unit_conv_PQinp"] = 1  # standard unit is kW and kVAR but here they are measured in W and VAR
        data["unit_conv_Vmeas"] = 1  # standard unit is kV but here they are measured in V
        data["P_name"] = "P"
        data["Q_name"] = "Q"
        data["ind_bus"] = ["1", "2", "3", "4", "5", "6"]
        data["ord_bus"] = 0
        data["ord_ph"] = 1
        data["ord_time"] = 2
        data["meas"] = True
        data["DeltaT"] = 10 / 60  # 2 minutes
        if data["meas"] == 1:
            data["P_meas_name"] = "P"
            data["Q_meas_name"] = "Q"
            data["V_meas_name"] = "V"
            data["I_meas_name"] = "I"
            data["meas_location"] = [{"from": "0", "to": "1", "tran/line": "tran"},
                                     {"from": "1", "to": "2", "tran/line": "line"},
                                     {"from": "2", "to": "3", "tran/line": "line"},
                                     {"from": "1", "to": "4", "tran/line": "line"},
                                     {"from": "1", "to": "5", "tran/line": "line"},
                                     {"from": "5", "to": "6", "tran/line": "line"}]
        PPP = np.abs(np.true_divide(data["value"][data["P_name"]], data["unit_conv_PQinp"]))
        QQQ = np.abs(np.true_divide(data["value"][data["Q_name"]], data["unit_conv_PQinp"]))
        VVV = np.true_divide(data["value"][data["V_meas_name"]], data["unit_conv_Vmeas"])
        III = data["value"][data["I_meas_name"]]
        # Fidning index of measured data from the name, "grid_formers[0]" as we have one connecting point
        nn = find_n(grid_formers[0]["bus"], data["meas_location"])
        P_CP = (data["value"][data["P_meas_name"]][nn][0]
                + data["value"][data["P_meas_name"]][nn][1]
                + data["value"][data["P_meas_name"]][nn][2])
        Q_CP = -np.abs(data["value"][data["Q_meas_name"]][nn][0]
                       + data["value"][data["Q_meas_name"]][nn][1]
                       + data["value"][data["Q_meas_name"]][nn][2])
        ttt = 5
        data["P"] = np.zeros((6, 3, int(230400 / ttt)))
        data["Q"] = np.zeros((6, 3, int(230400 / ttt)))
        data["V_CP"] = np.zeros((6, 3, int(230400 / ttt)))
        data["I_CP"] = np.zeros((6, 3, int(230400 / ttt)))
        for ph in range(3):
            for n in range(6):
                data["P"][n][ph] = signal.resample(PPP[n][ph], int(230400 / ttt))
                data["Q"][n][ph] = signal.resample(QQQ[n][ph], int(230400 / ttt))
                data["V_CP"][n][ph] = signal.resample(VVV[n][ph], int(230400 / ttt))
                data["I_CP"][n][ph] = signal.resample(III[n][ph], int(230400 / ttt))
        data["P_CP"] = signal.resample(P_CP, int(230400 / ttt))
        data["Q_CP"] = signal.resample(Q_CP, int(230400 / ttt))
        for ph in range(3):
            data["P"][0][ph] = data["P"][0][ph] - data["P"][1][ph] - data["P"][3][ph] - data["P"][4][ph]
            data["P"][1][ph] = data["P"][1][ph] - data["P"][2][ph]
            data["P"][4][ph] = data["P"][4][ph] - data["P"][5][ph]
            data["Q"][0][ph] = data["Q"][0][ph] - data["Q"][1][ph] - data["Q"][3][ph] - data["Q"][4][ph]
            data["Q"][1][ph] = data["Q"][1][ph] - data["Q"][2][ph]
            data["Q"][4][ph] = data["Q"][4][ph] - data["Q"][5][ph]
    # Common elements of data from different test cases
    if True:
        data["Nt"] = int(np.size(data["P"], data["ord_time"]) / Day)
        data["Nf"] = np.size(data["P"], data["ord_bus"])
        if data["meas"] is True:
            data["meas_number"] = np.size(data["value"][data["P_meas_name"]], data["ord_bus"])
    return data


def sim_loadflow_time_series(grid_inp, meas_inp, title):
    """" Completed
    This function is used to run the load flow in simulation
    Inputs: - grid_inp: output dictionary of function "grid_topology_sim" capturing specification of grid
            - meas_inp: output dictionary of function "sim_meas_time_series" capturing specification of recorded data
            - title: name of plotted figures
    Output: a dictionary gives the load flow of the grid
        data = {"V": np.zeros((np.size(grid_inp["buses"], 0), 3, meas_inp["Nt"])),
                "I": np.zeros(
                        (np.size(grid_inp["transformers"], 0) + np.size(grid_inp["lines"], 0), 3, meas_inp["Nt"])),
                "P_in": np.zeros(meas_inp["Nt"]),
                "Q_in": np.zeros(meas_inp["Nt"]),
        }
    Figures:
            - distribution grid input active and reactive powers,
            - grid voltages,
            - grid currents
    """
    grid_t = grid()
    grid_t.read(grid_inp)
    meas = {"V": np.zeros((np.size(grid_inp["buses"], 0), 3, meas_inp["Nt"])),
            "I": np.zeros(
                (np.size(grid_inp["transformers"], 0) + np.size(grid_inp["lines"], 0), 3, meas_inp["Nt"])),
            "P_in": np.zeros(meas_inp["Nt"]),
            "Q_in": np.zeros(meas_inp["Nt"])}
    for t in range(meas_inp["Nt"]):
        for n in range(meas_inp["Nf"]):
            nn = find_n(meas_inp["ind_bus"][n], grid_inp["grid_feeders"])
            # inp active and reactive power of feeders
            grid_inp["grid_feeders"][nn]["kW"] = [meas_inp["P"][n][0][t], meas_inp["P"][n][1][t],
                                                  meas_inp["P"][n][2][t]]
            grid_inp["grid_feeders"][nn]["kvar"] = [meas_inp["Q"][n][0][t], meas_inp["Q"][n][1][t],
                                                    meas_inp["Q"][n][2][t]]
        # inp volt is calc by sec volt of trans as follow; "grid_formers[0]" as we have one connecting point
        if meas_inp["meas"] is True:
            grid_inp["grid_formers"][0]["kV"] = [meas_inp["V_CP"][0][0][t] * grid_inp["buses"][0]["U_kV"]
                                                 / (grid_inp["buses"][1]["U_kV"] * 1000),
                                                 meas_inp["V_CP"][0][1][t] * grid_inp["buses"][0]["U_kV"]
                                                 / (grid_inp["buses"][1]["U_kV"] * 1000),
                                                 meas_inp["V_CP"][0][2][t] * grid_inp["buses"][0]["U_kV"]
                                                 / (grid_inp["buses"][1]["U_kV"] * 1000)]
        grid_t.read(grid_inp)
        grid_t.pf()
        for n in range((np.size(grid_inp["buses"], 0))):
            for phi in range(3):
                meas["V"][n][phi][t] = np.abs(grid_t.buses[n]["v_" + meas_inp["phase_name"][phi] + "n"])
        for n in range(np.size(grid_inp["lines"], 0)):
            for phi in range(3):
                meas["I"][n][phi][t] = np.abs(grid_t.lines[n]["i_k_" + meas_inp["phase_name"][phi] + "_m"])
        # finding the bus index from its name, "grid_formers[0]" as we have one connecting point
        nn = find_n(grid_inp["grid_formers"][0]["bus"], grid_inp["buses"])
        # Calculating the input active and reactive powers
        meas["P_in"][t] = (grid_t.buses[nn]["p_a"] + grid_t.buses[nn]["p_b"] + grid_t.buses[nn]["p_c"])
        meas["Q_in"][t] = (grid_t.buses[nn]["q_a"] + grid_t.buses[nn]["q_b"] + grid_t.buses[nn]["q_c"])
    # Distribution grid input active and reactive powers
    figuring(grid_inp, meas_inp, meas, "Power", title)
    # Distribution grid voltages
    figuring(grid_inp, meas_inp, meas, "Voltages", title)
    # Distribution grid currents
    figuring(grid_inp, meas_inp, meas, "Currents", title)
    return meas


def figuring(grid_inp, meas_inp, meas, fig_type, title):
    """" Completed: This function is used for figuring
    """
    w, h = 10, 4
    plt.style.use('bmh')
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    times = pd.date_range('01-01-2018', periods=meas_inp["Nt"], freq='10MIN')
    x_lab = times.strftime('%H:%M')
    times = pd.date_range('01-01-2018 02:00', periods=6, freq='4H')
    x_tik = times.strftime('%H:%M')
    if fig_type == "Power":
        fig_n = 1
        Figure = {"fig_" + str(fig_n): plt.figure()}
        Figure["fig_" + str(fig_n)].set_figheight(h * 2)
        Figure["fig_" + str(fig_n)].set_figwidth(w)
        Figure["ax_" + str(fig_n) + "_1"] = Figure["fig_" + str(fig_n)].add_subplot(2, 1, 1)
        Figure["ax_" + str(fig_n) + "_2"] = Figure["fig_" + str(fig_n)].add_subplot(2, 1, 2)
        Figure["ax_" + str(fig_n) + "_1"].set_title(title, fontsize=18)
        Figure["ax_" + str(fig_n) + "_1"].plot(x_lab, - meas["P_in"][0:(meas_inp["Nt"])] / 1000,
                                               label='With Control', linewidth=2, linestyle='-')
        Figure["ax_" + str(fig_n) + "_1"].set_ylabel("$P_{CP}$ (kW)", fontsize=18)
        Figure["ax_" + str(fig_n) + "_2"].plot(x_lab, - meas["Q_in"][0:(meas_inp["Nt"])] / 1000,
                                               label='With Control', linewidth=2, linestyle='-')
        Figure["ax_" + str(fig_n) + "_2"].set_xlabel("time", fontsize=18)
        Figure["ax_" + str(fig_n) + "_2"].set_ylabel("$Q_{CP}$ (kVar)", fontsize=18)
        if meas_inp["meas"] is True:
            Figure["ax_" + str(fig_n) + "_1"].plot(x_lab, np.abs(meas_inp["P_CP"][0:(meas_inp["Nt"])]),
                                                   label='Without Control', linewidth=2, linestyle='--')
            Figure["ax_" + str(fig_n) + "_2"].plot(x_lab, np.abs(meas_inp["Q_CP"][0:(meas_inp["Nt"])]),
                                                   label='Without Control', linewidth=2, linestyle='--')
        Figure["ax_" + str(fig_n) + "_1"].legend(fontsize=18)
        Figure["ax_" + str(fig_n) + "_2"].legend(fontsize=18)
        plt.sca(Figure["ax_" + str(fig_n) + "_1"])
        plt.xticks(x_tik, x_tik)
        plt.sca(Figure["ax_" + str(fig_n) + "_2"])
        plt.xticks(x_tik, x_tik)
    elif fig_type == "Voltages":
        fig_n = 2
        Figure = {"fig_" + str(fig_n): plt.figure()}
        Figure["fig_" + str(fig_n)].set_figheight(h * np.size(grid_inp["buses"], 0))
        Figure["fig_" + str(fig_n)].set_figwidth(w)
        for nn in range((np.size(grid_inp["buses"], 0))):
            Figure["ax_" + str(fig_n) + "_" + str(nn + 1)] = Figure["fig_" + str(fig_n)].add_subplot(
                np.size(grid_inp["buses"], 0), 1, nn + 1)
            Figure["ax_" + str(fig_n) + "_" + str(nn + 1)].plot(x_lab,
                                                                (meas["V"][nn][0][0:(meas_inp["Nt"])]
                                                                 + meas["V"][nn][1][0:(meas_inp["Nt"])]
                                                                 + meas["V"][nn][2][0:(meas_inp["Nt"])]) / 3,
                                                                label='With Control', linewidth=2, linestyle='-')
            Figure["ax_" + str(fig_n) + "_" + str(nn + 1)].set_ylabel("$V_{}$".format(nn) + "(V)", fontsize=18)
            plt.sca(Figure["ax_" + str(fig_n) + "_" + str(nn + 1)])
            plt.xticks(x_tik, x_tik)
        Figure["ax_" + str(fig_n) + "_" + str(1)].set_title(title, fontsize=18)
        Figure["ax_" + str(fig_n) + "_" + str(1)].set_xlabel("time", fontsize=18)
        if meas_inp["meas"] is True:
            for n in range(meas_inp["meas_number"]):
                nn = find_n(meas_inp["meas_location"][n]["to"], grid_inp["buses"])
                Figure["ax_" + str(fig_n) + "_" + str(nn + 1)].plot(x_lab,
                                                                    (meas_inp["V_CP"][n][0][0:(meas_inp["Nt"])]
                                                                     + meas_inp["V_CP"][n][1][0:(meas_inp["Nt"])]
                                                                     + meas_inp["V_CP"][n][2][0:(meas_inp["Nt"])]) / 3,
                                                                    label='Without Control', linewidth=2,
                                                                    linestyle='--')
                Figure["ax_" + str(fig_n) + "_" + str(nn + 1)].legend(fontsize=18)
    elif fig_type == "Currents":
        fig_n = 3
        Figure = {"fig_" + str(fig_n): plt.figure()}
        Figure["fig_" + str(fig_n)].set_figheight(h * np.size(grid_inp["lines"], 0))
        Figure["fig_" + str(fig_n)].set_figwidth(w)
        for n in range(np.size(grid_inp["lines"], 0)):
            Figure["ax_" + str(fig_n) + "_" + str(n + 1)] = Figure["fig_" + str(fig_n)].add_subplot(
                np.size(grid_inp["lines"], 0), 1, n + 1)
            Figure["ax_" + str(fig_n) + "_" + str(n + 1)].plot(x_lab,
                                                               meas["I"][n][0][0:(meas_inp["Nt"])]
                                                               + meas["I"][n][1][0:(meas_inp["Nt"])]
                                                               + meas["I"][n][2][0:(meas_inp["Nt"])],
                                                               label='With Control', linewidth=2, linestyle='-')
            Figure["ax_" + str(fig_n) + "_" + str(n + 1)].set_ylabel("$I_{}$".format(n + 1) + "(A)", fontsize=18)
            plt.sca(Figure["ax_" + str(fig_n) + "_" + str(n + 1)])
            plt.xticks(x_tik, x_tik)
        Figure["ax_" + str(fig_n) + "_" + str(1)].set_title(title, fontsize=18)
        Figure["ax_" + str(fig_n) + "_" + str(1)].set_xlabel("time", fontsize=18)
        if meas_inp["meas"] is True:
            for n in range(meas_inp["meas_number"]):
                if meas_inp["meas_location"][n]["tran/line"] == "line":
                    nn = find_n(meas_inp["meas_location"][n]["to"], grid_inp["lines"])
                    Figure["ax_" + str(fig_n) + "_" + str(nn + 1)].plot(x_lab,
                                                                        np.abs(
                                                                            meas_inp["I_CP"][n][0][0:(meas_inp["Nt"])]
                                                                            + meas_inp["I_CP"][n][1][0:(meas_inp["Nt"])]
                                                                            + meas_inp["I_CP"][n][2][
                                                                              0:(meas_inp["Nt"])]),
                                                                        label='Without Control', linewidth=2,
                                                                        linestyle='--')
                    Figure["ax_" + str(fig_n) + "_" + str(nn + 1)].legend(fontsize=18)
    elif fig_type == "DA_Offers":
        fig_n = 4
        Figure = {"fig_" + str(fig_n): plt.figure()}
        Figure["fig_" + str(fig_n)].set_figheight(h * 2)
        Figure["fig_" + str(fig_n)].set_figwidth(w)
        Figure["ax_" + str(fig_n) + "_1"] = Figure["fig_" + str(fig_n)].add_subplot(2, 1, 1)
        Figure["ax_" + str(fig_n) + "_2"] = Figure["fig_" + str(fig_n)].add_subplot(2, 1, 2)
        # Figure["ax_" + str(fig_n) + "_1"].set_title(title, fontsize=18)
        Figure["ax_" + str(fig_n) + "_1"].plot(x_lab, meas["DA_PP"][0:(meas_inp["Nt"])],
                                               label='$p_{t}^{(DA)}$', linewidth=2, linestyle='-')
        Figure["ax_" + str(fig_n) + "_1"].set_ylabel("Active power (kW)", fontsize=18)
        Figure["ax_" + str(fig_n) + "_2"].plot(x_lab, meas["DA_QQ"][0:(meas_inp["Nt"])],
                                               label='$q_{t}^{(DA)}$', linewidth=2, linestyle='-')
        Figure["ax_" + str(fig_n) + "_2"].set_xlabel("time", fontsize=18)
        Figure["ax_" + str(fig_n) + "_2"].set_ylabel("Reactive power (kVAR)", fontsize=18)
        Figure["ax_" + str(fig_n) + "_1"].plot(x_lab, meas["DA_P+"][0:(meas_inp["Nt"])],
                                               label='$p_{t}^{(DA)}+r_t^{(p\u2191)}$', linewidth=2, linestyle='--')
        Figure["ax_" + str(fig_n) + "_2"].plot(x_lab, meas["DA_Q+"][0:(meas_inp["Nt"])],
                                               label='$q_{t}^{(DA)}+r_t^{(q\u2191)}$', linewidth=2, linestyle='--')
        Figure["ax_" + str(fig_n) + "_1"].plot(x_lab, meas["DA_P-"][0:(meas_inp["Nt"])],
                                               label='$p_{t}^{(DA)}-r_t^{(p\u2193)}$', linewidth=2, linestyle=':')
        Figure["ax_" + str(fig_n) + "_2"].plot(x_lab, meas["DA_Q-"][0:(meas_inp["Nt"])],
                                               label='$q_{t}^{(DA)}-r_t^{(q\u2193)}$', linewidth=2, linestyle=':')
        Figure["ax_" + str(fig_n) + "_1"].legend(fontsize=18, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.29))
        Figure["ax_" + str(fig_n) + "_2"].legend(fontsize=18, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.29))
        # Figure["ax_" + str(fig_n) + "_2"].legend().set_visible(False)
        plt.sca(Figure["ax_" + str(fig_n) + "_1"])
        plt.xticks(x_tik, x_tik)
        plt.sca(Figure["ax_" + str(fig_n) + "_2"])
        plt.xticks(x_tik, x_tik)
        plt.subplots_adjust(hspace=0.4)
    if True:
        now = datetime.now()
        tomorrow = str(now.year) + str(now.month) + str(now.day + 1)
        plt.savefig(os.path.join('Figures',
                                 title.replace('\\', '').replace('$', '').replace('.', '') + '_' + tomorrow + '.pdf'),
                    bbox_inches='tight')


def forecast_defining(pv_hist, dem_P_hist, dem_Q_hist):
    """" Not Completed: CM#0
    This function is used to make the forecast input of
    Inputs: - pv_hist
            - dem_P_hist
            - dem_Q_hist
    Output: a dictionary gives the forecast values of the PV and demands
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
    for n in range(dem_P_hist.shape[0]):
        fore_inp["Dem_P"][n] = -(dem_P_hist[n][0] + dem_P_hist[n][1] + dem_P_hist[n][2]) / 3  # in [kW]
        fore_inp["Dem_Q"][n] = (dem_Q_hist[n][0] + dem_Q_hist[n][1] + dem_Q_hist[n][2]) / 3  # in [kW]
        fore_inp["Dem_P_zeta+"][n] = 0 * fore_inp["Dem_P"][n]
        fore_inp["Dem_P_zeta-"][n] = 0 * fore_inp["Dem_P"][n]
        fore_inp["Dem_Q_zeta+"][n] = 0 * fore_inp["Dem_Q"][n]
        fore_inp["Dem_Q_zeta-"][n] = 0 * fore_inp["Dem_Q"][n]
    return fore_inp


def rt_simulation(grid_inp, meas_inp, fore_inp, DA_result, t):
    _ = meas_inp
    rt_meas_inp = {}
    rt_meas_inp["delta"] = DA_result["delta"]
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
            fore_inp["confidence"]) \
                                 - (fore_inp["Dem_P_zeta+"][d][t] + fore_inp["Dem_P_zeta-"][d][t]) * a
        a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
        rt_meas_inp["DM_Q"][d] = fore_inp["Dem_Q"][d][t] + fore_inp["Dem_Q_zeta+"][d][t] / norm.ppf(
            fore_inp["confidence"]) \
                                 - (fore_inp["Dem_Q_zeta+"][d][t] + fore_inp["Dem_Q_zeta-"][d][t]) * a
    rt_meas_inp["ST_SOC_des"] = np.zeros(np.size(grid_inp["storage_elements"], 0))
    rt_meas_inp["ST_SOC_t_1"] = np.zeros(np.size(grid_inp["storage_elements"], 0))
    for s in range(np.size(grid_inp["storage_elements"], 0)):
        rt_meas_inp["ST_SOC_des"][s] = DA_result["Solution_ST_SOC"][s][t]
        if t == 0:
            a = max([-b, min([b, gauss(0, 1)])]) / norm.ppf(fore_inp["confidence"])
            rt_meas_inp["ST_SOC_t_1"][s] = \
                (fore_inp["ST_SOC_0"] + fore_inp["ST_SOC_zeta+"] / norm.ppf(fore_inp["confidence"])
                 * (fore_inp["ST_SOC_zeta+"] + fore_inp["ST_SOC_zeta-"]) * a) \
                * grid_inp["storage_elements"][s]["SOC_max_kWh"]
        else:
            rt_meas_inp["ST_SOC_t_1"][s] = DA_result["Solution_ST_SOC_RT"][s]
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
        rt_meas_inp["ConPoint_P_DA_EN"][f] = DA_result["DA_P"][t]
        rt_meas_inp["ConPoint_Q_DA_EN"][f] = DA_result["DA_Q"][t]
        rt_meas_inp["ConPoint_P_DA_RS_pos"][f] = DA_result["DA_RP_pos"][t]
        rt_meas_inp["ConPoint_P_DA_RS_neg"][f] = DA_result["DA_RP_neg"][t]
        rt_meas_inp["ConPoint_Q_DA_RS_pos"][f] = DA_result["DA_RQ_pos"][t]
        rt_meas_inp["ConPoint_Q_DA_RS_neg"][f] = DA_result["DA_RQ_neg"][t]
    return rt_meas_inp


def cases_load_sim(Cases_Name, Scen_num):
    data = {}
    l_case = 0
    for case in Cases_Name:
        data[l_case] = {}
        data[l_case]['MonteCarlo_Scen'] = Scen_num
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


def find_n(bus_name, buses):
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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
