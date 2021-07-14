"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
import os
import sys
import pickle


#### Defining meta parameters
dir_path = r"C:\Users\mohammad.rayati\Desktop\DiGriFlex_Code"  # Defining directory path of the code
# dir_path = r"C:\Users\labo-reine-iese\Desktop\DiGriFlex_Code"
network_name = "Case_4bus_DiGriFlex"  # Defining the network
# network_name = "Case_LabVIEW"
python64_path = r"C:\Users\mohammad.rayati\AppData\Local\Programs\Python\Python39\python.exe"
# python64_path = r"C:\Users\labo-reine-iese\AppData\Local\Programs\Python\Python39\python.exe"


#### Temporary variables for dayahead scheduling (CM#5 and CM#6 and CM#1)
P_SC, Q_SC = [7] * 144, [2] * 144
RPP_SC, RPN_SC, RQP_SC, RQN_SC = [0.4] * 144, [0.4] * 144, [0.2] * 144, [0.2] * 144
SOC_dersired = [10] * 144
prices_vec = [0, 1, 100, 1000]


file_to_store = open(dir_path + r"/Data/tmp_da.pickle", "wb")
pickle.dump(P_SC, file_to_store)
pickle.dump(Q_SC, file_to_store)  # Ligne_U[0]
pickle.dump(RPP_SC, file_to_store)
pickle.dump(RPN_SC, file_to_store)
pickle.dump(RQP_SC, file_to_store)
pickle.dump(RQN_SC, file_to_store)
pickle.dump(SOC_dersired, file_to_store)
pickle.dump(prices_vec, file_to_store)
file_to_store.close()
