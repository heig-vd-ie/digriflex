"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
#### To Do lists:
## - CM#0: The input of Cinergia must be equal to the load of school multiplied by fac_P, fac_Q = 0.1, 0.1
## - CM#0: @AuxiliaryFunctions: R and X of lines, efficiencies data of battery, X of PVs (function "reine_parameters()")
## - CM#0: For the test, I consider two things for SoC: first minimum SoC in Aux. Functions, then result_SoC in DiGriFlex_DA
## - CM#0: I seperate simulation and test files in the folder "Result"
## - CM#1: Finding power output from the irradiance and temperature
## - CM#2: Embedding the code of Pasquale for day_ahead forecasting


#### Importing packages
from DiGriFlex_RT import interface_control_digriflex
# from DiGriFlex_DA import dayahead_digriflex
import os

# dayahead_digriflex(1)
dir_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/Desktop/DiGriFlex_Code"
Vec_Inp0 = open(dir_path + r"\Data\test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
Vec_Out0 = interface_control_digriflex(Vec_Inp0)
print(Vec_Out0)
