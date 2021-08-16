"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
#### To Do lists:
## - CM#0: The input of Cinergia must be equal to the load of school multiplied by 0.1
## - CM#0: Figuring the output of the codes using jupyter notebook and the package plotly (using the csv and pickle).
## - CM#0: DiGriFlex_sim, Main_file
## - CM#0: @AuxiliaryFunctions: R and X of lines, efficiencies data of battery, X of PVs (function "reine_parameters()")
## - CM#1: Finding power output from the irradiance and temperature
## - CM#2: Embedding the code of Pasquale for day_ahead forecasting


#### Importing packages
from DiGriFlex_RT import interface_control_digriflex
from DiGriFlex_DA import dayahead_digriflex


dayahead_digriflex(0.9)

# Vec_Inp0 = open(dir_path + r"\Data\test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
# Vec_Out0 = interface_control_digriflex(Vec_Inp0)
# print(Vec_Out0)
