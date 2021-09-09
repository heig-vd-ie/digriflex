"""@author: MYI, #Python version: 3.6.8 [32 bit]"""
#### Importing packages
from DiGriFlex_RT import interface_control_digriflex
# from DiGriFlex_DA import dayahead_digriflex
import os

# dayahead_digriflex(1)
dir_path = r"C:/Users/" + os.environ.get('USERNAME') + r"/Desktop/DiGriFlex_Code"
Vec_Inp0 = open(dir_path + r"\Data\test_douglas_interface.txt", encoding="ISO-8859-1").read().splitlines()
Vec_Out0 = interface_control_digriflex(Vec_Inp0)
print(Vec_Out0)
