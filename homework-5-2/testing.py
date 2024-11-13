
from H_2pz_overlap import psi_2p_z

print(psi_2p_z(1,1,1))

import os
os.chdir(os.getcwd().split("homework-5-2")[0]+"homework-4-1")
from compute_work_adiabatic import pressure_adi

os.chdir(os.getcwd().split("homework-4-1")[0]+"homework-5-2")

print(pressure_adi(2,1,1.4,1,300))