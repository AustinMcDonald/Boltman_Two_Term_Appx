import numpy as np
from scipy import interpolate
from scipy import integrate
import time
import pandas as pd

import matplotlib.pyplot as plt


from TTA_Functions import *
from MERT_6_PLUS_Functions import *

e0_Xe  = e_Xe[e_Xe<10]
Qm0_Xe = Qm_Xe[e_Xe<10]


M6P = MERT_6_PLUS()

M6P.A  = -5.13
M6P.D  = 909
M6P.F  = -2336
M6P.G  = 1093
M6P.A1 = 22.3
M6P.H  = 53.9
M6P.A2 = 12.9

M6P.k = np.sqrt(e0_Xe/13.065)/M6P.alpha_0


nscale_Xe=1
edNde_Xe = []
ne  = np.linspace(0,100,101,endpoint=True)* nscale_Xe
edNde1 = np.log(0.001) + (np.log(0.2) - np.log(0.001)) * ne/(100*nscale_Xe)
edNde1 = np.exp(edNde1)
ne  = np.linspace(1,35,35,endpoint=True)* nscale_Xe
edNde2 = np.log(0.2)   + (np.log(1) - np.log(0.2)) * ne/(35*nscale_Xe)
edNde2 = np.exp(edNde2)
ne  = np.linspace(1,50,50,endpoint=True)* nscale_Xe
edNde3 = np.log(1)   + (np.log(max(e0_Xe)) - np.log(1)) * ne/(50*nscale_Xe)
edNde3 = np.exp(edNde3)

edNde_Xe.append([0])
edNde_Xe.append(edNde1)
edNde_Xe.append(edNde2)
edNde_Xe.append(edNde3)
edNde_Xe  = np.array([item for sublist in edNde_Xe for item in sublist])

M_Xe  = 131.293*0.9315*1e+9 # [eV]
E_scan_Xe = [0.08, 0.25, 0.5, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 35, 50, 75, 100]

xx=30

c  = 3e+8;                        # [m/s]
kB = 8.617*1e-5;                  # [eV/K]
Td2Vcm2 = 1e-17;              # conversion of Townsend to V/cm2
To = 293.15
Po = 1
Zo = 1



TTA = TTA_class()

TTA.P = 1
TTA.T = 300
TTA.Z = 1
TTA.isthermOff = 1*0
TTA.E      = E_scan_Xe[xx]*100
TTA.No = 2.687e+25 * 273/293 * 1.013 # number of molecules per unit volume at 20 deg, 1 atm [m^-3]
TTA.m  = (0.511*1e+6)/c**2           # [eV/c^2]
TTA.kT = kB*TTA.T                        # [eV]
TTA.M  = M_Xe/c**2
TTA.N  = TTA.No * 293/TTA.T * TTA.P/1.013 * TTA.Z
TTA.K0 = 1/6 * (TTA.M)/(TTA.m) * (TTA.E/TTA.N)**2 * 1/(TTA.kT**2)
TTA.edNde = edNde_Xe
TTA.edNde_ = TTA.edNde/TTA.kT                # [a.u.]

TTA.e_     = e0_Xe/TTA.kT                    # [a.u.]


#TTA.Qm     = Qm0_Xe*1e-20                # [m^2]
TTA.Qm  = WEIGHT_Q(e0_Xe, Q_MT(M6P), Qm0_Xe, 0.9, 1)*1e-20

TTA.Qm_interp_master = interpolate.interp1d(TTA.e_, TTA.Qm,  kind='linear', fill_value="extrapolate")


####################################################
####################################################

t0 = time.time()

dNdeCALC(TTA)
vv = vdCALC_optim(TTA)
dt = DTCALC(TTA)
dl = DLCALC(TTA)

t1 = time.time()
print("Time, ",t1-t0)


print(vv)
print(dt)
print(dl)

""" t0 = time.time()

DifL = []
VDt = []
DifT = []
TD = []

for E_scan in E_scan_Xe:
    TTA.E      = E_scan*100
    TTA.K0 = 1/6 * (TTA.M)/(TTA.m) * (TTA.E/TTA.N)**2 * 1/(TTA.kT**2)

    dNdeCALC(TTA)
    vv = vdCALC_optim(TTA)
    dt = DTCALC(TTA)
    dl = DLCALC(TTA)
    density_convert(TTA)

    DifL.append(TTA.NDL)
    DifT.append(TTA.NDT)
    VDt.append(TTA.NMU)
    TD.append(TTA.TD)

t1 = time.time()
print("Time elapsed, ",t1-t0) """
