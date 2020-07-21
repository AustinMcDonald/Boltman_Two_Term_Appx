
import numpy as np
from scipy import interpolate
from scipy import integrate
import pandas as pd


Td2Vcm2 = 1e-17;              # conversion of Townsend to V/cm2


def density_convert(Obj):
    Obj.NDL = Obj.N*Obj.DL/1e+22*1e-2
    Obj.NDT = Obj.N*Obj.DT/1e+22*1e-2
    Obj.NMU = Obj.vd * Obj.N / (Obj.E) / 100 / 1e+22
    Obj.TD  = Obj.E/100 * Obj.P/(Td2Vcm2 * Obj.N * 1e-6)
    return True

    
class TTA_class:
    def __init__(self):
        """
        Fill the empty variables. 
        """
        self.No = 0
        self.m  = 0
        self.kB = 0
        self.kT = 0

        self.e_     = 0
        self.edNde_ = 0
        self.E      = 0
        self.Qm     = 0

        self.N  = 0
        self.K0 = 0

        self.dNde = np.array([])
        self.fe   = np.array([])
        self.eKT  = np.array([])




def Kint_A(e__,  Qm_interp_master, K0, isthermOFF):
    Qm_interp = Qm_interp_master(e__)
    K         = K0/((Qm_interp**2)*e__)
    Kint      = 1/(K + 1)
    if(isthermOFF):
        Kint = 1/K
    return Kint

def moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def dNdeCALC(Obj):

    feEXP = -999*np.ones(len(Obj.edNde_))
    for i in range(1,len(Obj.edNde_)):

        X = Obj.edNde_[Obj.edNde_<=Obj.edNde_[i]]
        Y = Kint_A(X,  Obj.Qm_interp_master, Obj.K0, Obj.isthermOff)
        feEXP[i] = integrate.trapz(Y, X)

    #Define output variables
    feEXP[0] = 0
    fe      = np.exp(-feEXP)
    dNde    = np.sqrt(Obj.edNde_)*fe
    eKT     = Obj.edNde_ # think i can kill this

    #Normalization
    dNdeInt = integrate.trapz(dNde,Obj.edNde_)
    fe      = fe/dNdeInt
    dNde    = dNde/dNdeInt

    Obj.dNde = dNde
    Obj.fe   = fe
    Obj.eKT  = eKT
    
    return dNde, fe, eKT

def vdCALC_optim(Obj):

    dfede  = np.diff(Obj.fe)/np.diff(Obj.eKT)
    edfede = moving_average(Obj.eKT,2) 
    dfede  = interpolate.interp1d(edfede, dfede,  kind='linear', fill_value="extrapolate")
    dfede  = dfede(Obj.eKT)

    Obj.dfede_interp_master = interpolate.interp1d(Obj.edNde_, dfede,  kind='linear')
    Obj.fe_interp_master    = interpolate.interp1d(Obj.edNde_, Obj.fe,  kind='linear')

    emax = Obj.e_[-1]                                                  

    X = Obj.edNde_[Obj.edNde_<=emax]
    Y = vdint(X, Obj.Qm_interp_master, Obj.dfede_interp_master)
    tmp = integrate.trapz(Y,X)
    vd = - 1/(3*Obj.N) * Obj.E/Obj.kT * ((2*Obj.kT/Obj.m)**0.5) * tmp

    Obj.vd = vd

    return vd

def vdint(e__, Qm_interp_master, dfede_interp_master):
    Qm_interp = Qm_interp_master(e__)
    dfede_interp = dfede_interp_master(e__)    
    vdint        = (e__*dfede_interp)/Qm_interp
    return vdint


def DTCALC(Obj):
    emax = Obj.e_[-1]   
    
    X = Obj.edNde_[Obj.edNde_<=emax]

    Y = DTint(X, Obj.Qm_interp_master, Obj.fe_interp_master)

    tmp = integrate.trapz(Y,X)
    DT = 1/3 * ((2*Obj.kT/Obj.m)**0.5) * 1/Obj.N * tmp
    
    Obj.DT = DT
    return DT


def DTint(e__, Qm_interp_master, fe_interp_master):
    Qm_interp = Qm_interp_master(e__)
    fe_interp = fe_interp_master(e__)    
    Dint        = (e__*fe_interp)/Qm_interp
    return Dint



def DLCALC(Obj):

    emax = Obj.e_[-1] 
                                                  
    Obj.K1 = 3 * Obj.kT * Obj.vd * Obj.N/Obj.E * (Obj.m/Obj.kT/2)**0.5  #[m^-1 s^-1]
    Obj.K2 = (2*Obj.kT/Obj.m)**0.5 * 1/(3*Obj.N) * Obj.E/Obj.kT      #[m^2  s^-1]

    f1_ = np.zeros(len(Obj.edNde_))
    
    for i in range(1,len(Obj.edNde_)):
        f1_[i]   = f1(Obj.edNde_[i], Obj)
        

    df1de  = np.diff(f1_)/np.diff(Obj.eKT)
    edfede = moving_average(Obj.eKT,2) 
    df1de  = interpolate.interp1d(edfede, df1de,  kind='linear', fill_value="extrapolate")
    df1de  = df1de(Obj.eKT)
    
    X = np.linspace(0,emax,500)
    Y = argDT(X, f1_, df1de, Obj)
    integrand = integrate.trapz(Y,X)

    DL = Obj.DT + integrand
    Obj.DL = DL

    return DL


def f1(e__, Obj):
    fe_interp = Obj.fe_interp_master(e__)
    
    X = Obj.edNde_[Obj.edNde_<=e__]
    Y = argF1(X, Obj)
    tmp = integrate.trapz(Y,X)
    f1  = -fe_interp/Obj.E * Obj.kT * tmp
    return f1

def argF1(e__, Obj):
    Qm_interp = Obj.Qm_interp_master(e__)
    fe_interp = Obj.fe_interp_master(e__)
    argF1     = np.zeros(len(e__))  
    
    K = Obj.K0/((Qm_interp**2)*e__)
    K[0]=0
    Kint = 1/(1/K + 1)
    if(Obj.isthermOff):
        Kint = np.ones(len(e__))# not sure if first element should be 0
        
    for i in range(1,len(e__)):
        X = Obj.edNde_[Obj.edNde_<=e__[i]]
        Y = arg_arg_F1(X, Obj)
        tmp = integrate.trapz(Y,X)
        
        argF1[i] = Kint[i] * (1 + Qm_interp[i]/(e__[i]*fe_interp[i])*tmp)
    return argF1

def arg_arg_F1(e__, Obj):
    Qm_interp = Obj.Qm_interp_master(e__)
    fe_interp = Obj.fe_interp_master(e__)
    dfede_interp = Obj.dfede_interp_master(e__)
                              
    arg_arg_F1   = (e__ * dfede_interp) / Qm_interp +  Obj.K1 * e__**(1/2)* fe_interp
    return arg_arg_F1

def argDT(e__, f1, df1de, Obj):
    Qm_interp = Obj.Qm_interp_master(e__)

    f1_interp    = interpolate.interp1d(Obj.edNde_, f1,  kind='linear', fill_value="extrapolate")
    f1_interp    = f1_interp(e__)

    df1de_interp    = interpolate.interp1d(Obj.edNde_, df1de,  kind='linear',fill_value="extrapolate")
    df1de_interp    = df1de_interp(e__)
    
    argDT = Obj.K2 * e__/Qm_interp*df1de_interp + Obj.vd*(e__**0.5)*f1_interp
    return argDT


