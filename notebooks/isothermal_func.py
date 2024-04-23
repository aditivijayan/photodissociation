import numpy as np
import numexpr as ne
import math
import scipy.integrate as integrate
import h5py
from functions import *
#------------Functions required for estimating mass flux for Isothermal Mass Distribution--------------#

def IsoUax_int(a, gamma, x):
    Ua_x = ((gamma*np.exp(-x)-1)*np.log(a))**(0.5 )   #intermediate
    return Ua_x

def IsoUax_fSA(a, gamma, x):
    Ua_x = (gamma*np.exp(-x)*(a-1.) - np.log(a))**(0.5 )   #fixed solid angle
    return Ua_x


def IsodMassFlux_HI(x, a, gamma, xi, M, p):
    pm = getPm(x, M)    
    if(p==1 ):
        U_ax    = IsoUax_int(a, gamma, x) 
        fHI     = IsofHI_integrand(a, gamma, x, xi, 1) 
    elif(p==2):
        U_ax    = IsoUax_fSA(a, gamma, x) 
        fHI     = IsofHI_integrand(a, gamma, x, xi, 2) 
    if(fHI>1.0):
        fHI = 1.
    return (pm  *  fHI)
    
        
def IsodMassFlux_tot(x, gamma, M, p):
    pm = getPm(x, M)
    return (pm)

def IsofHI_integrand(a, gamma, x, xi, p):
    prefac = xi * np.exp(-x) 
    if(p==1):
        Uax = IsoUax_int(a, gamma, x)
    elif(p==2):
        Uax = IsoUax_fSA(a, gamma, x)
        
    integrand = prefac * (a**(p-2))/Uax
    return integrand
