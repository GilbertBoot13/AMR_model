"""
we will create the basic structure for the 6 classes in the model
(S) susceptible cells (no ARG on chromosome) + no plasmid
N
(S_P0) susceptible cells (no ARG on chromosome) + plasmid with no ARG
Y
(R_P1) resistant cells with no ARG on chromosome + plasmid with ARG
Y
(R_C) resistant cells with ARG on chromosome (no plasmid)

(R_CP0) resistant cells with ARG on chromosome + plasmid with no ARG

(R_CP1) resistant cells with ARG on chromosome + plasmid with ARG

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def hill_E(c, Emax, H, MIC):
    return 1 - Emax * c**H / (MIC**H + c**H)


c = 2
Emax, H = 2.0, 2.0
MICs, MICr = 8.0, 2000.0

Es = hill_E(c, Emax, H, MICs)
Er = hill_E(c, Emax, H, MICr)


def amrmodel(x, t, params):
    r, Nmax, Es, Er, deltaMS, deltaMR, beta, alpha_P,alpha_Rp, alpha_Rc, epsil = params
    S, S_P0, R_P1, R_C, R_CP0,R_CP1 = x

    N = S, S_P0, R_P1, R_C, R_CP0,R_CP1
    
    growth = r * (1 - N / Nmax)
    conjSxS_P0 = beta * S * (S_P0 +R_CP0) / N
    
    conjSxR_P1 = beta * S * (R_P1 +R_CP1) / N
    
    conjR_CxR_CP0 = beta * R_C * (S_P0 +R_CP0) / N
    
    conjR_CxR_CP1 = beta * R_C * (R_P1 +R_CP1) / N
    
    segrS_P0xS = growth * (1 - (alpha_P)) * Es * S_P0
    
    segrR_P1xS = growth * (1 - (alpha_P + alpha_Rp)) * Er * R_P1
    
    segrR_CP0xR_C = growth * (1 - (alpha_P + alpha_Rc)) * Er * R_CP0
    
    segrR_CP1xR_C = growth * (1 - (alpha_P + alpha_Rc + alpha_Rp)) * Er * R_CP1

    dS = (growth * Es * S
          - deltaMS * S
          - conjSxS_P0
          - conjSxR_P1
          + epsil * segrS_P0xS
          + epsil * segrR_P1xS)

    dS_P0 = ((1 - epsil) * segrS_P0xS
          - deltaMR * S_P0
          + conjSxS_P0)
    
    dR_P1 = ((1 - epsil) * segrR_P1xS
          - deltaMR * R_P1
          + conjSxR_P1)
    
    dR_C = (growth * Er * R_C
          - deltaMS * R_C
          - conjR_CxR_CP0
          - conjR_CxR_CP1
          + epsil * segrR_CP0xR_C
          + epsil * segrR_CP1xR_C)
    
    dR_CP0 = ((1 - epsil) * segrR_CP0xR_C
          - deltaMR * R_CP0
          + conjR_CxR_CP0)
    
    dR_CP1 = ((1 - epsil) * segrR_CP1xR_C
          - deltaMR * R_CP1
          + conjR_CxR_CP1)

    return [dS, dS_P0, dR_P1, dR_C, dR_CP0, dR_CP1]


r = 0.5
Nmax = 6.71e7
deltaMR = 0.025
deltaMS = deltaMR * 1.02
beta = 0.001
alpha_P = 0
alpha_Rp = 0
alpha_Rc = 0
epsil = 0.000144

params = [r, Nmax, Es, Er, deltaMS, deltaMR, beta,
           alpha_P,alpha_Rc, alpha_Rp, epsil]

x0 = [100, 100]

maxt = 500.0
tstep = 0.1
t = np.arange(0, maxt, tstep)

out = odeint(amrmodel, x0, t, args=(params,))

print("Es =", Es, " Er =", Er)
print("final S, R =", out[-1])
print("final resistant fraction =", out[-1, 1] / out[-1].sum())
print("final N / Nmax =", out[-1].sum() / Nmax)
print("min value anywhere =", out.min())

plt.plot(t, out[:, 0], label='Sensitive cells')
plt.plot(t, out[:, 1], label='Resistant cells')
plt.legend()
plt.xlabel('time (h)')
plt.ylabel('cells')
plt.savefig('amr_check.png', dpi=110)