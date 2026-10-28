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
we are working from the previous model created by Arya et al 2021
i believe this does not include seperate effects for ARG on chromosomes
vs plasmids.

We can start with the susceptible cell since that contains ARG on neither
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def hill_E(c, Emax, H, MIC):
    return 1 - Emax * c**H / (MIC**H + c**H)


c = 10.0
Emax, H = 2.0, 2.0
MICs, MICr = 8.0, 2000.0

Es = hill_E(c, Emax, H, MICs)
Er = hill_E(c, Emax, H, MICr)


def amrmodel(x, t, params):
    r, Nmax, Es, Er, deltaS, deltaR, beta, alpha, epsil = params
    S, R = x

    N = S + R
    growth = r * (1 - N / Nmax)
    conj = beta * S * R / N
    divR = growth * (1 - alpha) * Er * R

    dS = (growth * Es * S
          - deltaS * S
          - conj
          + epsil * divR)

    dR = ((1 - epsil) * divR
          - deltaR * R
          + conj)

    return [dS, dR]


r = 0.5
Nmax = 6.71e7
deltaR = 0.025
deltaS = deltaR * 3
beta = 0.001
alpha = 0.1
epsil = 0.000144

params = [r, Nmax, Es, Er, deltaS, deltaR, beta, alpha, epsil]

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