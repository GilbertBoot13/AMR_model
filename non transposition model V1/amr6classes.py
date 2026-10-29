"""
Six-class ODE model: antibiotic resistance gene (ARG) carriage on plasmid vs chromosome.

Combining the slurry-tank
model of Baker et al. (2016) with the metal
co-selection model of Arya et al. (2021), and splits the
single resistant class of both papers into four, so that the same ARG can be carried
chromosomally, on a plasmid, or both.

Classes
    S      no chromosomal ARG, no plasmid
    S_P0   no chromosomal ARG, plasmid without ARG
    R_P1   no chromosomal ARG, plasmid carrying ARG
    R_C    chromosomal ARG, no plasmid
    R_CP0  chromosomal ARG, plasmid without ARG
    R_CP1  chromosomal ARG, plasmid carrying ARG

Taken unchanged from the source papers
    Logistic growth r(1 - N/Nmax)                              Baker Eq 1-2, Arya Eq 1-2
    Frequency-dependent conjugation beta*recipients*donors/N   Baker Eq 1-2, Arya Eq 1-2
    Antibiotic effect via Hill function Es, Er                 Baker Eq 7-8
    Segregational plasmid loss epsilon                         Arya Eq 1-2, Table 1
    Differential death rate from metal co-selection            Arya Eq 1-3

Changed or added here
    1. No inflow. Baker's slurry inflow, growing tank volume and antibiotic decay
       (Baker Eq 4-6) are dropped, so this is a closed population. Antibiotic
       concentration c is therefore a fixed parameter rather than A(t)/V(t).
    2. zeta replaces Arya Eq 3. Arya converts a metal concentration M into a death
       rate ratio; here zeta = deltaMS/deltaMR is set directly, so no metal-specific
       parameters are needed. 
    3. Metal resistance gene sits on the plasmid backbone, so both P0 and P1 confer
       it.So R_C is antibiotic-resistant but metal-sensitive.
    4. Cost of resistance alpha is split three ways: alpha_P (plasmid carriage),
       alpha_Rp (ARG expressed from plasmid), alpha_Rc (ARG expressed from chromosome).
    5. Plasmid-bearing cells cannot receive a second
       plasmid (entry exclusion, and P0/P1 share a backbone so are incompatible).
       Only S and R_C are recipients.
    6. No transposition.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

LABELS = ['S', 'S_P0', 'R_P1', 'R_C', 'R_CP0', 'R_CP1']


def hill_E(c, Emax, H, MIC):
    #Baker Eq 7-8. Growth multiplier under antibiotic concentration c.
     return 1 - Emax * c**H / (MIC**H + c**H)


def amrmodel(x, t, p):
    # Parameters are looked up by name, so ordering cannot go wrong.
    r, Nmax = p['r'], p['Nmax']
    Es, Er = p['Es'], p['Er']
    deltaMS, deltaMR = p['deltaMS'], p['deltaMR']
    beta, epsil = p['beta'], p['epsil']
    alpha_P, alpha_Rp, alpha_Rc = p['alpha_P'], p['alpha_Rp'], p['alpha_Rc']

    S, S_P0, R_P1, R_C, R_CP0, R_CP1 = x

    N = S + S_P0 + R_P1 + R_C + R_CP0 + R_CP1
    growth = r * (1 - N / Nmax)

    # --- Conjugation -------------------------------------------------------
    # Donor pools, grouped by plasmid cargo. Ex. R_CP0 donates P0 and R_CP1
    # donates P1.
    donP0 = S_P0 + R_CP0
    donP1 = R_P1 + R_CP1

    # Only plasmid-free cells (S, R_C) are recipients.
    conj_S_P0  = beta * S   * donP0 / N
    conj_S_P1  = beta * S   * donP1 / N
    conj_RC_P0 = beta * R_C * donP0 / N
    conj_RC_P1 = beta * R_C * donP1 / N

    # --- Division of plasmid-bearing classes -------------------------------
    # Each class divides at the logistic rate, reduced by its own costs and by
    # its antibiotic multiplier (Es if antibiotic-sensitive, Er if resistant).
    # A fraction epsil of these divisions produces a plasmid-free daughter
    # (segregational loss); the remaining (1 - epsil) stay in the class. 
    #
    # Cost structure :
    #   S_P0   plasmid only               
    #   R_P1   plasmid + ARG on plasmid
    #   R_CP0  plasmid + ARG on chromosome
    #   R_CP1  carries two ARG copies. max() rather than a sum assumes expression
    #          is regulated to a single-gene level, so the cell pays the larger of
    #          the two ARG costs, not both. Switch to a sum for additive dosage.
    div_S_P0  = growth * (1 - alpha_P)                            * Es * S_P0
    div_R_P1  = growth * (1 - alpha_P - alpha_Rp)                 * Er * R_P1
    div_R_CP0 = growth * (1 - alpha_P - alpha_Rc)                 * Er * R_CP0
    div_R_CP1 = growth * (1 - alpha_P - max(alpha_Rc, alpha_Rp))  * Er * R_CP1

    # --- Equations ---------------------------------------------------------
    # Death: deltaMS for metal-sensitive classes (S, R_C), deltaMR for every
    # plasmid-bearing class, since the metal resistance gene is on the plasmid
    # backbone. 

    # S: grows under antibiotic penalty Es, dies at the metal-sensitive rate,
    # loses cells to conjugation, and regains them when S_P0 or R_P1 segregate.
    dS = (growth * Es * S
          - deltaMS * S
          - conj_S_P0 - conj_S_P1
          + epsil * (div_S_P0 + div_R_P1))

    dS_P0 = ((1 - epsil) * div_S_P0
             - deltaMR * S_P0
             + conj_S_P0)

    dR_P1 = ((1 - epsil) * div_R_P1
             - deltaMR * R_P1
             + conj_S_P1)

    # R_C: antibiotic-resistant (Er) but metal-sensitive (deltaMS). 
    dR_C = (growth * (1 - alpha_Rc) * Er * R_C
            - deltaMS * R_C
            - conj_RC_P0 - conj_RC_P1
            + epsil * (div_R_CP0 + div_R_CP1))

    dR_CP0 = ((1 - epsil) * div_R_CP0
              - deltaMR * R_CP0
              + conj_RC_P0)

    dR_CP1 = ((1 - epsil) * div_R_CP1
              - deltaMR * R_CP1
              + conj_RC_P1)

    return [dS, dS_P0, dR_P1, dR_C, dR_CP0, dR_CP1]


if __name__ == "__main__":
    from massbalancecheck import check_mass_balance


    c = 2.0                    #Baker's concentration is of this order
                               #(theta/(gamma*V) ~ 1-10 ug/L), will sweep.
   
    Emax, H = 2.0, 2.0         # From Baker et al. (2016) 
    MICs, MICr = 8.0, 2000.0   # From Baker et al. (2016) 

    deltaMR = 0.025            # From Arya et al. (2021) 
    zeta = 1.20                # swept 1 - 10 in Arya et al. (2021), 
                               # models a weak death rate currently
    params = dict(
        r=0.5,               # From Baker et al. (2016)
        Nmax=6.71e7,         # From Arya et al. (2021)
        Es=hill_E(c, Emax, H, MICs),
        Er=hill_E(c, Emax, H, MICr),
        deltaMR=deltaMR,             
        deltaMS=deltaMR * zeta,
        beta=0.001,                   # From Baker et al. (2016) 
        
        alpha_P=0.07,        # both papers take sum cost of resistance on plasmid 
        alpha_Rp=0.03,       # at 0.1, split values are assumptions to test.
        
        alpha_Rc=0.015,      # assummed at 1/2 of alpha_Rp since plasmids
                             # can contain many copies of R gene.
        
        epsil=0.000144,        # From Arya et al. (2021)
    )

    # conjugation and segregation only move cells between classes. A residual
    # above floating-point noise means a flow term is dropped or duplicated.
    # a test for bugs.
    
    print("mass balance residual = %.3e" % check_mass_balance(params, amrmodel))

    x0 = [100.0] * 6          # equal start: no class favoured by initial conditions
    t = np.arange(0, 4000, 0.5)
    out = odeint(amrmodel, x0, t, args=(params,))

    print("Es = %.4f   Er = %.4f" % (params['Es'], params['Er']))
    print("min value anywhere = %.4g" % out.min())   # negative => invalid run
    print("final N / Nmax = %.4f" % (out[-1].sum() / params['Nmax']))
    print("\nfinal state:")
    tot = out[-1].sum()
    for lab, v in zip(LABELS, out[-1]):
        print("  %-6s %12.4e   (%.4f)" % (lab, v, v / tot))

    for i, lab in enumerate(LABELS):
        plt.plot(t, out[:, i], label=lab)
    # log scale: classes span many orders of magnitude
    plt.yscale('log'); plt.legend(); plt.xlabel('time (h)'); plt.ylabel('cells')
    plt.savefig('amr6.png', dpi=110)