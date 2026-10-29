"""Mass-balance check for the six-class AMR model.

Conjugation and segregation only move cells between classes, so they must
cancel when the six derivatives are summed. Whatever is left over has to equal
total births minus total deaths. A non-zero residual means a bookkeeping
error: a flow added to one equation but not subtracted from another, a wrong
epsil split, a duplicated term, or a wrong per-class growth term.
"""
import numpy as np


def expected_net(x, p):
    """Births minus deaths, written out independently of the model."""
    r, Nmax = p['r'], p['Nmax']
    Es, Er = p['Es'], p['Er']
    deltaMS, deltaMR = p['deltaMS'], p['deltaMR']
    alpha_P, alpha_Rp, alpha_Rc = p['alpha_P'], p['alpha_Rp'], p['alpha_Rc']

    S, S_P0, R_P1, R_C, R_CP0, R_CP1 = x

    N = S + S_P0 + R_P1 + R_C + R_CP0 + R_CP1
    growth = r * (1 - N / Nmax)

    births = growth * (
        Es * S
        + (1 - alpha_P) * Es * S_P0
        + (1 - alpha_P - alpha_Rp) * Er * R_P1
        + (1 - alpha_Rc) * Er * R_C
        + (1 - alpha_P - alpha_Rc) * Er * R_CP0
        + (1 - alpha_P - max(alpha_Rc, alpha_Rp)) * Er * R_CP1
    )
    deaths = deltaMS * (S + R_C) + deltaMR * (S_P0 + R_P1 + R_CP0 + R_CP1)
    return births - deaths


def check_mass_balance(p, model, n=500, seed=0, tol=1e-9):
    """Raise AssertionError if the model's flows don't balance.

    Costs must be non-zero, or the per-class growth terms aren't exercised
    and a wrong cost cannot show up.
    """
    if p['alpha_P'] == p['alpha_Rp'] == p['alpha_Rc'] == 0:
        raise ValueError("set non-zero costs before testing, or growth terms are untested")

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        x = rng.uniform(0.0, 2e7, 6)
        total = sum(model(x, 0.0, p))
        residual = total - expected_net(x, p)
        scale = max(abs(total), 1.0)          # relative: values are ~1e6
        worst = max(worst, abs(residual) / scale)

    assert worst < tol, "mass balance violated: relative residual %.3e" % worst
    return worst