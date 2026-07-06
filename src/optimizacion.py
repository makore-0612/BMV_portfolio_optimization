import numpy as np
import pandas as pd
from scipy.optimize import minimize


def pesos_gmv(Sigma, ridge=1e-6):
    activos = Sigma.columns
    n = len(activos)
    invS = np.linalg.inv(Sigma.values + ridge * np.eye(n))
    unos = np.ones(n)

    pesos = (invS @ unos) / (unos @ invS @ unos)

    return pd.Series(pesos, index=activos)


def pesos_frontera(mu, Sigma, r_objetivo, ridge=1e-6):
    activos = mu.index
    n = len(activos)
    unos = np.ones(n)
    invS = np.linalg.inv(Sigma.values + ridge * np.eye(n))
    mu_arr = mu.values

    A = unos @ invS @ unos
    B = unos @ invS @ mu_arr
    C = mu_arr @ invS @ mu_arr
    D = A * C - B ** 2

    lam = (C - B * r_objetivo) / D
    gam = (A * r_objetivo - B) / D

    pesos = lam * (invS @ unos) + gam * (invS @ mu_arr)

    return pd.Series(pesos, index=activos)


def frontera_eficiente_sin_restricciones(mu, Sigma, n_points=200, ridge=1e-6):
    n = len(mu)
    unos = np.ones(n)
    invS = np.linalg.inv(Sigma.values + ridge * np.eye(n))
    mu_arr = mu.values

    A = unos @ invS @ unos
    B = unos @ invS @ mu_arr
    C = mu_arr @ invS @ mu_arr
    D = A * C - B ** 2

    r_grid = np.linspace(mu_arr.min() * 0.8, mu_arr.max() * 1.2, n_points)
    sigma2 = np.maximum((A * r_grid ** 2 - 2 * B * r_grid + C) / D, 0)
    sigma = np.sqrt(sigma2)

    r_gmv = B / A
    sigma_gmv = np.sqrt(1 / A)
    mask_eficiente = r_grid >= r_gmv

    return {
        "sigma_eficiente": sigma[mask_eficiente],
        "r_eficiente": r_grid[mask_eficiente],
        "sigma_completa": sigma,
        "r_completa": r_grid,
        "r_gmv": r_gmv,
        "sigma_gmv": sigma_gmv,
    }


def pesos_tangencia(mu, Sigma, r_f, ridge=1e-6):
    activos = mu.index
    n = len(activos)
    invS = np.linalg.inv(Sigma.values + ridge * np.eye(n))
    exceso = mu.values - r_f

    numerador = invS @ exceso
    pesos = numerador / (np.ones(n) @ numerador)

    return pd.Series(pesos, index=activos)


def linea_mercado_capitales(r_f, mu_tangencia, sigma_tangencia, sigma_max, n_points=100):
    sigma_cml = np.linspace(0, sigma_max, n_points)
    pendiente = (mu_tangencia - r_f) / sigma_tangencia
    r_cml = r_f + pendiente * sigma_cml

    return sigma_cml, r_cml


def pesos_gmv_restringido(Sigma, permitir_cortos=False, ridge=1e-6):
    activos = Sigma.columns
    n = len(activos)
    Sigma_reg = Sigma.values + ridge * np.eye(n)

    def objetivo(x):
        return x @ Sigma_reg @ x

    restricciones = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    limites = [(None, None) for _ in range(n)] if permitir_cortos else [(0, 1) for _ in range(n)]
    x0 = np.ones(n) / n

    resultado = minimize(objetivo, x0, method="SLSQP", bounds=limites, constraints=restricciones)

    return pd.Series(resultado.x, index=activos)


def pesos_frontera_restringida(mu, Sigma, r_objetivo, permitir_cortos=False, ridge=1e-6):
    activos = mu.index
    n = len(activos)
    Sigma_reg = Sigma.values + ridge * np.eye(n)
    mu_arr = mu.values

    def objetivo(x):
        return x @ Sigma_reg @ x

    restricciones = [
        {"type": "ineq", "fun": lambda x: (mu_arr @ x) - r_objetivo},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
    ]
    limites = [(None, None) for _ in range(n)] if permitir_cortos else [(0, 1) for _ in range(n)]
    x0 = np.ones(n) / n

    resultado = minimize(objetivo, x0, method="SLSQP", bounds=limites, constraints=restricciones)

    return pd.Series(resultado.x, index=activos)


def frontera_eficiente_restringida(mu, Sigma, n_points=50, permitir_cortos=False, ridge=1e-6):
    pesos_gmv_r = pesos_gmv_restringido(Sigma, permitir_cortos=permitir_cortos, ridge=ridge)
    r_gmv = float(mu.values @ pesos_gmv_r.values)
    sigma_gmv = float(np.sqrt(pesos_gmv_r.values @ Sigma.values @ pesos_gmv_r.values))

    r_grid = np.linspace(r_gmv, mu.max(), n_points)
    sigma_grid = np.empty(n_points)

    for i, r in enumerate(r_grid):
        pesos_r = pesos_frontera_restringida(mu, Sigma, r, permitir_cortos=permitir_cortos, ridge=ridge)
        sigma_grid[i] = np.sqrt(pesos_r.values @ Sigma.values @ pesos_r.values)

    return {
        "sigma_eficiente": sigma_grid,
        "r_eficiente": r_grid,
        "r_gmv": r_gmv,
        "sigma_gmv": sigma_gmv,
    }


def pesos_tangencia_restringida(mu, Sigma, r_f, permitir_cortos=False, ridge=1e-6):
    activos = mu.index
    n = len(activos)
    Sigma_reg = Sigma.values + ridge * np.eye(n)
    mu_arr = mu.values

    def objetivo(x):
        rendimiento = mu_arr @ x - r_f
        riesgo = np.sqrt(x @ Sigma_reg @ x)
        return -rendimiento / riesgo

    restricciones = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    limites = [(None, None) for _ in range(n)] if permitir_cortos else [(0, 1) for _ in range(n)]
    x0 = np.ones(n) / n

    resultado = minimize(objetivo, x0, method="SLSQP", bounds=limites, constraints=restricciones)

    return pd.Series(resultado.x, index=activos)


def resumen_portafolio(pesos, mu, Sigma, r_f=None):
    mu_portafolio = float(pesos.values @ mu.values)
    sigma_portafolio = float(np.sqrt(pesos.values @ Sigma.values @ pesos.values))

    resumen = {
        "pesos_pct": (pesos * 100).sort_values(ascending=False),
        "mu": mu_portafolio,
        "sigma": sigma_portafolio,
    }

    if r_f is not None:
        resumen["sharpe"] = (mu_portafolio - r_f) / sigma_portafolio

    return resumen


def optimizar_portafolio(mu, Sigma, tipo="gmv", r_f=0.0, r_objetivo=None, permitir_cortos=True, ridge=1e-6):
    if tipo == "gmv":
        if permitir_cortos:
            pesos = pesos_gmv(Sigma, ridge=ridge)
        else:
            pesos = pesos_gmv_restringido(Sigma, permitir_cortos=False, ridge=ridge)

    elif tipo == "tangencia":
        if permitir_cortos:
            pesos = pesos_tangencia(mu, Sigma, r_f, ridge=ridge)
        else:
            pesos = pesos_tangencia_restringida(mu, Sigma, r_f, permitir_cortos=False, ridge=ridge)

    elif tipo == "objetivo":
        if r_objetivo is None:
            raise ValueError("Se requiere r_objetivo para tipo='objetivo'")
        if permitir_cortos:
            pesos = pesos_frontera(mu, Sigma, r_objetivo, ridge=ridge)
        else:
            pesos = pesos_frontera_restringida(mu, Sigma, r_objetivo, permitir_cortos=False, ridge=ridge)

    else:
        raise ValueError(f"Tipo de portafolio desconocido: {tipo}")

    return resumen_portafolio(pesos, mu, Sigma, r_f=r_f)
