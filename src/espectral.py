import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import optimizacion


def descomposicion_espectral(Sigma):
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma.values)
    orden = np.argsort(eigenvalues)[::-1]

    return eigenvalues[orden], eigenvectors[:, orden]


def varianza_explicada(eigenvalues):
    return eigenvalues / eigenvalues.sum()


def num_componentes_umbral(eigenvalues, umbral=0.90):
    varianza_acumulada = np.cumsum(varianza_explicada(eigenvalues))

    return int(np.argmax(varianza_acumulada >= umbral) + 1)


def _modelo_4(mu_arr, mu_P, eigenvalues, eigenvectors, I):
    n = len(mu_arr)
    lambda_I = eigenvalues[:I]
    P_I = eigenvectors[:, :I]

    def objetivo(x):
        z_I = P_I.T @ x
        return np.sum(lambda_I * (z_I ** 2))

    restricciones = [
        {"type": "ineq", "fun": lambda x: (mu_arr @ x) - mu_P},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
    ]
    limites = [(None, None) for _ in range(n)]
    x0 = np.ones(n) / n

    resultado = minimize(objetivo, x0, method="SLSQP", bounds=limites, constraints=restricciones)

    return resultado.x, resultado.fun


def modelo_6(mu, mu_P, eigenvalues, eigenvectors, I, Sigma, x_bar=None):
    activos = mu.index
    n = len(activos)
    mu_arr = mu.values
    Sigma_arr = Sigma.values

    if x_bar is None:
        x_bar, _ = _modelo_4(mu_arr, mu_P, eigenvalues, eigenvectors, I)

    lambda_I_vec = np.zeros(len(eigenvalues))
    lambda_I_vec[:I] = eigenvalues[:I]
    Sigma_I = eigenvectors @ np.diag(lambda_I_vec) @ eigenvectors.T
    D = Sigma_arr - Sigma_I

    const_term = x_bar @ D @ x_bar
    grad_term = 2 * (D @ x_bar)

    def objetivo(x):
        z = eigenvectors.T @ x
        var_aprox = np.sum(eigenvalues[:I] * (z[:I] ** 2))
        e_hat = const_term + grad_term @ (x - x_bar)
        return var_aprox + max(e_hat, 0)

    restricciones = [
        {"type": "ineq", "fun": lambda x: (mu_arr @ x) - mu_P},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
    ]

    resultado = minimize(objetivo, x_bar, method="SLSQP", bounds=[(None, None)] * n, constraints=restricciones)

    return pd.Series(resultado.x, index=activos), resultado.fun


def cotas_modelo_6(mu, Sigma, mu_P, I, x_bar=None):
    eigenvalues, eigenvectors = descomposicion_espectral(Sigma)
    pesos, m_i_optimo = modelo_6(mu, mu_P, eigenvalues, eigenvectors, I, Sigma, x_bar=x_bar)

    cota_inferior = m_i_optimo
    cota_superior = float(pesos.values @ Sigma.values @ pesos.values)

    pesos_exacto = optimizacion.pesos_frontera(mu, Sigma, mu_P)
    V_optimo = float(pesos_exacto.values @ Sigma.values @ pesos_exacto.values)

    error_relativo_pct = (cota_superior - V_optimo) / V_optimo * 100

    return {
        "pesos": pesos,
        "cota_inferior": cota_inferior,
        "cota_superior": cota_superior,
        "V_optimo": V_optimo,
        "error_relativo_pct": error_relativo_pct,
    }


def convergencia_modelo_6(mu, Sigma, mu_P, rango_I=None):
    eigenvalues, eigenvectors = descomposicion_espectral(Sigma)
    n = len(mu)

    if rango_I is None:
        rango_I = range(1, n + 1)

    pesos_exacto = optimizacion.pesos_frontera(mu, Sigma, mu_P)
    V_optimo = float(pesos_exacto.values @ Sigma.values @ pesos_exacto.values)

    rango_I = np.array(list(rango_I))
    cotas_inferiores = np.empty(len(rango_I))
    cotas_superiores = np.empty(len(rango_I))

    x_bar = None
    for i, I in enumerate(rango_I):
        x_bar, _ = _modelo_4(mu.values, mu_P, eigenvalues, eigenvectors, I)
        pesos, m_i_optimo = modelo_6(mu, mu_P, eigenvalues, eigenvectors, I, Sigma, x_bar=x_bar)
        cotas_inferiores[i] = m_i_optimo
        cotas_superiores[i] = float(pesos.values @ Sigma.values @ pesos.values)

    error_relativo_pct = (cotas_superiores - V_optimo) / V_optimo * 100

    return {
        "rango_I": rango_I,
        "cota_inferior": cotas_inferiores,
        "cota_superior": cotas_superiores,
        "V_optimo": V_optimo,
        "error_relativo_pct": error_relativo_pct,
    }


def frontera_espectral(mu, Sigma, n_points=50, umbral_varianza=0.90):
    eigenvalues, eigenvectors = descomposicion_espectral(Sigma)
    k = num_componentes_umbral(eigenvalues, umbral=umbral_varianza)

    resultado_cerrado = optimizacion.frontera_eficiente_sin_restricciones(mu, Sigma, n_points=2)
    r_gmv = resultado_cerrado["r_gmv"]

    r_grid = np.linspace(r_gmv, mu.max(), n_points)
    sigma_grid = np.empty(n_points)

    x_bar = None
    for i, r in enumerate(r_grid):
        x_bar, _ = _modelo_4(mu.values, r, eigenvalues, eigenvectors, k)
        pesos, _ = modelo_6(mu, r, eigenvalues, eigenvectors, k, Sigma, x_bar=x_bar)
        sigma_grid[i] = np.sqrt(pesos.values @ Sigma.values @ pesos.values)

    return {
        "sigma_eficiente": sigma_grid,
        "r_eficiente": r_grid,
        "k_componentes": k,
    }


def optimizar_portafolio_espectral(mu, Sigma, mu_P, umbral_varianza=0.90, k_manual=None, r_f=None):
    eigenvalues, eigenvectors = descomposicion_espectral(Sigma)
    k = k_manual if k_manual is not None else num_componentes_umbral(eigenvalues, umbral=umbral_varianza)

    pesos, _ = modelo_6(mu, mu_P, eigenvalues, eigenvectors, k, Sigma)

    resumen = optimizacion.resumen_portafolio(pesos, mu, Sigma, r_f=r_f)
    resumen["k_componentes"] = k

    return resumen
