import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from statsmodels.stats.multitest import multipletests


def estimar_mu_historico(rendimientos_log, factor_anual=252):
    return rendimientos_log.mean() * factor_anual


def estimar_mu_bayes_stein(rendimientos_log, factor_anual=252, ridge=1e-6):
    R = rendimientos_log.values
    T, N = R.shape

    mu = R.mean(axis=0)
    Sigma = np.cov(R, rowvar=False) + ridge * np.eye(N)
    Sigma_inv = np.linalg.inv(Sigma)

    unos = np.ones(N)
    mu_min = (unos @ Sigma_inv @ mu) / (unos @ Sigma_inv @ unos)

    desviacion = mu - mu_min * unos
    lam = (N + 2) / ((N + 2) + T * (desviacion @ Sigma_inv @ desviacion))

    mu_bayes_stein = (1 - lam) * mu + lam * mu_min * unos

    return pd.Series(mu_bayes_stein * factor_anual, index=rendimientos_log.columns)


def regresion_capm(rendimientos_log, rendimientos_benchmark, tasa_libre_riesgo_anual=0.0, factor_anual=252):
    r_f_diario = tasa_libre_riesgo_anual / factor_anual
    exceso_mercado = rendimientos_benchmark.values - r_f_diario
    exceso_activos = rendimientos_log.values - r_f_diario

    T = len(exceso_mercado)
    X = np.column_stack([np.ones(T), exceso_mercado])
    XtX_inv = np.linalg.inv(X.T @ X)
    coefs = XtX_inv @ X.T @ exceso_activos

    alpha = coefs[0]
    beta = coefs[1]
    residuos = exceso_activos - X @ coefs
    grados_libertad = T - 2

    sigma2_residual = (residuos ** 2).sum(axis=0) / grados_libertad
    error_estandar_alpha = np.sqrt(sigma2_residual * XtX_inv[0, 0])
    t_stat_alpha = np.divide(
        alpha, error_estandar_alpha,
        out=np.full_like(alpha, np.nan), where=error_estandar_alpha > 0,
    )
    p_valor_alpha = 2 * (1 - stats.t.cdf(np.abs(t_stat_alpha), df=grados_libertad))

    prima_mercado_anual = exceso_mercado.mean() * factor_anual
    mu_capm = tasa_libre_riesgo_anual + beta * prima_mercado_anual

    activos = rendimientos_log.columns

    return {
        "mu_capm": pd.Series(mu_capm, index=activos),
        "beta": pd.Series(beta, index=activos),
        "alpha_diario": pd.Series(alpha, index=activos),
        "error_estandar_alpha": pd.Series(error_estandar_alpha, index=activos),
        "t_stat_alpha": pd.Series(t_stat_alpha, index=activos),
        "p_valor_alpha": pd.Series(p_valor_alpha, index=activos),
        "residuos": pd.DataFrame(residuos, index=rendimientos_log.index, columns=activos),
        "grados_libertad": grados_libertad,
        "prima_mercado_anual": prima_mercado_anual,
    }


def estimar_mu_capm(rendimientos_log, rendimientos_benchmark, tasa_libre_riesgo_anual=0.0, factor_anual=252):
    resultado = regresion_capm(rendimientos_log, rendimientos_benchmark, tasa_libre_riesgo_anual, factor_anual)
    return resultado["mu_capm"]


def estimar_mu(rendimientos_log, metodo="historico", factor_anual=252, **kwargs):
    if metodo == "historico":
        return estimar_mu_historico(rendimientos_log, factor_anual=factor_anual)

    if metodo == "bayes_stein":
        return estimar_mu_bayes_stein(rendimientos_log, factor_anual=factor_anual)

    if metodo == "capm":
        return estimar_mu_capm(
            rendimientos_log,
            kwargs["rendimientos_benchmark"],
            tasa_libre_riesgo_anual=kwargs.get("tasa_libre_riesgo_anual", 0.0),
            factor_anual=factor_anual,
        )

    raise ValueError(f"Metodo de estimacion de mu desconocido: {metodo}")


def estimar_sigma_muestral(rendimientos_log, factor_anual=252):
    return rendimientos_log.cov() * factor_anual


def estimar_sigma_ledoit_wolf(rendimientos_log, factor_anual=252):
    modelo = LedoitWolf().fit(rendimientos_log.values)
    sigma_diaria = modelo.covariance_
    return pd.DataFrame(sigma_diaria * factor_anual, index=rendimientos_log.columns, columns=rendimientos_log.columns)


def estimar_sigma(rendimientos_log, metodo="muestral", factor_anual=252):
    if metodo == "muestral":
        return estimar_sigma_muestral(rendimientos_log, factor_anual=factor_anual)

    if metodo == "ledoit_wolf":
        return estimar_sigma_ledoit_wolf(rendimientos_log, factor_anual=factor_anual)

    raise ValueError(f"Metodo de estimacion de sigma desconocido: {metodo}")


def prueba_fdr_alphas(resultado_capm, nivel_significancia=0.05):
    p_valores = resultado_capm["p_valor_alpha"]
    rechazar, p_ajustado, _, _ = multipletests(p_valores.values, alpha=nivel_significancia, method="fdr_bh")

    return pd.DataFrame({
        "alpha": resultado_capm["alpha_diario"],
        "p_valor": p_valores,
        "p_valor_ajustado_fdr": p_ajustado,
        "significativo_fdr": rechazar,
    }, index=p_valores.index)


def prueba_grs(resultado_capm, rendimientos_benchmark, tasa_libre_riesgo_anual=0.0, factor_anual=252, ridge=1e-6):
    alpha = resultado_capm["alpha_diario"].values
    residuos = resultado_capm["residuos"].values
    grados_libertad = resultado_capm["grados_libertad"]
    T, N = residuos.shape

    sigma_residuos = (residuos.T @ residuos) / grados_libertad + ridge * np.eye(N)

    r_f_diario = tasa_libre_riesgo_anual / factor_anual
    exceso_mercado = rendimientos_benchmark.values - r_f_diario
    mu_m = exceso_mercado.mean()
    sigma2_m = exceso_mercado.var(ddof=1)

    sharpe_mercado_cuadrado = (mu_m ** 2) / sigma2_m

    estadistico_grs = ((T - N - 1) / N) * (alpha @ np.linalg.inv(sigma_residuos) @ alpha) / (1 + sharpe_mercado_cuadrado)
    p_valor_grs = 1 - stats.f.cdf(estadistico_grs, N, T - N - 1)

    return {
        "estadistico_grs": estadistico_grs,
        "p_valor_grs": p_valor_grs,
        "grados_libertad_num": N,
        "grados_libertad_den": T - N - 1,
    }
