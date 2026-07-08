import numpy as np
import plotly.graph_objects as go


def fig_precios_normalizados(precios):
    precios_norm = precios / precios.iloc[0] * 100

    fig = go.Figure()
    for columna in precios_norm.columns:
        fig.add_trace(go.Scatter(x=precios_norm.index, y=precios_norm[columna], mode="lines", name=columna))

    fig.update_layout(
        height = 900,
        title="Precios normalizados (base 100)",
        xaxis_title="Fecha", yaxis_title="Precio (base 100)",
        legend_title="Activo", template="plotly_white",
        paper_bgcolor = '#F4F6F9'
    )

    return fig


def fig_rendimientos_diarios(rendimientos_log):
    fig = go.Figure()
    for columna in rendimientos_log.columns:
        fig.add_trace(go.Scatter(x=rendimientos_log.index, y=rendimientos_log[columna], mode="lines", name=columna))

    fig.update_layout(
        height = 900,
        title="Rendimientos logarítmicos diarios",
        xaxis_title="Fecha", yaxis_title="Rendimiento",
        legend_title="Activo", template="plotly_white",
        paper_bgcolor = '#F4F6F9'
    )

    return fig


def fig_heatmap_covarianza(Sigma, modo="covarianza"):
    if modo == "correlacion":
        desviaciones = np.sqrt(np.diag(Sigma.values))
        denominador = np.outer(desviaciones, desviaciones)
        matriz = np.divide(Sigma.values, denominador, out=np.full_like(Sigma.values, np.nan), where=denominador > 0)
        titulo = "Correlación entre activos"
        zmid = 0
    else:
        matriz = Sigma.values
        titulo = "Covarianza entre activos (anualizada)"
        zmid = None

    fig = go.Figure(data=go.Heatmap(
        z=matriz, x=list(Sigma.columns), y=list(Sigma.columns),
        colorscale="RdBu", zmid=zmid, reversescale=True,
    ))
    fig.update_layout(height = 900, title=titulo, template="plotly_white", paper_bgcolor = '#F4F6F9')

    return fig


def fig_frontera(mu, Sigma, resultado_clasico, gmv, tangencia=None, cml=None, resultado_espectral=None, punto_espectral=None):
    fig = go.Figure()

    std_activos = np.sqrt(np.diag(Sigma.values))
    fig.add_trace(go.Scatter(
        x=std_activos, y=mu.values, mode="markers", name="Activos individuales",
        marker=dict(color="gray", size=6, opacity=0.6), text=list(mu.index), hoverinfo="text",
    ))

    if "sigma_completa" in resultado_clasico:
        fig.add_trace(go.Scatter(
            x=resultado_clasico["sigma_completa"], y=resultado_clasico["r_completa"],
            mode="lines", line=dict(color="black", dash="dot", width=1), opacity=0.4, name="Frontera completa",
        ))

    fig.add_trace(go.Scatter(
        x=resultado_clasico["sigma_eficiente"], y=resultado_clasico["r_eficiente"],
        mode="lines", line=dict(color="black", width=2.5), name="Frontera eficiente",
    ))

    fig.add_trace(go.Scatter(
        x=[gmv["sigma"]], y=[gmv["mu"]], mode="markers", name="Mínima varianza (GMV)",
        marker=dict(color="orange", size=14, symbol="diamond"),
    ))

    if tangencia is not None:
        fig.add_trace(go.Scatter(
            x=[tangencia["sigma"]], y=[tangencia["mu"]], mode="markers", name="Tangencia",
            marker=dict(color="crimson", size=14, symbol="star"),
        ))

    if cml is not None:
        sigma_cml, r_cml = cml
        fig.add_trace(go.Scatter(
            x=sigma_cml, y=r_cml, mode="lines", line=dict(color="crimson", dash="dash"), name="CML",
        ))

    if resultado_espectral is not None:
        fig.add_trace(go.Scatter(
            x=resultado_espectral["sigma_eficiente"], y=resultado_espectral["r_eficiente"],
            mode="lines", line=dict(color="seagreen", width=2),
            name=f"Frontera espectral (k={resultado_espectral['k_componentes']})",
        ))

    if punto_espectral is not None:
        fig.add_trace(go.Scatter(
            x=[punto_espectral["sigma"]], y=[punto_espectral["mu"]], mode="markers", name="Modelo 6",
            marker=dict(color="seagreen", size=13, symbol="square"),
        ))

    fig.update_layout(
        height = 900,
        title="Frontera eficiente", xaxis_title="Riesgo (σ anualizado)", yaxis_title="Retorno esperado (μ anualizado)",
        template="plotly_white", hovermode="closest",
        paper_bgcolor = '#F4F6F9'
    )

    return fig


def fig_scree(eigenvalues, k):
    componentes = list(range(1, len(eigenvalues) + 1))
    varianza_explicada = eigenvalues / eigenvalues.sum() * 100
    varianza_acumulada = np.cumsum(varianza_explicada)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=componentes, y=varianza_explicada, name="Varianza individual", marker=dict(color='#1E3A8A')))
    fig.add_trace(go.Scatter(x=componentes, y=varianza_acumulada, mode="lines", name="Varianza acumulada",
                              yaxis="y2", line=dict(color="steelblue")))
    fig.add_vline(x=k, line_dash="dot", line_color="red", annotation_text=f"k={k}")

    fig.update_layout(
        height = 900,
        title="Varianza explicada por componente",
        xaxis_title="Componente",
        yaxis=dict(title="Varianza individual (%)"),
        yaxis2=dict(title="Varianza acumulada (%)", overlaying="y", side="right", range=[0, 105]),
        template="plotly_white",
        paper_bgcolor = '#F4F6F9'
    )

    return fig


def fig_convergencia(resultado_convergencia):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=resultado_convergencia["rango_I"], y=resultado_convergencia["error_relativo_pct"],
        mode="lines+markers", name="Error relativo de la cota superior", marker=dict(size=4),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        height = 900,
        title="Convergencia del Modelo 6",
        xaxis_title="Componentes retenidos (I)", yaxis_title="Error relativo (%)",
        template="plotly_white",
        paper_bgcolor = '#F4F6F9'
    )

    return fig


def fig_pesos_barra(pesos_pct, n_top=25):
    pesos_top = pesos_pct.head(n_top)

    fig = go.Figure(go.Bar(x=list(pesos_top.index), y=pesos_top.values, marker=dict(color="steelblue")))
    fig.update_layout(
        height = 900,
        title="Pesos del portafolio (%)", xaxis_title="Activo", yaxis_title="Peso (%)", template="plotly_white",
        paper_bgcolor = '#F4F6F9'
    )

    return fig
