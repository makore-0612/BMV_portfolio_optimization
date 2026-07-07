import hashlib

import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import cargar_config
from src import datos, estimacion, optimizacion, espectral, graficas

config = cargar_config()
factor_anual = config["estimacion"]["factor_anualizacion"]

CACHE_ANALISIS = {}
CACHE_ESPECTRAL = {}
CACHE_FRONTERA_CLASICA = {}

def construir_opciones_activos(precios):
    activos_disponibles = sorted(precios.columns)
    opciones = [{"label": ticker, "value": ticker} for ticker in activos_disponibles]
    return activos_disponibles, opciones


precios_iniciales = datos.obtener_precios(config)
fecha_min = precios_iniciales.index.min()
fecha_max = precios_iniciales.index.max()
fecha_inicio_default = fecha_max - pd.Timedelta(days=182)

ACTIVOS_DISPONIBLES, OPCIONES_ACTIVOS = construir_opciones_activos(precios_iniciales)
OPCIONES_METODO_MU = [
    {"label": "Histórico", "value": "historico"},
    {"label": "Bayes-Stein", "value": "bayes_stein"},
    {"label": "CAPM", "value": "capm"},
]
OPCIONES_METODO_SIGMA = [
    {"label": "Muestral", "value": "muestral"},
    {"label": "Ledoit-Wolf", "value": "ledoit_wolf"},
]


def clave_cache(activos, fecha_inicio, fecha_fin, metodo_mu, metodo_sigma):
    crudo = f"{sorted(activos)}|{fecha_inicio}|{fecha_fin}|{metodo_mu}|{metodo_sigma}"
    return hashlib.md5(crudo.encode()).hexdigest()


app = dash.Dash(__name__)
server = app.server
app.title = "Optimización de portafolios - BMV"

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

TAB_STYLE = {"border": "1px solid #d8dbe6", "padding": "10px"}
TAB_SELECTED_STYLE = {
    "border": "2px solid #202A44", "borderTop": "3px solid #202A44",
    "padding": "10px", "backgroundColor": "white", "fontWeight": "bold",
}

app.layout = html.Div(className="app-contenedor", style={"display": "flex", "fontFamily": "'IBM Plex Sans', 'Manrope', 'Helvetica Neue', 'Inter', Arial, sans-serif"
, 'backgroundColor': '#F4F6F9'}, children=[

    html.Div(className="sidebar", style={"width": "320px", "padding": "16px", "borderRight": "1px solid #ddd", 'backgroundColor': '#202A44',
        'padding': '20px', 'borderRadius': '8px'}, children=[
        html.Div(className="logo", children=[
            html.Img(src="assets/logo.png", style={"width": "100px", "height": "auto"}),
        ]),
        html.H1("Controles", style={"color": "white"}),
        html.P("Selecciona los parámetros que desees utilizar para el proceso, luego presiona Calcular.", style={"color": "white"}),

        html.Label("Activos", style={"color": "white"}),
        dcc.Dropdown(id="dropdown-activos", options=OPCIONES_ACTIVOS, value=ACTIVOS_DISPONIBLES, multi=True),
        html.Button("Seleccionar todos", id="boton-todos-activos", n_clicks=0, style={"marginTop": "6px"}),

        html.Label("Periodo", style={"marginTop": "16px", "display": "block", "color": "white"}),
        dcc.DatePickerRange(
            id="date-picker-rango",
            min_date_allowed=fecha_min, max_date_allowed=fecha_max,
            start_date=fecha_inicio_default, end_date=fecha_max,
        ),

        html.Label("Ventas en corto (frontera clásica)", style={"marginTop": "16px", "display": "block", "color": "white"}),
        dcc.RadioItems(id="radio-cortos", options=[{"label": " Sí", "value": "si"}, {"label": " No", "value": "no"}], value="no", inline=True,
                       labelStyle={"color": "white"}),

        html.Label("Activo libre de riesgo", style={"marginTop": "16px", "display": "block", "color": "white"}),
        dcc.Checklist(id="switch-rf", options=[{"label": " Incluir", "value": "si"}], value=[], labelStyle={"color": "white"}),
        dcc.Input(id="input-rf", type="number", value=0.0, step=0.001, placeholder="tasa anual (ej. 0.07)", debounce=True, min = 0, max = 1),

        html.Label("Método de estimación de μ", style={"marginTop": "16px", "display": "block", "color": "white"}),
        dcc.Dropdown(id="dropdown-metodo-mu", options=OPCIONES_METODO_MU, value="bayes_stein", clearable=False),

        html.Label("Método de estimación de Σ", style={"marginTop": "16px", "display": "block", "color": "white"}),
        dcc.Dropdown(id="dropdown-metodo-sigma", options=OPCIONES_METODO_SIGMA, value="ledoit_wolf", clearable=False),

        html.Button("Calcular", id="boton-calcular", n_clicks=0, style={"marginTop": "20px", "width": "100%"}),
        html.Button("Actualizar datos", id="boton-actualizar-datos", n_clicks=0, style={"marginTop": "8px", "width": "100%"}),

        dcc.Loading(type="default", children=[
            html.Div(id="div-alertas", style={"marginTop": "16px", "color": "white"}),
        ]),

        html.H6("Creado en conjunto con Milena Fernanda Rivera e Ignacio Chuquiure a partir de las notas del curso 'Introducción a las Finanzas y a la Empresa' impartido en el IIMAS - UNAM por el profesor Eduardo Selim Martínez Mayorga junto al profesor adjunto Luis Enrique Villalón Pineda.", style={"color": '#9E9E9E'}),

        html.Hr(style={'borderTop': '3px solid #FFFFFF', 'margin': '20px 0', 'opacity': '0.4'}),

        html.Div(className="footer-sidebar", style={
            'marginTop': 'auto', 'paddingTop': '20px', 'fontSize': '13px',
            'color': '#aab2c8', 'textAlign': 'center'
        }, children=[
            html.Span("© Ángel Zamora, 2026"),
            html.Span(" · "),
            html.A("About", href="https://makore-0612.github.io/zam_portfolio/", target="_blank", style={'color': '#aab2c8'}),
            html.Span(" · "),
            html.A("Github", href="https://github.com/makore-0612", target="_blank", style={'color': '#aab2c8'}),
            html.Span(" · "),
            html.A("LinkedIn", href="https://www.linkedin.com/in/%C3%A1ngel-z-072674378/", target="_blank", style={'color': '#aab2c8'}),
        ]),

        dcc.Store(id="store-clave-analisis"),
        dcc.Store(id="store-clave-espectral"),
    ]),

    html.Div(className="contenido", style={"flex": "1", "padding": "16px", "minWidth": "0"}, children=[
        dcc.Tabs(id="tabs-principal", value="tab-series", children=[

            dcc.Tab(label="Series de tiempo", value="tab-series", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                dcc.Graph(id="graph-precios"),
                dcc.Graph(id="graph-rendimientos"),
            ]),

            dcc.Tab(label="Covarianza", value="tab-covarianza", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                dcc.RadioItems(id="radio-modo-heatmap", options=[
                    {"label": " Covarianza", "value": "covarianza"}, {"label": " Correlación", "value": "correlacion"}],
                    value="correlacion", inline=True, style={"marginTop": "12px"}),
                dcc.Graph(id="graph-heatmap"),
            ]),

            dcc.Tab(label="Frontera eficiente", value="tab-frontera", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                html.Div(style = {'marginTop': '12px'}, children = [
                    dcc.Markdown(r"""
#### ¿Qué hace el Modelo 6?

Dado el espectro de $\Sigma$ (eigenvalores $\lambda_i$ ordenados de mayor a menor, con eigenvectores $p_i$), el Modelo 6 separa la varianza del portafolio en una parte exacta (los primeros $I$ componentes) y una aproximación linealizada del residuo (los componentes descartados):

$$x^\top \Sigma x \;=\; \underbrace{\sum_{i=1}^{I} \lambda_i (p_i^\top x)^2}_{\text{exacto}} \;+\; \underbrace{\max\big(\hat{e}(x),\, 0\big)}_{\text{residuo linealizado}}$$

donde $\hat{e}(x) = \bar{x}^\top D \bar{x} + 2(D\bar{x})^\top(x - \bar{x})$ es la linealización de Taylor de primer orden del residuo $D = \Sigma - \Sigma_I$ alrededor de un punto de referencia $\bar{x}$ (la solución de un modelo anterior, que ignora el residuo por completo). El problema se resuelve numéricamente sujeto a $\mathbf{1}^\top x = 1$ y $\mu^\top x = \mu_P$, permitiendo ventas en corto.
                    """, mathjax=True, style={"backgroundColor": "white", "padding": "14px", "borderRadius": "6px"}),
                    html.Div(style={'height': '30px'}),
                    html.Label("Retorno objetivo (μ_P) para el Modelo 6"),
                    dcc.Input(id="input-mu-objetivo", type="number", value=0.05, step=0.001, min = 0, max = 0.1),
                    html.Div(style={'height': '30px'}),
                    html.Label("Presiona el siguiente botón para recalcular la Frontera Espectral cada vez que se seleccionen nuevos parámetros: "),
                    html.Button("Calcular frontera espectral y convergencia", id="boton-espectral", n_clicks=0,
                                style={"marginTop": "10px"}),
                    html.H5("Nota: Dependiendo del número de activos seleccionados el cálculo podría tomar un tiempo.", style={"color": '#9E9E9E'}),
                    html.Div(style={'height': '30px'}),
                ]),
                html.H3("Da un click en cualquier punto de la frontera eficiente para obtener los pesos del portafolio asociado."),
                dcc.Graph(id="graph-frontera"),
                html.H4("Pesos del portafolio seleccionado"),
                html.Div(style={'height': '20px'}),
                html.Div(id="div-tabla-pesos"),
                html.Div(style={'height': '20px'}),
                dcc.Graph(id="graph-pesos"),
            ]),

            dcc.Tab(label="Corte espectral", value="tab-espectral", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                html.Div(style={"marginTop": "12px", 'padding' : '10px'}, children=[
                    html.Label("Umbral de varianza explicada"),
                    dcc.Slider(id="slider-umbral-varianza", min=0.5, max=0.99, step=0.01, value=0.90,
                               marks={0.5: "50%", 0.75: "75%", 0.90: "90%", 0.99: "99%"}),
                ]),
                html.Div(id="div-resumen-modelo6", style={"marginTop": "12px"}),
                dcc.Graph(id="graph-scree"),
                html.H4("Comparación con el punto de la frontera eficiente exacto para el retorno objetivo μ_P"),
                dcc.Graph(id="graph-comparacion-pesos"),
            ]),

            dcc.Tab(label="Diagnósticos", value="tab-diagnosticos", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                html.H5("Nota: No se tendrá información sin ejecutar el cálculo de la frontera espectral y convergencia en la pestaña Frontera eficiente.", style={"color": '#9E9E9E'}),
                dcc.Loading(type="circle", children=[dcc.Graph(id="graph-convergencia")]),
                html.Div(id="div-diagnosticos", style={"marginTop": "12px"}),
            ]),
        ]),
    ]),
])


@app.callback(
    Output("dropdown-activos", "value", allow_duplicate=True),
    Input("boton-todos-activos", "n_clicks"),
    State("dropdown-activos", "options"),
    prevent_initial_call=True,
)
def seleccionar_todos_activos(n_clicks, opciones):
    return [o["value"] for o in opciones]


DIAS_MINIMOS_VENTANA = 7


@app.callback(
    Output("date-picker-rango", "start_date"),
    Output("date-picker-rango", "end_date"),
    Input("date-picker-rango", "start_date"),
    Input("date-picker-rango", "end_date"),
    State("date-picker-rango", "max_date_allowed"),
    prevent_initial_call=True,
)
def validar_rango_fechas(start_date, end_date, max_date_allowed):
    if start_date is None or end_date is None:
        return dash.no_update, dash.no_update

    disparador = dash.callback_context.triggered[0]["prop_id"]

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    maximo = pd.Timestamp(max_date_allowed)
    minimo = pd.Timedelta(days=DIAS_MINIMOS_VENTANA)

    if end > maximo:
        end = maximo

    if end - start < minimo:
        if "start_date" in disparador:
            end = min(start + minimo, maximo)
            start = end - minimo
        else:
            start = end - minimo

    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    if start_str == start_date and end_str == end_date:
        return dash.no_update, dash.no_update

    return start_str, end_str


@app.callback(
    Output("date-picker-rango", "min_date_allowed"),
    Output("date-picker-rango", "max_date_allowed"),
    Output("div-alertas", "children", allow_duplicate=True),
    Output("dropdown-activos", "options"),
    Output("dropdown-activos", "value", allow_duplicate=True),
    Input("boton-actualizar-datos", "n_clicks"),
    State("dropdown-activos", "value"),
    prevent_initial_call=True,
)
def actualizar_datos(n_clicks, activos_seleccionados):
    precios = datos.obtener_precios(config, forzar_descarga=True)
    datos.obtener_benchmark(config, forzar_descarga=True)
    CACHE_ANALISIS.clear()
    CACHE_ESPECTRAL.clear()
    CACHE_FRONTERA_CLASICA.clear()

    activos_disponibles, opciones = construir_opciones_activos(precios)
    seleccion_vigente = [a for a in (activos_seleccionados or []) if a in activos_disponibles]

    return precios.index.min(), precios.index.max(), "Datos actualizados desde Yahoo Finance.", opciones, seleccion_vigente


@app.callback(
    Output("store-clave-analisis", "data"),
    Output("div-alertas", "children"),
    Input("boton-calcular", "n_clicks"),
    State("dropdown-activos", "value"),
    State("date-picker-rango", "start_date"),
    State("date-picker-rango", "end_date"),
    State("dropdown-metodo-mu", "value"),
    State("dropdown-metodo-sigma", "value"),
    prevent_initial_call=True,
)
def calcular_analisis(n_clicks, activos, fecha_inicio, fecha_fin, metodo_mu, metodo_sigma):
    if not activos or len(activos) < 2:
        return dash.no_update, "Selecciona al menos 2 activos."

    clave = clave_cache(activos, fecha_inicio, fecha_fin, metodo_mu, metodo_sigma)

    precios = datos.obtener_precios(config)[activos]
    benchmark = datos.obtener_benchmark(config)

    ventana = precios.loc[fecha_inicio:fecha_fin]
    reporte_faltantes = datos.perfilar_faltantes(ventana)

    ventana_filtrada = datos.eliminar_columnas_incompletas(ventana, umbral=config["limpieza"]["umbral_faltantes_columna"])
    columnas_eliminadas = sorted(set(ventana.columns) - set(ventana_filtrada.columns))

    if ventana_filtrada.shape[1] < 2:
        return dash.no_update, "Muy pocos activos con datos suficientes en ese periodo. Amplía el rango o cambia los activos."

    ventana_imputada = datos.imputar_precios(
        ventana_filtrada,
        umbral_racha_corta=config["limpieza"]["umbral_faltantes_racha_corta"],
        ventana=config["limpieza"]["ventana_imputacion_corta"],
    )
    rendimientos_log = datos.calcular_rendimientos_log(ventana_imputada)

    benchmark_ventana = benchmark.loc[fecha_inicio:fecha_fin]
    rendimientos_benchmark = datos.calcular_rendimientos_log(benchmark_ventana.to_frame(name=config["benchmark"])).iloc[:, 0]
    rendimientos_benchmark = rendimientos_benchmark.reindex(rendimientos_log.index)

    mu = estimacion.estimar_mu(
        rendimientos_log, metodo=metodo_mu, factor_anual=factor_anual, rendimientos_benchmark=rendimientos_benchmark,
    )
    Sigma = estimacion.estimar_sigma(rendimientos_log, metodo=metodo_sigma, factor_anual=factor_anual)

    CACHE_ANALISIS[clave] = {
        "precios_ventana": ventana_imputada,
        "rendimientos_log": rendimientos_log,
        "rendimientos_benchmark": rendimientos_benchmark,
        "mu": mu,
        "Sigma": Sigma,
        "reporte_faltantes": reporte_faltantes,
    }

    mensaje = f"Listo: {ventana_imputada.shape[1]} activos, {ventana_imputada.shape[0]} días."
    if columnas_eliminadas:
        mensaje += f" Excluidos por faltantes (>{config['limpieza']['umbral_faltantes_columna']*100:.0f}%): {', '.join(columnas_eliminadas)}."

    return clave, mensaje


@app.callback(
    Output("graph-precios", "figure"),
    Output("graph-rendimientos", "figure"),
    Input("store-clave-analisis", "data"),
)
def render_series_tiempo(clave):
    if not clave or clave not in CACHE_ANALISIS:
        return dash.no_update, dash.no_update

    resultado = CACHE_ANALISIS[clave]
    return graficas.fig_precios_normalizados(resultado["precios_ventana"]), graficas.fig_rendimientos_diarios(resultado["rendimientos_log"])


@app.callback(
    Output("graph-heatmap", "figure"),
    Input("store-clave-analisis", "data"),
    Input("radio-modo-heatmap", "value"),
)
def render_heatmap(clave, modo):
    if not clave or clave not in CACHE_ANALISIS:
        return dash.no_update

    Sigma = CACHE_ANALISIS[clave]["Sigma"]
    return graficas.fig_heatmap_covarianza(Sigma, modo=modo)


def _r_f_seleccionado(switch_rf, valor_rf):
    if switch_rf and "si" in switch_rf:
        return float(valor_rf or 0.0)
    return None


@app.callback(
    Output("graph-frontera", "figure"),
    Input("store-clave-analisis", "data"),
    Input("radio-cortos", "value"),
    Input("switch-rf", "value"),
    Input("input-rf", "value"),
    Input("store-clave-espectral", "data"),
)
def render_frontera(clave, cortos, switch_rf, valor_rf, clave_espectral):
    if not clave or clave not in CACHE_ANALISIS:
        return dash.no_update

    resultado = CACHE_ANALISIS[clave]
    mu, Sigma = resultado["mu"], resultado["Sigma"]
    permitir_cortos = cortos == "si"
    r_f = _r_f_seleccionado(switch_rf, valor_rf)

    clave_frontera = (clave, permitir_cortos)
    if clave_frontera not in CACHE_FRONTERA_CLASICA:
        if permitir_cortos:
            resultado_clasico = optimizacion.frontera_eficiente_sin_restricciones(mu, Sigma)
            pesos_gmv = optimizacion.pesos_gmv(Sigma)
        else:
            resultado_clasico = optimizacion.frontera_eficiente_restringida(mu, Sigma, n_points=25, permitir_cortos=False)
            pesos_gmv = optimizacion.pesos_gmv_restringido(Sigma, permitir_cortos=False)

        CACHE_FRONTERA_CLASICA[clave_frontera] = {
            "resultado_clasico": resultado_clasico,
            "gmv": optimizacion.resumen_portafolio(pesos_gmv, mu, Sigma),
        }

    resultado_clasico = CACHE_FRONTERA_CLASICA[clave_frontera]["resultado_clasico"]
    gmv = CACHE_FRONTERA_CLASICA[clave_frontera]["gmv"]

    tangencia = None
    cml = None
    if r_f is not None:
        if permitir_cortos:
            pesos_tan = optimizacion.pesos_tangencia(mu, Sigma, r_f)
        else:
            pesos_tan = optimizacion.pesos_tangencia_restringida(mu, Sigma, r_f, permitir_cortos=False)
        tangencia = optimizacion.resumen_portafolio(pesos_tan, mu, Sigma, r_f=r_f)
        sigma_max = float(np.sqrt(np.diag(Sigma.values)).max())
        cml = optimizacion.linea_mercado_capitales(r_f, tangencia["mu"], tangencia["sigma"], sigma_max)

    resultado_espectral = None
    if clave_espectral and clave_espectral in CACHE_ESPECTRAL:
        resultado_espectral = CACHE_ESPECTRAL[clave_espectral].get("frontera_espectral")

    return graficas.fig_frontera(mu, Sigma, resultado_clasico, gmv, tangencia=tangencia, cml=cml, resultado_espectral=resultado_espectral)


@app.callback(
    Output("graph-pesos", "figure"),
    Output("div-tabla-pesos", "children"),
    Input("graph-frontera", "clickData"),
    State("graph-frontera", "figure"),
    State("store-clave-analisis", "data"),
    State("store-clave-espectral", "data"),
    State("radio-cortos", "value"),
)
def mostrar_pesos_click(click_data, figura, clave, clave_espectral, cortos):
    if not click_data or not clave or clave not in CACHE_ANALISIS:
        return dash.no_update, dash.no_update

    resultado = CACHE_ANALISIS[clave]
    mu, Sigma = resultado["mu"], resultado["Sigma"]
    permitir_cortos = cortos == "si"

    punto = click_data["points"][0]
    nombre_traza = figura["data"][punto["curveNumber"]]["name"]

    if nombre_traza == "Modelo 6" and clave_espectral in CACHE_ESPECTRAL:
        pesos_pct = CACHE_ESPECTRAL[clave_espectral]["punto_espectral"]["pesos_pct"]
    else:
        r_objetivo = punto["y"]
        if permitir_cortos:
            pesos = optimizacion.pesos_frontera(mu, Sigma, r_objetivo)
        else:
            pesos = optimizacion.pesos_frontera_restringida(mu, Sigma, r_objetivo, permitir_cortos=False)
        pesos_pct = optimizacion.resumen_portafolio(pesos, mu, Sigma)["pesos_pct"]

    activos = pesos_pct.head(300).index.tolist()
    valores = pesos_pct.head(300).values

    celda_style = {"padding": "6px 10px", "whiteSpace": "nowrap", "border": "1px solid #ddd", "textAlign": "center"}

    tabla = html.Div(
        style={"overflowX": "auto", "maxWidth": "100%"},  # para poder hacer scroll horizontal sin crecer la pagina
        children=[
            html.Table(
                className="tabla-pesos",
                style={"borderCollapse": "collapse"},
                children=[
                    html.Thead(
                        html.Tr([html.Th("Activo", style=celda_style)] + [html.Th(a, style=celda_style) for a in activos])
                    ),
                    html.Tbody(
                        html.Tr([html.Td("Peso (%)", style=celda_style)] + [html.Td(f"{p:.2f}", style=celda_style) for p in valores])
                    ),
                ],
            )
        ],
    )

    return graficas.fig_pesos_barra(pesos_pct), tabla


@app.callback(
    Output("graph-scree", "figure"),
    Output("div-resumen-modelo6", "children"),
    Output("graph-comparacion-pesos", "figure"),
    Input("store-clave-analisis", "data"),
    Input("slider-umbral-varianza", "value"),
    Input("input-mu-objetivo", "value"),
    Input("switch-rf", "value"),
    Input("input-rf", "value"),
)
def render_espectral_rapido(clave, umbral_varianza, mu_P, switch_rf, valor_rf):
    if not clave or clave not in CACHE_ANALISIS or mu_P is None:
        return dash.no_update, dash.no_update, dash.no_update

    resultado = CACHE_ANALISIS[clave]
    mu, Sigma = resultado["mu"], resultado["Sigma"]

    eigenvalues, eigenvectors = espectral.descomposicion_espectral(Sigma)
    k = espectral.num_componentes_umbral(eigenvalues, umbral=umbral_varianza)

    r_f = _r_f_seleccionado(switch_rf, valor_rf)

    try:
        resumen_espectral = espectral.optimizar_portafolio_espectral(mu, Sigma, mu_P=mu_P, umbral_varianza=umbral_varianza, r_f=r_f)
    except Exception:
        return graficas.fig_scree(eigenvalues, k), "No fue posible resolver el Modelo 6 para ese retorno objetivo.", dash.no_update

    cotas = espectral.cotas_modelo_6(mu, Sigma, mu_P, k)
    pesos_exacto = optimizacion.pesos_frontera(mu, Sigma, mu_P)
    resumen_exacto = optimizacion.resumen_portafolio(pesos_exacto, mu, Sigma)

    texto_resumen = [
        html.H2(f"Componentes usados (k): {k} / {len(mu)}"),
        html.H3(f"μ: {resumen_espectral['mu']:.4f}   σ: {resumen_espectral['sigma']:.4f}"
               + (f"   Sharpe: {resumen_espectral['sharpe']:.4f}" if "sharpe" in resumen_espectral else "")),
        html.H3(f"Error relativo vs. óptimo exacto: {cotas['error_relativo_pct']:.2f}%",
               style={"color": "#a33" if cotas["error_relativo_pct"] > 20 else "inherit"}),
        html.P("El corte espectral siempre permite ventas en corto (igual que el modelo original)."),
    ]

    fig_comparacion = graficas.fig_pesos_barra(resumen_espectral["pesos_pct"])
    fig_comparacion.add_trace(go.Scatter(
        x=list(resumen_exacto["pesos_pct"].head(25).index), y=resumen_exacto["pesos_pct"].head(25).values,
        mode="markers", name="Óptimo exacto", marker=dict(color="crimson", size=8),
    ))

    return graficas.fig_scree(eigenvalues, k), texto_resumen, fig_comparacion


@app.callback(
    Output("store-clave-espectral", "data"),
    Output("graph-convergencia", "figure"),
    Input("boton-espectral", "n_clicks"),
    State("store-clave-analisis", "data"),
    State("input-mu-objetivo", "value"),
    State("slider-umbral-varianza", "value"),
    prevent_initial_call=True,
)
def calcular_espectral_completo(n_clicks, clave, mu_P, umbral_varianza):
    if not clave or clave not in CACHE_ANALISIS or mu_P is None:
        return dash.no_update, dash.no_update

    resultado = CACHE_ANALISIS[clave]
    mu, Sigma = resultado["mu"], resultado["Sigma"]

    clave_espectral = f"{clave}|{mu_P}|{umbral_varianza}"

    frontera_esp = espectral.frontera_espectral(mu, Sigma, n_points=20, umbral_varianza=umbral_varianza)
    convergencia = espectral.convergencia_modelo_6(mu, Sigma, mu_P)
    punto_espectral = espectral.optimizar_portafolio_espectral(mu, Sigma, mu_P=mu_P, umbral_varianza=umbral_varianza)

    CACHE_ESPECTRAL[clave_espectral] = {
        "frontera_espectral": frontera_esp,
        "convergencia": convergencia,
        "punto_espectral": punto_espectral,
    }

    return clave_espectral, graficas.fig_convergencia(convergencia)


@app.callback(
    Output("div-diagnosticos", "children"),
    Input("store-clave-analisis", "data"),
    State("dropdown-metodo-mu", "value"),
)
def render_diagnosticos(clave, metodo_mu):
    if not clave or clave not in CACHE_ANALISIS:
        return dash.no_update

    resultado = CACHE_ANALISIS[clave]

    if metodo_mu != "capm":
        reporte = resultado["reporte_faltantes"]
        return html.Div([
            html.P(f"Filas completas: {reporte['filas_completas']} / {reporte['filas_totales']}"),
            html.P("Las pruebas FDR/GRS solo aplican cuando el método de μ es CAPM."),
        ])

    resultado_capm = estimacion.regresion_capm(resultado["rendimientos_log"], resultado["rendimientos_benchmark"])
    tabla_fdr = estimacion.prueba_fdr_alphas(resultado_capm)
    grs = estimacion.prueba_grs(resultado_capm, resultado["rendimientos_benchmark"])

    n_significativos = int(tabla_fdr["significativo_fdr"].sum())

    return html.Div([
        html.H3(f"Prima de mercado anualizada: {resultado_capm['prima_mercado_anual']:.4f}"),
        html.H3(f"Activos con alpha significativo (FDR): {n_significativos} / {len(tabla_fdr)}"),
        html.H3(f"GRS: estadístico={grs['estadistico_grs']:.4f}, p-valor={grs['p_valor_grs']:.4f} "
               f"(F({grs['grados_libertad_num']}, {grs['grados_libertad_den']}))"),
    ])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
