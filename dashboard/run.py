import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, dash_table, State, no_update, callback_context
import re
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy.stats import rankdata
from dash.exceptions import PreventUpdate
from flask import send_from_directory

app = Dash(__name__, title="Feature Selection Evaluation")
server = app.server

# Static file serving routes
@server.route('/about.html')
def serve_about():
    return send_from_directory('.', 'about.html')

@server.route('/citation.html')
def serve_citation():
    return send_from_directory('.', 'citation.html')

@server.route('/documentation.html')
def serve_documentation():
    return send_from_directory('.', 'documentation.html')

@server.route('/benchmarking.html')
def serve_benchmarking():
    return send_from_directory('.', 'benchmarking.html')

@server.route('/downloads.html')
def serve_downloads():
    return send_from_directory('.', 'downloads.html')

@server.route('/references.html')
def serve_references():
    return send_from_directory('.', 'references.html')

@server.route('/files/<path:filename>')
def download_file(filename):
    return send_from_directory('files', filename, as_attachment=True)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

DATA_DIR = 'resources'

BENCHMARK_METRICS = ['CLSACC', 'NMI', 'ACC', 'AUC']
BENCHMARK_METHODS = ['Variance', 'Correlation', 'Laplacian', 'Random', 'VCSDFS', 'LIDFS', 'SCFS', 'MCFS']

STYLE_MAP = {
    'Variance':    {'color': '#e6194b', 'marker': 'circle',      'plt': 'o'},
    'Correlation': {'color': '#3cb44b', 'marker': 'square',      'plt': 's'},
    'Laplacian':   {'color': "#b39d13", 'marker': 'diamond',     'plt': 'D'},
    'Random':      {'color': '#4361ee', 'marker': 'cross',       'plt': 'P'},
    'VCSDFS':      {'color': '#f58231', 'marker': 'x',           'plt': 'X'},
    'LIDFS':       {'color': '#911eb4', 'marker': 'triangle-up', 'plt': '^'},
    'SCFS':        {'color': "#1dc9f0", 'marker': 'pentagon',    'plt': 'p'},
    'MCFS':        {'color': '#f032e6', 'marker': 'star',        'plt': '*'}
}

PERCENTAGE_RANGES = {
    '10Percent':  {'label': '0.5% – 10%',  'cols': [str(np.round(p, 3)) for p in np.arange(0.005, 0.1001, 0.005)]},
    '100Percent': {'label': '5% – 100%',   'cols': [str(np.round(p, 2)) for p in np.arange(0.05, 1.001, 0.05)]}
}

# Runtime Plot Constants
MARKERS_LIST = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'P', 'H']
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ─── Data Loading ─────────────────────────────────────────────

def load_data():
    data_dict = {}
    pattern = re.compile(r'^(.+?)_(CLSACC|NMI|ACC|AUC)_(10Percent|100Percent)\.csv$')
    if not os.path.exists(DATA_DIR):
        return data_dict
    for f in os.listdir(DATA_DIR):
        m = pattern.match(f)
        if m:
            method, metric, suffix = m.groups()
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            df.columns = [str(c) for c in df.columns]
            data_dict[f"{method}|{metric}|{suffix}"] = df.to_dict('records')
    return data_dict

INITIAL_DATA = load_data()

# Runtime data (built-in) - loaded once
RUNTIME_DATA = {'features': {}, 'instances': {}}

def load_runtime_data():
    global RUNTIME_DATA
    for rtype in ['features', 'instances']:
        path = f"{DATA_DIR}/time_analysis_{rtype}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            RUNTIME_DATA[rtype] = df.set_index('Method').to_dict('index')
        else:
            RUNTIME_DATA[rtype] = {}

load_runtime_data()

# ─── Styles ─────────────────────────────────────────────

BLUE_BTN   = {'padding': '10px 22px', 'cursor': 'pointer', 'backgroundColor': '#4a6bff', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'fontWeight': '600', 'fontSize': '14px'}
ORANGE_BTN = {'padding': '10px 22px', 'cursor': 'pointer', 'backgroundColor': '#f39c12', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'fontWeight': '600', 'fontSize': '14px'}
GREEN_BTN  = {'padding': '11px 32px', 'cursor': 'pointer', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'fontWeight': '600', 'fontSize': '14px'}
PURPLE_BTN = {'padding': '11px 32px', 'cursor': 'pointer', 'backgroundColor': '#8e44ad', 'color': 'white', 'border': 'none', 'borderRadius': '6px', 'fontWeight': '600', 'fontSize': '14px'}

HEADER_STYLE = {
    'backgroundColor': '#f8f9fc',
    'padding': '16px 32px',
    'borderBottom': '1px solid #e2e8f0',
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'flexWrap': 'wrap',
    'gap': '16px'
}

CARD_STYLE = {
    'backgroundColor': '#ffffff',
    'borderRadius': '10px',
    'border': '1px solid #e2e8f0',
    'boxShadow': '0 2px 12px rgba(0,0,0,0.05)',
    'padding': '20px'
}

# ────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────

app.layout = html.Div(style={'backgroundColor': '#f8f9fc', 'minHeight': '100vh', 'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"}, children=[

    dcc.Store(id='exclusion-store', data={'datasets': [], 'methods': []}),
    dcc.Store(id='custom-data-store', data={}),
    dcc.Store(id='state-for-download'),
    dcc.Download(id='download-line-plot'),
    dcc.Download(id='download-cd-plot'),
    dcc.Download(id='download-runtime-plot'),

    # Header
    html.Div(style=HEADER_STYLE, children=[
        html.Div([
            html.H1('Feature Selection Evaluation', style={'margin': '0', 'fontSize': '1.9rem', 'fontWeight': '700', 'color': '#1e293b'}),
            html.Div(style={'marginTop': '8px', 'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}, children=[
                html.A(html.Button('About', style=BLUE_BTN), href='/about.html'),
                html.A(html.Button('Benchmarking', style=BLUE_BTN), href='/benchmarking.html'),
                html.A(html.Button('Documentation', style=BLUE_BTN), href='/documentation.html'),
                html.A(html.Button('Downloads', style=BLUE_BTN), href='/downloads.html'),
                html.A(html.Button('Cite', style=BLUE_BTN), href='/citation.html'),
                html.A(html.Button('References', style=BLUE_BTN), href='/references.html'),
                dcc.Upload(id='upload-data', multiple=True, children=html.Button('Import', style=ORANGE_BTN)),
                html.Button('Exclude', id='btn-exclude-toggle', style=ORANGE_BTN),
            ])
        ]),
        html.Div(style={'display': 'flex', 'gap': '16px', 'alignItems': 'flex-end'}, children=[
            html.Div([html.Label('Dataset', style={'fontSize': '13px', 'fontWeight': '600', 'color': '#475569'}), dcc.Dropdown(id='dataset-dropdown', style={'width': '190px'})]),
            html.Div([html.Label('Metric',   style={'fontSize': '13px', 'fontWeight': '600', 'color': '#475569'}), dcc.Dropdown(id='metric-dropdown', options=[{'label': m, 'value': m} for m in BENCHMARK_METRICS], value='CLSACC', style={'width': '130px'})]),
            html.Div([html.Label('Range',    style={'fontSize': '13px', 'fontWeight': '600', 'color': '#475569'}), dcc.Dropdown(id='range-dropdown', options=[{'label': v['label'], 'value': k} for k, v in PERCENTAGE_RANGES.items()], value='10Percent', style={'width': '170px'})]),
        ])
    ]),

    # Exclusion panel
    html.Div(id='exclude-container', style={'display': 'none', 'padding': '16px 32px', 'backgroundColor': '#fefce8', 'borderBottom': '1px solid #fef08a'}, children=[
        html.Div(style={'maxWidth': '640px', 'margin': '0 auto'}, children=[
            dcc.Textarea(id='exclude-input', value='DATASETS = []\nMETHODS = []', style={'width': '100%', 'height': '90px', 'fontFamily': 'monospace', 'fontSize': '13px', 'padding': '8px'}),
            html.Button('Apply Exclusion', id='btn-exclude-apply', style={**ORANGE_BTN, 'marginTop': '12px'})
        ])
    ]),

    # Main content
    html.Div(style={'padding': '24px 32px', 'display': 'flex', 'flexDirection': 'column', 'gap': '24px'}, children=[

        html.Div(style=CARD_STYLE, children=[
            dcc.Graph(id='line-plot', style={'height': '68vh', 'marginBottom': '12px'}),
            html.Div(style={'textAlign': 'center'}, children=[
                html.Button('Download Line Plot (PDF)', id='btn-download-line', style=GREEN_BTN)
            ])
        ]),

        html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px'}, children=[

            html.Div(style=CARD_STYLE, children=[
                html.H3(id='table-title', style={'margin': '0 0 16px 0', 'fontSize': '1.32rem', 'borderLeft': '5px solid #27ae60', 'paddingLeft': '12px', 'color': '#1e293b'}),
                dash_table.DataTable(
                    id='score-table',
                    markdown_options={"html": True},
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '10px 8px', 'fontSize': '13.5px', 'minWidth': '80px'},
                    style_header={'backgroundColor': '#f1f5f9', 'fontWeight': '600', 'borderBottom': '2px solid #cbd5e1', 'color': '#1e293b'},
                    style_data_conditional=[{'if': {'column_id': 'Method'}, 'fontWeight': '500', 'textAlign': 'left'}]
                ),
                html.Div(style={'textAlign': 'center', 'marginTop': '20px'}, children=[
                    html.Button('Generate LaTeX Table', id='btn-latex', style={**BLUE_BTN, 'backgroundColor': '#27ae60'}),
                    dcc.Textarea(id='latex-output', style={'display': 'none', 'width': '100%', 'height': '260px', 'marginTop': '16px', 'fontSize': '13px', 'fontFamily': 'monospace', 'padding': '10px'})
                ])
            ]),

            html.Div(style=CARD_STYLE, children=[
                html.H3("Critical Difference Diagram (Nemenyi post-hoc)", style={'margin': '0 0 16px 0', 'fontSize': '1.32rem', 'borderLeft': '5px solid #8e44ad', 'paddingLeft': '12px', 'color': '#1e293b'}),
                dcc.Graph(id='cd-diagram', style={'height': '560px', 'marginBottom': '12px'}),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.Button('Download CD Diagram (PDF)', id='btn-download-cd', style=GREEN_BTN)
                ])
            ]),

            html.Div(style=CARD_STYLE, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '16px'}, children=[
                    html.H3("Runtime Scalability Analysis", style={'margin': '0', 'fontSize': '1.32rem', 'borderLeft': '5px solid #4a6bff', 'paddingLeft': '12px', 'color': '#1e293b'}),
                    dcc.Dropdown(
                        id='runtime-type-dropdown',
                        options=[{'label': 'Instances Experiment', 'value': 'instances'}, {'label': 'Features Experiment', 'value': 'features'}],
                        value='features',
                        clearable=False,
                        style={'width': '220px'}
                    )
                ]),
                dcc.Graph(id='runtime-plot', style={'height': '560px', 'marginBottom': '12px'}),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.Button('Download Runtime Plot (PDF)', id='btn-download-runtime', style=GREEN_BTN)
                ])
            ]),
        ])
    ])
])

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def get_filtered_data(metric, rng, exclusion, custom):
    full = {**INITIAL_DATA, **(custom or {})}
    ex_methods = exclusion.get('methods', [])
    keys = [k for k in full if f"|{metric}|{rng}" in k and k.split('|')[0] not in ex_methods]
    return full, keys

def get_effective_runtime_data(rtype, exclusion, custom):
    base = RUNTIME_DATA.get(rtype, {})
    custom_key = f'custom_runtime_{rtype}'
    custom_data = (custom or {}).get(custom_key, {})
    
    # Merge: custom overrides built-in
    merged = {**base, **custom_data}
    
    # Apply exclusion
    excluded_methods = set(exclusion.get('methods', []))
    filtered = {m: v for m, v in merged.items() if m not in excluded_methods}
    
    if not filtered:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(filtered, orient='index').reset_index(names='Method')
    return df

# ────────────────────────────────────────────────
# Callbacks
# ────────────────────────────────────────────────

@app.callback(
    [Output('exclude-container', 'style'), Output('exclusion-store', 'data')],
    [Input('btn-exclude-toggle', 'n_clicks'), Input('btn-exclude-apply', 'n_clicks')],
    State('exclude-input', 'value'),
    prevent_initial_call=True
)
def toggle_exclude_panel(n_toggle, n_apply, text):
    ctx = callback_context
    if not ctx.triggered: raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'btn-exclude-toggle':
        return {'display': 'block'}, no_update
    try:
        ds = eval(re.search(r'DATASETS\s*=\s*(\[.*?\])', text, re.DOTALL).group(1))
        me = eval(re.search(r'METHODS\s*=\s*(\[.*?\])', text, re.DOTALL).group(1))
        return {'display': 'none'}, {'datasets': ds, 'methods': me}
    except:
        return {'display': 'none'}, no_update

@app.callback(
    Output('custom-data-store', 'data'),
    Input('upload-data', 'contents'),
    [State('upload-data', 'filename'), State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def store_uploaded_data(contents, filenames, current):
    if not contents: raise PreventUpdate
    data = current or {}
    
    for content, fname in zip(contents, filenames):
        if not content: continue
        _, b64 = content.split(',')
        decoded = base64.b64decode(b64)
        
        # Performance files
        match = re.match(r'^(.+?)_(CLSACC|NMI|ACC|AUC)_(10Percent|100Percent)\.csv$', fname)
        if match:
            method, metric, suffix = match.groups()
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df.columns = [str(c) for c in df.columns]
            key = f"{method}|{metric}|{suffix}"
            data[key] = df.to_dict('records')
            continue
        
        # Runtime files
        content_str = decoded.decode('utf-8')
        if fname == "time_analysis_features.csv":
            df = pd.read_csv(io.StringIO(content_str))
            data['custom_runtime_features'] = df.set_index('Method').to_dict('index')
        elif fname == "time_analysis_instances.csv":
            df = pd.read_csv(io.StringIO(content_str))
            data['custom_runtime_instances'] = df.set_index('Method').to_dict('index')
    
    return data

@app.callback(
    Output('state-for-download', 'data'),
    Input('metric-dropdown', 'value'),
    Input('range-dropdown', 'value'),
    Input('dataset-dropdown', 'value'),
    Input('exclusion-store', 'data'),
    Input('custom-data-store', 'data')
)
def sync_state_for_download(metric, rng, ds, exclusion, custom):
    return {
        'metric': metric,
        'range': rng,
        'selected_dataset': ds,
        'exclusion': exclusion,
        'custom_keys': list(custom.keys()) if custom else []
    }

@app.callback(
    [Output('line-plot', 'figure'),
     Output('score-table', 'data'),
     Output('score-table', 'columns'),
     Output('dataset-dropdown', 'options'),
     Output('dataset-dropdown', 'value'),
     Output('table-title', 'children'),
     Output('cd-diagram', 'figure'),
     Output('runtime-plot', 'figure')],
    [Input('dataset-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('range-dropdown', 'value'),
     Input('exclusion-store', 'data'),
     Input('custom-data-store', 'data'),
     Input('runtime-type-dropdown', 'value')]
)
def update_all_views(selected_ds, metric, rng, exclusion, custom, rtype):
    full_data, keys = get_filtered_data(metric, rng, exclusion, custom or {})
    
    empty = go.Figure(); empty.update_layout(title="No data available")
    
    if not keys:
        return empty, [], [], [], None, "No data", empty, empty

    p_cols = PERCENTAGE_RANGES[rng]['cols']
    x_labels = [f'{float(c)*100:g}%' for c in p_cols]

    all_ds = set()
    for k in keys:
        all_ds.update(pd.DataFrame(full_data[k])['Dataset'].unique())
    unique_datasets = sorted(list(all_ds - set(exclusion.get('datasets', []))))

    active_ds = selected_ds if selected_ds in unique_datasets else (unique_datasets[0] if unique_datasets else None)

    # ─── Line plot ────────────────────────────────────────
    line_fig = go.Figure()
    if active_ds:
        for key in keys:
            method = key.split('|')[0]
            df = pd.DataFrame(full_data[key])
            sub = df[df['Dataset'] == active_ds]
            if sub.empty: continue
            y = sub[p_cols].apply(pd.to_numeric, errors='coerce').mean().values
            style = STYLE_MAP.get(method, {'color': '#666', 'marker': 'circle'})

            if method == 'Random':
                std = sub[p_cols].apply(pd.to_numeric, errors='coerce').std().values
                line_fig.add_trace(go.Scatter(
                    x=x_labels + x_labels[::-1],
                    y=np.concatenate([y + std, y - std][::-1]),
                    fill='toself', fillcolor='rgba(67,97,238,0.18)', line=dict(color='rgba(0,0,0,0)'), showlegend=False
                ))

            line_fig.add_trace(go.Scatter(
                x=x_labels, y=y, mode='lines+markers', name=method,
                line=dict(color=style['color'], width=2.8),
                marker=dict(symbol=style['marker'], size=9)
            ))

    line_fig.update_layout(
        template='plotly_white',
        yaxis=dict(
            range=[0, 1.05],
	        autorange=False,
            title="Performance",
            showgrid=True,
            gridcolor='rgba(210, 210, 220, 0.65)',
            gridwidth=1,
            zeroline=False
        ),
        xaxis_title="Selected features (%)",
        xaxis_showgrid=False,
        margin=dict(l=50, r=30, t=20, b=50)
    )


    # ─── Table ────────────────────────────────────────────
    rows = []
    rankings = {}
    for ds in unique_datasets:
        scores = []
        for k in keys:
            df = pd.DataFrame(full_data[k])
            sub = df[df['Dataset'] == ds]
            if not sub.empty:
                val = sub[p_cols].apply(pd.to_numeric).mean(axis=1).mean()
                if not np.isnan(val):
                    scores.append((k.split('|')[0], val))
        if scores:
            sorted_vals = sorted([v for _, v in scores], reverse=True)
            rankings[ds] = {
                'best': sorted_vals[0] if sorted_vals else None,
                'second': sorted_vals[1] if len(sorted_vals) > 1 else None,
                'lookup': dict(scores)
            }

    methods = sorted({k.split('|')[0] for k in keys})
    for m in methods:
        row = {'Method': m}
        for ds in unique_datasets:
            info = rankings.get(ds, {})
            val = info.get('lookup', {}).get(m)
            if val is None:
                row[ds] = "—"
            else:
                fmt = f"{val:.4f}"
                if val == info.get('best'):
                    row[ds] = f"**{fmt}**"
                elif val == info.get('second'):
                    row[ds] = f"<u>{fmt}</u>"
                else:
                    row[ds] = fmt
        rows.append(row)

    columns = [{"name": c, "id": c, "presentation": "markdown"} for c in ['Method'] + unique_datasets]

    # ─── CD figure ────────────────────────────────────────
    cd_fig = create_cd_figure(full_data, keys, metric, rng, exclusion.get('datasets', []))

    # ─── Runtime plot ─────────────────────────────────────
    rt_fig = go.Figure()
    df_rt = get_effective_runtime_data(rtype, exclusion, custom or {})
    
    if not df_rt.empty:
        x_cols = [c for c in df_rt.columns if c != 'Method']
        try:
            x_vals = [int(c) for c in x_cols]
        except ValueError:
            x_vals = list(range(len(x_cols)))  # fallback

        for _, row in df_rt.iterrows():
            y_vals = row[x_cols].astype(float).values
            valid = ~pd.isna(y_vals) & (y_vals != -1)
            if valid.sum() > 0:
                rt_fig.add_trace(go.Scatter(
                    x=np.array(x_vals)[valid],
                    y=y_vals[valid],
                    name=row['Method'],
                    mode='lines+markers'
                ))

    rt_fig.update_layout(
        template='plotly_white',
        yaxis_type="log",
        xaxis_title=f"Number of {rtype.capitalize()}",
        yaxis_title="Runtime (s)",
        yaxis_showgrid=True,
        yaxis_gridcolor='rgba(210, 210, 220, 0.7)',
        yaxis_gridwidth=1,
        xaxis_showgrid=False,
        margin=dict(l=50, r=30, t=20, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5)
    )

    title = f"Metric: {metric}   |   {PERCENTAGE_RANGES[rng]['label']}"

    return line_fig, rows, columns, [{'label': d, 'value': d} for d in unique_datasets], active_ds, title, cd_fig, rt_fig

def create_cd_figure(full_data, keys, metric, rng, excluded_ds):
    if not keys:
        fig = go.Figure(); fig.update_layout(title="No data"); return fig

    methods = sorted({k.split('|')[0] for k in keys})
    p_cols = PERCENTAGE_RANGES[rng]['cols']

    perf_matrix = []
    ref_df = pd.DataFrame(full_data[keys[0]])
    for ds in sorted(set(ref_df['Dataset']) - set(excluded_ds)):
        row = []
        valid = all(
            f"{m}|{metric}|{rng}" in full_data and
            not pd.DataFrame(full_data[f"{m}|{metric}|{rng}"])[pd.DataFrame(full_data[f"{m}|{metric}|{rng}"])['Dataset'] == ds].empty
            for m in methods
        )
        if not valid: continue
        for m in methods:
            df = pd.DataFrame(full_data[f"{m}|{metric}|{rng}"])
            sub = df[df['Dataset'] == ds]
            row.append(sub[p_cols].apply(pd.to_numeric).mean(axis=1).mean())
        perf_matrix.append(row)

    if len(perf_matrix) < 3 or len(methods) < 2:
        fig = go.Figure(); fig.update_layout(title="Not enough data for CD"); return fig

    perf_matrix = np.array(perf_matrix)
    ranks = np.apply_along_axis(rankdata, 1, -perf_matrix)
    avg_ranks = ranks.mean(axis=0)

    n_ds, n_m = perf_matrix.shape
    cd = 2.728 * np.sqrt(n_m * (n_m + 1) / (6.0 * n_ds))

    fig = go.Figure()
    colors = ['#e74c3c' if i == np.argmin(avg_ranks) else '#3498db' for i in range(n_m)]

    fig.add_trace(go.Bar(y=methods, x=avg_ranks, orientation='h', marker_color=colors,
                         text=[f"{r:.2f}" for r in avg_ranks], textposition='auto'))

    best = min(avg_ranks)
    fig.add_shape(type="line", x0=best, x1=best + cd, y0=-1.1, y1=-1.1,
                  line=dict(color="#1f2937", width=6))
    fig.add_annotation(x=best + cd/2, y=-2.0,
                       text=f"CD = {cd:.2f}",
                       showarrow=False, font=dict(size=14, color="#111827"),
                       bgcolor="white", bordercolor="#9ca3af", borderwidth=1, borderpad=6,
                       align='center')

    fig.update_layout(
        xaxis_title="Average rank (lower is better)",
        yaxis_title="Method",
        template='plotly_white',
        height=max(520, 65 + 48 * n_m),
        margin=dict(l=160, r=40, t=20, b=140),
        font=dict(family="Segoe UI", size=13.5)
    )
    return fig

# ─── Download Callbacks ───────────────────────────────────────

@app.callback(
    Output('download-line-plot', 'data'),
    Input('btn-download-line', 'n_clicks'),
    [State('state-for-download', 'data'), State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def download_line_plot(n, state, custom):
    if not n or not state: raise PreventUpdate
    metric, rng, ds = state['metric'], state['range'], state.get('selected_dataset')
    excl = state.get('exclusion', {'methods':[], 'datasets':[]})
    full, keys = get_filtered_data(metric, rng, excl, custom)
    if not ds or not keys: raise PreventUpdate
    p_cols = PERCENTAGE_RANGES[rng]['cols']
    x_vals = [float(c)*100 for c in p_cols]
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    
    for k in keys:
        m = k.split('|')[0]
        df = pd.DataFrame(full[k])
        sub = df[df['Dataset'] == ds]
        if sub.empty: continue
        y = sub[p_cols].apply(pd.to_numeric).mean().values
        s = STYLE_MAP.get(m, {'color': '#555', 'plt': 'o'})
        
        # ─── Add shaded area for Random method ───────────────────────
        if m == 'Random':
            std = sub[p_cols].apply(pd.to_numeric).std().values
            ax.fill_between(
                x_vals, y - std, y + std,
                color='#4361ee', alpha=0.2, edgecolor='none', linewidth=0
            )
        # ──────────────────────────────────────────────────────────────
        
        ax.plot(x_vals, y, label=m, marker=s['plt'], color=s['color'], lw=2.4)
    
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Selected features (%)")
    ax.set_ylabel("Performance")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return dcc.send_bytes(buf.getvalue(), filename=f"line_{ds}_{metric}_{rng}.pdf")

@app.callback(
    Output('download-cd-plot', 'data'),
    Input('btn-download-cd', 'n_clicks'),
    [State('state-for-download', 'data'), State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def download_cd_plot(n, state, custom):
    if not n or not state: raise PreventUpdate
    metric, rng = state['metric'], state['range']
    excl = state.get('exclusion', {'methods':[], 'datasets':[]})
    full, keys = get_filtered_data(metric, rng, excl, custom)
    if not keys: raise PreventUpdate
    methods = sorted({k.split('|')[0] for k in keys})
    p_cols, excluded_ds = PERCENTAGE_RANGES[rng]['cols'], excl.get('datasets', [])
    perf_matrix = []
    ref_df = pd.DataFrame(full[keys[0]])
    for ds in sorted(set(ref_df['Dataset']) - set(excluded_ds)):
        row = []
        valid = all(f"{m}|{metric}|{rng}" in full and not pd.DataFrame(full[f"{m}|{metric}|{rng}"])[pd.DataFrame(full[f"{m}|{metric}|{rng}"])['Dataset'] == ds].empty for m in methods)
        if not valid: continue
        for m in methods:
            df = pd.DataFrame(full[f"{m}|{metric}|{rng}"])
            row.append(df[df['Dataset'] == ds][p_cols].apply(pd.to_numeric).mean(axis=1).mean())
        perf_matrix.append(row)
    if len(perf_matrix) < 3: raise PreventUpdate
    perf_matrix = np.array(perf_matrix)
    ranks = np.apply_along_axis(rankdata, 1, -perf_matrix)
    avg_ranks = ranks.mean(axis=0); n_ds, n_m = perf_matrix.shape
    cd = 2.728 * np.sqrt(n_m * (n_m + 1) / (6.0 * n_ds))
    buf = io.BytesIO(); fig_height = max(7.0, 0.65 * n_m + 2.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_height), dpi=160)
    y_pos = np.arange(len(methods))
    colors = ['#c0392b' if i == np.argmin(avg_ranks) else '#2980b9' for i in range(n_m)]
    ax.barh(y_pos, avg_ranks, color=colors, height=0.70, align='center')
    ax.set_yticks(y_pos); ax.set_yticklabels(methods, fontsize=13); ax.invert_yaxis()
    ax.set_xlabel('Average rank (lower is better)'); best = min(avg_ranks)
    ax.plot([best, best + cd], [-0.55, -0.55], color='#1f2937', linewidth=7)
    ax.text(best + cd + 0.75, -0.55, f'CD = {cd:.2f}', ha='center', va='center', fontweight='bold')
    plt.subplots_adjust(left=0.32, bottom=0.18); plt.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return dcc.send_bytes(buf.getvalue(), filename=f"CD_{metric}_{rng}.pdf")

@app.callback(
    Output('download-runtime-plot', 'data'),
    Input('btn-download-runtime', 'n_clicks'),
    [State('runtime-type-dropdown', 'value'),
     State('exclusion-store', 'data'),
     State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def download_runtime_plot(n, rtype, exclusion, custom):
    if not n: raise PreventUpdate
    
    df = get_effective_runtime_data(rtype, exclusion or {'methods':[]}, custom or {})
    if df.empty: raise PreventUpdate
    
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    x_cols = [c for c in df.columns if c != 'Method']
    try:
        x_vals = np.array([int(c) for c in x_cols])
    except ValueError:
        x_vals = np.arange(len(x_cols))
    
    colors = itertools.cycle(DEFAULT_COLORS)
    markers = itertools.cycle(MARKERS_LIST)
    
    for _, row in df.iterrows():
        y = row[x_cols].astype(float).values
        valid = ~pd.isna(y) & (y != -1)
        if valid.sum() > 0:
            ax.plot(
                x_vals[valid], y[valid],
                label=row['Method'],
                marker=next(markers),
                color=next(colors),
                lw=1.8
            )
    
    ax.grid(axis='y', which='major', linestyle=':', linewidth=0.65, alpha=0.6, color='gray')
    ax.set_yscale('log')
    ax.set_xlabel(f"Number of {rtype.capitalize()}")
    ax.set_ylabel("Runtime (s)")
    
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=10,
        fontsize=9.5,
        frameon=True,
        edgecolor='0.8',
        columnspacing=1.1,
        handlelength=2.2,
        handletextpad=0.6
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(buf, format='pdf', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return dcc.send_bytes(buf.getvalue(), filename=f"runtime_{rtype}.pdf")

@app.callback(
    [Output('latex-output', 'value'), Output('latex-output', 'style')],
    Input('btn-latex', 'n_clicks'),
    State('score-table', 'data'),
    prevent_initial_call=True
)
def generate_latex(n, table_data):
    if not n: raise PreventUpdate
    df = pd.DataFrame(table_data)
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(r'\*\*(.*?)\*\*', r'\\textbf{\1}', regex=True)
        df[col] = df[col].astype(str).str.replace(r'<u>(.*?)</u>', r'\\underline{\1}', regex=True)
    latex = df.to_latex(index=False, escape=False, column_format='l' + 'c'*(len(df.columns)-1))
    return latex, {'display': 'block', 'width': '100%', 'height': '280px', 'marginTop': '16px', 'fontSize': '13px', 'fontFamily': 'monospace', 'padding': '10px'}

if __name__ == '__main__':
    app.run(debug=False, port=8000)