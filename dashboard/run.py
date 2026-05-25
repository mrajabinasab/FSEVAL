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
import networkx as nx
import math
import operator
from scipy.stats import rankdata, studentized_range
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

@server.route('/random/index.html')
def serve_random_file():
    return send_from_directory('random', 'index.html')

@server.route('/random/')
def serve_random_dir():
    return send_from_directory('random', 'index.html')

@server.route('/random/<path:filename>')
def serve_random_assets(filename):
    return send_from_directory('random', filename)

@server.route('/files/<path:filename>')
def download_file(filename):
    return send_from_directory('files', filename, as_attachment=True)


DATA_DIR = 'resources'

BENCHMARK_METRICS = ['CLSACC', 'NMI', 'ACC', 'AUC', 'AAD']
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
    '10Percent':  {'label': '0.5% to 10%',  'cols': [str(np.round(p, 3)) for p in np.arange(0.005, 0.1001, 0.005)]},
    '100Percent': {'label': '5% to 100%',   'cols': [str(np.round(p, 2)) for p in np.arange(0.05, 1.001, 0.05)]}
}

MARKERS_LIST = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'P', 'H']
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


class UnifiedPlotter:
    @staticmethod
    def graph_ranks(avranks, names, p_values=None, cd=None, title="", name="", color_mode='color'):
        plt.style.use('default')
        nnames = list(names)
        sums = np.array(avranks)
        k = len(sums)

        if color_mode == 'bw':
            colors = ['black'] * k
        else:
            colormap = matplotlib.colormaps['Dark2']
            colors = [colormap(i / k) for i in range(k)]

        lowv = min(1, int(math.floor(min(sums))))
        highv = max(k, int(math.ceil(max(sums))))

        title_gap   = 0.07
        cline       = 0.32
        step_height = 0.16
        bottom_pad  = 0.12
        mid = math.ceil(k / 2)

        g = nx.Graph()
        g.add_nodes_from(nnames)

        if cd is not None:
            for i in range(k):
                for j in range(i + 1, k):
                    if abs(sums[i] - sums[j]) < cd:
                        g.add_edge(nnames[i], nnames[j])
        elif p_values is not None:
            for p in p_values:
                if not p[3]:
                    g.add_edge(p[0], p[1])

        cliques = sorted(
            list(nx.find_cliques(g)),
            key=lambda x: np.min([sums[nnames.index(n)] for n in x])
        )
        active_cliques = [c for c in cliques if len(c) > 1]

        curr_clq_y = cline + 0.08
        for _ in active_cliques:
            curr_clq_y += 0.09

        sorted_idx    = np.argsort(sums)
        label_elbow_y = curr_clq_y + 0.02

        label_ys = []
        for i in range(k):
            if i < mid:
                ly = label_elbow_y + i * step_height
            else:
                ly = label_elbow_y + (k - i - 1) * step_height
            label_ys.append(ly)

        max_label_y = max(label_ys)
        true_max_y  = max_label_y + bottom_pad

        width      = 12
        fig_height = true_max_y * 3.5

        fig = plt.figure(figsize=(width, fig_height))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(true_max_y, 0)

        textspace  = 3.2
        scalewidth = width - 2 * textspace

        def xpos(rank):
            rel = (rank - lowv) / (highv - lowv) if highv > lowv else 0
            return (textspace + scalewidth * (1 - rel)) / width

        display_title = f"{name} - {title}"
        ax.text(0.5, title_gap, display_title,
                ha='center', va='top',
                fontsize=24, fontweight='bold')

        ax.plot([textspace / width, (width - textspace) / width],
                [cline, cline], color='black', lw=3)

        midv = (lowv + highv) / 2.0
        for val, label in [(lowv, str(lowv)),
                           (midv, f'{midv:.1f}' if midv != int(midv) else str(int(midv))),
                           (highv, str(highv))]:
            p = xpos(val)
            ax.plot([p, p], [cline - 0.03, cline], color='black', lw=2)
            ax.text(p, cline - 0.04, label,
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        curr_clq_y = cline + 0.08
        for clq in active_cliques:
            r_vals = [sums[nnames.index(n)] for n in clq]
            p1, p2 = xpos(min(r_vals)), xpos(max(r_vals))
            ax.plot([p1, p2], [curr_clq_y, curr_clq_y],
                    color='black', lw=9, solid_capstyle='butt', zorder=5)
            curr_clq_y += 0.09

        for i, idx in enumerate(sorted_idx):
            rank, name_i = sums[idx], nnames[idx]
            m_col      = colors[i % len(colors)]
            side_right = i < mid

            px = xpos(rank)
            lx = (width - textspace + 0.5) / width if side_right \
                 else (textspace - 0.5) / width
            ly = label_ys[i]

            ax.plot([px, px, lx], [cline, ly, ly],
                    color=m_col, lw=3.5, solid_capstyle='round', zorder=3)

            ha = 'left' if side_right else 'right'
            
            ax.text(lx, ly - 0.010,
                    name_i,
                    color=m_col,
                    ha=ha, va='bottom',
                    fontsize=18, fontweight='bold')
            
            rank_label = f"MARS Score: {rank:.2f}" if title == "MARS" else f"Rank: {rank:.2f}"
            ax.text(lx, ly + 0.015,
                    rank_label,
                    color=m_col,
                    ha=ha, va='top',
                    fontsize=11, fontweight='normal', alpha=0.85)

        return fig


def load_data():
    data_dict = {}
    pattern = re.compile(r'^(.+?)_(CLSACC|NMI|ACC|AUC|AAD|Stability_CLSACC|Stability_NMI|Stability_ACC|Stability_AUC|Stability_AAD)_(10Percent|100Percent)\.csv$')
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

app.layout = html.Div(style={'backgroundColor': '#f8f9fc', 'minHeight': '100vh', 'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"}, children=[

    dcc.Store(id='exclusion-store', data={'datasets': [], 'methods': []}),
    dcc.Store(id='custom-data-store', data={}),
    dcc.Store(id='state-for-download'),
    dcc.Download(id='download-line-plot'),
    dcc.Download(id='download-cd-standard'),
    dcc.Download(id='download-cd-mars'),
    dcc.Download(id='download-runtime-plot'),
    dcc.Download(id='download-stability-plot'),

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

    html.Div(id='exclude-container', style={'display': 'none', 'padding': '16px 32px', 'backgroundColor': '#fefce8', 'borderBottom': '1px solid #fef08a'}, children=[
        html.Div(style={'maxWidth': '640px', 'margin': '0 auto'}, children=[
            dcc.Textarea(id='exclude-input', value='DATASETS = []\nMETHODS = []', style={'width': '100%', 'height': '90px', 'fontFamily': 'monospace', 'fontSize': '13px', 'padding': '8px'}),
            html.Button('Apply Exclusion', id='btn-exclude-apply', style={**ORANGE_BTN, 'marginTop': '12px'})
        ])
    ]),

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
                html.H3(id='stability-title', style={'margin': '0 0 16px 0', 'fontSize': '1.32rem', 'borderLeft': '5px solid #f39c12', 'paddingLeft': '12px', 'color': '#1e293b'}),
                dcc.Graph(id='stability-bar-plot', style={'height': '450px'}),
                html.Div(style={'textAlign': 'center', 'marginTop': '12px'}, children=[
                    html.Button('Download Stability Plot (PDF)', id='btn-download-stability', style=GREEN_BTN)
                ])
            ]),

            # === NEW SIDE-BY-SIDE CD SECTION ===
            html.Div(style=CARD_STYLE, children=[
                html.H3("Critical Difference Diagrams", style={'margin': '0 0 20px 0', 'fontSize': '1.32rem', 'borderLeft': '5px solid #8e44ad', 'paddingLeft': '12px', 'color': '#1e293b'}),
                html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("Standard (Nemenyi)", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id='cd-standard', style={'height': '620px'}),
                        html.Div(style={'textAlign': 'center', 'marginTop': '12px'}, children=[
                            html.Button('Download Standard CD (PDF)', id='btn-download-cd-standard', style=GREEN_BTN)
                        ])
                    ]),
                    html.Div(style={'flex': '1'}, children=[
                        html.H4("MARS (Weighted Ranks)", style={'textAlign': 'center', 'color': '#8e44ad', 'marginBottom': '10px'}),
                        dcc.Graph(id='cd-mars', style={'height': '620px'}),
                        html.Div(style={'textAlign': 'center', 'marginTop': '12px'}, children=[
                            html.Button('Download MARS CD (PDF)', id='btn-download-cd-mars', style=PURPLE_BTN)
                        ])
                    ])
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


def get_filtered_data(metric, rng, exclusion, custom):
    full = {**INITIAL_DATA, **(custom or {})}
    ex_methods = exclusion.get('methods', [])
    perf_keys = [k for k in full if f"|{metric}|{rng}" in k
                 and 'Stability' not in k
                 and k.split('|')[0] not in ex_methods]
    return full, perf_keys


def get_effective_runtime_data(rtype, exclusion, custom):
    base = RUNTIME_DATA.get(rtype, {})
    custom_key = f'custom_runtime_{rtype}'
    custom_data = (custom or {}).get(custom_key, {})
    
    merged = {**base, **custom_data}
    
    excluded_methods = set(exclusion.get('methods', []))
    filtered = {m: v for m, v in merged.items() if m not in excluded_methods}
    
    if not filtered:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(filtered, orient='index').reset_index(names='Method')
    return df


def compute_weighted_rank_matrix(data_matrix):
    n_ds, n_clfs = data_matrix.shape
    weighted_ranks = np.zeros_like(data_matrix, dtype=float)

    for i in range(n_ds):
        row = data_matrix[i, :]
        mx, mn = np.max(row), np.min(row)

        temp = (-row).argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(n_clfs) + 1.0

        if mx == mn:
            weighted_ranks[i, :] = ranks
            continue

        not_min_mask = row > mn
        weights = np.zeros(n_clfs, dtype=float)

        weights[not_min_mask] = (mx - mn) / (row[not_min_mask] - mn)

        w_vals = np.sort(weights[not_min_mask])
        if len(w_vals) > 1:
            delta = np.max(np.diff(w_vals))
        elif len(w_vals) > 0:
            delta = w_vals[0]
        else:
            delta = 1.0

        weights[~not_min_mask] = (w_vals[-1] + delta) if len(w_vals) > 0 else 1.0

        weighted_ranks[i, :] = ranks * weights

    return weighted_ranks


def compute_cd_mars(wr_mat, n_ds):
    k = wr_mat.shape[1]
    theo_std = math.sqrt((k**2 - 1) / 12)
    obs_std = wr_mat.std()
    cd_base = 2.8 * math.sqrt(k * (k + 1) / (6 * n_ds))
    cd_mars = cd_base * (obs_std / theo_std)
    return cd_mars


def create_cd_figures(full_data, keys, metric, rng, excluded_ds):
    """Return both Standard and MARS matplotlib figures"""
    if not keys:
        empty = plt.figure()
        plt.close(empty)
        return empty, empty

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
        empty = plt.figure()
        plt.close(empty)
        return empty, empty

    perf_matrix = np.array(perf_matrix)
    n_ds, n_m = perf_matrix.shape
    is_lower_better = (metric == 'AAD')

    # === STANDARD ===
    ranking_matrix = perf_matrix if is_lower_better else -perf_matrix
    ranks = np.apply_along_axis(rankdata, 1, ranking_matrix)
    std_ranks = ranks.mean(axis=0)
    cd_std = 2.728 * np.sqrt(n_m * (n_m + 1) / (6.0 * n_ds))
    fig_std = UnifiedPlotter.graph_ranks(std_ranks, methods, cd=cd_std, title="Standard", name=metric)

    # === MARS ===
    X = perf_matrix if not is_lower_better else -perf_matrix
    wr_mat = compute_weighted_rank_matrix(X)
    mars_scores = wr_mat.mean(axis=0)
    cd_mars = compute_cd_mars(wr_mat, n_ds)
    fig_mars = UnifiedPlotter.graph_ranks(mars_scores, methods, cd=cd_mars, title="MARS", name=metric)

    return fig_std, fig_mars


@app.callback(
    [Output('line-plot', 'figure'),
     Output('score-table', 'data'),
     Output('score-table', 'columns'),
     Output('dataset-dropdown', 'options'),
     Output('dataset-dropdown', 'value'),
     Output('table-title', 'children'),
     Output('cd-standard', 'figure'),
     Output('cd-mars', 'figure'),
     Output('runtime-plot', 'figure'),
     Output('stability-bar-plot', 'figure'),  
     Output('stability-title', 'children')],
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
        return empty, [], [], [], None, "No data", empty, empty, empty, empty, "No data"

    p_cols = PERCENTAGE_RANGES[rng]['cols']
    x_labels = [f'{float(c)*100:g}%' for c in p_cols]

    all_ds = set()
    for k in keys:
        all_ds.update(pd.DataFrame(full_data[k])['Dataset'].unique())
    unique_datasets = sorted(list(all_ds - set(exclusion.get('datasets', []))))

    active_ds = selected_ds if selected_ds in unique_datasets else (unique_datasets[0] if unique_datasets else None)
    is_lower_better = (metric == 'AAD')
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
            range=[0, 1.05] if metric != 'AAD' else None,
            autorange=metric == 'AAD',
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
            sorted_vals = sorted([v for _, v in scores], reverse=not is_lower_better)
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

    stab_fig = go.Figure()
    stab_title = f"Stability Analysis: {metric}"
    
    if active_ds:
        stab_methods = []
        stab_values = []
        stab_colors = []

        for m in methods:
            stab_key = f"{m}|Stability_{metric}|{rng}"
            if stab_key in full_data:
                df_s = pd.DataFrame(full_data[stab_key])
                sub_s = df_s[df_s['Dataset'] == active_ds]
                if not sub_s.empty:
                    val = sub_s['Stability'].iloc[0]
                    stab_methods.append(m)
                    stab_values.append(val)
                    stab_colors.append(STYLE_MAP.get(m, {'color': '#666'})['color'])
        
        if stab_methods:
            stab_fig.add_trace(go.Bar(
                x=stab_methods,
                y=stab_values,
                marker_color=stab_colors,
                text=[f"{v:.4f}" for v in stab_values],
                textposition='auto'
            ))
            stab_fig.update_layout(
                template='plotly_white',
                xaxis_title="Method",
                yaxis_title="Stability Score",
                margin=dict(l=50, r=30, t=20, b=50)
            )
        else:
            stab_fig.update_layout(title="No Stability data found for this selection")

    # Create both CD figures
    fig_std_mat, fig_mars_mat = create_cd_figures(full_data, keys, metric, rng, exclusion.get('datasets', []))

    # Convert to Plotly images for display
    def mat_to_plotly(fig_mat):
        if fig_mat is None or len(fig_mat.axes) == 0:
            empty = go.Figure(); empty.update_layout(title="No data")
            return empty
        buf = io.BytesIO()
        fig_mat.savefig(buf, format='png', bbox_inches='tight', dpi=180)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_mat)
        plotly_fig = go.Figure(go.Image(source=f'data:image/png;base64,{img_base64}'))
        plotly_fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=620
        )
        return plotly_fig

    cd_std_plotly = mat_to_plotly(fig_std_mat)
    cd_mars_plotly = mat_to_plotly(fig_mars_mat)

    rt_fig = go.Figure()
    df_rt = get_effective_runtime_data(rtype, exclusion, custom or {})
    
    if not df_rt.empty:
        x_cols = [c for c in df_rt.columns if c != 'Method']
        try:
            x_vals = [int(c) for c in x_cols]
        except ValueError:
            x_vals = list(range(len(x_cols)))

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

    return line_fig, rows, columns, [{'label': d, 'value': d} for d in unique_datasets], \
           active_ds, title, cd_std_plotly, cd_mars_plotly, rt_fig, stab_fig, stab_title


# ==================== DOWNLOAD CALLBACKS ====================

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

        if m == 'Random':
            std = sub[p_cols].apply(pd.to_numeric).std().values
            ax.fill_between(
                x_vals, y - std, y + std,
                color='#4361ee', alpha=0.2, edgecolor='none', linewidth=0
            )
        
        ax.plot(x_vals, y, label=m, marker=s['plt'], color=s['color'], lw=2.4)
    
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    if metric != 'AAD':
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
    Output('download-stability-plot', 'data'),
    Input('btn-download-stability', 'n_clicks'),
    [State('state-for-download', 'data'), State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def download_stability_plot(n, state, custom):
    if not n or not state: raise PreventUpdate
    metric, rng = state['metric'], state['range']
    ds = state.get('selected_dataset')
    excl = state.get('exclusion', {'methods': [], 'datasets': []})
    full, _ = get_filtered_data(metric, rng, excl, custom)
    if not ds: raise PreventUpdate

    methods = sorted({k.split('|')[0] for k in full if f"|{metric}|{rng}" in k and 'Stability' not in k})
    stab_methods, stab_values, stab_colors = [], [], []

    for m in methods:
        stab_key = f"{m}|Stability_{metric}|{rng}"
        if stab_key in full:
            df_s = pd.DataFrame(full[stab_key])
            sub_s = df_s[df_s['Dataset'] == ds]
            if not sub_s.empty:
                stab_methods.append(m)
                stab_values.append(sub_s['Stability'].iloc[0])
                stab_colors.append(STYLE_MAP.get(m, {'color': '#666'})['color'])

    if not stab_methods: raise PreventUpdate

    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.bar(stab_methods, stab_values, color=stab_colors)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_xlabel("Method")
    ax.set_ylabel("Stability Score")
    ax.set_title(f"Stability: {metric} | {ds} | {PERCENTAGE_RANGES[rng]['label']}")
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    plt.tight_layout()
    plt.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return dcc.send_bytes(buf.getvalue(), filename=f"stability_{ds}_{metric}_{rng}.pdf")


@app.callback(
    Output('download-cd-standard', 'data'),
    Input('btn-download-cd-standard', 'n_clicks'),
    [State('state-for-download', 'data'), State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def download_cd_standard(n, state, custom):
    if not n or not state: raise PreventUpdate
    metric, rng = state['metric'], state['range']
    excl = state.get('exclusion', {'methods':[], 'datasets':[]})
    full, keys = get_filtered_data(metric, rng, excl, custom)
    if not keys: raise PreventUpdate
    fig_std, _ = create_cd_figures(full, keys, metric, rng, excl.get('datasets', []))
    buf = io.BytesIO()
    fig_std.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(fig_std)
    return dcc.send_bytes(buf.getvalue(), filename=f"CD_Standard_{metric}_{rng}.pdf")


@app.callback(
    Output('download-cd-mars', 'data'),
    Input('btn-download-cd-mars', 'n_clicks'),
    [State('state-for-download', 'data'), State('custom-data-store', 'data')],
    prevent_initial_call=True
)
def download_cd_mars(n, state, custom):
    if not n or not state: raise PreventUpdate
    metric, rng = state['metric'], state['range']
    excl = state.get('exclusion', {'methods':[], 'datasets':[]})
    full, keys = get_filtered_data(metric, rng, excl, custom)
    if not keys: raise PreventUpdate
    _, fig_mars = create_cd_figures(full, keys, metric, rng, excl.get('datasets', []))
    buf = io.BytesIO()
    fig_mars.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(fig_mars)
    return dcc.send_bytes(buf.getvalue(), filename=f"CD_MARS_{metric}_{rng}.pdf")


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
    Output('state-for-download', 'data'),
    [Input('metric-dropdown', 'value'),
     Input('range-dropdown', 'value'),
     Input('dataset-dropdown', 'value'),
     Input('exclusion-store', 'data'),
     Input('custom-data-store', 'data')]
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
        
        match = re.match(r'^(.+?)_(CLSACC|NMI|ACC|AUC|AAD|Stability_CLSACC|Stability_NMI|Stability_ACC|Stability_AUC|Stability_AAD)_(10Percent|100Percent)\.csv$', fname)
        if match:
            method, metric, suffix = match.groups()
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df.columns = [str(c) for c in df.columns]
            key = f"{method}|{metric}|{suffix}"
            data[key] = df.to_dict('records')
            continue
        
        content_str = decoded.decode('utf-8')
        if fname == "time_analysis_features.csv":
            df = pd.read_csv(io.StringIO(content_str))
            data['custom_runtime_features'] = df.set_index('Method').to_dict('index')
        elif fname == "time_analysis_instances.csv":
            df = pd.read_csv(io.StringIO(content_str))
            data['custom_runtime_instances'] = df.set_index('Method').to_dict('index')
    
    return data


if __name__ == '__main__':
    app.run(debug=False, port=8000)
