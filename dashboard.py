"""
Customer Segmentation — Interactive Dashboard
Run with:  python3 dashboard.py
Then open http://127.0.0.1:8050 in a browser.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, dash_table
import joblib

# ─── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("customer_segmentation.csv")

# Guarantee cluster & cluster_name columns exist
if "cluster" not in df.columns:
    pipeline = joblib.load("gmm_clustering_pipeline.joblib")
    df["cluster"] = pipeline.predict(df.drop(columns=["cluster_name"], errors="ignore"))

if "cluster_name" not in df.columns:
    df["cluster_name"] = df["cluster"].astype(str)

FEATURE_COLS = [c for c in df.columns if c not in ("cluster", "cluster_name")]

# Pre-compute stats
cluster_means = df.groupby("cluster")[FEATURE_COLS].mean()
cluster_medians = df.groupby("cluster")[FEATURE_COLS].median()
cluster_counts = df["cluster"].value_counts().sort_index()
persona_map = df.groupby("cluster")["cluster_name"].first().to_dict()
n_clusters = df["cluster"].nunique()

# Normalize for radar / heatmap
from sklearn.preprocessing import MinMaxScaler
cluster_means_norm = pd.DataFrame(
    MinMaxScaler().fit_transform(cluster_means),
    index=cluster_means.index,
    columns=cluster_means.columns,
)

# ─── Colour palette ─────────────────────────────────────────────────────────
COLORS = px.colors.qualitative.T10[:n_clusters]
CLUSTER_COLOR_MAP = {i: COLORS[idx] for idx, i in enumerate(sorted(df["cluster"].unique()))}

# ─── App ─────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Customer Segmentation Dashboard",
)

# ──────────────────────────────  TABS  ──────────────────────────────────────


def _kpi_card(title, value, colour="#3498db"):
    return html.Div(
        [
            html.H4(title, style={"margin": "0", "fontSize": "13px", "color": "#888"}),
            html.H2(
                value,
                style={
                    "margin": "4px 0 0 0",
                    "fontSize": "28px",
                    "color": colour,
                    "fontWeight": "700",
                },
            ),
        ],
        style={
            "background": "#fff",
            "borderRadius": "12px",
            "padding": "18px 24px",
            "boxShadow": "0 2px 8px rgba(0,0,0,.06)",
            "flex": "1",
            "minWidth": "160px",
            "textAlign": "center",
        },
    )


# ──── TAB 1: Overview ────────────────────────────────────────────────────────
def build_overview_tab():
    # KPI cards
    kpi_row = html.Div(
        [
            _kpi_card("Total Customers", f"{len(df):,}"),
            _kpi_card("Segments", str(n_clusters), "#e67e22"),
            _kpi_card("Avg Balance", f"${df['BALANCE'].mean():,.0f}", "#27ae60"),
            _kpi_card("Avg Purchases", f"${df['PURCHASES'].mean():,.0f}", "#8e44ad"),
        ],
        style={"display": "flex", "gap": "16px", "marginBottom": "20px", "flexWrap": "wrap"},
    )

    # Cluster size bar
    fig_bar = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        color=cluster_counts.index.astype(str),
        color_discrete_sequence=COLORS,
        labels={"x": "Cluster", "y": "Customers"},
        title="Customers per Cluster",
        text=cluster_counts.values,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, template="plotly_white", height=380)

    # Cluster size pie
    fig_pie = px.pie(
        names=[f"C{i} – {persona_map[i]}" for i in cluster_counts.index],
        values=cluster_counts.values,
        color_discrete_sequence=COLORS,
        title="Cluster Proportions",
        hole=0.4,
    )
    fig_pie.update_layout(height=380, template="plotly_white")

    return html.Div(
        [
            kpi_row,
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_bar), style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=fig_pie), style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ]
    )



# ──── TAB 4: Cluster Deep Dive ──────────────────────────────────────────────
def build_deep_dive_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Select Cluster:", style={"fontWeight": "600", "marginRight": "10px"}),
                    dcc.Dropdown(
                        id="cluster-dropdown",
                        options=[
                            {"label": f"Cluster {i} – {persona_map[i]}", "value": i}
                            for i in sorted(df["cluster"].unique())
                        ],
                        value=sorted(df["cluster"].unique())[0],
                        style={"width": "400px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "20px"},
            ),
            html.Div(id="deep-dive-content"),
        ]
    )


# ──── TAB 5: Data Explorer ──────────────────────────────────────────────────
def build_data_tab():
    show_cols = ["cluster", "cluster_name"] + FEATURE_COLS[:10]
    return html.Div(
        [
            html.H4("Customer Data (first 500 rows)", style={"marginBottom": "12px"}),
            html.Div(
                [
                    html.Label("Filter by cluster:", style={"marginRight": "8px"}),
                    dcc.Dropdown(
                        id="data-cluster-filter",
                        options=[{"label": "All", "value": "all"}]
                        + [{"label": f"C{i} – {persona_map[i]}", "value": i} for i in sorted(df["cluster"].unique())],
                        value="all",
                        style={"width": "350px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
            ),
            html.Div(id="data-table-container"),
        ]
    )


# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Customer Segmentation Dashboard",
                    style={"margin": "0", "fontSize": "26px", "fontWeight": "700"},
                ),
                html.P(
                    "GMM Clustering with 8 Segments — Interactive Analysis",
                    style={"margin": "4px 0 0 0", "fontSize": "14px", "color": "#888"},
                ),
            ],
            style={
                "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                "color": "white",
                "padding": "24px 32px",
                "borderRadius": "0 0 16px 16px",
                "marginBottom": "20px",
            },
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            children=[
                dcc.Tab(label="📋 Overview", value="tab-overview"),
                dcc.Tab(label="🔍 Cluster Deep Dive", value="tab-deep"),
                dcc.Tab(label="📂 Data Explorer", value="tab-data"),
            ],
            style={"marginBottom": "16px"},
        ),
        html.Div(id="tab-content", style={"padding": "0 20px 20px 20px"}),
    ],
    style={
        "fontFamily": "'Segoe UI', 'Inter', sans-serif",
        "background": "#f5f6fa",
        "minHeight": "100vh",
    },
)


# ─── Callbacks ───────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-overview":
        return build_overview_tab()
    elif tab == "tab-profiles":
        return build_profiles_tab()
    elif tab == "tab-spending":
        return build_spending_tab()
    elif tab == "tab-deep":
        return build_deep_dive_tab()
    elif tab == "tab-data":
        return build_data_tab()
    return html.Div("Select a tab.")


@app.callback(Output("deep-dive-content", "children"), Input("cluster-dropdown", "value"))
def deep_dive(cluster_id):
    if cluster_id is None:
        return html.Div()

    c_data = df[df["cluster"] == cluster_id]
    means = cluster_means.loc[cluster_id]
    meds = cluster_medians.loc[cluster_id]

    # KPIs
    kpis = html.Div(
        [
            _kpi_card("Customers", f"{len(c_data):,}", CLUSTER_COLOR_MAP[cluster_id]),
            _kpi_card("Share", f"{len(c_data)/len(df)*100:.1f}%"),
            _kpi_card("Avg Balance", f"${means['BALANCE']:,.0f}", "#27ae60"),
            _kpi_card("Avg Purchases", f"${means['PURCHASES']:,.0f}", "#8e44ad"),
            _kpi_card("Avg Credit Limit", f"${means['CREDIT_LIMIT']:,.0f}", "#e67e22"),
        ],
        style={"display": "flex", "gap": "12px", "marginBottom": "20px", "flexWrap": "wrap"},
    )

    # Radar for this cluster
    radar_features = [
        "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
        "PAYMENTS", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT",
    ]
    vals = cluster_means_norm.loc[cluster_id, radar_features].tolist()
    vals += vals[:1]

    fig_radar = go.Figure(
        go.Scatterpolar(
            r=vals,
            theta=radar_features + [radar_features[0]],
            fill="toself",
            line=dict(color=CLUSTER_COLOR_MAP[cluster_id], width=3),
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        title=f"Cluster {cluster_id} Profile",
        height=400,
        template="plotly_white",
        showlegend=False,
    )

    # Feature comparison bar (cluster vs overall)
    comp = pd.DataFrame({"Cluster Mean": means, "Overall Mean": df[FEATURE_COLS].mean()})
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="Overall", x=comp.index, y=comp["Overall Mean"], marker_color="#bdc3c7"))
    fig_comp.add_trace(
        go.Bar(name=f"Cluster {cluster_id}", x=comp.index, y=comp["Cluster Mean"], marker_color=CLUSTER_COLOR_MAP[cluster_id])
    )
    fig_comp.update_layout(
        barmode="group",
        title="Cluster vs Overall Averages",
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45,
    )

    # Stats table
    stats_df = pd.DataFrame({"Mean": means.round(1), "Median": meds.round(1)})
    stats_table = dash_table.DataTable(
        data=stats_df.reset_index().rename(columns={"index": "Feature"}).to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["Feature", "Mean", "Median"]],
        style_table={"maxHeight": "350px", "overflowY": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
    )

    return html.Div(
        [
            html.H3(
                f"Cluster {cluster_id} — {persona_map[cluster_id]}",
                style={"color": CLUSTER_COLOR_MAP[cluster_id]},
            ),
            kpis,
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_radar), style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=fig_comp), style={"flex": "1.2"}),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
            html.H4("Feature Statistics", style={"marginTop": "20px"}),
            stats_table,
        ]
    )


@app.callback(Output("data-table-container", "children"), Input("data-cluster-filter", "value"))
def update_data_table(cluster_val):
    filtered = df if cluster_val == "all" else df[df["cluster"] == cluster_val]
    filtered = filtered.head(500)
    show_cols = ["cluster", "cluster_name"] + FEATURE_COLS

    return dash_table.DataTable(
        data=filtered[show_cols].round(2).to_dict("records"),
        columns=[{"name": c, "id": c} for c in show_cols],
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px", "minWidth": "100px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
        ],
    )


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Dashboard starting at http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)
