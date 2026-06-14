"""
Gutbiome Fingerprinting — Interactive Atlas Dashboard (Deep Sea Scientific Edition)
frontend/app.py
"""

import json
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────── CONFIG ────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

# Deep Sea Scientific Palette
C_BG         = "#0a0f1d"
C_SURFACE    = "#0d1526"
C_BORDER     = "rgba(255,255,255,0.06)"
C_TEAL       = "#00f5d4"
C_TEAL_DIM   = "#028090"
C_CORAL      = "#ff5a5f"
C_AMBER      = "#f77f00"
C_TEXT       = "#cdd9e5"
C_MUTED      = "#5e7a8a"
C_GRID       = "#111d2e"

PALETTE = [C_TEAL, "#0096c7", "#48cae4", C_AMBER, C_CORAL,
           "#90e0ef", "#f4a261", "#e76f51", "#2ec4b6", "#cbf3f0"]

st.set_page_config(
    page_title="Gutbiome Atlas",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Glassmorphism & Custom Layout Styles Override
st.markdown(f"""
<style>
    .stApp {{ background-color: {C_BG}; color: {C_TEXT}; }}
    header, [data-testid="stHeader"] {{ background: transparent !important; }}
    
    /* Sidebar Overhaul */
    section[data-testid="stSidebar"] {{
        background-color: {C_SURFACE} !important;
        border-right: 1px solid {C_BORDER};
    }}
    section[data-testid="stSidebar"] .stMarkdown h2 {{
        color: {C_TEAL} !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    /* Glassmorphic Cards */
    .glass-card {{
        background: rgba(13, 21, 38, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {C_BORDER};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }}
    
    .metric-title {{ color: {C_MUTED}; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 4px; }}
    .metric-value {{ color: #ffffff; font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .metric-delta {{ color: {C_TEAL}; font-size: 12px; margin-top: 4px; font-weight: 500; }}
    
    /* Custom Indicator Pill */
    .status-pill {{
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
        border: 1px solid rgba(0, 245, 212, 0.2);
        background: rgba(0, 245, 212, 0.05);
        color: {C_TEAL};
    }}
    .status-pill.error {{
        border: 1px solid rgba(255, 90, 95, 0.2);
        background: rgba(255, 90, 95, 0.05);
        color: {C_CORAL};
    }}
    
    /* Sidebar Input Tweaks */
    div[data-baseweb="textarea"] {{ background-color: {C_BG} !important; border: 1px solid {C_BORDER} !important; border-radius: 8px !important; }}
    textarea {{ color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }}
    
    /* Section Labels */
    .section-label {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {C_TEAL_DIM};
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 12px;
    }}
    
    /* Navigation Tabs */
    button[data-baseweb="tab"] {{ color: {C_MUTED} !important; font-weight: 600 !important; font-size: 14px !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {C_TEAL} !important; border-bottom-color: {C_TEAL} !important; }}
    div[data-穩testid="stHorizontalBlock"] {{ gap: 16px !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── HELPERS ───────────────────────────────────────

def safe_get(obj, *keys, default=None):
    for k in keys:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(k)
        elif isinstance(obj, list) and isinstance(k, int):
            obj = obj[k] if k < len(obj) else None
        else:
            return default
    return obj if obj is not None else default

@st.cache_data(ttl=30, show_spinner=False)
def fetch_samples() -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/api/v1/samples", params={"limit": 2000}, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", data) if isinstance(data, dict) else data
        
        if not isinstance(items, list):
            return pd.DataFrame()
            
        flattened_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            flat = {k: v for k, v in item.items() if not isinstance(v, (dict, list))}
            
            # Extract coordinates explicitly
            if "umap_coords" in item and isinstance(item["umap_coords"], dict):
                flat["umap_x"] = item["umap_coords"].get("x")
                flat["umap_y"] = item["umap_coords"].get("y")
            elif "umap_x" not in flat and "coordinates" in item:
                coords = item["coordinates"]
                if isinstance(coords, list) and len(coords) >= 2:
                    flat["umap_x"], flat["umap_y"] = coords[0], coords[1]
                    
            flattened_items.append(flat)
            
        return pd.DataFrame(flattened_items)
    except Exception as e:
        st.error(f"Could not load samples: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def fetch_clusters() -> list:
    try:
        r = requests.get(f"{API_BASE}/api/v1/clusters", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("clusters", data) if isinstance(data, dict) else data
    except Exception:
        return []

def build_umap_figure(df: pd.DataFrame, color_by: str, pred_point: dict | None = None) -> go.Figure:
    if df.empty or "umap_x" not in df.columns or "umap_y" not in df.columns:
        return go.Figure()

    df["_color_str"] = df[color_by].astype(str) if color_by in df.columns else df.iloc[:, 0].astype(str)

    fig = px.scatter(
        df, x="umap_x", y="umap_y", color="_color_str",
        color_discrete_sequence=PALETTE,
        hover_data={"sample_id": True, "disease": True, "country": True, "age": True, "umap_x": False, "umap_y": False, "_color_str": False},
        labels={"_color_str": color_by.replace("_", " ").title()},
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=5.5, opacity=0.75, line=dict(width=0)))

    if pred_point and pred_point.get("umap_x") is not None and pred_point.get("umap_y") is not None:
        fig.add_trace(go.Scatter(
            x=[pred_point["umap_x"]], y=[pred_point["umap_y"]],
            mode="markers",
            marker=dict(symbol="star", size=20, color=C_CORAL, line=dict(color="#ffffff", width=1.5)),
            name="⭐ Target Profile",
            hovertemplate="<b>Analyzed Sample Location</b><br>UMAP: (%{x:.3f}, %{y:.3f})<extra></extra>",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(10,15,29,0.6)", bordercolor=C_BORDER, borderwidth=1, font=dict(size=11, color=C_TEXT)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Dimension 1", gridcolor=C_GRID, zeroline=False, showticklabels=False),
        yaxis=dict(title="Dimension 2", gridcolor=C_GRID, zeroline=False, showticklabels=False),
        height=540,
    )
    return fig

def dist_donut(counts: dict, title: str) -> go.Figure:
    if not counts: return go.Figure()
    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        textinfo="percent", textfont=dict(size=10, color="#ffffff"),
        marker=dict(colors=PALETTE, line=dict(color=C_SURFACE, width=2)),
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(font=dict(size=10, color=C_TEXT), bgcolor="transparent", orientation="h", y=-0.1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=10, b=5), height=220,
    )
    return fig

def horiz_bar(counts: dict, title: str, color: str = C_TEAL) -> go.Figure:
    if not counts: return go.Figure()
    df = pd.DataFrame(list(counts.items()), columns=["label", "value"]).sort_values("value")
    fig = go.Figure(go.Bar(
        x=df["value"], y=df["label"], orientation="h",
        marker=dict(color=color, opacity=0.8, line=dict(color=color, width=0)),
        text=df["value"].round(3), textposition="outside", textfont=dict(color=C_TEXT, size=10)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT),
        xaxis=dict(gridcolor=C_GRID, zeroline=False, title="Score Value"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=30, t=10, b=10),
        height=max(180, 30 * len(df)),
    )
    return fig

# ─────────────────────────── SIDEBAR — SIMULATOR ───────────────────────────

with st.sidebar:
    st.markdown("## 🧬 Sample Simulator")
    st.caption("Enter composition markers to predict population group membership.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    example_payload = json.dumps({
        "Bacteroides": 0.24, "Firmicutes": 0.18, "Prevotella": 0.09,
        "Faecalibacterium": 0.07, "Ruminococcus": 0.05, "Akkermansia": 0.04,
        "Bifidobacterium": 0.03, "Lachnospiraceae": 0.06, "Blautia": 0.05, "Parabacteroides": 0.19
    }, indent=2)

    raw_input = st.text_area("Taxonomic Profile Vector (JSON)", value=example_payload, height=260)

    # Live validation check
    try:
        parsed_vector = json.loads(raw_input)
        total_sum = sum(float(v) for v in parsed_vector.values())
        if abs(total_sum - 1.0) < 1e-4:
            st.markdown('<div class="status-pill">✓ Abundances sum to 1.0000 — normalized</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-pill error">⚠ Sums to {total_sum:.4f} (Expected exactly 1.0)</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="status-pill error">⚠ Invalid JSON format structure</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_prediction = st.button("🔬 Predict Biome Cluster", use_container_width=True)

    if run_prediction:
        try:
            payload = json.loads(raw_input)
            with st.spinner("Executing inference matrix..."):
                r = requests.post(f"{API_BASE}/api/v1/fingerprint/predict", json={"abundances": payload}, timeout=15)
                r.raise_for_status()
                st.session_state["pred_result"] = r.json()
        except Exception as e:
            st.sidebar.error(f"Prediction Pipeline Failed: {e}")

    pred_result = st.session_state.get("pred_result", {})

    if pred_result:
        st.markdown("<div class='section-label'>Prediction Metrics</div>", unsafe_allow_html=True)
        
        c_id = safe_get(pred_result, "cluster_id", default="—")
        d_class = safe_get(pred_result, "diversity_class", default="—")
        n_class = safe_get(pred_result, "novelty_class", default="—")
        
        st.markdown(f"""
            <div class="glass-card" style="padding:14px; margin-bottom:10px; border-left: 3px solid {C_TEAL};">
                <div class="metric-title">Assigned Group Cluster</div>
                <div class="metric-value" style="color:{C_TEAL}">#{c_id}</div>
            </div>
            <div class="glass-card" style="padding:14px; margin-bottom:10px; border-left: 3px solid {C_TEAL_DIM};">
                <div class="metric-title">Alpha Diversity Index</div>
                <div class="metric-value" style="font-size:20px;">{d_class}</div>
            </div>
            <div class="glass-card" style="padding:14px; margin-bottom:10px; border-left: 3px solid {C_CORAL};">
                <div class="metric-title">Outlier Novelty Class</div>
                <div class="metric-value" style="font-size:20px; color:{C_CORAL}">{n_class}</div>
            </div>
        """, unsafe_allow_html=True)

        ux, uy = safe_get(pred_result, "umap_x"), safe_get(pred_result, "umap_y")
        if ux is not None and uy is not None:
            st.caption(f"📍 Mapped Atlas Coordinates: ({ux:.3f}, {uy:.3f})")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("Gutbiome Atlas v1.0 · Systems Biology Core")

# ─────────────────────────── MAIN CONTENT ──────────────────────────────────

st.markdown("<h1 style='margin-bottom:0px; padding-bottom:0px;'>🌊 Gutbiome Atlas</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#5e7a8a; font-size:14px; margin-top:4px;'>Unsupervised population-level gut microbiome discovery from clinical stool datasets</p>", unsafe_allow_html=True)

df_samples = fetch_samples()
clusters_raw = fetch_clusters()

cluster_map = {}
for c in clusters_raw:
    cid = safe_get(c, "cluster_id", default=safe_get(c, "id"))
    if cid is not None:
        cluster_map[str(cid)] = c

tab_atlas, tab_cohort = st.tabs(["🗺️ Interactive Map Atlas", "👥 Structural Cohort Insights"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — INTERACTIVE ATLAS MAP
# ═══════════════════════════════════════════════════════════════════════════

with tab_atlas:
    if df_samples.empty:
        st.markdown(
            f'<div class="glass-card" style="text-align:center; padding: 60px 20px; color:{C_MUTED};">'
            f'<div style="font-size:40px; margin-bottom:12px;">🌐</div>Atlas data will appear here once the backend is running.<br>'
            f'<span style="font-size:12px; color:{C_CORAL}">No samples returned from /api/v1/samples.</span></div>',
            unsafe_allow_html=True
        )
    else:
        col_ctrl, col_rf = st.columns([3, 1])
        with col_ctrl:
            color_options = ["cluster_id", "disease", "country", "sex"]
            available_colors = [c for c in color_options if c in df_samples.columns] or df_samples.columns.tolist()
            color_by = st.selectbox("Color mapping vector attribute:", options=available_colors, index=0)
        with col_rf:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗘 Refresh Map Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        pred_point = st.session_state.get("pred_result") or None
        fig_umap = build_umap_figure(df_samples, color_by, pred_point=pred_point)
        st.plotly_chart(fig_umap, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-label'>Global Repository Statistics</div>", unsafe_allow_html=True)
        
        n_clusters = df_samples["cluster_id"].nunique() if "cluster_id" in df_samples.columns else "—"
        n_samples  = len(df_samples)
        diseases   = df_samples["disease"].nunique() if "disease" in df_samples.columns else "—"
        countries  = df_samples["country"].nunique() if "country" in df_samples.columns else "—"

        m1, m2, m3, m4 = st.columns(4)
        for col, title, value in zip([m1, m2, m3, m4], 
                                     ["Total Processed Profiles", "Identified Clusters", "Phenotype Targets", "Source Countries"],
                                     [f"{n_samples:,}", str(n_clusters), str(diseases), str(countries)]):
            with col:
                st.markdown(f'<div class="glass-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — COHORT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════

with tab_cohort:
    if not cluster_map:
        st.markdown(
            f'<div class="glass-card" style="text-align:center; padding:40px; color:{C_MUTED};">'
            'Cohort composition profiling options will populate once valid cluster metadata clusters are parsed.</div>',
            unsafe_allow_html=True
        )
    else:
        # User selection setup
        cluster_options = sorted(list(cluster_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        selected_cluster = st.selectbox("Select Target Biome Group Cluster For Fine-Grain Analysis:", options=cluster_options, index=0)
        
        # Scoping safely bounded within the resolution logic
        c_data = cluster_map.get(str(selected_cluster), {})

        # Bio Narrative Callout
        narrative = safe_get(c_data, "narrative", default="Clinical annotation logs omitted or unassigned for this cluster node.")
        st.markdown(f"""
            <div class="glass-card" style="background:linear-gradient(135deg, rgba(13,21,38,0.8) 0%, rgba(2,128,144,0.08) 100%); border-left:3px solid {C_TEAL}; color:#e2e8f0; line-height:1.6;">
                <span style="color:{C_TEAL}; font-weight:700; font-size:12px; text-transform:uppercase; display:block; margin-bottom:6px;">Functional Group Clinical Interpretation</span>
                {narrative}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Cohort Target Architecture</div>", unsafe_allow_html=True)
        age_stats = safe_get(c_data, "age_stats", default={})
        avg_div   = safe_get(c_data, "avg_diversity_score", default=safe_get(c_data, "diversity_score"))
        size      = safe_get(c_data, "size", default="—")
        
        age_mean  = safe_get(age_stats, "mean")
        age_min   = safe_get(age_stats, "min")
        age_max   = safe_get(age_stats, "max")
        age_std   = safe_get(age_stats, "std")

        cm1, cm2, cm3, cm4 = st.columns(4)
        
        with cm1:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Cohort Size</div><div class="metric-value">{f"{size:,}" if isinstance(size, int) else str(size)}</div></div>', unsafe_allow_html=True)
        with cm2:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Mean Entropy Shannon Score</div><div class="metric-value">{f"{avg_div:.3f}" if isinstance(avg_div, (int, float)) else "—"}</div></div>', unsafe_allow_html=True)
        with cm3:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-title">Mean Age Profile</div>
                    <div class="metric-value">{f"{age_mean:.1f}" if isinstance(age_mean, (int, float)) else "—"}</div>
                    <div class="metric-delta">± {age_std:.1f} yrs deviation</div>
                </div>
            """, unsafe_allow_html=True)
        with cm4:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Age Bound Spans</div><div class="metric-value">{f"{int(age_min)}–{int(age_max)}" if isinstance(age_min, (int, float)) and isinstance(age_max, (int, float)) else "—"}</div></div>', unsafe_allow_html=True)

        # Demographics Chart Panel Rows
        st.markdown("<div class='section-label'>Demographic Partition Distribution Matrices</div>", unsafe_allow_html=True)
        meta_dist = safe_get(c_data, "metadata_dist", default={})
        
        c_dist = safe_get(meta_dist, "country", default={})
        s_dist = safe_get(meta_dist, "sex", default=safe_get(meta_dist, "gender", default={}))
        d_dist = safe_get(meta_dist, "disease", default=safe_get(meta_dist, "phenotype", default={}))

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown('<div class="metric-title" style="text-align:center;">Geographic Cohort Isolation</div>', unsafe_allow_html=True)
            if c_dist: st.plotly_chart(dist_donut(c_dist, "Country"), use_container_width=True, config={"displayModeBar": False})
            else: st.markdown('<div class="glass-card" style="text-align:center;color:#5e7a8a;padding:40px 10px;">No country demographics</div>', unsafe_allow_html=True)
        with d2:
            st.markdown('<div class="metric-title" style="text-align:center;">Biological Sex Factor</div>', unsafe_allow_html=True)
            if s_dist: st.plotly_chart(dist_donut(s_dist, "Sex"), use_container_width=True, config={"displayModeBar": False})
            else: st.markdown('<div class="glass-card" style="text-align:center;color:#5e7a8a;padding:40px 10px;">No biological sex trends</div>', unsafe_allow_html=True)
        with d3:
            st.markdown('<div class="metric-title" style="text-align:center;">Clinical Phenotype Phenotype</div>', unsafe_allow_html=True)
            if d_dist: st.plotly_chart(dist_donut(d_dist, "Disease"), use_container_width=True, config={"displayModeBar": False})
            else: st.markdown('<div class="glass-card" style="text-align:center;color:#5e7a8a;padding:40px 10px;">No clinical phenotypes</div>', unsafe_allow_html=True)

        # Bacterial Signature Feature Space Charts
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Bacterial Signature — Defining Microbes</div>', unsafe_allow_html=True)

        dom_microbes = safe_get(c_data, "dominant_microbes", default=[])
        top_features = safe_get(c_data, "top_features", default={})

        if isinstance(top_features, dict) and top_features:
            feature_counts = top_features
        elif isinstance(dom_microbes, list) and dom_microbes:
            feature_counts = {str(m): round(1.0 - i * 0.07, 4) for i, m in enumerate(dom_microbes)}
        elif isinstance(dom_microbes, dict) and dom_microbes:
            feature_counts = dom_microbes
        else:
            feature_counts = {}

        if feature_counts:
            st.plotly_chart(
                horiz_bar(feature_counts, "Relative Importance / Prevalence", color=C_TEAL),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.markdown(
                '<div class="glass-card" style="text-align:center;color:#5e7a8a;padding:40px 10px;">'
                'No dominant microbe signature available for this cluster...</div>',
                unsafe_allow_html=True
            )