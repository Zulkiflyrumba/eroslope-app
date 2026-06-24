import streamlit as st
import numpy as np
import pandas as pd
import ezdxf
import tempfile
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid
from shapely import vectorized
from shapely.ops import linemerge, polygonize
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
import alphashape
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import requests

from matplotlib_scalebar.scalebar import ScaleBar

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle
)


from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from PIL import Image as PILImage

st.set_page_config(page_title="Geo AI", layout="wide")

import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
try:
    from xgboost import XGBRegressor, XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
# =====================================================
# LANDING PAGE
# =====================================================

if "home_page" not in st.session_state:
    st.session_state.home_page = True

if st.session_state.home_page:

    st.markdown("""
    <style>

    .stApp{
        background: linear-gradient(
            135deg,
            #00151a 0%,
            #02111d 100%
        );
    }

    .hero-card{
        width:100%;
        padding:35px;
        border-radius:25px;
        background:rgba(0,0,0,0.35);
        border:1px solid rgba(255,255,255,0.15);
        box-shadow:0 0 40px rgba(0,255,150,0.12);
    }

    .hero-title{
        font-size:70px;
        font-weight:900;
        color:white;
        line-height:0.9;
    }

    .hero-subtitle{
        margin-top:15px;
        color:#8FFF7A;
        font-size:22px;
        font-weight:700;
    }

    .hero-desc{
        margin-top:20px;
        color:white;
        font-size:18px;
        line-height:1.7;
    }

    .hero-tag{
        display:inline-block;
        margin-top:20px;
        margin-right:8px;
        padding:8px 16px;
        border-radius:20px;
        background:rgba(0,255,120,0.12);
        border:1px solid rgba(0,255,120,0.25);
        color:white;
    }

    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.2, 2])

    with left:

        st.markdown("""
        <div style="
            padding:35px;
            border-radius:25px;
            background:rgba(0,0,0,0.35);
            border:1px solid rgba(255,255,255,0.15);
            box-shadow:0 0 40px rgba(0,255,150,0.12);
        ">

        <div style="
                color:white;
                font-size:70px;
                line-height:0.9;
                font-weight:900;
                margin-bottom:15px;
            ">
            GEOTECHNICAL<br>
            INTELLIGENCE
            </div>

        <div style="
                color:#8FFF7A;
                margin-bottom:20px;
            ">
            Turning Monitoring Data Into Engineering Decisions
            </div>
    
        <div style="
                color:white;
                font-size:18px;
                line-height:1.8;
            ">
            Advanced platform combining AI,
            Machine Learning,
            Numerical Modelling,
            Digital Twin,
            and Predictive Analytics
            for smarter mining operations.
            </div>

        <div style="margin-top:20px">
                <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">AI</span>
                <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">Machine Learning</span>
                <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">Digital Twin</span>
                <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">Numerical Modelling</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image("sump.jpg", use_container_width=True)
            st.markdown(
                "<center><h4 style='color:white'>SUMP MONITORING</h4></center>",
                unsafe_allow_html=True
            )

        with c2:
            st.image("water.jpg", use_container_width=True)
            st.markdown(
                "<center><h4 style='color:white'>WATER MANAGEMENT</h4></center>",
                unsafe_allow_html=True
            )

        with c3:
            st.image("slope.jpg", use_container_width=True)
            st.markdown(
                "<center><h4 style='color:white'>PIT STABILITY</h4></center>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,1,2])

    with c2:
        if st.button(
            "▶︎START ANALYSIS",
            use_container_width=True
        ):
            st.session_state.home_page = False
            st.rerun()

    st.stop()
    
    # ================= CONFIG =================
st.set_page_config(page_title="Erosion + ML Predictor", layout="wide")

if "analysis_method" not in st.session_state:
    st.session_state["analysis_method"] = None

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "erosion_area" not in st.session_state:
    st.session_state["erosion_area"] = 0.0

if "online_rainfall" not in st.session_state:
    st.session_state["online_rainfall"] = None

if "sedimentation_area" not in st.session_state:
    st.session_state["sedimentation_area"] = 0.0

if "max_zone" not in st.session_state:
    st.session_state["max_zone"] = 0.0

# ================= CSS PREMIUM =================
def load_css():
    st.markdown("""
    <style>

    /* ===== BASE ===== */
    .stApp {
        background: linear-gradient(135deg, #002D2D, #001A1A);
        color: #E6F4F1;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ===== HEADER ===== */
    .header {
        background: rgba(0, 128, 96, 0.15);
        padding: 20px;
        border-radius: 18px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(149,191,71,0.2);
        margin-bottom: 20px;
    }

    /* ===== BUTTON ===== */
    .stButton>button {
        background: linear-gradient(135deg, #008060, #5E8E3E);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(149,191,71,0.5);
    }

    /* ===== INPUT ===== */
    .stTextInput>div>div>input {
        background: rgba(255,255,255,0.05);
        color: white;
        border: 1px solid rgba(149,191,71,0.2);
        border-radius: 10px;
    }

    /* ===== CHAT USER ===== */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #003B3B, #002D2D);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #95BF47;
    }

    /* ===== CHAT AI ===== */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: linear-gradient(135deg, #001A1A, #002D2D);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #008060;
    }

    /* ===== METRIC ===== */
    [data-testid="metric-container"] {
        background: rgba(0,128,96,0.1);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(149,191,71,0.2);
    }

    /* ===== TAB ===== */
    .stTabs [role="tab"] {
        color: #95BF47;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #008060;
    }

    /* ===== ANIMATION ===== */
    [data-testid="stChatMessage"] {
        animation: fadeIn 0.4s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity:0; transform: translateY(8px);}
        to {opacity:1; transform: translateY(0);}
    }

    </style>
    """, unsafe_allow_html=True)

load_css()

# ================= HEADER =================
import streamlit as st

header_container = st.container()

with header_container:
    st.image("header.png", use_container_width=True)

# ================= HEADER =================
st.markdown('<div class="header">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image("logo.png", width=120)

with col2:
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:white; margin-bottom:0;">
            EroSlope & Machine Learning
        </h1>
        <p style="color:rgba(255,255,255,0.7); font-size:14px;">
            Advanced Geotechnical Analysis for Erosion, Stability, and Predictive Modeling
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("logo_2.png", width=120)

st.markdown('</div>', unsafe_allow_html=True)


tab1, tab2 = st.tabs(["Erosion Mapping", "Machine Learning"])

# =========================================================
# =================== TAB 1: EROSION ======================
# =========================================================
with tab1:

    st.subheader("A. Input DXF")
    col1, col2 = st.columns(2)

    with col1:
        kontur_dxf = st.file_uploader("Upload Kontur DXF", type=["dxf"])
    with col2:
        boundary_dxf = st.file_uploader("Upload Boundary DXF", type=["dxf"])

    st.subheader("B. Parameter Flow")
    col3, col4 = st.columns(2)

    analysis_method = st.radio(
        "Metode Analisis Sedimen",
        [
            "Hjulstrom Diagram",
            "Shields Diagram",
            "Partheniades + Flow Accumulation"
        ]
    )

    with col3:
        velocity_hulu = st.number_input("Velocity Hulu (m/s)", 0.01, 5.0, 0.5)

    with col4:

        if analysis_method == "Hjulstrom Diagram":

            grain_size = st.number_input(
                "Grain Size d (mm)",
                0.001,
                10.0,
                0.1
            )

        elif analysis_method == "Shields Diagram":

            grain_size = st.number_input(
                "D50 Sedimen (mm)",
                0.001,
                100.0,
                0.5
            )

            rho_water = st.number_input(
                "Density Water (kg/m3)",
                1000,
                1200,
                1000
            )

            rho_soil = st.number_input(
                "Density Sediment (kg/m3)",
                1200,
                3500,
                2650
            )

        elif analysis_method == "Partheniades + Flow Accumulation":

            tau_critical = st.number_input(
                "Critical Shear Stress τc (Pa)",
                0.01,
                100.0,
                2.0
            )

            erodibility_M = st.number_input(
                "Erodibility Coefficient M",
                0.0001,
                10.0,
                0.05,
                format="%.4f"
            )

            flow_weight = st.slider(
                "Flow Accumulation Weight",
                0.1,
                10.0,
                3.0,
                0.1
            )

            flow_depth = st.number_input(
                "Assumed Flow Depth (m)",
                0.01,
                10.0,
                0.30
            )

            rho_water = st.number_input(
                "Density Water (kg/m3)",
                1000,
                1200,
                1000
            )

    st.subheader("Visualisasi")

    colv1, colv2 = st.columns(2)

    with colv1:
        vertical_exaggeration = st.slider(
            "Vertical Exaggeration",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5
        )

    with colv2:
        shadow_strength = st.slider(
            "Shadow Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05
        )
    # ===== OPEN METEO =====
        def get_online_rainfall(lat, lon):

            try:

                from datetime import date

                today = date.today()

                url = (
                    f"https://archive-api.open-meteo.com/v1/archive"
                    f"?latitude={lat}"
                    f"&longitude={lon}"
                    f"&start_date=2025-06-01"
                    f"&end_date=2025-06-30"
                    f"&daily=precipitation_sum"
                    f"&timezone=auto"
                )

                response = requests.get(url)

                data = response.json()

                rainfall = np.nanmean(
                    data["daily"]["precipitation_sum"]
                )

                return rainfall

            except:

                return None
    # ================= SOURCE =================
    rain_factor = st.slider(
        "Extreme Rainfall Factor",
        1.0,
        5.0,
        1.0,
        0.5
    )

    st.subheader("Online Rainfall")

    latitude = st.number_input(
        "Latitude",
        value=-2.28
    )

    longitude = st.number_input(
        "Longitude",
        value=115.41
    )

    if st.button("Get Online Rainfall"):

        online_rainfall = get_online_rainfall(
            latitude,
            longitude
        )

        st.session_state["online_rainfall"] = online_rainfall


    if st.session_state["online_rainfall"] is not None:

        st.success(
            f"Average Rainfall : "
            f"{st.session_state['online_rainfall']:.2f} mm/day"
        )

    else:

        st.error(
            "Unable to retrieve rainfall data."
        )

    st.subheader("C. Skenario")

    source_type = st.selectbox(
        "Pilih sumber aliran",
        ["Hujan (Uniform)", "Satu Titik (Point Source)"]
    )

    point_x, point_y = None, None
    if source_type == "Satu Titik (Point Source)":
        colp1, colp2 = st.columns(2)
        with colp1:
            point_x = st.number_input("Koordinat X Hulu")
        with colp2:
            point_y = st.number_input("Koordinat Y Hulu")

    # ================= UTIL =================
    def save_uploaded_dxf(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.read())
            return tmp.name

    def fix_geom(geom):
        if not geom.is_valid:
            geom = make_valid(geom)
        return geom


    from shapely.ops import linemerge, polygonize
    from shapely.geometry import LineString

    def read_boundary_polygon(uploaded_dxf):
        path = save_uploaded_dxf(uploaded_dxf)
        doc = ezdxf.readfile(path)

        lines = []

        for e in doc.modelspace():

            # ===== polyline langsung jadi polygon =====
            if e.dxftype() == "LWPOLYLINE" and e.closed:
                pts = [(p[0], p[1]) for p in e.get_points()]
                return fix_geom(Polygon(pts))

            elif e.dxftype() == "POLYLINE" and e.is_closed:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                return fix_geom(Polygon(pts))

            # ===== kumpulin semua garis =====
            elif e.dxftype() == "LINE":
                start = (e.dxf.start.x, e.dxf.start.y)
                end = (e.dxf.end.x, e.dxf.end.y)
                lines.append(LineString([start, end]))

            elif e.dxftype() == "LWPOLYLINE" and not e.closed:
                pts = [(p[0], p[1]) for p in e.get_points()]
                lines.append(LineString(pts))

            elif e.dxftype() == "POLYLINE" and not e.is_closed:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                lines.append(LineString(pts))

        if len(lines) == 0:
            return None

        # ===== gabung garis =====
        merged = linemerge(lines)

        # ===== coba jadi polygon otomatis =====
        polygons = list(polygonize(merged))
        if len(polygons) > 0:
            return fix_geom(polygons[0])

        # ===== fallback: paksa tutup =====
        coords = []
        for line in lines:
            coords.extend(list(line.coords))

        coords = list(dict.fromkeys(coords))

        if len(coords) < 3:
            return None

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        return fix_geom(Polygon(coords))

    def read_contours(uploaded_dxf):
        
        path = save_uploaded_dxf(uploaded_dxf)
        doc = ezdxf.readfile(path)

        contours = []
        for e in doc.modelspace():

            if e.dxftype() == "POLYLINE" and e.is_3d_polyline:
                pts = [(v.dxf.location.x,
                        v.dxf.location.y,
                        v.dxf.location.z) for v in e.vertices]
                contours.append(pts)

            elif e.dxftype() == "LWPOLYLINE":
                elev = e.dxf.elevation
                pts = [(p[0], p[1], elev) for p in e.get_points()]
                contours.append(pts)

        return contours

    def read_section_dxf(uploaded_dxf):

        path = save_uploaded_dxf(
            uploaded_dxf
        )

        doc = ezdxf.readfile(path)

        for e in doc.modelspace():

            if e.dxftype() == "LWPOLYLINE":

                pts = [
                    (p[0], p[1])
                    for p in e.get_points()
                ]

                return pts

            elif e.dxftype() == "LINE":

                return [
                    (
                        e.dxf.start.x,
                        e.dxf.start.y
                    ),
                    (
                        e.dxf.end.x,
                        e.dxf.end.y
                    )
                ]

        return None

    # ================= HJULSTROM =================
    def hjulstrom_zone(v, d_mm):
        d = d_mm / 1000 + 1e-6
        v_erosion = 0.1 * (d**-0.4)
        v_deposition = 0.01 * (d**-0.2)
        score = (v - v_deposition) / (v_erosion - v_deposition + 1e-6)
        return np.clip(score * 2, 0, 2)

    def shields_zone(
            velocity,
            slope,
            rho_water,
            rho_soil,
            d_mm
    ):

        g = 9.81

        d = d_mm / 1000

        tau = (
            rho_water *
            g *
            slope
        )

        theta = (
            tau /
            (
                (rho_soil-rho_water)
                * g
                * d
                + 1e-9
            )
        )

        if theta < 0.03:
            return 0

        elif theta < 0.06:
            return 1

        else:
            return 2

    def partheniades_erosion(
            slope,
            flow_density,
            rho_water,
            flow_depth,
            tau_critical,
            erodibility_M,
            flow_weight
    ):

        g = 9.81

        tau = (
            rho_water *
            g *
            flow_depth *
            slope
        )

        erosion_rate = np.zeros_like(tau)

        active = tau > tau_critical

        erosion_rate[active] = (
            erodibility_M *
            (
                tau[active] /
                tau_critical
                - 1.0
            )
        )

        flow_factor = (
            flow_density /
            (
                np.nanpercentile(
                    flow_density,
                    95
                ) + 1e-9
            )
        )

        flow_factor = np.clip(
            flow_factor,
            0,
            3
        )

        risk = (
            erosion_rate *
            (
                1 +
                flow_weight *
                flow_factor
            )
        )

        return risk

    # ================= FLOW =================
    def trace_multiple_flow(x0, y0, grid_x, grid_y, dz_dx, dz_dy, grid_z, boundary, n_stream=20):

        all_paths = []

        for angle in np.linspace(-0.5, 0.5, n_stream):

            path_x, path_y, path_z = [x0], [y0], []
            x, y = x0, y0

            for _ in range(800):

                ix = np.abs(grid_x[:,0] - x).argmin()
                iy = np.abs(grid_y[0,:] - y).argmin()

                vx = -dz_dx[ix, iy]
                vy = -dz_dy[ix, iy]

                vx += angle * 0.2
                vy += angle * 0.2

                norm = np.sqrt(vx**2 + vy**2)
                if norm < 1e-6:
                    break

                x += (vx / norm) * 0.5
                y += (vy / norm) * 0.5

                if not boundary.contains(Point(x, y)):
                    break

                z = grid_z[ix, iy]
                path_x.append(x)
                path_y.append(y)
                path_z.append(z)

            all_paths.append((path_x, path_y, path_z))

        return all_paths

    # ================= RUN =================
    if st.button("RUN ANALYSIS"):

        st.session_state["analysis_done"] = True

        if not kontur_dxf:
            st.error("Upload Kontur DXF dulu")
            st.stop()

        contours = read_contours(kontur_dxf)

        x_all = []
        y_all = []
        z_all = []

        for contour in contours:
            for x, y, z in contour:
                x_all.append(x)
                y_all.append(y)
                z_all.append(z)

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        z_all = np.array(z_all)

        if len(x_all) < 10:
            st.error("Kontur tidak terbaca")
            st.stop()

        # ================= AUTO BOUNDARY =================

        pts = np.column_stack(
            [x_all, y_all]
        )

        if boundary_dxf:
            boundary = read_boundary_polygon(boundary_dxf)
        else:
            boundary = alphashape.alphashape(pts, 0.001)

        bx, by = boundary.exterior.xy

        # ================= GRID =================

        grid_x, grid_y = np.mgrid[
            x_all.min():x_all.max():800j,
            y_all.min():y_all.max():800j
        ]

        dx = np.abs(grid_x[1, 0] - grid_x[0, 0])
        dy = np.abs(grid_y[0, 1] - grid_y[0, 0])
        cell_area = dx * dy

        grid_z = griddata(
            (x_all, y_all),
            z_all,
            (grid_x, grid_y),
            method="linear"
        )

        inside = vectorized.contains(
            boundary,
            grid_x,
            grid_y
        )

        x_mesh = grid_x[inside]
        y_mesh = grid_y[inside]
        z_mesh = grid_z[inside]

        # Titik untuk TIN
        points = np.column_stack([
            x_mesh,
            y_mesh
        ])

        # Triangulasi
        tri = Delaunay(points)

        # Filter triangle yang benar-benar berada di dalam boundary
        valid_triangles = []

        for simplex in tri.simplices:

            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]

            cx = (p1[0] + p2[0] + p3[0]) / 3
            cy = (p1[1] + p2[1] + p3[1]) / 3

            if boundary.contains(Point(cx, cy)):
                valid_triangles.append(simplex)

        valid_triangles = np.array(valid_triangles)

        surface_mask = inside.copy()

        grid_z[~surface_mask] = np.nan

        # isi lubang kecil
        if np.isnan(grid_z).any():

            nearest = griddata(
                (x_all, y_all),
                z_all,
                (grid_x, grid_y),
                method="nearest"
            )

            grid_z[np.isnan(grid_z)] = nearest[np.isnan(grid_z)]

        grid_z = gaussian_filter(
            grid_z,
            sigma=0.5
        )

        # ================= SLOPE =================

        dz_dx, dz_dy = np.gradient(grid_z)

        slope = np.sqrt(
            dz_dx**2 +
            dz_dy**2
        )

        slope[slope < 1e-6] = 1e-6

        velocity_field = (
            velocity_hulu *
            np.sqrt(
                slope /
                np.nanmean(slope)
        )
        )

        velocity_field *= np.sqrt(
            rain_factor
        )
    

        def calculate_sedimentation(
            slope,
            velocity,
            flow_density
        ):

            slope_norm = 1 - (
                slope /
                np.nanmax(slope)
            )

            velocity_norm = 1 - (
                velocity /
                np.nanmax(velocity)
            )

            flow_norm = (
                flow_density /
                (
                    np.nanmax(flow_density)
                    + 1e-9
                )
            )

            sediment_index = (
                slope_norm * 0.4 +
                velocity_norm * 0.3 +
                flow_norm * 0.3
            )

            return sediment_index
        # ================= FLOW =================

        flow_paths = []

        flow_count = np.zeros_like(
            grid_z
        )

        if (
            source_type ==
            "Satu Titik (Point Source)"
        ):

            if not boundary.contains(
                Point(point_x, point_y)
            ):
                st.error(
                    "Point Source berada di luar area kontur"
                )
                st.stop()

            flow_paths = trace_multiple_flow(
                point_x,
                point_y,
                grid_x,
                grid_y,
                dz_dx,
                dz_dy,
                grid_z,
                boundary
            )

        # ================= FLOW MASK =================

        if source_type == "Hujan (Uniform)":

            flow_mask = ~np.isnan(grid_z)

        else:

            flow_mask = np.zeros_like(grid_z)

            for px, py, _ in flow_paths:

                for x, y in zip(px, py):

                    ix = np.abs(
                        grid_x[:,0] - x
                    ).argmin()

                    iy = np.abs(
                        grid_y[0,:] - y
                    ).argmin()

                    flow_mask[ix, iy] = 1

                    flow_count[ix, iy] += 1

            flow_mask = (
                gaussian_filter(
                    flow_mask,
                    sigma=2
                ) > 0.05
            )

        flow_density = gaussian_filter(
            flow_count,
            sigma=5
        )

        convergence_zone = (
            flow_density >
            np.percentile(
                flow_density,
                95
            )
        )

        

        overflow_index = (
            flow_density *
            velocity_field
        )

        overflow_index = (
            overflow_index /
            (
                np.nanmax(overflow_index)
                + 1e-9
            )
        )

        overflow_zone = (
            overflow_index > 0.8
        )

        sediment_map = calculate_sedimentation(
            slope,
            velocity_field,
            flow_density
        )

        sediment_map[~inside] = np.nan

            
        # ================= EROSION =================

        zone_map = np.zeros_like(grid_z)

        valid = (
            flow_mask &
            (~np.isnan(grid_z))
        )

        st.write(
            "Jumlah cell valid:",
            int(np.sum(valid))
        )

        if np.any(valid):

            if analysis_method == "Hjulstrom Diagram":

                zone_map[valid] = np.vectorize(
                    hjulstrom_zone,
                    otypes=[float]
                )(
                    velocity_field[valid],
                    grain_size
                )

            elif analysis_method == "Shields Diagram":

                zone_map[valid] = np.vectorize(
                    shields_zone,
                    otypes=[float]
                )(
                    velocity_field[valid],
                    slope[valid],
                    rho_water,
                    rho_soil,
                    grain_size
                )

            else:

                risk_map = partheniades_erosion(
                    slope=slope,
                    flow_density=flow_density,
                    rho_water=rho_water,
                    flow_depth=flow_depth,
                    tau_critical=tau_critical,
                    erodibility_M=erodibility_M,
                    flow_weight=flow_weight
                )

                zone_map = risk_map.copy()

                zone_map = (
                    zone_map /
                    (
                        np.nanpercentile(
                            zone_map,
                            99
                        ) + 1e-9
                    )
                )

                zone_map = np.clip(
                    zone_map,
                    0,
                    2
                )

        # ==========================================
        # ================= MASK OUTSIDE BOUNDARY =================

        zone_map[~inside] = np.nan

        flow_density[~inside] = np.nan

        overflow_zone = overflow_zone & inside

        convergence_zone = convergence_zone & inside

        # ================= LEGEND CONFIG =================

        if analysis_method == "Hjulstrom Diagram":

            legend_title = "Hjulstrom Erosion Risk"

            colorbar_ticks = [0,1,2]

            colorbar_text = [
                "Deposisi",
                "Transisi",
                "Erosi"
            ]

        elif analysis_method == "Shields Diagram":

            legend_title = "Shields Sediment Mobility"

            colorbar_ticks = [0,1,2]

            colorbar_text = [
                "Stable",
                "Incipient Motion",
                "Transport"
            ]

        else:

            legend_title = "Partheniades Erosion Index"

            colorbar_ticks = [
                0,
                0.5,
                1.0,
                1.5,
                2.0
            ]

            colorbar_text = [
                "Very Low",
                "Low",
                "Moderate",
                "High",
                "Extreme"
            ]

        # ================= PLOT =================

        fig = go.Figure()

        fig.add_trace(
            go.Mesh3d(
                x=x_mesh,
                y=y_mesh,
                z=z_mesh * vertical_exaggeration,

                i=valid_triangles[:,0],
                j=valid_triangles[:,1],
                k=valid_triangles[:,2],

                intensity=zone_map[inside],

                colorscale=[
                    [0,"green"],
                    [0.5,"yellow"],
                    [1,"red"]
                ],

                showscale=True,

                colorbar=dict(
                    title=legend_title,
                    tickvals=colorbar_ticks,
                    ticktext=colorbar_text
                ),

                flatshading=False
            )
        )

        critical_y, critical_x = np.where(
            overflow_zone & inside
        )

        df_overflow = pd.DataFrame({

            "X": grid_x[
                critical_y,
                critical_x
            ],

            "Y": grid_y[
                critical_y,
                critical_x
            ],

            "RL": grid_z[
                critical_y,
                critical_x
            ],

            "FlowDensity": flow_density[
                critical_y,
                critical_x
            ]
        })

        df_overflow = df_overflow.sort_values(
            "FlowDensity",
            ascending=False
        )

        top10 = df_overflow.head(10)

        top10 = top10.copy()

        top10.insert(
            0,
            "Rank",
            np.arange(
                1,
                len(top10)+1
            )
        )

        st.subheader(
            "Numerical Modelling"
        )

        fig.add_trace(
                go.Scatter3d(
                    x=grid_x[
                        critical_y,
                        critical_x
                    ],
                    y=grid_y[
                        critical_y,
                        critical_x
                    ],
                    z=grid_z[
                        critical_y,
                        critical_x
                    ] * vertical_exaggeration,

                    mode="markers",

                    marker=dict(
                        size=4,
                        color="red"
                    ),

                    name="Overflow Risk"
                )
            )

        for px, py, pz in flow_paths:

            fig.add_trace(
                go.Scatter3d(
                    x=px,
                    y=py,
                    z=np.array(pz) * vertical_exaggeration,
                    mode="lines",
                    line=dict(
                        color="cyan",
                        width=4
                    ),
                    showlegend=False
                )
            )

        fig.update_layout(

            title=dict(
                text=f"3D Erosion Analysis - {analysis_method}",
                x=0.5
            ),

            height=850,

            scene=dict(
                aspectmode="data"
            )
        )
    
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "2D Risk Map"
        )

        fig2 = go.Figure()

        for contour in contours:

            xs = [p[0] for p in contour]
            ys = [p[1] for p in contour]

            fig2.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(
                        color="white",
                        width=0.2
                    ),
                    showlegend=False
                )
            )

        bx, by = boundary.exterior.xy

        bx = np.array(bx)
        by = np.array(by)

        fig2.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(
                    color="Magenta",
                    width=2
                ),
                name="Boundary"
            )
        )

        erosion_y, erosion_x = np.where(
            (~np.isnan(zone_map)) &
            (zone_map > 1.0) &
            inside
        )

        fig2.add_trace(
            go.Scatter(
                x=grid_x[
                    erosion_y,
                    erosion_x
                ],

                y=grid_y[
                    erosion_y,
                    erosion_x
                ],

                mode="markers",

                marker=dict(
                    size=3,
                    color="orange"
                ),

                name="Erosion"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=grid_x[
                    critical_y,
                    critical_x
                ],

                y=grid_y[
                    critical_y,
                    critical_x
                ],

                mode="markers",

                marker=dict(
                    size=5,
                    color="red"
                ),

                name="Overflow"
            )
        )

        conv_y, conv_x = np.where(
        convergence_zone & inside
    )

        fig2.add_trace(
            go.Scatter(
                x=grid_x[
                    conv_y,
                    conv_x
                ],

                y=grid_y[
                    conv_y,
                    conv_x
                ],

                mode="markers",

                marker=dict(
                    size=5,
                    color="purple"
                ),

                name="Flow Convergence"
            )
        )

        fig2.update_layout(

            height=800,

            xaxis=dict(
                title="Easting",
                showgrid=True,
                gridwidth=1,
                tickformat=".0f"
            ),

            yaxis=dict(
                title="Northing",
                showgrid=True,
                gridwidth=1,
                scaleanchor="x",
                tickformat=".0f"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        erosion_area = (
            np.sum(zone_map > 1.5)
            * cell_area
        ) / 10000

        sedimentation_area = (
            np.sum(sediment_map > 0.7)
            * cell_area
        ) / 10000

        st.session_state["grid_x"] = grid_x
        st.session_state["grid_y"] = grid_y
        st.session_state["grid_z"] = grid_z
        st.session_state["zone_map"] = zone_map
        st.session_state["sediment_map"] = sediment_map
        st.session_state.get("analysis_method", None)
        st.session_state["max_zone"] = float(
            np.nanmax(zone_map)
        )
        st.session_state["erosion_area"] = float(
            erosion_area
        )
        st.session_state["sedimentation_area"] = float(
            sedimentation_area
        )
        st.session_state["boundary"] = boundary
        st.session_state["cell_area"] = cell_area
        st.session_state["inside"] = inside


        erosion_ratio = (
            np.sum(zone_map > 1.0)
            /
            np.sum(~np.isnan(grid_z))
        )

        st.metric(
            "Area Potensi Erosi (%)",
            f"{erosion_ratio*100:.1f}%"
        )
        

        if analysis_method == "Partheniades + Flow Accumulation":

            st.metric(
                "Maximum Erosion Index",
                f"{np.nanmax(zone_map):.3f}"
            )

            st.metric(
                "Average Erosion Index",
                f"{np.nanmean(zone_map):.3f}"
            )

        st.markdown("---")

    # =====================================================
# CROSS SECTION TOOL
# =====================================================

if st.session_state.get("analysis_done", False):

    st.markdown("---")
    st.subheader("Cross Section Analysis")

    grid_x = st.session_state["grid_x"]
    grid_y = st.session_state["grid_y"]
    grid_z = st.session_state["grid_z"]

    zone_map = st.session_state["zone_map"]
    sediment_map = st.session_state["sediment_map"]

    section_mode = st.radio(
        "Section Type",
        [
            "Straight Line",
            "Polyline DXF"
        ]
    )

    # ==========================================
    # STRAIGHT LINE
    # ==========================================

    if section_mode == "Straight Line":

        col1, col2 = st.columns(2)

        with col1:

            x1 = st.number_input(
                "Start X",
                value=float(np.nanmin(grid_x))
            )

            y1 = st.number_input(
                "Start Y",
                value=float(np.nanmin(grid_y))
            )

        with col2:

            x2 = st.number_input(
                "End X",
                value=float(np.nanmax(grid_x))
            )

            y2 = st.number_input(
                "End Y",
                value=float(np.nanmax(grid_y))
            )

    # ==========================================
    # POLYLINE DXF
    # ==========================================

    else:

        section_dxf = st.file_uploader(
            "Upload Polyline Section DXF",
            type=["dxf"],
            key="section_polyline"
        )

    # ==========================================
    # BUTTON
    # ==========================================

    if st.button("Generate Cross Section"):

        # ----------------------------
        # STRAIGHT
        # ----------------------------

        if section_mode == "Straight Line":

            n_samples = 500

            xs = np.linspace(
                x1,
                x2,
                n_samples
            )

            ys = np.linspace(
                y1,
                y2,
                n_samples
            )

            st.session_state["section_line"] = [
                (x1, y1),
                (x2, y2)
            ]

        # ----------------------------
        # POLYLINE
        # ----------------------------

        else:

            if section_dxf is None:

                st.warning(
                    "Upload DXF terlebih dahulu"
                )

                st.stop()

            vertices = read_section_dxf(
                section_dxf
            )

            if vertices is None:

                st.error(
                    "Polyline tidak terbaca"
                )

                st.stop()

            st.session_state["section_line"] = vertices

            xs = []
            ys = []

            for i in range(
                len(vertices)-1
            ):

                p1 = vertices[i]
                p2 = vertices[i+1]

                xs.extend(
                    np.linspace(
                        p1[0],
                        p2[0],
                        100
                    )
                )

                ys.extend(
                    np.linspace(
                        p1[1],
                        p2[1],
                        100
                    )
                )

            xs = np.array(xs)
            ys = np.array(ys)

            n_samples = len(xs)

        # ==================================
        # PROFILE
        # ==================================

        elev_profile = []
        erosion_profile = []
        sediment_profile = []

        for xx, yy in zip(xs, ys):

            ix = np.abs(
                grid_x[:,0] - xx
            ).argmin()

            iy = np.abs(
                grid_y[0,:] - yy
            ).argmin()

            elev_profile.append(
                grid_z[ix, iy]
            )

            erosion_profile.append(
                zone_map[ix, iy]
            )

            sediment_profile.append(
                sediment_map[ix, iy]
            )

        distance = np.arange(
            len(xs)
        )

        # ==================================
        # ELEVATION PROFILE
        # ==================================

        fig_cs = go.Figure()

        fig_cs.add_trace(
            go.Scatter(
                x=distance,
                y=elev_profile,
                name="Elevation"
            )
        )

        fig_cs.update_layout(
            title="Elevation Profile",
            xaxis_title="Distance",
            yaxis_title="Elevation (m)",
            height=500
        )

        st.plotly_chart(
            fig_cs,
            use_container_width=True
        )

        # ==================================
        # RISK PROFILE
        # ==================================

        fig_risk = go.Figure()

        fig_risk.add_trace(
            go.Scatter(
                x=distance,
                y=erosion_profile,
                name="Erosion Risk"
            )
        )

        fig_risk.add_trace(
            go.Scatter(
                x=distance,
                y=sediment_profile,
                name="Sedimentation"
            )
        )

        fig_risk.update_layout(
            title="Risk Profile",
            height=500
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

        st.session_state["section_distance"] = distance
        st.session_state["section_elevation"] = elev_profile
        st.session_state["section_erosion"] = erosion_profile
        st.session_state["section_sediment"] = sediment_profile

    if st.session_state.get("analysis_done", False) and st.button("GENERATE EXECUTIVE REPORT"):

        grid_x = st.session_state["grid_x"]
        grid_y = st.session_state["grid_y"]
        grid_z = st.session_state["grid_z"]
        boundary = st.session_state["boundary"]
        zone_map = st.session_state["zone_map"]
        sediment_map = st.session_state["sediment_map"]
        analysis_method = st.session_state["analysis_method"]
        erosion_area = st.session_state["erosion_area"]
        sedimentation_area = st.session_state["sedimentation_area"]
        max_zone = st.session_state["max_zone"]


        st.info("Generating report...")

        fig, ax = plt.subplots(
            figsize=(16.5,11.7)
        )

        from matplotlib.colors import LightSource

        ls = LightSource(
            azdeg=315,
            altdeg=45
        )

        hillshade = ls.hillshade(
            grid_z,
            vert_exag=3
        )

        ax.imshow(
            hillshade,
            cmap="copper",
            alpha=0.35,
            extent=[
                np.nanmin(grid_x),
                np.nanmax(grid_x),
                np.nanmin(grid_y),
                np.nanmax(grid_y)
            ],
            origin="lower"
        )

        ax.contour(
            grid_x,
            grid_y,
            grid_z,
            levels=20,
            colors="black",
            linewidths=0.4,
            alpha=0.6
        )

        risk_mask = np.where(
            zone_map > 1.0,
            zone_map,
            np.nan
        )

        erosion_plot = ax.contourf(
            grid_x,
            grid_y,
            zone_map,
            levels=15,
            cmap="RdYlGn_r",
            alpha=0.65
        )

        sediment_plot = ax.contourf(
            grid_x,
            grid_y,
            sediment_map,
            levels=[0.5,0.7,0.85,1],
            cmap="Blues",
            alpha=0.50
        )

        bx, by = boundary.exterior.xy

        ax.plot(
            bx,
            by,
            color="black",
            linewidth=2.5
        )

        # ==========================
        # DRAW CROSS SECTION LINE
        # ==========================

        if "section_line" in st.session_state:

            p1, p2 = st.session_state["section_line"]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="cyan",
                linewidth=3,
                linestyle="--"
            )

            ax.text(
                p1[0],
                p1[1],
                "A",
                fontsize=12,
                fontweight="bold"
            )

            ax.text(
                p2[0],
                p2[1],
                "A'",
                fontsize=12,
                fontweight="bold"
            )

        cbar = plt.colorbar(
            erosion_plot,
            ax=ax,
            shrink=0.7
        )

        cbar.set_label(
            "Erosion Risk Index"
        )
        
        ax.annotate(
            '',
            xy=(0.82,0.12),      # ujung panah (arah North)
            xytext=(0.92,0.12),  # pangkal panah
            xycoords='axes fraction',

            arrowprops=dict(
                arrowstyle='-|>',
                lw=2,
                color='black'
            )
        )

        ax.text(
            0.93,
            0.12,
            'N',
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='center'
        )

        scalebar = ScaleBar(
            1,
            units="m",
            location="lower left"
        )

        ax.set_xlim(
            np.nanmin(grid_x),
            np.nanmax(grid_x)
        )

        ax.set_ylim(
            np.nanmin(grid_y),
            np.nanmax(grid_y)
        )

        ax.set_title(
            "EroSlope Risk Assessment Map",
            fontsize=18,
            fontweight="bold"
        )

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        ax.add_artist(
            scalebar
        )

        ax.text(
            0.02,
            0.98,
            f"""
        Analysis : {analysis_method}

        Erosion Area :
        {erosion_area:.2f} Ha

        Sedimentation Area :
        {sedimentation_area:.2f} Ha

        Max Risk :
        {max_zone:.2f}
        """,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(
                facecolor='white',
                alpha=0.8
            )
        )
        # 👇 TAMBAH INI LANGSUNG SETELAHNYA
        

        ax.set_aspect('equal', adjustable='box')

        plt.savefig(
            "Erosion_Map.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        pdf_file = "Executive_Report.pdf"

        doc = SimpleDocTemplate(pdf_file)

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "EXECUTIVE EROSION REPORT",
                styles["Title"]
            )
        )

        story.append(Spacer(1,20))

        story.append(
            Image(
                "Erosion_Map.png",
                width=500,
                height=350
            )
        )

        story.append(Spacer(1,15))

        summary = f"""
        <b>RISK SUMMARY</b><br/>
        Analysis Method : {analysis_method}<br/>
        Potential Erosion Area : {erosion_area:.2f} Ha<br/>
        Potential Sedimentation Area : {sedimentation_area:.2f} Ha<br/>
        Maximum Risk Index : {max_zone:.2f}
        """

        story.append(
            Paragraph(
                summary,
                styles["BodyText"]
            )
        )

        story.append(Spacer(1,20))

        from reportlab.platypus import PageBreak

        # ===================================
        # CROSS SECTION PAGE
        # ===================================

        if "section_distance" in st.session_state:

            fig_sec, ax_sec = plt.subplots(
                figsize=(10,4)
            )

            ax_sec.plot(
                st.session_state["section_distance"],
                st.session_state["section_elevation"],
                linewidth=2
            )

            ax_sec.set_title(
                "Cross Section A-A'"
            )

            ax_sec.set_xlabel(
                "Distance (m)"
            )

            ax_sec.set_ylabel(
                "Elevation (m)"
            )

            plt.tight_layout()

            plt.savefig(
                "CrossSection.png",
                dpi=300
            )

            plt.close()

            story.append(PageBreak())

            story.append(
                Paragraph(
                    "CROSS SECTION A-A'",
                    styles["Heading1"]
                )
            )

            story.append(
                Image(
                    "CrossSection.png",
                    width=500,
                    height=220
                )
            )
        
        story.append(PageBreak())

        story.append(
            Paragraph(
                "AUTOMATIC GEOTECHNICAL MITIGATION",
                styles["Heading1"]
            )
        )

        if max_zone < 0.5:

            mitigation = """
            Stable Zone Detected.

            Recommended Actions:
            • Routine inspection
            • Maintain drainage geometry
            • Annual monitoring
            """

        elif max_zone < 1.0:

            mitigation = """
            Initial Erosion Zone Detected.

            Recommended Actions:
            • Revegetation
            • Surface protection mat
            • Berm installation
            • Reduce runoff concentration
            """

        elif max_zone < 2.0:

            mitigation = """
            Moderate Erosion Zone Detected.

            Recommended Actions:
            • Riprap D50 150-300 mm
            • Check Dam Installation
            • Drop Structure
            • Velocity Reduction Below 1 m/s
            """

        else:

            mitigation = """
            Extreme Erosion Zone Detected.

            Potential Hazard:
            • Scouring
            • Headcut Migration
            • Drainage Failure
            • Slope Instability

            Recommended Actions:
            • Drainage Redesign
            • Concrete Lining
            • Reno Mattress
            • Detention Pond
            • Reduce Slope Below 5%
            • Target Velocity Below 1 m/s
            """

        story.append(
            Paragraph(
                mitigation,
                styles["BodyText"]
            )
        )

        doc.build(story)

        with open(pdf_file, "rb") as f:

            st.download_button(
                label="Download Executive Report",
                data=f,
                file_name="Executive_Report.pdf",
                mime="application/pdf"
            )


# =========================================================
# =================== TAB 2: ML ============================
# =========================================================
with tab2:

    st.subheader("Submit Data Training")
    train_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if train_file:
        df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)

        st.dataframe(df)

        target = st.selectbox("Target", df.columns)
        features = st.multiselect("Fitur", df.columns.drop(target))
        model_type = st.selectbox("Model", ["Regression", "Classification"])
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100


        # ================= CLEANING =================
        st.subheader("Data Cleaning")
        clean_option = st.radio("Cleaning", ["Tanpa Cleaning", "Hapus Outlier (IQR)"])

        df_clean = df.copy()

        if clean_option == "Hapus Outlier (IQR)":
            num_cols = df_clean[features + [target]].select_dtypes(include=np.number).columns

            Q1 = df_clean[num_cols].quantile(0.25)
            Q3 = df_clean[num_cols].quantile(0.75)
            IQR = Q3 - Q1

            mask = ~((df_clean[num_cols] < (Q1 - 1.5 * IQR)) |
                     (df_clean[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

            removed = len(df_clean) - np.sum(mask)
            df_clean = df_clean[mask]

            st.warning(f"Outlier terhapus: {removed} data")

        st.dataframe(df_clean)

        def klasifikasi_fs(fs):
            if fs < 1:
                return "FAIL"
            elif fs < 1.2:
                return "CRITICAL"
            else:
                return "STABLE"

         # ================= VISUAL AWAL =================
        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=np.round(corr.values * 100, 1),
            texttemplate="%{text}%",
            textfont={"size":12},
            hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
        ))
        
        fig_corr.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Parameter",
            yaxis_title="Parameter"
            )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Scatter Plot + Trendline")

        num_cols = df_clean.select_dtypes(include=np.number).columns

        col1, col2 = st.columns(2)

        with col1:
            x_cols = st.multiselect(
                "Pilih X (bisa lebih dari 1)",
                num_cols,
                default=[num_cols[0]] if len(num_cols) > 0 else []
            )

        with col2:
            y_cols = st.multiselect(
                "Pilih Y (bisa lebih dari 1)",
                num_cols,
                default=[num_cols[1]] if len(num_cols) > 1 else []
            )

        st.session_state["df_clean"] = df_clean

        fig_scatter = go.Figure()

        # ================= LOOP =================
        if x_cols and y_cols:
            for x_col in x_cols:
                for y_col in y_cols:

                    valid = df_clean[[x_col, y_col]].dropna()

                    if len(valid) < 3:
                        continue

                    x = valid[x_col]
                    y = valid[y_col]

                    # SCATTER
                    fig_scatter.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name=f"{x_col} vs {y_col}"
                    ))

                    # TRENDLINE
                    coeffs = np.polyfit(x, y, 1)
                    trend = np.poly1d(coeffs)

                    x_range = np.linspace(x.min(), x.max(), 100)

                    fig_scatter.add_trace(go.Scatter(
                        x=x_range,
                        y=trend(x_range),
                        mode='lines',
                        name=f"Trend {x_col}-{y_col}",
                        line=dict(dash='dash')
                    ))

            fig_scatter.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="X",
                yaxis_title="Y",
                legend=dict(orientation="h")
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Pilih minimal 1 X dan 1 Y")

        # ================= TIME SERIES =================
        if "Date" in df.columns:
            st.subheader("Time Series")
            df["Date"] = pd.to_datetime(df["Date"])

            ts_cols = st.multiselect("Parameter TS", df.columns,
                                     default=[c for c in df.columns if "water" in c.lower() or "mud" in c.lower()])

            fig_ts = go.Figure()
            for c in ts_cols:
                fig_ts.add_trace(go.Scatter(x=df["Date"], y=df[c], mode='lines', name=c))
            st.plotly_chart(fig_ts, use_container_width=True)

         # ================= VIOLIN (PINDAH KE ATAS) =================
        st.subheader("Violin Plot")

        cols_multi = st.multiselect("Pilih Parameter", num_cols, default=list(num_cols[:2]))

        if cols_multi:
            fig_v = go.Figure()
            for col in cols_multi:
                fig_v.add_trace(go.Violin(y=df_clean[col], name=col, box_visible=True))
            st.plotly_chart(fig_v, use_container_width=True)
        # ================= AUTO MODEL =================
        st.subheader("Auto Model Selection")

        if st.button("Cek Model Terbaik"):

            X = df_clean[features]
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            results = []

            if model_type == "Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor()
                }
                if xgb_available:
                    models["XGBoost"] = XGBRegressor()

                for name, m in models.items():
                    m.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    results.append({"Model": name, "Score": r2_score(y_test, pred)})

            else:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier()
                }
                if xgb_available:
                    models["XGBoost"] = XGBClassifier(eval_metric='logloss')

                for name, m in models.items():
                    m.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    results.append({"Model": name, "Score": accuracy_score(y_test, pred)})

            df_result = pd.DataFrame(results).sort_values(by="Score", ascending=False)

            st.dataframe(df_result)
            st.bar_chart(df_result.set_index("Model"))

            st.session_state["best_model"] = df_result.iloc[0]["Model"]

        # ================= TRAIN =================
        st.subheader("Train Model")

        model_choice = st.selectbox(
            "Model",
            ["Auto (Best)", "Linear Regression", "Random Forest"] + (["XGBoost"] if xgb_available else [])
        )

        if st.button("Train Model"):

            X = df_clean[features]
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if model_choice == "Auto (Best)" and "best_model" in st.session_state:
                model_choice = st.session_state["best_model"]

            # ================= SELECT MODEL =================
            if model_type == "Regression":
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        random_state=42
                    )
                elif model_choice == "XGBoost":
                    model = XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42
                    )

            else:
                if model_choice == "Linear Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier()
                elif model_choice == "XGBoost":
                    model = XGBClassifier(eval_metric='logloss')

            # ================= FIT =================
            model.fit(X_train, y_train)

            # ================= PREDICT =================
            y_pred = model.predict(X_test)

            # ================= SAVE =================
            st.session_state["model"] = model
            st.session_state["features"] = features

            # ================= EVALUATION =================
            st.subheader("Model Evaluation")

            if model_type == "Regression":
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # MAPE (hindari divide by zero)
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-6, y_test))) * 100
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:.3f}")
                col3.metric("MSE", f"{mse:.3f}")
                col4.metric("RMSE", f"{rmse:.3f}")
                col5.metric("MAPE (%)", f"{mape:.2f}")
            else:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                st.dataframe(confusion_matrix(y_test, y_pred))

    # ================= PREDIKSI =================
    st.subheader("Submit Data Prediksi")

    pred_file = st.file_uploader("Upload Data Baru", type=["csv", "xlsx"], key="predict")

    if pred_file and "model" in st.session_state:

        df_pred = pd.read_csv(pred_file) if pred_file.name.endswith(".csv") else pd.read_excel(pred_file)

        model = st.session_state["model"]
        features = st.session_state["features"]

        df_pred["FS_PRED"] = model.predict(df_pred[features])
        
        # Auto klasifikasi
        df_pred["STATUS"] = df_pred["FS_PRED"].apply(klasifikasi_fs)

        st.dataframe(df_pred)

        st.download_button(
            "Download Hasil",
            df_pred.to_csv(index=False),
            "hasil_prediksi.csv"
        )

    if "df_pred" in locals():
        st.session_state["df_pred"] = df_pred

    # ================= TIME SERIES PREDIKSI =================
    st.markdown("### Time Series Prediction (Dual Axis)")

    df_pred = st.session_state.get("df_pred", None)

    if df_pred is not None:

        if "Date" in df_pred.columns:

            df_pred["Date"] = pd.to_datetime(df_pred["Date"], errors='coerce')
            df_pred = df_pred.dropna(subset=["Date"])
            df_pred = df_pred.sort_values("Date")

            all_cols = [col for col in df_pred.columns if col not in ["Date", "STATUS"]]

            col1, col2 = st.columns(2)

            with col1:
                primary_cols = st.multiselect(
                    "Primary Y (Line - kiri)",
                    options=all_cols,
                    default=[col for col in ["WL", "ML"] if col in all_cols]
                )

            with col2:
                secondary_cols = st.multiselect(
                    "Secondary Y (Area - kanan)",
                    options=all_cols,
                    default=[col for col in ["FS_PRED"] if col in all_cols]
                )

            fig = go.Figure()

            # ================= PRIMARY (LINE)
            for col in primary_cols:
                fig.add_trace(go.Scatter(
                    x=df_pred["Date"],
                    y=df_pred[col],
                    mode='lines',
                    name=f"{col} (Primary)",
                    yaxis="y1"
                ))

            # ================= SECONDARY (AREA)
            for col in secondary_cols:
                fig.add_trace(go.Scatter(
                    x=df_pred["Date"],
                    y=df_pred[col],
                    mode='lines',
                    name=f"{col} (Secondary)",
                    yaxis="y2",
                    fill='tozeroy',
                    opacity=0.4
                ))

            # ================= UNSTABLE MARKER
            if "STATUS" in df_pred.columns and "FS_PRED" in df_pred.columns:
                unstable = df_pred[df_pred["STATUS"] != "STABLE"]

                fig.add_trace(go.Scatter(
                    x=unstable["Date"],
                    y=unstable["FS_PRED"],
                    mode='markers',
                    name='UNSTABLE',
                    marker=dict(size=10, symbol='x'),
                    yaxis="y2"
                ))

            # ================= FS LIMIT
            fig.add_hline(
                y=1.5,
                line_dash="dash",
                annotation_text="FS Limit",
                yref="y2"
            )

            # ================= LAYOUT
            fig.update_layout(
                template="plotly_dark",
                height=550,
                hovermode="x unified",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Primary Axis", side="left"),
                yaxis2=dict(title="Secondary Axis", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Kolom 'Date' tidak ditemukan di data prediksi")

    else:
        st.info("Upload data prediksi dulu untuk menampilkan grafik")

# ================= MQG AI =================
import os
import pandas as pd
import streamlit as st
import numpy as np

from PyPDF2 import PdfReader
from docx import Document

# ===== AI =====
from sentence_transformers import SentenceTransformer
import faiss

st.subheader("MQG AI Assistant (FAISS Mode)")

# ================= LOAD DOKUMEN =================
def load_documents(folder_path="docs"):
    text_data = ""

    if not os.path.exists(folder_path):
        return ""

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        try:
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text_data += f.read() + "\n"

            elif file.endswith(".csv"):
                df = pd.read_csv(file_path)
                text_data += df.to_string() + "\n"

            elif file.endswith(".xlsx"):
                df = pd.read_excel(file_path)
                text_data += df.to_string() + "\n"

            elif file.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text_data += page.extract_text() or ""
                text_data += "\n"

            elif file.endswith(".docx"):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text_data += para.text + "\n"

        except:
            continue

    return text_data

knowledge_base = load_documents("docs")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ================= CHUNKING =================
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ================= VECTOR DB + FAISS =================
@st.cache_data
def build_faiss_index(text):
    chunks = chunk_text(text)

    if len(chunks) == 0:
        return [], None

    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return chunks, index

doc_chunks, faiss_index = build_faiss_index(knowledge_base)

# ================= SEMANTIC SEARCH =================
def semantic_search(query, chunks, index, top_k=3):
    if index is None or len(chunks) == 0:
        return ["Tidak ada data"]

    query_vector = model.encode([str(query)])
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    return [chunks[i] for i in indices[0]]

# ================= INTERPRETASI =================
def interpretasi_ml(df_pred):
    if df_pred is None:
        return "Tidak ada data ML"

    try:
        min_fs = df_pred["FS_PRED"].min()
        avg_fs = df_pred["FS_PRED"].mean()

        status = "STABLE"
        if min_fs < 1:
            status = "FAILURE"
        elif min_fs < 1.2:
            status = "CRITICAL"

        return f"""
FS min: {min_fs:.2f}
FS avg: {avg_fs:.2f}
Status: {status}
"""
    except:
        return "Error baca FS_PRED"

# ================= AI RESPONSE =================
def geo_ai_response(prompt, knowledge, df_pred):

    prompt = str(prompt).lower()
    ml_info = interpretasi_ml(df_pred)

    if "fs" in prompt:
        return f"📊 ANALISA FS:\n{ml_info}"

    elif "piping" in prompt:
        return "💧 Piping → kontrol hydraulic gradient & drainase"

    elif "erosi" in prompt:
        return "🌧️ Erosi → riprap & vegetasi"

    else:
        results = semantic_search(prompt, doc_chunks, faiss_index)

        context = "\n\n---\n\n".join(results)

        return f"""
📚 HASIL PALING RELEVAN:

{context}

📊 ML:
{ml_info}
"""

# ================= CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "audience.png" if msg["role"] == "user" else "ai_bot.png"

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt = st.chat_input("Tanya geoteknik...")

df_pred = st.session_state.get("df_pred", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="audience.png"):
        st.markdown(prompt)

    result = geo_ai_response(prompt, knowledge_base, df_pred)

    with st.chat_message("assistant", avatar="ai_bot.png"):st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})

    
