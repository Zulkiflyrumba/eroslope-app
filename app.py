import streamlit as st
import numpy as np
import pandas as pd
import ezdxf
import tempfile
import time
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid
from shapely import vectorized
from shapely.ops import linemerge, polygonize, unary_union
import os

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.spatial import Delaunay
from scipy.optimize import brentq
import alphashape
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import requests

from matplotlib_scalebar.scalebar import ScaleBar

from reportlab.platypus import (
    SimpleDocTemplate,
    BaseDocTemplate,
    PageTemplate,
    Frame,
    NextPageTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)


from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

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
# LOGIN GATE
# =====================================================
#
# Autentikasi sederhana berbasis session_state + password hashing (SHA-256 + salt).
# Tidak butuh library eksternal tambahan (streamlit-authenticator dsb) supaya tetap
# ringan di-deploy. CATATAN PENTING UNTUK PRODUKSI:
#   - Kredensial di bawah ini HARUS dipindah ke st.secrets (Streamlit Cloud "Secrets")
#     atau environment variable / database — JANGAN commit password asli ke repo.
#   - Untuk multi-user dengan role granular / audit log, pertimbangkan backend
#     database (mis. Postgres + tabel users) daripada dict di kode.
import hashlib
import hmac
import secrets as _secrets_mod

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def _load_user_db():
    """Ambil kredensial dari st.secrets kalau tersedia (disarankan untuk produksi),
    fallback ke default demo (WAJIB diganti sebelum dipakai di lapangan)."""
    try:
        if "auth_users" in st.secrets:
            return dict(st.secrets["auth_users"])
    except Exception:
        pass  # st.secrets belum dikonfigurasi -> pakai fallback demo di bawah
    # ---- fallback default (GANTI SEBELUM PRODUKSI) ----
    default_salt = "geoai-default-salt-ganti-ini"
    return {
        "admin": {
            "name": "Administrator",
            "role": "admin",
            "salt": default_salt,
            "password_hash": _hash_password("admin123", default_salt),
        },
        "surveyor": {
            "name": "Surveyor Lapangan",
            "role": "user",
            "salt": default_salt,
            "password_hash": _hash_password("surveyor123", default_salt),
        },
    }

USER_DB = _load_user_db()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["auth_username"] = None
    st.session_state["auth_name"] = None
    st.session_state["auth_role"] = None

def _attempt_login(username, password):
    user = USER_DB.get(username)
    if not user:
        return False
    computed = _hash_password(password, user["salt"])
    if hmac.compare_digest(computed, user["password_hash"]):
        st.session_state["authenticated"] = True
        st.session_state["auth_username"] = username
        st.session_state["auth_name"] = user.get("name", username)
        st.session_state["auth_role"] = user.get("role", "user")
        return True
    return False

if not st.session_state["authenticated"]:

    # ================= MINING & WATER MANAGEMENT LOGIN =================
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        /* PENTING: .stApp / stAppViewContainer / stHeader HARUS transparan.
           Kalau elemen-elemen ini diberi background sendiri, background itu akan
           menutupi video (video ada di belakangnya secara stacking-order), sehingga
           video tidak pernah kelihatan walau tag <video>-nya sudah benar. */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .main{
            background: transparent !important;
        }
        .stApp{
            font-family:'Inter', sans-serif;
        }
        [data-testid="stHeader"]{
            background: transparent !important;
        }

        /* Video latar belakang aktivitas tambang open pit (footage stok gratis, bukan
           milik/dibuat sendiri — sumber: Pexels, lisensi Pexels License: bebas dipakai
           komersial/non-komersial tanpa atribusi wajib. Ganti mwm-video-src kalau mau
           pakai video lain dari penyedia stok gratis lain, mis. Coverr/Mixkit).
           z-index paling negatif -> paling belakang. */
        .mwm-bg-video{
            position:fixed !important; inset:0; width:100vw; height:100vh;
            object-fit:cover; z-index:-3; pointer-events:none;
        }

        .mwm-bg-svg{
            position:fixed !important; inset:0; width:100vw; height:100vh;
            z-index:-2; pointer-events:none; opacity:0.9;
            display:none; /* default disembunyikan; hanya dimunculkan via JS kalau video GAGAL load,
                              supaya ilustrasi ini tidak menutupi video saat video berhasil tampil */
        }

        /* Lapisan penggelap TERPISAH dari .stApp (bukan background .stApp lagi),
           dibuat setipis mungkin supaya video tetap terlihat jelas. Sisi ATAS dibuat
           lebih pekat & lebih tinggi supaya judul EROSLOPE kontras di atas video,
           sisi BAWAH tetap digelapkan supaya form tetap terbaca. */
        .mwm-bg-overlay{
            position:fixed !important; inset:0; width:100vw; height:100vh;
            z-index:-1; pointer-events:none;
            background:
                linear-gradient(180deg, rgba(2,6,8,0.72) 0%, rgba(3,8,11,0.48) 14%, rgba(4,10,14,0.14) 30%, rgba(4,10,14,0.10) 55%, rgba(3,8,11,0.36) 100%);
        }

        .mwm-title{
            font-family:'Rajdhani', sans-serif;
            font-weight:700;
            font-size:38px;
            letter-spacing:0.04em;
            text-align:center;
            color:#EAF6EE;
            margin-bottom:2px;
            text-shadow: 0 2px 18px rgba(0,0,0,0.6);
        }

        .mwm-subtitle{
            text-align:center;
            color:#8FD9C4;
            font-size:13px;
            letter-spacing:0.14em;
            text-transform:uppercase;
            margin-bottom:0;
            opacity:0.9;
        }

        /* Kartu login SEKARANG ditarget lewat st.container(key="mwm_login_card"),
           bukan lewat <div class="mwm-card"> manual yang dibuka/ditutup lintas
           beberapa st.markdown() -- pola lama itu bikin div-nya jadi kotak kosong
           sendiri karena tiap st.markdown() dirender sebagai elemen terpisah di
           Streamlit, tidak bersarang menjadi satu. */
        .st-key-mwm_login_card{
            position:relative;
            padding:34px 34px 26px;
            background:
                linear-gradient(160deg, rgba(0,0,0,0.16), rgba(0,0,0,0.16)),
                linear-gradient(160deg, rgba(10,22,20,0.42), rgba(6,16,15,0.50));
            border:1px solid rgba(143,217,196,0.30);
            border-radius:10px;
            box-shadow:
                0 0 0 1px rgba(196,151,74,0.12),
                0 12px 40px rgba(0,0,0,0.45);
            backdrop-filter: blur(2px);
        }

        .st-key-mwm_login_card::before{
            content:"";
            position:absolute; top:0; left:0; right:0; height:3px;
            background:linear-gradient(90deg, #C4974A, #3FA9A0, #4C8C4A);
            border-radius:10px 10px 0 0;
        }

        /* Backdrop hijau tua transparan di belakang expander kredensial demo +
           footer, supaya tulisannya tetap kebaca di atas video (sebelumnya
           langsung di atas video tanpa lapisan apa pun). */
        [data-testid="stExpander"]{
            background: linear-gradient(160deg, rgba(6,16,15,0.62), rgba(4,10,9,0.72)) !important;
            border:1px solid rgba(143,217,196,0.22) !important;
            border-radius:8px !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] p, [data-testid="stExpander"] span{
            color:#CFEDE2 !important;
        }

        .mwm-label{
            color:#8FD9C4;
            font-size:11px;
            font-weight:600;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin-bottom:16px;
            border-bottom:1px solid rgba(143,217,196,0.25);
            padding-bottom:9px;
        }

        div[data-testid="stForm"]{
            background:transparent; border:none; padding:0;
        }

        .stTextInput>div>div>input{
            background:rgba(143,217,196,0.06) !important;
            border:1px solid rgba(143,217,196,0.30) !important;
            border-radius:6px !important;
            color:#EAF6EE !important;
            font-family:'Inter', sans-serif !important;
        }
        .stTextInput>div>div>input:focus{
            border-color:#C4974A !important;
            box-shadow:0 0 0 3px rgba(196,151,74,0.18) !important;
        }
        .stTextInput label{ color:#8FD9C4 !important; font-size:12px !important; letter-spacing:0.04em; }

        div[data-testid="stFormSubmitButton"] button{
            background:linear-gradient(90deg, #3FA9A0, #4C8C4A) !important;
            color:#06110F !important;
            font-family:'Rajdhani', sans-serif !important;
            font-weight:700 !important;
            letter-spacing:0.04em !important;
            border:none !important;
            border-radius:6px !important;
            box-shadow:0 4px 18px rgba(63,169,160,0.35) !important;
            transition:transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover{
            transform:translateY(-1px) !important;
            box-shadow:0 6px 26px rgba(196,151,74,0.45) !important;
        }

        .mwm-footer{
            text-align:center; color:rgba(143,217,196,0.55);
            font-size:10.5px; letter-spacing:0.08em; margin-top:18px;
        }
        </style>

        <!-- ================================================================
             VIDEO BACKGROUND — aktivitas operasional open-pit mining/quarry
             (tampak udara area quarry dengan kolam tambang turquoise).
             Sumber: stok video gratis Pexels (bukan rekaman/aset milik user),
             lisensi Pexels License → boleh dipakai bebas tanpa royalti/atribusi.
             Kreator: David Pickup | Advertising & Marketing.
             Halaman: https://www.pexels.com/video/aerial-view-of-quarry-with-turquoise-pond-28668454/
             Kalau video gagal load (mis. tidak ada akses internet di server),
             SVG dekoratif open-pit di bawahnya otomatis tampil sebagai fallback.
             ================================================================ -->
        <video class="mwm-bg-video" id="mwmBgVideo" autoplay muted loop playsinline preload="auto"
               poster="https://images.pexels.com/videos/28668454/pexels-photo-28668454.jpeg?auto=compress&cs=tinysrgb&h=627&fit=crop&w=1200"
               onerror="var f=document.getElementById('mwmVideoFallbackNote'); if(f){f.style.display='block';}
                        var s=document.querySelector('.mwm-bg-svg'); if(s){s.style.display='block';}
                        this.style.display='none';">
          <source src="https://videos.pexels.com/video-files/28668454/12447079_2560_1440_30fps.mp4" type="video/mp4">
        </video>

        <div class="mwm-bg-overlay"></div>

        <!-- Catatan tersembunyi: kalau video gagal dimuat (mis. server tidak punya akses
             internet keluar / domain videos.pexels.com diblokir firewall), pesan ini akan
             muncul kecil di pojok agar mudah didiagnosa. SVG dekoratif tetap tampil sebagai
             fallback visual di belakangnya. -->
        <div id="mwmVideoFallbackNote" style="display:none; position:fixed; left:10px; bottom:6px;
             z-index:9999; color:rgba(234,246,238,0.55); font-size:10px; font-family:'Inter',sans-serif;">
          Video latar tidak dapat dimuat (cek akses internet ke videos.pexels.com) — menampilkan grafis fallback.
        </div>

        <svg class="mwm-bg-svg" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#1B3A3E"/>
              <stop offset="55%" stop-color="#123028"/>
              <stop offset="100%" stop-color="#0A1A1A"/>
            </linearGradient>
            <linearGradient id="pit" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#8A6A45"/>
              <stop offset="100%" stop-color="#5A4530"/>
            </linearGradient>
            <linearGradient id="water" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#4FBFB0"/>
              <stop offset="100%" stop-color="#256E68"/>
            </linearGradient>
            <linearGradient id="slope" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#4C8C4A"/>
              <stop offset="100%" stop-color="#2E5C31"/>
            </linearGradient>
          </defs>

          <rect x="0" y="0" width="1600" height="900" fill="url(#sky)"/>

          <!-- open-pit mine benches (stepped terraces), sisi kiri -->
          <g opacity="0.85">
            <polygon points="0,560 420,560 380,610 0,610" fill="url(#pit)"/>
            <polygon points="0,610 380,610 340,660 0,660" fill="#77592F"/>
            <polygon points="0,660 340,660 300,710 0,710" fill="#6B4F2C"/>
            <polygon points="0,710 300,710 260,760 0,760" fill="#5F4628"/>
            <polygon points="0,760 260,760 220,810 0,810" fill="#533E24"/>
            <polygon points="0,810 220,810 190,900 0,900" fill="#473520"/>
          </g>

          <!-- haul road ribbon melintasi bench -->
          <path d="M 30,570 C 160,600 120,650 260,680 C 360,700 300,740 200,770"
                fill="none" stroke="#D8C79A" stroke-width="10" opacity="0.35" stroke-linecap="round"/>

          <!-- rehabilitasi: lereng hijau bervegetasi, sisi kanan -->
          <g opacity="0.9">
            <path d="M 1600,900 L 1600,520 C 1420,560 1300,600 1180,660 C 1080,710 1020,760 980,900 Z" fill="url(#slope)"/>
            <!-- pohon-pohon sederhana -->
            <g fill="#1F4522">
              <circle cx="1260" cy="640" r="22"/><rect x="1256" y="655" width="8" height="22"/>
              <circle cx="1340" cy="600" r="26"/><rect x="1335" y="618" width="9" height="26"/>
              <circle cx="1420" cy="660" r="20"/><rect x="1416" y="674" width="8" height="20"/>
              <circle cx="1120" cy="720" r="22"/><rect x="1116" y="736" width="8" height="24"/>
              <circle cx="1480" cy="610" r="24"/><rect x="1475" y="627" width="9" height="24"/>
              <circle cx="1050" cy="780" r="20"/><rect x="1046" y="794" width="8" height="20"/>
            </g>
          </g>

          <!-- sediment pond / check dam di tengah-bawah -->
          <g>
            <rect x="380" y="800" width="720" height="14" fill="#3A4348" opacity="0.9"/>
            <ellipse cx="740" cy="840" rx="340" ry="60" fill="url(#water)" opacity="0.88"/>
            <path d="M 430,835 C 520,820 620,850 740,835 C 860,820 950,850 1040,835"
                  fill="none" stroke="#CDEDE7" stroke-width="3" opacity="0.35"/>
            <path d="M 450,855 C 560,842 660,868 760,853 C 880,838 960,862 1030,850"
                  fill="none" stroke="#CDEDE7" stroke-width="2.5" opacity="0.25"/>
            <!-- pipa drainase menuju pond -->
            <rect x="700" y="770" width="16" height="35" fill="#8C9AA0" opacity="0.8"/>
          </g>

          <!-- overlay penggelap ringan supaya kartu login tetap kontras -->
          <rect x="0" y="0" width="1600" height="900" fill="#04100E" opacity="0.28"/>
        </svg>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="mwm-title">⛏ EROSLOPE — MINE WATER &amp; EROSION CONTROL</div>'
        '<div class="mwm-subtitle">Digital Terrain · Hydrological Modelling · Erosion, Sedimentation &amp; Rehabilitation Management</div>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    login_col1, login_col2, login_col3 = st.columns([1, 1.05, 1])
    with login_col2:

        # PENTING: pakai st.container(key=...) supaya SEMUA widget di dalamnya
        # (label, form, input, tombol) benar-benar bersarang dalam satu elemen
        # DOM yang sama (.st-key-mwm_login_card) dan kebagian background kartu.
        # Cara lama (buka <div class="mwm-card"> di satu st.markdown lalu tutup
        # di st.markdown lain) TIDAK bersarang di Streamlit -- tiap st.markdown()
        # jadi elemen sejajar terpisah, jadi div pembukanya sendiri jadi kotak
        # kosong (itu kotak hitam kosong yang terlihat di atas label "Masuk ke
        # Sistem" sebelumnya).
        with st.container(key="mwm_login_card"):
            st.markdown('<div class="mwm-label">Masuk ke Sistem</div>', unsafe_allow_html=True)

            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Masuk", use_container_width=True)

            if submitted:
                if _attempt_login(login_username.strip(), login_password):
                    st.success(f"Berhasil masuk — {st.session_state['auth_name']} ({st.session_state['auth_role']})")
                    st.rerun()
                else:
                    st.error("Username atau password salah.")

        with st.expander("⚙ Kredensial demo (WAJIB diganti untuk pemakaian nyata)"):
            st.code(
                "admin    / admin123      (role: admin)\n"
                "surveyor / surveyor123   (role: user)",
                language="text"
            )
            st.caption(
                "Ganti lewat Streamlit Secrets (st.secrets['auth_users']) sebelum deploy ke lapangan. "
                "Jangan pakai password default ini untuk data proyek nyata."
            )

        st.markdown('<div class="mwm-footer">Mine Site Drainage · Sediment Control · Slope Rehabilitation Monitoring</div>', unsafe_allow_html=True)


    st.stop()

# ================= HELPER: safe image loader =================
import os as _os_header

def _safe_image(path, **kwargs):
    """st.image() crash kalau file asset tidak ada di server deploy (umum terjadi kalau
    header.png/logo.png/sump.jpg dsb tidak ikut ter-upload). Fallback aman: skip diam-diam kalau hilang."""
    if _os_header.path.exists(path):
        st.image(path, **kwargs)
        return True
    return False

# ================= SIDEBAR: tema mining/water + info user + quick tools geoteknik =================
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"]{
        background:
            radial-gradient(ellipse 480px 320px at 15% 0%, rgba(196,151,74,0.10), transparent 60%),
            radial-gradient(ellipse 420px 380px at 100% 100%, rgba(63,169,160,0.12), transparent 60%),
            linear-gradient(165deg, #0E211F 0%, #142B26 55%, #0B1A19 100%) !important;
        border-right: 1px solid rgba(143,217,196,0.14);
    }
    .mwm-user-card{
        background: linear-gradient(160deg, rgba(63,169,160,0.14), rgba(76,140,74,0.08));
        border: 1px solid rgba(143,217,196,0.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .mwm-user-card .name{ color:#EAF6EE; font-weight:700; font-size:15px; }
    .mwm-user-card .role{ color:#8FD9C4; font-size:11.5px; letter-spacing:0.06em; text-transform:uppercase; }
    .mwm-side-divider{
        height:1px; margin:14px 0;
        background: linear-gradient(90deg, transparent, rgba(143,217,196,0.35), transparent);
        border:none;
    }
    .mwm-side-heading{
        color:#8FD9C4; font-size:11px; font-weight:700; letter-spacing:0.10em;
        text-transform:uppercase; margin: 4px 0 8px 0;
    }
    .mwm-status-pill{
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:11px; font-weight:700; letter-spacing:0.03em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:

    st.markdown(
        f"""<div class="mwm-user-card">
                <div class="name">{st.session_state['auth_name']}</div>
                <div class="role">Role: {st.session_state['auth_role']}</div>
            </div>""",
        unsafe_allow_html=True
    )

    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["auth_username"] = None
        st.session_state["auth_name"] = None
        st.session_state["auth_role"] = None
        st.rerun()

    st.markdown('<hr class="mwm-side-divider"/>', unsafe_allow_html=True)

    # ---- Ringkasan risiko cepat (otomatis dari hasil analisis, kalau sudah pernah dijalankan) ----
    st.markdown('<div class="mwm-side-heading">Ringkasan Risiko</div>', unsafe_allow_html=True)
    _seg_results_side = st.session_state.get("segment_results", {})
    if _seg_results_side:
        _max_seg_label, _max_seg_score = None, -1
        for _sid, _seg in _seg_results_side.items():
            if _seg.get("max_zone", 0) > _max_seg_score:
                _max_seg_score = _seg.get("max_zone", 0)
                _max_seg_label = _seg.get("label", _sid)
        _pill_color = (
            "#7A1F1F" if _max_seg_score >= 2.0 else
            "#B7622A" if _max_seg_score >= 1.0 else
            "#B7950B" if _max_seg_score >= 0.5 else
            "#2E5C31"
        )
        _pill_text = (
            "MERAH — KRITIS" if _max_seg_score >= 2.0 else
            "ORANYE — SIAGA" if _max_seg_score >= 1.0 else
            "KUNING — WASPADA" if _max_seg_score >= 0.5 else
            "HIJAU — NORMAL"
        )
        st.caption(f"Segmen dianalisis: **{len(_seg_results_side)}**")
        st.caption(f"Segmen risiko tertinggi: **{_max_seg_label}**")
        st.markdown(
            f'<span class="mwm-status-pill" style="background:{_pill_color}; color:#fff;">{_pill_text}</span>',
            unsafe_allow_html=True
        )
    else:
        st.caption("Belum ada analisis dijalankan pada sesi ini.")

    st.markdown('<hr class="mwm-side-divider"/>', unsafe_allow_html=True)

    # ---- Quick tools untuk engineer geoteknik lapangan ----
    st.markdown('<div class="mwm-side-heading">Quick Tools</div>', unsafe_allow_html=True)

    if st.button("Reset Data Hujan Online", use_container_width=True,
                 help="Hapus cache data hujan tersimpan agar bisa diambil ulang dari sumber terbaru."):
        st.session_state["online_rainfall"] = None
        st.session_state["online_rainfall_meta"] = None
        st.session_state["online_rainfall_errors"] = []
        st.success("Cache data hujan direset — silakan ambil ulang di panel utama.")

    with st.expander("SOP Tanggap Darurat Erosi/Limpasan"):
        st.markdown(
            "1. Amankan area & personel di sekitar titik limpasan/gerusan.\n"
            "2. Hentikan aktivitas alat berat di sekitar tebing/tanggul terdampak.\n"
            "3. Hubungi pengawas geoteknik & K3 lapangan.\n"
            "4. Dokumentasikan lokasi (koordinat), waktu, dan kondisi visual.\n"
            "5. Pasang tanda peringatan/barrier sementara.\n"
            "6. Lapor ke tim Quality & Geotechnical Department untuk asesmen lanjutan."
        )

    with st.expander("Rumus Cepat"):
        st.latex(r"V = \frac{1}{n} R^{2/3} S^{1/2} \quad \text{(Manning)}")
        st.latex(r"\theta = \frac{\tau_0}{(\rho_s-\rho_w) g D_{50}} \quad \text{(Shields)}")
        st.latex(r"E = M\left(\frac{\tau_0}{\tau_c}-1\right) \quad \text{(Partheniades)}")

    st.markdown('<hr class="mwm-side-divider"/>', unsafe_allow_html=True)
    st.caption(f"Sesi berjalan sejak login · {pd.Timestamp.now().strftime('%H:%M, %d %b %Y')}")



# =====================================================
# LANDING PAGE
# =====================================================

if "home_page" not in st.session_state:
    st.session_state.home_page = True

if st.session_state.home_page:

    st.markdown("""
    <style>

    .stApp{
        background: linear-gradient(135deg, #00151a 0%, #02111d 100%);
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
        font-size:60px;
        font-weight:900;
        color:white;
        line-height:0.95;
    }

    .hero-subtitle{
        margin-top:15px;
        color:#8FFF7A;
        font-size:20px;
        font-weight:700;
    }

    .hero-desc{
        margin-top:20px;
        color:white;
        font-size:17px;
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
        <div class="hero-card">

        <div class="hero-title">
            GEOTECHNICAL<br>INTELLIGENCE
        </div>

        <div class="hero-subtitle">
            Turning Monitoring Data Into Engineering Decisions
        </div>

        <div class="hero-desc">
            Advanced platform combining AI, Machine Learning, Numerical Modelling,
            Digital Twin, and Predictive Analytics for smarter mining operations.
        </div>

        <div style="margin-top:20px">
            <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">AI</span>
            <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">Machine Learning</span>
            <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">Digital Twin</span>
            <span style="padding:8px 16px;border-radius:20px;background:#006b4f;color:white;">Numerical Modelling</span>
        </div>

        </div>
        """, unsafe_allow_html=True)

    with right:

        c1, c2, c3 = st.columns(3)

        feat_images = [
            ("sump.jpg", "", "SUMP MONITORING"),
            ("water.jpg", "", "WATER MANAGEMENT"),
            ("slope.jpg", "", "PIT STABILITY"),
        ]

        for col, (img_path, emoji_fallback, title) in zip([c1, c2, c3], feat_images):
            with col:
                if not _safe_image(img_path, use_container_width=True):
                    st.markdown(
                        f"""
                        <div style="
                            background:rgba(0,0,0,0.3);
                            border:1px solid rgba(255,255,255,0.15);
                            border-radius:16px;
                            padding:40px 0;
                            text-align:center;
                            font-size:44px;
                        ">{emoji_fallback}</div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f"<center><h4 style='color:white'>{title}</h4></center>",
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 2])

    with c2:
        if st.button("▶︎ START ANALYSIS", use_container_width=True):
            st.session_state.home_page = False
            st.rerun()

    st.stop()

if "analysis_method" not in st.session_state:
    st.session_state["analysis_method"] = None

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "erosion_area" not in st.session_state:
    st.session_state["erosion_area"] = 0.0

if "online_rainfall" not in st.session_state:
    st.session_state["online_rainfall"] = None

if "online_rainfall_meta" not in st.session_state:
    st.session_state["online_rainfall_meta"] = None

if "online_rainfall_errors" not in st.session_state:
    st.session_state["online_rainfall_errors"] = []

if "sedimentation_area" not in st.session_state:
    st.session_state["sedimentation_area"] = 0.0

if "max_zone" not in st.session_state:
    st.session_state["max_zone"] = 0.0

# ================= CSS PREMIUM (tema lama: dark green/teal + tekstur topografi) =================
def load_css():
    st.markdown("""
    <style>

    /* ===== BASE ===== */
    .stApp {
        background: linear-gradient(135deg, #002D2D, #001A1A);
        color: #E6F4F1;
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp::before {
        content:"";
        position:fixed;
        inset:0;
        background: url("https://www.transparenttextures.com/patterns/topography.png");
        background-size:900px;
        opacity:0.12;
        pointer-events:none;
        mix-blend-mode:screen;
    }

    .stApp {
        background:
        radial-gradient(circle at top left, rgba(0,255,150,0.12), transparent 35%),
        radial-gradient(circle at bottom right, rgba(149,191,71,0.10), transparent 40%),
        linear-gradient(135deg, #031A1A 0%, #022424 30%, #001515 70%, #000E0E 100%);
        color: white;
    }

    /* ===== HEADER =====
       Dihapus atas permintaan -- dulu ada kotak/background di sekitar
       logo+judul header, sekarang logo & judul dirender polos tanpa
       pembungkus bergaya apa pun (lihat bagian HEADER pada badan skrip). */

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

    [data-testid="stFileUploader"] {
        border:2px dashed rgba(0,255,170,0.35);
        border-radius:18px;
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(15px);
        padding:10px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color:#00E0A8;
        background: rgba(0,255,170,0.05);
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
    .stTabs [role="tab"]{
        background:rgba(255,255,255,0.04);
        border-radius:14px;
        margin-right:8px;
        padding:10px 22px;
        font-weight:600;
        transition:0.3s;
        color:#95BF47;
    }

    .stTabs [role="tab"]:hover{
        background:rgba(0,255,170,0.08);
        color:white;
    }

    .stTabs [aria-selected="true"]{
        background: linear-gradient(135deg, #00C896, #008060);
        color:white;
        box-shadow: 0 8px 25px rgba(0,200,150,0.30);
    }

    /* ===== ANIMATION ===== */
    [data-testid="stChatMessage"] {
        animation: fadeIn 0.4s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity:0; transform: translateY(8px);}
        to {opacity:1; transform: translateY(0);}
    }

    /* ===== MQG AI FLOATING CHAT (tombol emoji + bubble popup) =====
       Dulu section "MQG AI Assistant" dirender polos di bawah semua tab (di
       luar with tab1/tab2/tab3, jadi selalu tampil apa pun tab yang aktif --
       makanya terasa 'muncul terus & mengganggu'). Sekarang jadi tombol
       melayang di pojok kanan-bawah; panel chat cuma muncul saat tombolnya
       diklik, dan balik jadi ikon emoji lagi saat diklik ulang. */
    .st-key-mqg_chat_fab {
        position: fixed;
        bottom: 22px;
        right: 22px;
        z-index: 99999;
        width: 60px;
    }
    .st-key-mqg_chat_fab button {
        border-radius: 50% !important;
        width: 58px !important;
        height: 58px !important;
        font-size: 24px !important;
        padding: 0 !important;
        box-shadow: 0 4px 18px rgba(0,0,0,0.45) !important;
        border: 1px solid rgba(149,191,71,0.4) !important;
    }
    .st-key-mqg_chat_panel {
        position: fixed;
        bottom: 92px;
        right: 22px;
        z-index: 99998;
        width: 380px;
        max-width: 92vw;
        max-height: 65vh;
        overflow-y: auto;
        background: rgba(10, 20, 18, 0.98);
        border: 1px solid rgba(149,191,71,0.35);
        border-radius: 16px;
        padding: 16px 16px 8px 16px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
    }

    </style>
    """, unsafe_allow_html=True)

load_css()

# ================= HEADER =================
header_container = st.container()
with header_container:
    _safe_image("header.png", use_container_width=True)

# NOTE: dulu dibuka lewat st.markdown('<div class="header">') lalu ditutup
# PERBAIKAN: dulu dibuka lewat st.markdown('<div class="header">') lalu ditutup
# st.markdown('</div>') di ujung -- pola itu menghasilkan div "header" KOSONG
# sendiri (jadi bar hijau gelap tanpa isi di layar) karena tiap st.markdown()/
# st.columns() dirender Streamlit sebagai elemen terpisah, tidak benar-benar
# bersarang jadi satu <div>. Sempat dicoba dibungkus st.container(key=...) supaya
# kotak hijaunya beneran melingkupi logo/judul -- tapi ternyata yang diinginkan
# BUKAN kotaknya diperbaiki, melainkan dihilangkan sama sekali. Jadi sekarang
# logo & judul dirender polos tanpa background/border/padding pembungkus apa pun.
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if not _safe_image("logo.png", width=120):
        st.markdown("<div style='font-size:34px;'></div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:white; margin-bottom:0;">
            EroSlope &amp; Machine Learning
        </h1>
        <p style="color:rgba(255,255,255,0.7); font-size:14px;">
            Advanced Geotechnical Analysis for Erosion, Stability, and Predictive Modeling
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if not _safe_image("logo_2.png", width=120):
        st.markdown("<div style='font-size:34px; text-align:right;'></div>", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["Erosion Mapping", "Machine Learning", "Monitoring Deviation"])

# =========================================================
# =================== TAB 1: EROSION ======================
# =========================================================
with tab1:

    st.subheader("A. Input DXF & Segmen Sekat/Channel")

    st.caption(
        "Tiap segmen = 1 lokasi sekat/channel dengan DXF (kontur & boundary) "
        "dan parameter sendiri-sendiri. Properti tidak digeneralisir ke segmen lain."
    )

    if "segments" not in st.session_state:
        st.session_state["segments"] = ["seg_1"]
    if "segment_counter" not in st.session_state:
        st.session_state["segment_counter"] = 1

    def _add_segment():
        st.session_state["segment_counter"] += 1
        st.session_state["segments"].append(
            f"seg_{st.session_state['segment_counter']}"
        )

    def _remove_segment(seg_id):
        st.session_state["segments"] = [
            s for s in st.session_state["segments"] if s != seg_id
        ]

    for idx, sid in enumerate(st.session_state["segments"]):

        default_name = "Segmen 1 (Main DXF)" if idx == 0 else f"Segmen {idx + 1}"
        label_now = st.session_state.get(f"seg_name_{sid}", default_name)

        with st.expander(f"{label_now}", expanded=(idx == 0)):

            colh1, colh2 = st.columns([3, 1])

            with colh1:
                st.text_input(
                    "Nama Segmen / Lokasi",
                    value=default_name,
                    key=f"seg_name_{sid}"
                )

            with colh2:
                st.write("")
                if len(st.session_state["segments"]) > 1:
                    if st.button("Hapus", key=f"del_{sid}"):
                        _remove_segment(sid)
                        st.rerun()

            st.radio(
                "Jenis Kondisi Segmen Ini",
                ["Desain Baru", "Eksisting (Rekonstruksi)"],
                key=f"design_type_{sid}",
                horizontal=True,
                help=(
                    "Desain Baru: cek kelayakan lokasi/geometri untuk "
                    "diimplementasikan -> output naratif LAYAK / REJECT + rekomendasi rekayasa. "
                    "Eksisting (Rekonstruksi): channel/sekat sudah ada -> output sebaran "
                    "erosi & sedimentasi + rekomendasi perbaikan/mitigasi."
                )
            )

            is_sub_segment = (idx > 0)

            if is_sub_segment:
                st.info(
                    "Sub-boundary — memakai kontur/DXF terrain yang SAMA dengan **Segmen 1 (Main)**. "
                    "Anda hanya perlu mengunggah boundary sub-area di bawah ini (garis batas yang lebih "
                    "kecil, ada DI DALAM boundary utama), bukan DXF kontur baru."
                )

                cold2 = st.container()

                with cold2:
                    st.file_uploader(
                        "Upload Boundary DXF (sub-area, wajib untuk sub-segmen)",
                        type=["dxf"],
                        key=f"boundary_dxf_{sid}"
                    )
            else:

                cold1, cold2 = st.columns(2)

                with cold1:
                    st.file_uploader(
                        "Upload Kontur DXF",
                        type=["dxf"],
                        key=f"kontur_dxf_{sid}"
                    )

                with cold2:
                    st.file_uploader(
                        "Upload Boundary DXF (opsional)",
                        type=["dxf"],
                        key=f"boundary_dxf_{sid}"
                    )

            if not is_sub_segment:
                st.file_uploader(
                    "Upload Orthophoto (opsional) — .ecw / .tif / .tiff / .jp2",
                    type=["ecw", "tif", "tiff", "jp2", "jp2000"],
                    key=f"orthophoto_{sid}",
                    help=(
                        "Opsional. Kalau diunggah, peta risiko 2D (rainbow kontur erosi/sedimentasi) "
                        "dan garis DXF akan ditampilkan teroverlay di atas orthophoto ini sebagai latar. "
                        "Format .ecw butuh driver ECW pada instalasi GDAL server (biasanya tersedia di "
                        "GDAL versi komersial/ArcGIS/Global Mapper, TIDAK selalu ada di server cloud "
                        "standar) — kalau gagal dibaca, convert dulu ke GeoTIFF (.tif) pakai QGIS/Global "
                        "Mapper lalu unggah ulang. Orthophoto harus sudah georeferenced pada sistem "
                        "koordinat yang SAMA dengan DXF (tidak ada reprojection otomatis)."
                    )
                )

            st.markdown("**Parameter Flow & Sedimen (khusus segmen ini)**")

            main_sid = st.session_state["segments"][0]

            if is_sub_segment:
                # Sub-segmen TIDAK memilih metode analisis sendiri lagi -- ikut
                # metode & parameter yang sama dengan Segmen 1 (Main), supaya
                # hasil antar segmen konsisten/comparable (bukan tercampur
                # beberapa metode analisis berbeda dalam satu proyek).
                analysis_method_seg = st.session_state.get(
                    f"analysis_method_{main_sid}", "Hjulstrom Diagram"
                )
                st.session_state[f"analysis_method_{sid}"] = analysis_method_seg

                st.caption(
                    f"Metode Analisis Sedimen: **{analysis_method_seg}** "
                    "(mengikuti Segmen 1 / Main — tidak dipilih ulang per sub-segmen)."
                )

                colp2 = st.container()
                colp2.number_input(
                    "Velocity Hulu (m/s)", 0.01, 5.0, 0.5,
                    key=f"velocity_hulu_{sid}"
                )
            else:
                colp1, colp2 = st.columns(2)

                analysis_method_seg = colp1.radio(
                    "Metode Analisis Sedimen",
                    [
                        "Hjulstrom Diagram",
                        "Shields Diagram",
                        "Partheniades + Flow Accumulation"
                    ],
                    key=f"analysis_method_{sid}"
                )

                colp2.number_input(
                    "Velocity Hulu (m/s)", 0.01, 5.0, 0.5,
                    key=f"velocity_hulu_{sid}"
                )

            if analysis_method_seg == "Hjulstrom Diagram":

                st.number_input(
                    "Grain Size d (mm)", 0.001, 10.0, 0.1,
                    key=f"grain_size_{sid}"
                )

            elif analysis_method_seg == "Shields Diagram":

                colsd1, colsd2, colsd3 = st.columns(3)

                colsd1.number_input(
                    "D50 Sedimen (mm)", 0.001, 100.0, 0.5,
                    key=f"grain_size_{sid}"
                )

                colsd2.number_input(
                    "Density Water (kg/m3)", 1000, 1200, 1000,
                    key=f"rho_water_{sid}"
                )

                colsd3.number_input(
                    "Density Sediment (kg/m3)", 1200, 3500, 2650,
                    key=f"rho_soil_{sid}"
                )

                st.number_input(
                    "Asumsi Kedalaman Aliran / Flow Depth (m)", 0.01, 10.0, 0.30,
                    key=f"flow_depth_{sid}",
                    help=(
                        "Dipakai untuk menghitung tegangan geser dasar τ = ρw·g·h·S "
                        "(depth-slope product). Sebelumnya parameter ini tidak dipakai "
                        "pada Shields sehingga τ terhitung tanpa suku kedalaman — sudah diperbaiki."
                    )
                )

            elif analysis_method_seg == "Partheniades + Flow Accumulation":

                colpp1, colpp2 = st.columns(2)

                colpp1.number_input(
                    "Critical Shear Stress \u03c4c (Pa)", 0.01, 100.0, 2.0,
                    key=f"tau_critical_{sid}"
                )

                colpp2.number_input(
                    "Erodibility Coefficient M", 0.0001, 10.0, 0.05,
                    format="%.4f",
                    key=f"erodibility_M_{sid}"
                )

                colpp3, colpp4 = st.columns(2)

                colpp3.slider(
                    "Flow Accumulation Weight", 0.1, 10.0, 3.0, 0.1,
                    key=f"flow_weight_{sid}"
                )

                colpp4.number_input(
                    "Assumed Flow Depth (m)", 0.01, 10.0, 0.30,
                    key=f"flow_depth_{sid}"
                )

                st.number_input(
                    "Density Water (kg/m3)", 1000, 1200, 1000,
                    key=f"rho_water_{sid}"
                )

            if not is_sub_segment:
                st.markdown("**Skenario Sumber Aliran (khusus segmen ini)**")

                source_type_seg = st.selectbox(
                    "Pilih sumber aliran",
                    ["Hujan (Uniform)", "Satu Titik (Point Source)"],
                    key=f"source_type_{sid}"
                )

                if source_type_seg == "Satu Titik (Point Source)":

                    point_method_seg = st.radio(
                        "Cara menentukan titik aliran",
                        ["Ketik Koordinat Manual", "Klik di Peta Desain"],
                        key=f"point_method_{sid}",
                        horizontal=True,
                        help=(
                            "'Klik di Peta Desain' menampilkan peta 2D sederhana berisi garis DXF & "
                            "boundary area kajian (muncul di bagian bawah setelah DXF selesai diproses) "
                            "— tinggal klik di titik yang dimaksud sebagai lokasi awal air mengalir."
                        )
                    )

                    if point_method_seg == "Ketik Koordinat Manual":
                        colp3, colp4 = st.columns(2)
                        colp3.number_input("Koordinat X Hulu", key=f"point_x_{sid}")
                        colp4.number_input("Koordinat Y Hulu", key=f"point_y_{sid}")
                    else:
                        st.info(
                            "Mode klik-peta aktif — scroll ke bagian **'Pilih Titik Aliran di Peta'** "
                            "(muncul setelah DXF segmen ini selesai diproses di bawah) untuk klik titik "
                            "awal aliran secara langsung di atas desain."
                        )
                        _cx, _cy = st.session_state.get(f"flow_click_xy_{sid}", (None, None))
                        if _cx is not None:
                            st.caption(f"Titik terpilih saat ini: X = {_cx:.3f}, Y = {_cy:.3f}")

                    st.number_input(
                        "Kedalaman/Ketebalan Air Awal di Titik Hulu (m)",
                        min_value=0.01, max_value=10.0, value=0.20, step=0.01,
                        key=f"point_depth_{sid}",
                        help=(
                            "Ketebalan lapisan air pada saat mulai mengalir dari titik hulu ini. "
                            "Dipakai untuk memberi kesan visual 'setebal apa' aliran air pada "
                            "simulasi 3D & animasi (bukan hasil hitungan hidrolika Manning — kalau "
                            "'Rational Method + Manning's Equation' di bawah diaktifkan, kedalaman "
                            "hasil hitungan itu yang dipakai untuk analisis erosi, nilai di sini "
                            "murni untuk visualisasi ketebalan aliran di peta 3D)."
                        )
                    )

                st.markdown("**Hidrologi & Hidrolika (opsional — Rational Method + Manning's Equation)**")

                st.caption(
                    "Jika diaktifkan, kecepatan & kedalaman aliran TIDAK lagi diambil dari input manual "
                    "di atas, melainkan dihitung dari debit rencana hasil hujan (metode Rasional + rumus "
                    "Mononobe untuk intensitas, umum dipakai di Indonesia) dan geometri channel via "
                    "Manning's Equation. Ini menjawab kelemahan versi sebelumnya di mana kecepatan hanya "
                    "angka asumsi manual, tidak terhubung ke data hujan maupun geometri channel aktual."
                )

                use_hydraulics = st.checkbox(
                    "Aktifkan perhitungan Rational Method + Manning's Equation",
                    key=f"use_hydraulics_{sid}",
                    value=False
                )

                if use_hydraulics:

                    colh1, colh2, colh3 = st.columns(3)

                    colh1.selectbox(
                        "Koefisien Limpasan C",
                        [
                            "0.85 - Lahan terbuka/terganggu tambang, minim vegetasi",
                            "0.70 - Tanah kompak, vegetasi jarang",
                            "0.55 - Tanah dengan vegetasi sedang",
                            "0.35 - Vegetasi rapat/hutan",
                            "Custom"
                        ],
                        key=f"runoff_c_label_{sid}"
                    )

                    if st.session_state.get(f"runoff_c_label_{sid}", "").startswith("Custom"):
                        colh1.number_input(
                            "Nilai C custom", 0.05, 1.0, 0.70,
                            key=f"runoff_c_custom_{sid}"
                        )

                    colh2.number_input(
                        "Waktu Konsentrasi tc (jam)", 0.05, 24.0, 1.0, 0.05,
                        key=f"tc_hours_{sid}",
                        help="Waktu air mengalir dari titik terjauh DAS ke titik keluaran channel."
                    )

                    colh3.number_input(
                        "Periode Ulang Hujan (tahun) — info",
                        1, 1000, 25,
                        key=f"return_period_{sid}",
                        help="Untuk dokumentasi laporan. Nilai curah hujan tetap memakai data R24 yang sudah diambil di atas."
                    )

                    colh4, colh5, colh6 = st.columns(3)

                    colh4.selectbox(
                        "Manning's Roughness (n)",
                        [
                            "0.030 - Saluran tanah alami",
                            "0.035 - Saluran berbatu/tidak rata",
                            "0.022 - Saluran tanah terkompaksi + rumput",
                            "0.013 - Beton/lining halus",
                            "Custom"
                        ],
                        key=f"manning_n_label_{sid}"
                    )

                    if st.session_state.get(f"manning_n_label_{sid}", "").startswith("Custom"):
                        colh4.number_input(
                            "Nilai n custom", 0.008, 0.15, 0.030,
                            key=f"manning_n_custom_{sid}"
                        )

                    colh5.number_input(
                        "Lebar Dasar Channel b (m)", 0.1, 100.0, 3.0,
                        key=f"channel_b_{sid}"
                    )

                    colh6.number_input(
                        "Kemiringan Tebing z (H:V)", 0.0, 5.0, 1.5,
                        key=f"channel_z_{sid}",
                        help="0 = dinding vertikal/persegi, 1.5 = umum untuk galian tanah."
                    )

                    st.number_input(
                        "Tinggi Total Channel/Tanggul (m) — untuk cek freeboard",
                        0.1, 30.0, 1.5,
                        key=f"channel_h_total_{sid}"
                    )

    st.button("Tambah Segmen (DXF Lain)", on_click=_add_segment)

    st.markdown("---")
    st.subheader("B. Parameter Umum (berlaku untuk semua segmen)")

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
    # ================= SUMBER DATA HUJAN (multi-opsi + fallback otomatis) =================
    #
    # Kode lama hanya pakai 1 sumber (Open-Meteo Archive), tanggal hardcoded ke Juni 2025
    # (jadi makin lama makin basi/salah), dan bare "except: return None" tanpa pesan error
    # sama sekali — pengguna tidak tahu kenapa gagal. Diperbaiki total di bawah:
    #   - 3 sumber: Open-Meteo Archive (ERA5 reanalysis, global), NASA POWER (global, tanpa
    #     API key), Open-Meteo Forecast (data terkini/prakiraan) — plus opsi input manual.
    #   - Fallback berurutan otomatis kalau satu sumber gagal/timeout.
    #   - Rentang tanggal relatif ke hari ini (bukan hardcode), bisa diatur pengguna.
    #   - Statistik yang relevan untuk desain erosi: rata-rata, hujan harian maksimum
    #     dalam periode (dipakai untuk kejadian kritis), dan jumlah hari data valid.

    from datetime import date, timedelta

    def _fetch_open_meteo_archive(lat, lon, start_date, end_date):
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=precipitation_sum&timezone=auto"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        vals = [v for v in data["daily"]["precipitation_sum"] if v is not None]
        if not vals:
            raise ValueError("Tidak ada data presipitasi pada rentang tanggal ini.")
        return {
            "source": "Open-Meteo Archive (ERA5 Reanalysis)",
            "mean": float(np.nanmean(vals)),
            "max": float(np.nanmax(vals)),
            "n_days": len(vals),
            "period": f"{start_date} s/d {end_date}",
        }

    def _fetch_open_meteo_forecast(lat, lon, past_days=16):
        # Sumber cadangan: endpoint forecast Open-Meteo juga menyediakan data historis
        # jangka pendek (past_days) — berguna kalau endpoint archive sedang down.
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum&past_days={past_days}&forecast_days=1&timezone=auto"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        vals = [v for v in data["daily"]["precipitation_sum"] if v is not None]
        if not vals:
            raise ValueError("Tidak ada data presipitasi dari endpoint forecast.")
        return {
            "source": "Open-Meteo Forecast (data terkini)",
            "mean": float(np.nanmean(vals)),
            "max": float(np.nanmax(vals)),
            "n_days": len(vals),
            "period": f"{past_days} hari terakhir",
        }

    def _fetch_nasa_power(lat, lon, start_date, end_date):
        # NASA POWER: global, gratis, tanpa API key. Format tanggal YYYYMMDD.
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=PRECTOTCORR&community=AG"
            f"&longitude={lon}&latitude={lat}"
            f"&start={sd}&end={ed}&format=JSON"
        )
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        series = data["properties"]["parameter"]["PRECTOTCORR"]
        # NASA POWER pakai -999 sebagai flag "no data"
        vals = [v for v in series.values() if v is not None and v > -900]
        if not vals:
            raise ValueError("Tidak ada data presipitasi valid dari NASA POWER.")
        return {
            "source": "NASA POWER (satelit/model global)",
            "mean": float(np.nanmean(vals)),
            "max": float(np.nanmax(vals)),
            "n_days": len(vals),
            "period": f"{start_date} s/d {end_date}",
        }

    _RAIN_SOURCES = {
        "Otomatis (coba semua sumber berurutan)": None,
        "Open-Meteo Archive (ERA5 Reanalysis)": _fetch_open_meteo_archive,
        "NASA POWER": _fetch_nasa_power,
        "Open-Meteo Forecast (data terkini, cadangan)": _fetch_open_meteo_forecast,
    }

    def get_online_rainfall_v2(lat, lon, start_date, end_date, source_choice):
        """Return (result_dict_or_None, list_of_error_messages)."""
        errors = []

        def _try(fn, label):
            try:
                if fn is _fetch_open_meteo_forecast:
                    return fn(lat, lon), None
                return fn(lat, lon, start_date, end_date), None
            except requests.exceptions.Timeout:
                return None, f"{label}: timeout (server tidak merespons)."
            except requests.exceptions.RequestException as e:
                return None, f"{label}: gagal koneksi ({e})."
            except (KeyError, ValueError) as e:
                return None, f"{label}: format data tidak sesuai ({e})."
            except Exception as e:
                return None, f"{label}: error tak terduga ({e})."

        if source_choice == "Otomatis (coba semua sumber berurutan)":
            for label, fn in _RAIN_SOURCES.items():
                if fn is None:
                    continue
                result, err = _try(fn, label)
                if result is not None:
                    return result, errors
                errors.append(err)
            return None, errors
        else:
            fn = _RAIN_SOURCES[source_choice]
            result, err = _try(fn, source_choice)
            if err:
                errors.append(err)
            return result, errors

    # ================= SOURCE =================
    rain_factor = st.slider(
        "Extreme Rainfall Factor",
        1.0,
        5.0,
        1.0,
        0.5
    )

    st.subheader("Sumber Data Hujan Online")

    rain_source_choice = st.selectbox(
        "Pilih sumber data hujan",
        list(_RAIN_SOURCES.keys()),
        help=(
            "'Otomatis' akan mencoba Open-Meteo Archive dulu, lalu NASA POWER, lalu Open-Meteo "
            "Forecast, sampai salah satu berhasil. Pilih manual kalau Anda ingin sumber tertentu "
            "(mis. NASA POWER untuk lokasi terpencil yang cakupan ERA5-nya kurang baik)."
        )
    )

    rain_stat_choice = st.radio(
        "Statistik hujan yang dipakai untuk analisis",
        ["Rata-rata harian (kondisi umum)", "Hujan harian maksimum (kejadian kritis/ekstrem)"],
        horizontal=True,
        help="Untuk analisis erosi/sedimentasi, hujan harian MAKSIMUM dalam periode biasanya lebih "
             "relevan sebagai skenario kritis dibanding rata-rata."
    )

    latitude = st.number_input(
        "Latitude",
        value=-2.28
    )

    longitude = st.number_input(
        "Longitude",
        value=115.41
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        rain_start_date = st.date_input(
            "Tanggal mulai", value=date.today() - timedelta(days=30)
        )
    with col_d2:
        rain_end_date = st.date_input(
            "Tanggal akhir", value=date.today() - timedelta(days=1)
        )

    if st.button("Ambil Data Hujan Online"):

        with st.spinner("Mengambil data hujan dari sumber online..."):
            result, errors = get_online_rainfall_v2(
                latitude, longitude,
                str(rain_start_date), str(rain_end_date),
                rain_source_choice
            )

        if result is not None:
            chosen_val = (
                result["mean"]
                if rain_stat_choice.startswith("Rata-rata")
                else result["max"]
            )
            st.session_state["online_rainfall"] = chosen_val
            st.session_state["online_rainfall_meta"] = result
        else:
            st.session_state["online_rainfall"] = None
            st.session_state["online_rainfall_meta"] = None
            st.session_state["online_rainfall_errors"] = errors

    if st.session_state.get("online_rainfall") is not None:

        meta = st.session_state.get("online_rainfall_meta", {})
        st.success(
            f"Hujan terpakai: **{st.session_state['online_rainfall']:.2f} mm/hari** "
            f"({rain_stat_choice.split(' (')[0]})"
        )
        st.caption(
            f"Sumber: {meta.get('source', '-')} | Periode: {meta.get('period', '-')} | "
            f"Hari data valid: {meta.get('n_days', '-')}"
        )

    else:
        st.error("Gagal mengambil data hujan online dari semua sumber yang dicoba.")
        for err in st.session_state.get("online_rainfall_errors", []):
            st.caption(f"• {err}")
        st.info(
            "Anda tetap bisa lanjut dengan memasukkan nilai hujan desain secara manual di parameter "
            "segmen (mis. dari data BMKG lokal / stasiun pos hujan setempat)."
        )

    manual_rainfall_override = st.number_input(
        "Atau masukkan hujan desain manual (mm/hari) — mengosongkan/0 berarti pakai hasil online",
        min_value=0.0, value=0.0, step=1.0,
        help="Isi ini kalau Anda punya data BMKG/stasiun lokal yang lebih akurat untuk DAS ini, "
             "atau kalau semua sumber online gagal."
    )
    if manual_rainfall_override > 0:
        st.session_state["online_rainfall"] = manual_rainfall_override
        st.session_state["online_rainfall_meta"] = {
            "source": "Input manual pengguna", "period": "-", "n_days": "-"
        }
        st.caption(f"Menggunakan nilai manual: {manual_rainfall_override:.2f} mm/hari")

    # ================= UTIL =================
    def save_uploaded_dxf(uploaded_file):
        # seek(0) WAJIB di sini: sejak sub-segmen bisa memakai ulang objek file yang
        # sama dengan Segmen 1 (Main) dalam satu run yang sama, tanpa reset posisi
        # pointer, pembacaan kedua akan mengembalikan bytes kosong (karena posisi
        # sudah di EOF setelah pembacaan pertama) -- menyebabkan sub-segmen gagal
        # parse DXF walau file-nya valid.
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.read())
            return tmp.name

    def _save_uploaded_file_keep_ext(uploaded_file):
        """Sama seperti save_uploaded_dxf tapi mempertahankan ekstensi asli file
        (penting untuk raster seperti .ecw/.tif/.jp2 -- beberapa driver GDAL
        mengandalkan ekstensi, bukan cuma konten file, untuk memilih driver yang tepat)."""
        uploaded_file.seek(0)
        orig_name = getattr(uploaded_file, "name", "") or ""
        ext = os.path.splitext(orig_name)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
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

    import re as _re_contour

    def _layer_elevation_hint(layer_name):
        """Fallback: cari angka elevasi dari nama layer, mis. 'C-TOPO-MAJR-100' -> 100.
        Hanya dipakai kalau entity punya Z=0 semua (indikasi DXF 2D-only per-layer-elevation)."""
        matches = _re_contour.findall(r"-?\d+(?:\.\d+)?", layer_name or "")
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

    def _flatten_entity(e, elevation_hints, contours, entities_used, entities_skipped):
        """Ekstrak titik 3D dari satu entity DXF. Dipanggil rekursif untuk INSERT (block)."""
        etype = e.dxftype()

        try:
            if etype == "POLYLINE" and e.is_3d_polyline:
                pts = [(v.dxf.location.x, v.dxf.location.y, v.dxf.location.z)
                       for v in e.vertices]
                if pts:
                    contours.append(pts)
                    entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "LWPOLYLINE":
                elev = e.dxf.elevation or 0.0
                pts_2d = list(e.get_points())
                if elev == 0.0:
                    hint = _layer_elevation_hint(e.dxf.layer)
                    if hint is not None:
                        elev = hint
                        elevation_hints.append((e.dxf.layer, hint))
                pts = [(p[0], p[1], elev) for p in pts_2d]
                if pts:
                    contours.append(pts)
                    entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "POLYLINE" and not e.is_3d_polyline:
                # polyline 2D dengan elevation attribute (jarang tapi ada)
                elev = getattr(e.dxf, "elevation", 0.0) or 0.0
                pts = [(v.dxf.location.x, v.dxf.location.y, elev) for v in e.vertices]
                if pts:
                    contours.append(pts)
                    entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "SPLINE":
                # flatten spline jadi polyline approx
                pts = [(p.x, p.y, p.z) for p in e.flattening(0.05)]
                if pts:
                    contours.append(pts)
                    entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "3DFACE":
                pts = [tuple(e.dxf.get(f"vtx{i}")) for i in range(4)
                       if e.dxf.hasattr(f"vtx{i}")]
                if len(pts) >= 3:
                    contours.append(pts)
                    entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "LINE":
                # segmen garis pendek (mis. tie-line tepi parit/tanggul, penghubung
                # antar cross-section) -- sebelumnya TIDAK ditangani sama sekali,
                # jadi diam-diam dibuang & bikin sebagian fitur channel bolong/tidak
                # terhubung di surface 3D.
                s, en = e.dxf.start, e.dxf.end
                pts = [(s.x, s.y, s.z), (en.x, en.y, en.z)]
                contours.append(pts)
                entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "POINT":
                loc = e.dxf.location
                contours.append([(loc.x, loc.y, loc.z)])
                entities_used[etype] = entities_used.get(etype, 0) + 1

            elif etype == "INSERT":
                # block reference: masuk ke entities di dalamnya (survey kadang taruh titik/kontur dalam block)
                entities_used["INSERT(expanded)"] = entities_used.get("INSERT(expanded)", 0) + 1
                for sub_e in e.virtual_entities():
                    _flatten_entity(sub_e, elevation_hints, contours, entities_used, entities_skipped)

            else:
                entities_skipped[etype] = entities_skipped.get(etype, 0) + 1

        except Exception:
            entities_skipped[etype] = entities_skipped.get(etype, 0) + 1

    def _densify_contour(pts, max_seg_len):
        """Sisipkan titik tambahan di sepanjang segmen polyline yang panjangnya melebihi
        max_seg_len (interpolasi linear X,Y,Z di antara dua vertex asli). DXF sering hanya
        punya vertex di titik belok (mis. garis kontur lurus panjang cuma 2 vertex ujung-
        ujung), sehingga area DI ANTARA vertex-vertex itu tidak punya titik pendukung sama
        sekali untuk alpha-shape/interpolasi -- itulah salah satu penyebab permukaan 3D bisa
        'kosong'/berlubang di suatu area walau garis DXF-nya sebenarnya melewati situ. Dengan
        densifikasi ini, garis (bukan cuma vertex-nya) yang ikut dimodelkan."""
        if len(pts) < 2 or not max_seg_len or max_seg_len <= 0:
            return pts
        out = [pts[0]]
        for i in range(1, len(pts)):
            x0, y0, z0 = pts[i - 1]
            x1, y1, z1 = pts[i]
            seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if seg_len > max_seg_len:
                n_extra = int(seg_len // max_seg_len)
                for k in range(1, n_extra + 1):
                    t = k / (n_extra + 1)
                    out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
            out.append(pts[i])
        return out

    def _load_orthophoto(uploaded_ortho):
        """Baca orthophoto georeferenced (ECW/GeoTIFF/JP2) via rasterio dan kembalikan
        array RGB (uint8, HxWx3) + extent (xmin,xmax,ymin,ymax) dalam koordinat aslinya.

        CATATAN PENTING soal ECW: format ECW (Erdas Compressed Wavelet) itu proprietary
        -- driver-nya butuh ECW/JP2 SDK dari Hexagon yang TIDAK open-source dan TIDAK
        otomatis ikut ter-install di GDAL versi standar/pip (apalagi di server cloud
        Linux). Kalau GDAL di server ini tidak dikompilasi dengan driver ECW, upload
        .ecw akan gagal dengan pesan error yang jelas di bawah (bukan silent fail) --
        solusinya convert dulu ke GeoTIFF (.tif) via QGIS/Global Mapper/ArcGIS lalu
        upload ulang sebagai .tif (didukung penuh tanpa dependency tambahan).

        Tidak melakukan reprojection apa pun -- orthophoto diasumsikan sudah dalam
        sistem koordinat yang sama dengan DXF (mis. UTM zona yang sama)."""
        try:
            import rasterio
        except ImportError:
            st.error(
                "Library 'rasterio' belum terinstall di server ini. Tambahkan "
                "'rasterio' ke requirements.txt untuk mengaktifkan fitur orthophoto."
            )
            return None

        tmp_path = _save_uploaded_file_keep_ext(uploaded_ortho)

        try:
            with rasterio.open(tmp_path) as src:
                bounds = src.bounds
                extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

                n_bands = min(src.count, 3)
                bands = src.read(list(range(1, n_bands + 1)))  # (bands, rows, cols)

                # normalisasi ke uint8 0-255 per band (tahan untuk data 16-bit/float juga)
                rgb = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
                for b in range(n_bands):
                    band = bands[b].astype(np.float64)
                    finite = band[np.isfinite(band)]
                    if finite.size == 0:
                        continue
                    lo, hi = np.nanpercentile(finite, [1, 99])
                    if hi <= lo:
                        hi = lo + 1.0
                    band_norm = np.clip((band - lo) / (hi - lo), 0, 1)
                    rgb[:, :, b] = (band_norm * 255).astype(np.uint8)
                if n_bands == 1:
                    rgb[:, :, 1] = rgb[:, :, 0]
                    rgb[:, :, 2] = rgb[:, :, 0]

                return {"rgb": rgb, "extent": extent}

        except Exception as e:
            err_txt = str(e).lower()
            if "ecw" in err_txt or "not recognized" in err_txt or "no driver" in err_txt:
                st.error(
                    "Gagal membaca file ECW — kemungkinan besar GDAL di server ini TIDAK "
                    "dikompilasi dengan driver ECW (SDK proprietary Hexagon, jarang tersedia "
                    "di server cloud Linux standar). Solusi: convert file .ecw ke GeoTIFF (.tif) "
                    "menggunakan QGIS (Raster > Convert > Translate) atau Global Mapper/ArcGIS, "
                    "lalu upload ulang sebagai .tif — proses analisis lain di aplikasi ini TIDAK "
                    "terpengaruh, fitur ini murni opsional."
                )
            else:
                st.error(f"Gagal membaca orthophoto: {e}")
            return None

    def _plot_dxf_overlay_2d(ax, contours_list, color="black", linewidth=0.5, alpha=0.9):
        """Gambar seluruh polyline/kontur DXF (proyeksi X,Y saja) di atas axes matplotlib
        yang sudah ada -- dipakai untuk overlay DXF di atas peta orthophoto + rainbow risk."""
        for c in contours_list:
            if len(c) < 2:
                continue
            xs = [p[0] for p in c]
            ys = [p[1] for p in c]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=5)

    def read_contours(uploaded_dxf, return_diagnostics=False):
        """Baca kontur dari DXF dengan cakupan entity lebih luas: POLYLINE 3D, LWPOLYLINE,
        SPLINE, 3DFACE, POINT, dan entity di dalam block INSERT. Mengembalikan diagnostics
        (jumlah titik, entity yang terpakai/terlewat, deteksi Z rata/blunder) agar pengguna
        bisa memverifikasi kontur terbaca dengan benar sebelum surface dibangun."""

        path = save_uploaded_dxf(uploaded_dxf)
        doc = ezdxf.readfile(path)

        contours = []
        elevation_hints = []
        entities_used = {}
        entities_skipped = {}

        for e in doc.modelspace():
            _flatten_entity(e, elevation_hints, contours, entities_used, entities_skipped)

        # ---- dedup titik persis sama (x,y,z) untuk hindari triangulasi degenerate ----
        cleaned_contours = []
        seen_global = set()
        n_raw = 0
        n_dup = 0
        for c in contours:
            new_c = []
            for pt in c:
                n_raw += 1
                key = (round(pt[0], 4), round(pt[1], 4), round(pt[2], 4))
                if key in seen_global:
                    n_dup += 1
                    continue
                seen_global.add(key)
                new_c.append(pt)
            if new_c:
                cleaned_contours.append(new_c)

        # ---- deteksi blunder: titik (x,y) sama tapi z beda jauh (indikasi error digitasi) ----
        xy_to_z = {}
        blunders = []
        for c in cleaned_contours:
            for x, y, z in c:
                key = (round(x, 2), round(y, 2))
                if key in xy_to_z and abs(xy_to_z[key] - z) > 0.01:
                    blunders.append((x, y, xy_to_z[key], z))
                else:
                    xy_to_z[key] = z

        # ---- densifikasi: sisipkan titik di sepanjang segmen polyline yang panjang ----
        # (bukan cuma andalkan vertex) -- lihat docstring _densify_contour di atas.
        # Ambang panjang segmen dihitung adaptif dari median panjang segmen ASLI data
        # ini sendiri (bukan angka fixed), supaya menyesuaikan skala/kerapatan tiap DXF.
        _seg_lens = []
        for c in cleaned_contours:
            for i in range(1, len(c)):
                x0, y0, _ = c[i - 1]
                x1, y1, _ = c[i]
                _seg_lens.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        _median_seg_len = float(np.median(_seg_lens)) if _seg_lens else 0.0
        # segmen yang lebih dari ~2.5x median dianggap "jarang vertex" & perlu disisipi.
        _densify_max_len = _median_seg_len * 2.5 if _median_seg_len > 0 else 0.0

        n_points_before_densify = sum(len(c) for c in cleaned_contours)
        if _densify_max_len > 0:
            cleaned_contours = [_densify_contour(c, _densify_max_len) for c in cleaned_contours]
        n_points_added_densify = sum(len(c) for c in cleaned_contours) - n_points_before_densify

        diagnostics = {
            "n_entities_used": entities_used,
            "n_entities_skipped": entities_skipped,
            "n_raw_points": n_raw,
            "n_duplicate_points": n_dup,
            "n_layer_elevation_fallback": len(elevation_hints),
            "layer_elevation_hints": elevation_hints[:20],
            "n_blunder_points": len(blunders),
            "blunder_sample": blunders[:10],
            "n_points_added_densify": n_points_added_densify,
            "densify_max_seg_len": _densify_max_len,
            "all_z_zero": all(
                abs(pt[2]) < 1e-9 for c in cleaned_contours for pt in c
            ) if cleaned_contours else True,
        }

        if return_diagnostics:
            return cleaned_contours, diagnostics
        return cleaned_contours

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
            d_mm,
            flow_depth=0.3
    ):
        """PERBAIKAN: rumus sebelumnya tau = rho_water*g*slope KEHILANGAN suku
        kedalaman aliran (flow_depth) sehingga secara dimensi SALAH (hasilnya
        bersatuan Pa/m, bukan Pa) dan bergantung penuh pada asumsi h yang
        tidak eksplisit. Formula tegangan geser dasar yang benar (depth-slope
        product, standar utk saluran terbuka relatif lebar):
            tau = rho_water * g * h * S
        """

        g = 9.81

        d = d_mm / 1000

        tau = (
            rho_water *
            g *
            flow_depth *
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
    # ================= HIDROLOGI & HIDROLIKA (Rational Method + Manning's Equation) =================
    def mononobe_intensity(r24_mm, t_hours):
        """Rumus Mononobe — estimasi intensitas hujan durasi-t dari hujan 24 jam (R24).
        Umum dipakai di praktik hidrologi Indonesia saat kurva IDF stasiun tidak tersedia.
        I(t) = (R24/24) * (24/t)^(2/3)   [mm/jam]
        """
        t_hours = max(t_hours, 1/60)
        return (r24_mm / 24.0) * (24.0 / t_hours) ** (2.0 / 3.0)

    def rational_method_q(c_runoff, intensity_mm_per_hr, area_km2):
        """Metode Rasional (SI): Q = C * I * A / 3.6   [m3/s], I dalam mm/jam, A dalam km2."""
        return c_runoff * intensity_mm_per_hr * area_km2 / 3.6

    def trapezoidal_geometry(h, b, z):
        A = b * h + z * h**2
        P = b + 2 * h * np.sqrt(1 + z**2)
        T = b + 2 * z * h
        R = A / P if P > 0 else 0.0
        return A, P, R, T

    def manning_velocity(R, n, S):
        S = max(S, 1e-6)
        return (1.0 / n) * (R ** (2.0/3.0)) * (S ** 0.5)

    def solve_manning_normal_depth(q_design, b, z, n, S, h_max=20.0):
        """Cari kedalaman normal h yang memenuhi Q(h) = q_design pada channel trapezoidal
        via Manning's Equation, memakai bisection (scipy.optimize.brentq)."""

        def f(h):
            if h <= 1e-6:
                return -q_design
            A, P, R, T = trapezoidal_geometry(h, b, z)
            V = manning_velocity(R, n, S)
            return A * V - q_design

        if q_design <= 0:
            return 0.0

        try:
            if f(h_max) < 0:
                return h_max
            h_n = brentq(f, 1e-5, h_max, xtol=1e-4)
            return h_n
        except Exception:
            return None

    def compute_segment_hydraulics(
        r24_mm, c_runoff, tc_hours, area_ha,
        manning_n, channel_b, channel_z, channel_h_total,
        representative_slope
    ):
        """Rangkaian lengkap: R24 -> intensitas (Mononobe) -> Q rencana (Rational Method)
        -> kedalaman & kecepatan normal (Manning's Equation) -> cek freeboard & Froude."""

        area_km2 = area_ha / 100.0
        intensity = mononobe_intensity(r24_mm, tc_hours)
        q_design = rational_method_q(c_runoff, intensity, area_km2)

        h_normal = solve_manning_normal_depth(
            q_design, channel_b, channel_z, manning_n, representative_slope
        )

        if h_normal is None:
            return None

        A, P, R, T = trapezoidal_geometry(h_normal, channel_b, channel_z)
        v_normal = manning_velocity(R, manning_n, representative_slope)

        hydraulic_depth = A / T if T > 0 else h_normal
        froude = v_normal / np.sqrt(9.81 * hydraulic_depth) if hydraulic_depth > 0 else None
        flow_regime = "Superkritis (Fr > 1)" if (froude and froude > 1) else "Subkritis (Fr < 1)"

        freeboard = channel_h_total - h_normal
        min_freeboard = max(0.3, 0.2 * h_normal)
        freeboard_ok = freeboard >= min_freeboard

        return {
            "intensity_mm_hr": intensity,
            "q_design_m3s": q_design,
            "h_normal_m": h_normal,
            "v_normal_ms": v_normal,
            "froude": froude,
            "flow_regime": flow_regime,
            "freeboard_m": freeboard,
            "min_freeboard_m": min_freeboard,
            "freeboard_ok": freeboard_ok,
            "top_width_m": T,
            "wetted_area_m2": A
        }

    def compute_d8_receivers(grid_z, dx, dy, inside):
        """Tentukan sel tujuan aliran (receiver) untuk tiap sel via steepest-descent D8
        (8 tetangga). Return (receiver_row, receiver_col, has_receiver) — has_receiver
        False berarti sel adalah local sink/pit atau di tepi boundary (tidak ada tetangga
        yang lebih rendah), sehingga aliran berhenti/keluar di situ."""
        ny, nx = grid_z.shape
        diag = float(np.hypot(dx, dy))
        neighbor_offsets = [
            (-1, -1, diag), (-1, 0, dy), (-1, 1, diag),
            (0, -1, dx), (0, 1, dx),
            (1, -1, diag), (1, 0, dy), (1, 1, diag),
        ]
        rows, cols = np.indices(grid_z.shape)
        best_slope = np.full(grid_z.shape, 0.0)
        receiver_r = np.full(grid_z.shape, -1, dtype=int)
        receiver_c = np.full(grid_z.shape, -1, dtype=int)
        finite_z = np.isfinite(grid_z) & inside

        for dr, dc, dist in neighbor_offsets:
            rr = np.clip(rows + dr, 0, ny - 1)
            cc = np.clip(cols + dc, 0, nx - 1)
            in_range = (rows + dr >= 0) & (rows + dr < ny) & (cols + dc >= 0) & (cols + dc < nx)
            z_neighbor = grid_z[rr, cc]
            candidate = in_range & finite_z & inside[rr, cc] & np.isfinite(z_neighbor)
            slope_to_neighbor = np.where(candidate, (grid_z - z_neighbor) / dist, -np.inf)
            better = candidate & (slope_to_neighbor > best_slope)
            best_slope = np.where(better, slope_to_neighbor, best_slope)
            receiver_r = np.where(better, rr, receiver_r)
            receiver_c = np.where(better, cc, receiver_c)

        has_receiver = (receiver_r >= 0)
        return receiver_r, receiver_c, has_receiver

    def compute_flow_accumulation_d8(grid_z, inside, dx, dy):
        """D8 flow accumulation asli: untuk setiap sel, hitung jumlah sel hulu (dalam satuan
        jumlah sel) yang alirannya melewati sel tersebut, dengan memproses seluruh sel valid
        terurut dari elevasi TERTINGGI ke TERENDAH (topological order, aman dari siklus karena
        receiver selalu lebih rendah dari sender). Ini menggantikan pendekatan lama yang hanya
        menelusuri ~20 garis dari 1 titik dan menghasilkan flow_count nol di mode Hujan (Uniform)."""
        ny, nx = grid_z.shape
        recv_r, recv_c, has_recv = compute_d8_receivers(grid_z, dx, dy, inside)

        recv_flat = np.where(has_recv, recv_r * nx + recv_c, -1).ravel()
        z_flat = grid_z.ravel()
        inside_flat = inside.ravel()

        order = np.argsort(-np.where(inside_flat, z_flat, -np.inf))
        acc = np.where(inside_flat, 1.0, 0.0)

        for i in order:
            r = recv_flat[i]
            if r >= 0:
                acc[r] += acc[i]

        return acc.reshape(ny, nx)

    def _find_downhill_escape(x, y, grid_x, grid_y, grid_z, inside, max_radius_cells=None):
        """Kalau steepest-descent 'macet' (gradien lokal ~0), cari sel terdekat
        dengan elevasi lebih rendah dalam radius yang MELEBAR bertahap, lalu
        arahkan aliran ke sana -- teknik standar 'flat area resolution' pada
        algoritma routing D8 (mis. Garbrecht & Martz, 1997) untuk kasus di mana
        permukaan hasil interpolasi (griddata linear + nearest-fill utk lubang)
        punya bidang datar/plateau lokal yang BUKAN sink asli, cuma artefak
        interpolasi -- gradiennya nol persis di situ padahal beberapa sel di
        sekitarnya (di luar plateau) sebenarnya masih lebih rendah & channel-nya
        masih berlanjut. Tanpa ini, trace berhenti prematur di tengah channel
        walau boundary & data DXF-nya masih lanjut jauh lebih ke hilir.

        REVISI: radius pencarian sebelumnya di-hardcap 40 sel -- kalau plateau/
        area datar lebih besar dari itu (umum terjadi pada pembelokan channel
        dengan data kontur jarang), escape gagal ditemukan dan aliran tetap
        berhenti walau boundary masih jauh, persis seperti dilaporkan (aliran
        stuck di dekat pembelokan kecil). Sekarang radius default melebar
        sampai MENCAKUP SELURUH GRID (max(ny, nx)), jadi selama masih ada
        satu saja sel bervalue lebih rendah di dalam boundary manapun posisinya,
        pasti ketemu -- baru dianggap sink asli kalau benar2 tidak ada sel lebih
        rendah sama sekali di seluruh grid dari titik itu.

        Return (new_x, new_y) titik terendah yang ditemukan, atau None kalau
        benar-benar tidak ada sel lebih rendah dalam radius maksimum (baru saat
        itu dianggap sink asli / ujung data)."""
        ix = np.abs(grid_x[:, 0] - x).argmin()
        iy = np.abs(grid_y[0, :] - y).argmin()
        z_here = grid_z[ix, iy]
        ny_max, nx_max = grid_z.shape

        if max_radius_cells is None:
            max_radius_cells = max(ny_max, nx_max)

        def _search(require_inside):
            for radius in range(2, max_radius_cells + 1, 2):
                r0, r1 = max(0, ix - radius), min(ny_max, ix + radius + 1)
                c0, c1 = max(0, iy - radius), min(nx_max, iy + radius + 1)
                sub_z = grid_z[r0:r1, c0:c1]
                valid = ~np.isnan(sub_z)
                if require_inside:
                    valid = valid & inside[r0:r1, c0:c1]
                candidate = valid & (sub_z < z_here - 1e-6)
                if candidate.any():
                    rel_r, rel_c = np.where(candidate)
                    zs = sub_z[rel_r, rel_c]
                    best = np.argmin(zs)
                    best_r = r0 + rel_r[best]
                    best_c = c0 + rel_c[best]
                    return float(grid_x[best_r, 0]), float(grid_y[0, best_c])
            return None

        # Pass 1 (ketat): hanya sel yang ditandai "inside" boundary poligon.
        result = _search(require_inside=True)
        if result is not None:
            return result

        # PERBAIKAN ("aliran stop padahal jalurnya masih landai/hampir flat"):
        # boundary hasil auto alpha-shape (atau boundary manual dgn digitasi
        # kurang presisi di area landai/bench/berm) kadang punya artefak lokal
        # (pinch/lubang kecil) yang membuat mask `inside` KELIRU menandai sel
        # tetangga yang sebenarnya masih bagian sah data DXF (grid_z-nya valid,
        # bukan NaN hasil interpolasi) sebagai "outside" -- terutama di area
        # yang topografinya sangat landai sehingga gradiennya sering menyentuh
        # ~0, jadi rutin butuh escape/flat-area-resolution persis di titik yang
        # kebetulan dekat artefak tsb. Kalau pencarian ketat (pass 1) gagal,
        # coba sekali lagi hanya mensyaratkan data valid (bukan NaN) TANPA
        # peduli mask `inside` -- lebih longgar, tapi tetap aman: langkah
        # berikutnya di trace_multiple_flow tetap dicek ulang terhadap boundary
        # (dgn buffer toleransi), jadi begitu memang sudah keluar area kajian
        # sungguhan, trace akan berhenti di situ juga, bukan gagal di sini.
        return _search(require_inside=False)

    def trace_multiple_flow(x0, y0, grid_x, grid_y, dz_dx, dz_dy, grid_z, boundary,
                             n_stream=20, inside=None, depth0=0.2, flow_acc=None):
        """Trace multiple steepest-descent streamlines dari titik (x0,y0).

        PERBAIKAN vs versi sebelumnya: sebelumnya z tiap titik path diambil dari
        indeks grid SEBELUM posisi berpindah (off-by-one), sehingga path_z
        memiliki 1 elemen lebih sedikit dari path_x/path_y dan elevasi yang
        ditampilkan tidak sinkron dengan posisi (x,y) sebenarnya — inilah salah
        satu penyebab tampilan streamline 3D terlihat 'melayang'/tidak menempel
        pas ke permukaan aktual. Sekarang z diambil pada posisi SETELAH pindah,
        dan titik awal (x0,y0,z0) turut disertakan sehingga x,y,z selalu
        panjangnya sama & konsisten satu sama lain.

        PERBAIKAN LEBIH LANJUT: sebelumnya iterasi dibatasi fixed range(800) --
        dengan step 0.5 satuan itu setara jarak tempuh maksimum ~400 satuan,
        yang BISA lebih pendek dari luas area kajian sebenarnya (mis. boundary
        berukuran 1-2 km), sehingga aliran berhenti karena kehabisan jatah
        iterasi, BUKAN karena benar-benar mencapai tepi boundary/sink. Sekarang
        jumlah iterasi maksimum dihitung ADAPTIF dari diagonal bounding-box
        boundary (dikali faktor aman 6x untuk mengakomodasi jalur yang berkelok/
        tidak lurus), supaya aliran betul-betul disimulasikan sampai keluar
        boundary atau mencapai titik tanpa gradien turun (sink/local minimum),
        bukan berhenti prematur oleh limit sembarang. Tetap ada hard-cap supaya
        tidak infinite-loop kalau ada kasus aneh (mis. boundary sangat kecil).

        REVISI TERBARU (fix "aliran stuck padahal cuma pembelokan kecil"):
        sebelumnya, begitu gradien lokal nol DAN _find_downhill_escape() gagal
        menemukan sel lebih rendah dalam radius terbatas (40 sel), aliran
        LANGSUNG dianggap sink & berhenti. Radius _find_downhill_escape()
        sekarang sudah dilebarkan sampai mencakup seluruh grid (lihat fungsi
        tsb), tapi sebagai lapis pengaman tambahan: kalau toh escape masih
        gagal (mis. titik itu benar2 titik terendah di seluruh grid yg masih
        tersambung), aliran TIDAK langsung berhenti selama masih punya arah
        gerak terakhir yang valid (momentum) -- aliran didorong terus memakai
        arah terakhir tsb sampai maksimum `max_momentum_steps` langkah, dengan
        harapan keluar dari plateau/artefak lokal dan kembali menemukan gradien
        turun yang sebenarnya atau mencapai boundary. Ini meniru perilaku air
        asli yang tetap punya inersia/momentum melewati bagian datar pendek.
        Baru kalau momentum juga habis tanpa progres, aliran dianggap benar2
        mencapai sink asli / dasar cekungan tertutup.

        Selain X,Y,Z, sekarang juga mengembalikan `path_depth` (ketebalan air
        perkiraan di tiap titik path) -- dimulai dari `depth0` (kedalaman air
        awal yang diisi user) dan membesar seiring bertambahnya flow_acc lokal
        (proxy jumlah tangkapan air di sel tsb, kalau `flow_acc` grid disediakan)
        supaya representasi 3D terlihat makin tebal ke arah hilir seperti aliran
        air sungai sungguhan, bukan garis tipis konstan.
        """

        step_size = 0.5
        minx, miny, maxx, maxy = boundary.bounds
        diag = float(np.hypot(maxx - minx, maxy - miny))
        max_iter = int(np.clip((diag / step_size) * 6, 800, 200000))
        max_momentum_steps = 400

        if inside is None:
            inside = ~np.isnan(grid_z)

        # PERBAIKAN ("aliran masih nyangkut/berhenti padahal mesh 3D-nya --
        # area hijau yang sama -- masih terlihat lanjut jauh melewati titik
        # itu"): sebelumnya kondisi berhenti trace memakai uji Shapely
        # `boundary.buffer(...).contains(Point)` TERPISAH dari mask `inside`
        # (raster hasil `vectorized.contains(boundary, grid_x, grid_y)`) yang
        # justru menentukan bentuk mesh 3D yang dirender (area hijau yang
        # dilihat user). Dua uji containment berbeda pada geometry yang sama
        # (satu Shapely scalar Point-in-polygon pada boundary yang dibuffer,
        # satu lagi raster grid) bisa TIDAK KONSISTEN persis di tepi/sudut
        # boundary yang rumit (hasil alpha-shape/digitasi manual) -- trace
        # berhenti duluan padahal sel gridnya sendiri masih ditandai "inside"
        # & dirender sebagai bagian mesh yang valid.
        #
        # Sekarang kondisi berhenti trace memakai PERSIS raster `inside` yang
        # sama dgn yang dipakai untuk mesh (look-up sel grid terdekat, bukan
        # uji polygon terpisah), supaya "kalau masih dirender sebagai area
        # valid, aliran juga masih boleh melewatinya" -- otomatis konsisten.
        # Raster ini didilasi beberapa sel (toleransi tepi, meniru cell buffer
        # sebelumnya tapi sedikit lebih longgar) supaya artefak tepi
        # sub-piksel (pinch/lubang kecil dari alpha-shape) tidak lagi memutus
        # trace prematur, sesuai laporan aliran masih "nyangkut" walau tepat
        # di sebelahnya area hijau masih lanjut.
        _inside_tol = binary_dilation(inside, iterations=3)
        _ny_max_chk, _nx_max_chk = grid_z.shape

        def _still_inside(px, py):
            _ixc = np.abs(grid_x[:, 0] - px).argmin()
            _iyc = np.abs(grid_y[0, :] - py).argmin()
            if _ixc < 0 or _ixc >= _ny_max_chk or _iyc < 0 or _iyc >= _nx_max_chk:
                return False
            return bool(_inside_tol[_ixc, _iyc])

        all_paths = []

        ix0 = np.abs(grid_x[:, 0] - x0).argmin()
        iy0 = np.abs(grid_y[0, :] - y0).argmin()
        z0 = grid_z[ix0, iy0]
        acc0 = float(flow_acc[ix0, iy0]) if flow_acc is not None else 1.0

        for angle in np.linspace(-0.5, 0.5, n_stream):

            path_x, path_y, path_z, path_depth = [x0], [y0], [z0], [depth0]
            x, y = x0, y0
            last_vx, last_vy = None, None
            stuck_streak = 0

            for _ in range(max_iter):

                ix = np.abs(grid_x[:, 0] - x).argmin()
                iy = np.abs(grid_y[0, :] - y).argmin()

                vx = -dz_dx[ix, iy]
                vy = -dz_dy[ix, iy]

                vx += angle * 0.2
                vy += angle * 0.2

                norm = np.sqrt(vx**2 + vy**2)

                if norm < 1e-6:
                    # gradien lokal nol -> coba lompati plateau/artefak interpolasi
                    # dulu sebelum menyerah (lihat catatan _find_downhill_escape)
                    escape = _find_downhill_escape(x, y, grid_x, grid_y, grid_z, inside)
                    if escape is not None:
                        ex, ey = escape
                        vx, vy = ex - x, ey - y
                        norm = np.sqrt(vx**2 + vy**2)
                        stuck_streak = 0
                    elif last_vx is not None and stuck_streak < max_momentum_steps:
                        # tidak ada sel lebih rendah SAMA SEKALI di seluruh grid dari
                        # sini -> dorong terus pakai arah gerak terakhir (momentum),
                        # jangan langsung menyerah, siapa tahu ini cuma titik pelana
                        # (saddle) sempit lalu gradien turun lagi ditemukan berikutnya
                        vx, vy = last_vx, last_vy
                        norm = np.sqrt(vx**2 + vy**2)
                        stuck_streak += 1
                    else:
                        # sudah dicoba escape + momentum ratusan langkah, tetap tidak
                        # ada progres -> baru dianggap sink asli / dasar cekungan
                        break

                    if norm < 1e-6:
                        break
                else:
                    stuck_streak = 0

                x += (vx / norm) * step_size
                y += (vy / norm) * step_size
                last_vx, last_vy = vx, vy

                if not _still_inside(x, y):
                    # mencapai tepi boundary -> kondisi berhenti yang sebenarnya diminta
                    break

                ix_new = np.abs(grid_x[:, 0] - x).argmin()
                iy_new = np.abs(grid_y[0, :] - y).argmin()
                z = grid_z[ix_new, iy_new]

                if np.isnan(z):
                    break

                path_x.append(x)
                path_y.append(y)
                path_z.append(z)

                if flow_acc is not None:
                    acc_here = float(flow_acc[ix_new, iy_new])
                    # ketebalan membesar mengikuti akar dari rasio flow accumulation
                    # (proxy debit ~ luas tangkapan) relatif titik awal -- pendekatan
                    # sederhana ala hydraulic geometry (lebar/kedalaman ~ Q^0.4-0.5)
                    growth = np.sqrt(max(acc_here, 1.0) / max(acc0, 1.0))
                    path_depth.append(float(np.clip(depth0 * growth, depth0, depth0 * 8)))
                else:
                    path_depth.append(path_depth[-1])

            all_paths.append((path_x, path_y, path_z, path_depth))

        return all_paths

    # ================= RECOMMENDATION ENGINE =================
    def generate_recommendation(
        design_type,
        seg_label,
        analysis_method,
        erosion_ratio,
        erosion_area,
        sedimentation_area,
        max_zone,
        overflow_count,
        convergence_count,
        boundary_area_ha
    ):

        notes = []
        score_bad = 0

        if erosion_ratio > 0.35:
            notes.append(
                f"Luas area berpotensi erosi {erosion_ratio*100:.1f}% dari total area — tergolong tinggi."
            )
            score_bad += 1
        elif erosion_ratio > 0.15:
            notes.append(
                f"Area berpotensi erosi {erosion_ratio*100:.1f}% dari total area — perlu perhatian."
            )

        if max_zone >= 1.8:
            notes.append(
                f"Indeks risiko maksimum {max_zone:.2f} mendekati batas ekstrem — kecepatan/tegangan "
                "geser jauh melampaui ambang kritis material."
            )
            score_bad += 1
        elif max_zone >= 1.2:
            notes.append(
                f"Indeks risiko maksimum {max_zone:.2f} berada pada zona transisi/moderat."
            )

        if overflow_count > 10:
            notes.append(
                f"Terdeteksi {overflow_count} titik overflow index tinggi — indikasi kapasitas "
                "penampang berpotensi terlampaui pada beberapa titik."
            )
            score_bad += 1

        if convergence_count > 10:
            notes.append(
                f"Terdapat {convergence_count} titik konvergensi aliran signifikan — berisiko "
                "memicu scouring lokal."
            )

        if boundary_area_ha > 0 and (sedimentation_area / boundary_area_ha) > 0.25:
            pct = sedimentation_area / boundary_area_ha * 100
            notes.append(
                f"Area berpotensi sedimentasi {sedimentation_area:.2f} Ha ({pct:.1f}% dari luas "
                "area) — cukup luas."
            )

        if design_type == "Desain Baru":

            if score_bad >= 2:
                status = "TIDAK DIREKOMENDASIKAN (REJECT)"
                color = "red"
                narrative = (
                    f"Berdasarkan geometri dan topografi hasil DXF pada {seg_label}, kombinasi "
                    "kemiringan, kecepatan aliran, dan pola konvergensi menghasilkan indikasi "
                    "risiko erosi/overflow yang tinggi apabila desain diimplementasikan tanpa "
                    "modifikasi. Lokasi dan/atau geometri saat ini perlu ditinjau ulang sebelum "
                    "konstruksi."
                )
                recs = [
                    "Redesain geometri channel (perlebar penampang / kurangi kemiringan "
                    "longitudinal) pada segmen dengan indeks risiko tertinggi.",
                    "Pertimbangkan relokasi alur pada zona konvergensi aliran signifikan.",
                    "Tambahkan lining/perkuatan (riprap, geotextile, atau beton) pada segmen "
                    "hulu sebelum konstruksi.",
                    "Lakukan survei tanah lanjutan untuk memverifikasi parameter sedimen yang digunakan."
                ]
            elif score_bad == 1:
                status = "LAYAK DENGAN CATATAN (CONDITIONAL)"
                color = "orange"
                narrative = (
                    f"Geometri desain pada {seg_label} secara umum dapat diimplementasikan, "
                    "namun terdapat indikator risiko (erosi/overflow/konvergensi) yang perlu "
                    "dimitigasi melalui detail desain tambahan agar performa jangka panjang terjaga."
                )
                recs = [
                    "Tambahkan proteksi (riprap/vegetasi) pada titik-titik dengan indeks erosi tinggi.",
                    "Evaluasi ulang dimensi penampang pada segmen dengan overflow index tinggi.",
                    "Rencanakan monitoring pasca-konstruksi pada 6-12 bulan pertama."
                ]
            else:
                status = "LAYAK DIIMPLEMENTASIKAN"
                color = "green"
                narrative = (
                    f"Geometri dan lokasi desain pada {seg_label} sesuai dengan kondisi topografi "
                    "dan parameter sedimen yang dianalisis. Risiko erosi, overflow, dan sedimentasi "
                    "berada pada level rendah-moderat sehingga desain dapat dilanjutkan ke tahap "
                    "konstruksi."
                )
                recs = [
                    "Lanjutkan ke detail desain (DED) dan verifikasi lapangan sebelum konstruksi.",
                    "Terapkan monitoring rutin pasca-konstruksi sebagai bagian dari QA/QC."
                ]

        else:

            if score_bad >= 2:
                status = "KONDISI KRITIS — PERLU PERBAIKAN SEGERA"
                color = "red"
                narrative = (
                    f"Kondisi eksisting {seg_label} menunjukkan sebaran erosi dan/atau overflow "
                    "yang signifikan. Tanpa tindakan perbaikan, degradasi channel berpotensi "
                    "berlanjut dan mengganggu fungsi drainase/pengendalian sedimen."
                )
                recs = [
                    "Prioritaskan perbaikan pada titik-titik overflow/erosi tertinggi (lihat tabel Top 10).",
                    "Lakukan pengerukan pada zona sedimentasi untuk mengembalikan kapasitas penampang.",
                    "Tambahkan check dam / sekat sedimen tambahan pada zona konvergensi aliran.",
                    "Tingkatkan frekuensi monitoring visual dan survei topografi berkala."
                ]
            elif score_bad == 1:
                status = "PERLU PERBAIKAN TERJADWAL"
                color = "orange"
                narrative = (
                    f"Kondisi eksisting {seg_label} secara umum masih berfungsi, namun terdapat "
                    "indikasi erosi/sedimentasi lokal yang bila dibiarkan dapat berkembang."
                )
                recs = [
                    "Jadwalkan perbaikan/lining lokal pada titik indeks erosi tinggi dalam 1 "
                    "siklus pemeliharaan.",
                    "Pantau perkembangan zona sedimentasi setiap periode monitoring."
                ]
            else:
                status = "KONDISI STABIL"
                color = "green"
                narrative = (
                    f"Kondisi eksisting {seg_label} menunjukkan tingkat erosi dan sedimentasi "
                    "yang relatif rendah dan masih dalam batas wajar operasional."
                )
                recs = [
                    "Lanjutkan monitoring rutin sesuai jadwal existing.",
                    "Tidak diperlukan tindakan perbaikan mendesak saat ini."
                ]

        return {
            "status": status,
            "color": color,
            "narrative": narrative,
            "recommendations": recs,
            "notes": notes
        }

    # ================= AI-BASED GEOTECHNICAL RECOMMENDATION =================
    def _get_anthropic_api_key():
        try:
            return st.secrets["anthropic_api_key"]
        except Exception:
            return None

    def generate_ai_recommendation(context, rule_based):
        """Minta narasi & rekomendasi geoteknik dari Claude (Anthropic API), berbasis
        angka HASIL RUNNING segmen ini (bukan template tetap) — tiap segmen akan
        mendapat narasi berbeda menyesuaikan kombinasi angka aktualnya.

        `rule_based` (dict dari generate_recommendation) dipakai sebagai fallback
        DAN sebagai anchor status/level supaya narasi AI tetap konsisten dengan
        klasifikasi TARP/status yang deterministik.
        """

        api_key = _get_anthropic_api_key()

        fallback = {
            "narrative": rule_based["narrative"],
            "recommendations": rule_based["recommendations"],
            "source": "rule-based (AI belum aktif — set st.secrets['anthropic_api_key'] untuk mengaktifkan narasi AI dinamis)"
        }

        if not api_key:
            return fallback

        prompt = f"""Anda adalah engineer geoteknik senior spesialis erosi, sedimentasi, dan hidrolika saluran tambang.
Berdasarkan HASIL RUNNING NUMERIK berikut (bukan asumsi), buat analisis dan rekomendasi rekayasa yang
SPESIFIK menyesuaikan angka-angka ini — jangan memberi jawaban generik yang bisa berlaku untuk kasus lain.

DATA HASIL RUNNING:
- Segmen: {context['seg_label']}
- Jenis kondisi: {context['design_type']}
- Metode analisis: {context['analysis_method']}
- Status klasifikasi (rule-based/TARP): {rule_based['status']}
- Luas area: {context['boundary_area_ha']:.2f} Ha
- Rasio area berpotensi erosi: {context['erosion_ratio']*100:.1f}%
- Luas area sedimentasi: {context['sedimentation_area']:.2f} Ha
- Indeks risiko maksimum: {context['max_zone']:.2f} (skala 0-2)
- Jumlah titik overflow signifikan: {context['overflow_count']}
- Jumlah titik konvergensi aliran: {context['convergence_count']}
- Kecepatan aliran representatif: {context['velocity_hulu']:.3f} m/s
- Grain size: {context['grain_size']:.3f} mm
- RMSE permukaan 3D vs titik DXF asli: {context.get('surface_rmse', 'N/A')}
{context.get('hydraulics_text', '')}

Tulis dalam Bahasa Indonesia, gaya laporan teknis profesional, dengan format:
1) Satu paragraf analisis (4-6 kalimat) yang secara eksplisit MERUJUK ANGKA-ANGKA di atas (bukan pernyataan umum).
2) Daftar 3-5 rekomendasi rekayasa konkret, masing-masing SATU KALIMAT, diurutkan dari prioritas tertinggi,
   dan sedapat mungkin sertakan angka/dimensi indikatif (mis. ukuran riprap, jarak check dam, target kecepatan).

Balas HANYA dalam format JSON valid seperti ini (tanpa markdown fence, tanpa teks lain):
{{"narrative": "...", "recommendations": ["...", "...", "..."]}}
"""

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            text = "".join(
                block.get("text", "") for block in data.get("content", [])
                if block.get("type") == "text"
            ).strip()
            text = text.replace("```json", "").replace("```", "").strip()

            import json as _json
            parsed = _json.loads(text)

            if not parsed.get("narrative") or not parsed.get("recommendations"):
                raise ValueError("Respons AI tidak lengkap")

            parsed["source"] = "AI (Claude) — dihasilkan dari angka hasil running segmen ini"
            return parsed

        except Exception as e:
            fallback["narrative"] += (
                f"\n\n[Catatan: narasi AI gagal diambil ({e}), menampilkan hasil rule-based sebagai fallback.]"
            )
            return fallback

    # ================= RUN =================
    # PERBAIKAN: sebelumnya gerbang blok ini HANYA `if st.button("RUN ANALYSIS"):`
    # (transient -- True cuma di rerun persis saat tombol diklik). Padahal di
    # DALAM blok ini ada interaksi lain yang men-trigger rerun Streamlit juga
    # (mis. klik titik di peta pada mode "Klik di Peta Desain" -- on_select pada
    # st.plotly_chart), dan setiap kali rerun terjadi, st.button(...) balik ke
    # False lagi -> seluruh blok (termasuk hasil 3D, peta klik itu sendiri, dsb)
    # LANGSUNG HILANG dari layar seolah "kembali ke awal", padahal analisis
    # sudah sempat dijalankan. Sekarang gerbang juga menerima flag persisten
    # st.session_state["analysis_done"] (di-set True begitu tombol diklik
    # sekali), supaya blok ini tetap tampil & re-render di rerun berikutnya --
    # termasuk saat klik-peta -- alih-alih ikut hilang mengikuti status tombol
    # yang cuma valid untuk satu rerun.
    if st.button("RUN ANALYSIS") or st.session_state.get("analysis_done", False):

        st.session_state["analysis_done"] = True
        st.session_state["segment_results"] = {}

        for seg_idx, sid in enumerate(st.session_state["segments"]):

            default_seg_name = "Segmen 1 (Main DXF)" if seg_idx == 0 else f"Segmen {seg_idx + 1}"
            seg_label = st.session_state.get(f"seg_name_{sid}", default_seg_name)

            kontur_dxf = st.session_state.get(f"kontur_dxf_{sid}")
            is_sub_segment = (seg_idx > 0)

            if is_sub_segment and not kontur_dxf:
                # Sub-boundary: pakai terrain/kontur yang SAMA dengan Segmen 1 (Main) --
                # sesuai desain baru, sub-segmen tidak lagi upload DXF kontur sendiri,
                # cukup boundary sub-area. Ambil dari segmen pertama di list.
                main_sid = st.session_state["segments"][0]
                kontur_dxf = st.session_state.get(f"kontur_dxf_{main_sid}")

            boundary_dxf = st.session_state.get(f"boundary_dxf_{sid}")
            orthophoto_file = st.session_state.get(f"orthophoto_{sid}")
            design_type = st.session_state.get(f"design_type_{sid}", "Desain Baru")
            analysis_method = st.session_state.get(f"analysis_method_{sid}", "Hjulstrom Diagram")
            velocity_hulu = st.session_state.get(f"velocity_hulu_{sid}", 0.5)
            grain_size = st.session_state.get(f"grain_size_{sid}", 0.1)
            rho_water = st.session_state.get(f"rho_water_{sid}", 1000)
            rho_soil = st.session_state.get(f"rho_soil_{sid}", 2650)
            tau_critical = st.session_state.get(f"tau_critical_{sid}", 2.0)
            erodibility_M = st.session_state.get(f"erodibility_M_{sid}", 0.05)
            flow_weight = st.session_state.get(f"flow_weight_{sid}", 3.0)
            flow_depth = st.session_state.get(f"flow_depth_{sid}", 0.30)
            if is_sub_segment:
                # Sub-segmen tidak lagi punya input "Skenario Sumber Aliran" sendiri
                # (lihat bagian input) -- WAJIB ikut source_type yang dipilih di
                # Segmen 1/Main, supaya analisisnya benar-benar tersambung dengan
                # Main. Sebelumnya key ini diam-diam jatuh ke default "Hujan
                # (Uniform)" untuk sub-segmen walau Main dipilih "Satu Titik (Point
                # Source)" -- itulah sebabnya sub-boundary terlihat seperti area
                # simulasi terpisah sendiri, tidak terhubung ke aliran dari Main.
                _main_sid_src = st.session_state["segments"][0]
                source_type = st.session_state.get(f"source_type_{_main_sid_src}", "Hujan (Uniform)")
            else:
                source_type = st.session_state.get(f"source_type_{sid}", "Hujan (Uniform)")
            point_method = st.session_state.get(f"point_method_{sid}", "Ketik Koordinat Manual")
            point_x = st.session_state.get(f"point_x_{sid}", 0.0)
            point_y = st.session_state.get(f"point_y_{sid}", 0.0)

            use_hydraulics = st.session_state.get(f"use_hydraulics_{sid}", False)
            runoff_c_label = st.session_state.get(f"runoff_c_label_{sid}", "0.70 - Tanah kompak, vegetasi jarang")
            runoff_c = (
                st.session_state.get(f"runoff_c_custom_{sid}", 0.70)
                if runoff_c_label.startswith("Custom")
                else float(runoff_c_label.split(" - ")[0])
            )
            tc_hours = st.session_state.get(f"tc_hours_{sid}", 1.0)
            manning_n_label = st.session_state.get(f"manning_n_label_{sid}", "0.030 - Saluran tanah alami")
            manning_n = (
                st.session_state.get(f"manning_n_custom_{sid}", 0.030)
                if manning_n_label.startswith("Custom")
                else float(manning_n_label.split(" - ")[0])
            )
            channel_b = st.session_state.get(f"channel_b_{sid}", 3.0)
            channel_z = st.session_state.get(f"channel_z_{sid}", 1.5)
            channel_h_total = st.session_state.get(f"channel_h_total_{sid}", 1.5)

            st.markdown("---")
            st.markdown(f"## {seg_label}")
            st.caption(f"Jenis kondisi: **{design_type}**  |  Metode: **{analysis_method}**")

            if not kontur_dxf:
                st.error(f"[{seg_label}] Upload Kontur DXF dulu — segmen ini dilewati.")
                continue

            if is_sub_segment and not boundary_dxf:
                st.error(
                    f"[{seg_label}] Ini sub-segmen (kontur sudah otomatis dari Segmen 1) — "
                    "wajib upload **Boundary DXF sub-area** supaya tahu bagian mana dari "
                    "boundary utama yang mau dianalisis. Tanpa ini, batas area akan otomatis "
                    "dibentuk dari SELURUH titik kontur (sama seperti Segmen 1) sehingga "
                    "hasilnya jadi duplikat, bukan sub-area. Segmen ini dilewati."
                )
                continue

            contours, contour_diag = read_contours(kontur_dxf, return_diagnostics=True)

            # ---- orthophoto (opsional) -- cache di session_state supaya tidak decode ulang tiap rerun ----
            orthophoto_data = None
            if orthophoto_file is not None:
                _ortho_cache_key = f"orthophoto_parsed_{sid}"
                _ortho_sig = (orthophoto_file.name, orthophoto_file.size)
                _cached = st.session_state.get(_ortho_cache_key)
                if _cached and _cached.get("sig") == _ortho_sig:
                    orthophoto_data = _cached.get("data")
                else:
                    with st.spinner(f"[{seg_label}] Membaca orthophoto ({orthophoto_file.name})..."):
                        orthophoto_data = _load_orthophoto(orthophoto_file)
                    st.session_state[_ortho_cache_key] = {"sig": _ortho_sig, "data": orthophoto_data}

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
                continue

            # ================= DIAGNOSTIK KONTUR (verifikasi sebelum surface dibangun) =================
            with st.expander(f"Diagnostik pembacaan DXF — {seg_label}", expanded=contour_diag["all_z_zero"] or contour_diag["n_blunder_points"] > 0):
                dcol1, dcol2, dcol3 = st.columns(3)
                dcol1.metric("Titik terbaca", f"{contour_diag['n_raw_points']:,}")
                dcol2.metric("Duplikat dibuang", f"{contour_diag['n_duplicate_points']:,}")
                dcol3.metric("Titik blunder (Z beda di XY sama)", f"{contour_diag['n_blunder_points']:,}")

                if contour_diag.get("n_points_added_densify", 0) > 0:
                    st.caption(
                        f"+{contour_diag['n_points_added_densify']:,} titik tambahan disisipkan otomatis "
                        f"di sepanjang segmen polyline yang panjangnya > "
                        f"{contour_diag.get('densify_max_seg_len', 0):.2f} m (garis DXF ikut dimodelkan "
                        "penuh, bukan cuma vertex-nya, supaya area di antara vertex yang jarang tidak "
                        "kosong/berlubang pada surface 3D)."
                    )

                st.write("**Entity terpakai:**", contour_diag["n_entities_used"] or "—")
                if contour_diag["n_entities_skipped"]:
                    st.write("**Entity dilewati (tipe tidak didukung):**", contour_diag["n_entities_skipped"])

                if contour_diag["all_z_zero"]:
                    st.error(
                        "Semua titik punya elevasi Z = 0. DXF ini kemungkinan kontur 2D murni "
                        "(elevasi disimpan di layer/atribut, bukan di geometri). Surface 3D TIDAK akan akurat. "
                        "Export ulang dari sumber (Civil3D/AutoCAD) sebagai 3D Polyline dengan Z asli, "
                        "atau gunakan opsi 'ambil elevasi dari nama layer' jika layer Anda mengikuti konvensi penamaan angka."
                    )
                elif contour_diag["n_layer_elevation_fallback"] > 0:
                    st.warning(
                        f"{contour_diag['n_layer_elevation_fallback']} entity elevasinya diambil dari nama layer "
                        f"(bukan dari geometri Z). Contoh: {contour_diag['layer_elevation_hints'][:5]}. "
                        "Mohon cross-check apakah ini benar sebelum melanjutkan."
                    )

                if contour_diag["n_blunder_points"] > 0:
                    st.warning(
                        f"Ditemukan {contour_diag['n_blunder_points']} titik dengan koordinat X,Y hampir sama "
                        "tetapi elevasi Z berbeda >1cm — indikasi kesalahan digitasi/duplikasi layer kontur. "
                        f"Contoh (x, y, z1, z2): {contour_diag['blunder_sample']}"
                    )

            # ================= AUTO BOUNDARY (scale-aware alpha-shape) =================

            pts = np.column_stack(
                [x_all, y_all]
            )

            def _estimate_alpha(points_xy):
                """Alpha lama (0.001) fixed dan salah-skala: untuk koordinat UTM (~500000)
                itu terlalu kecil (boundary jadi convex-hull kasar / gagal), untuk koordinat
                lokal kecil (0-100) itu terlalu besar (boundary bisa pecah/berlubang).
                Estimasi di sini berbasis jarak tetangga terdekat median (scale-aware)."""
                from scipy.spatial import cKDTree
                n = len(points_xy)
                sample = points_xy if n <= 3000 else points_xy[
                    np.random.choice(n, 3000, replace=False)
                ]
                tree = cKDTree(sample)
                dists, _ = tree.query(sample, k=2)
                nn_dist = np.median(dists[:, 1])
                if nn_dist <= 0 or not np.isfinite(nn_dist):
                    return 0.0  # fallback convex hull
                # radius alpha-shape ~ 2-3x jarak titik tetangga terdekat
                return 1.0 / (nn_dist * 2.5)

            if boundary_dxf:
                boundary = read_boundary_polygon(boundary_dxf)
                boundary = fix_geom(boundary)
            else:
                alpha_est = _estimate_alpha(pts)
                try:
                    boundary = alphashape.alphashape(pts, alpha_est)
                    if (boundary is None or boundary.is_empty
                            or not hasattr(boundary, "exterior")
                            or boundary.exterior is None):
                        raise ValueError("alpha-shape kosong/tidak valid")
                except Exception:
                    # fallback aman: convex hull (lebih kasar tapi selalu valid)
                    boundary = Polygon(pts).convex_hull
                    st.info(
                        f"[{seg_label}] Auto-boundary (alpha-shape adaptif) gagal membentuk poligon "
                        "yang valid dari titik kontur — menggunakan convex hull sebagai fallback. "
                        "Untuk batas area yang presisi mengikuti bentuk DAS asli, upload DXF boundary terpisah."
                    )
                boundary = fix_geom(boundary)

            # ---- tutup lubang kecil di boundary hasil alpha-shape/convex-hull ----
            # Alpha-shape adaptif kadang membentuk "lubang" (interior ring) di tengah
            # area kalau kerapatan titik kontur tidak merata di situ (bukan indikasi
            # ada area kosong sungguhan pada DXF). Lubang ini menyebabkan grid di
            # dalamnya ditandai "outside" (inside=False), sehingga muncul sebagai
            # bagian mesh 3D yang terputus/berlubang (background hitam terlihat di
            # tengah permukaan yang seharusnya menyambung). Lubang kecil (<3% luas
            # total polygon) ditutup otomatis; lubang besar tetap dipertahankan
            # (kemungkinan memang area yang sengaja dikecualikan, mis. kolam/pond).
            def _tutup_lubang_kecil(poly, rasio_luas_min=0.03):
                try:
                    if poly.geom_type != "Polygon" or not poly.interiors:
                        return poly, 0
                    luas_total = poly.area
                    n_ditutup = 0
                    interior_dipertahankan = []
                    for ring in poly.interiors:
                        luas_lubang = Polygon(ring).area
                        if luas_lubang < luas_total * rasio_luas_min:
                            n_ditutup += 1  # lubang kecil -> tidak dipertahankan (ditutup)
                        else:
                            interior_dipertahankan.append(ring)
                    if n_ditutup == 0:
                        return poly, 0
                    return Polygon(poly.exterior, interior_dipertahankan), n_ditutup
                except Exception:
                    return poly, 0

            boundary, _n_lubang_ditutup = _tutup_lubang_kecil(boundary)
            if _n_lubang_ditutup > 0:
                st.info(
                    f"[{seg_label}] Terdeteksi {_n_lubang_ditutup} lubang kecil pada boundary hasil "
                    "deteksi otomatis (kemungkinan artefak alpha-shape akibat kerapatan titik kontur "
                    "tidak merata) — lubang tersebut ditutup otomatis supaya permukaan 3D menyambung. "
                    "Kalau memang ada area yang SENGAJA berlubang (mis. kolam/pond di tengah area), "
                    "upload DXF Boundary manual agar bentuknya presisi sesuai desain."
                )

            bx, by = boundary.exterior.xy

            # ================= RESOLUSI GRID & SMOOTHING (adaptif + bisa diatur user) =================

            from scipy.spatial import cKDTree as _cKDTree_res
            _sample_n = min(len(pts), 2000)
            _sample_idx = np.random.choice(len(pts), _sample_n, replace=False) if len(pts) > _sample_n else np.arange(len(pts))
            _tree_res = _cKDTree_res(pts[_sample_idx])
            _dists_res, _ = _tree_res.query(pts[_sample_idx], k=2)
            _median_spacing = float(np.median(_dists_res[:, 1])) if len(_dists_res) else 1.0

            bbox_diag = float(np.hypot(x_all.max() - x_all.min(), y_all.max() - y_all.min()))
            # target: sel grid kira-kira separuh dari jarak titik median (Nyquist-ish), dibatasi 150-1200 px
            _suggested_res = int(np.clip(bbox_diag / max(_median_spacing / 2, 1e-6), 150, 1200))

            with st.expander(f"Pengaturan surface (resolusi grid & smoothing) — {seg_label}", expanded=False):
                grid_res = st.slider(
                    "Resolusi grid (titik per sisi)", 150, 1200, value=_suggested_res, step=50,
                    key=f"grid_res_{sid}",
                    help="Disarankan otomatis berdasar kerapatan titik kontur asli. Naikkan untuk detail lebih "
                         "tinggi (lebih lambat), turunkan jika terlalu berat."
                )
                smooth_sigma = st.slider(
                    "Smoothing permukaan (sigma gaussian)", 0.0, 2.0, value=0.0, step=0.1,
                    key=f"smooth_sigma_{sid}",
                    help="0 = permukaan mengikuti TIN/interpolasi linear apa adanya (paling akurat terhadap kontur "
                         "asli, direkomendasikan). Nilai >0 menghaluskan tapi bisa mengaburkan breakline/tebing tajam."
                )
                show_dxf_overlay_3d = st.checkbox(
                    "Tampilkan overlay garis DXF asli di atas surface 3D", value=True,
                    key=f"dxf_overlay_3d_{sid}",
                    help="Menggambar ulang polyline/kontur DXF mentah (sebelum diinterpolasi) sebagai garis di "
                         "atas permukaan 3D hasil model, untuk memverifikasi visual apakah surface sudah "
                         "menempel dengan benar pada data DXF aslinya. Matikan kalau grafik terasa berat."
                )

            grid_x, grid_y = np.mgrid[
                x_all.min():x_all.max():complex(0, grid_res),
                y_all.min():y_all.max():complex(0, grid_res)
            ]

            dx = np.abs(grid_x[1, 0] - grid_x[0, 0])
            dy = np.abs(grid_y[0, 1] - grid_y[0, 0])
            cell_area = dx * dy

            # ---- bersihkan titik blunder sebelum interpolasi/triangulasi ----
            # SEBELUMNYA: dedup cuma di presisi 6 desimal (np.round(...,6)) -- untuk
            # koordinat UTM itu setara ~0.000001 m, jadi PRAKTIS TIDAK MENYARING APA
            # PUN. Titik "blunder" (X,Y hampir sama tapi Z beda jauh -- sudah dideteksi
            # & ditampilkan di panel Diagnostik di atas, tapi sebelumnya cuma dilaporkan,
            # TIDAK benar-benar dibersihkan) tetap ikut masuk ke griddata() sebagai dua
            # sampel yang saling bertentangan pada lokasi (x,y) yang nyaris sama. Inilah
            # penyebab utama permukaan 3D muncul sebagai "spike"/piramida tajam yang
            # tidak sesuai bentuk DXF aslinya -- interpolasi linear di antara dua Z yang
            # jauh berbeda pada XY yang nyaris sama menghasilkan dinding/puncak nyaris
            # vertikal yang tidak ada di data sebenarnya.
            #
            # Perbaikan: kelompokkan titik dalam radius toleransi (berbasis jarak
            # median antar titik asli / 4, supaya tidak menggabungkan titik yang
            # memang berjarak wajar antar kontur), lalu ambil MEDIAN Z tiap kelompok
            # (median lebih tahan terhadap outlier dibanding rata-rata). Titik yang
            # tidak duplikat sama sekali tidak berubah.
            _snap_tol = max(_median_spacing / 4.0, 1e-4)
            _xy_key = np.round(
                np.column_stack([x_all, y_all]) / _snap_tol
            ).astype(np.int64)

            _df_dedup = pd.DataFrame({
                "kx": _xy_key[:, 0], "ky": _xy_key[:, 1],
                "x": x_all, "y": y_all, "z": z_all,
            })
            _grp = _df_dedup.groupby(["kx", "ky"], sort=False)
            _n_cluster_blunder = int((_grp["z"].transform("nunique") > 1).sum())
            _df_clean = _grp.agg(x=("x", "mean"), y=("y", "mean"), z=("z", "median")).reset_index(drop=True)

            x_in = _df_clean["x"].to_numpy()
            y_in = _df_clean["y"].to_numpy()
            z_in = _df_clean["z"].to_numpy()

            if _n_cluster_blunder > 0:
                st.info(
                    f"[{seg_label}] {_n_cluster_blunder} titik blunder (X,Y hampir sama, Z berbeda "
                    f"jauh, radius toleransi ≈{_snap_tol:.3f} m) dibersihkan otomatis sebelum "
                    "interpolasi permukaan (diambil median Z per kelompok) — sebelumnya titik-titik "
                    "ini ikut diinterpolasi apa adanya dan menyebabkan permukaan 3D muncul sebagai "
                    "puncak/spike tajam yang tidak sesuai bentuk DXF asli."
                )

            grid_z = griddata(
                (x_in, y_in),
                z_in,
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

            # isi lubang kecil (titik dalam boundary yang di luar convex-hull data linear interp)
            if np.isnan(grid_z).any():

                nearest = griddata(
                    (x_in, y_in),
                    z_in,
                    (grid_x, grid_y),
                    method="nearest"
                )

                grid_z[np.isnan(grid_z)] = nearest[np.isnan(grid_z)]

            # smoothing sekarang OPSIONAL (default 0 = tidak menghaluskan, mengikuti data aktual)
            if smooth_sigma > 0:
                grid_z = gaussian_filter(
                    grid_z,
                    sigma=smooth_sigma
                )

            # ================= SLOPE =================

            dz_dx, dz_dy = np.gradient(grid_z)

            slope = np.sqrt(
                dz_dx**2 +
                dz_dy**2
            )

            slope[slope < 1e-6] = 1e-6

            # ================= HIDROLOGI & HIDROLIKA (opsional) =================
            hydraulics_result = None

            if use_hydraulics:

                r24_source = st.session_state.get("online_rainfall_meta")
                r24_mm = (
                    r24_source.get("max")
                    if r24_source and r24_source.get("max")
                    else st.session_state.get("online_rainfall")
                )

                if not r24_mm:
                    st.warning(
                        f"[{seg_label}] Perhitungan Rational Method+Manning diaktifkan tapi data "
                        "curah hujan (R24) belum tersedia — ambil data hujan dulu di bagian "
                        "'Data Curah Hujan' atau isi manual. Memakai kecepatan/kedalaman manual sebagai fallback."
                    )
                else:
                    r24_mm_extreme = r24_mm * rain_factor
                    boundary_area_ha_for_hydro = float(boundary.area) / 10000.0
                    representative_slope = float(np.nanmean(slope[inside])) if np.any(inside) else float(np.nanmean(slope))

                    hydraulics_result = compute_segment_hydraulics(
                        r24_mm=r24_mm_extreme,
                        c_runoff=runoff_c,
                        tc_hours=tc_hours,
                        area_ha=boundary_area_ha_for_hydro,
                        manning_n=manning_n,
                        channel_b=channel_b,
                        channel_z=channel_z,
                        channel_h_total=channel_h_total,
                        representative_slope=representative_slope
                    )

                    if hydraulics_result:
                        velocity_hulu = hydraulics_result["v_normal_ms"]
                        flow_depth = hydraulics_result["h_normal_m"]

                        st.markdown("#### Hasil Rational Method + Manning's Equation")
                        colhy1, colhy2, colhy3, colhy4 = st.columns(4)
                        colhy1.metric("Intensitas Hujan (Mononobe)", f"{hydraulics_result['intensity_mm_hr']:.1f} mm/jam")
                        colhy2.metric("Debit Rencana Q", f"{hydraulics_result['q_design_m3s']:.3f} m³/s")
                        colhy3.metric("Kecepatan Normal V", f"{velocity_hulu:.3f} m/s")
                        colhy4.metric("Kedalaman Normal h", f"{flow_depth:.3f} m")

                        colhy5, colhy6 = st.columns(2)
                        colhy5.metric("Froude Number", f"{hydraulics_result['froude']:.2f} — {hydraulics_result['flow_regime']}" if hydraulics_result['froude'] else "-")
                        fb_status = "Cukup" if hydraulics_result["freeboard_ok"] else "KURANG"
                        colhy6.metric(
                            "Freeboard",
                            f"{hydraulics_result['freeboard_m']:.2f} m ({fb_status})",
                            help=f"Minimum disarankan ≈ {hydraulics_result['min_freeboard_m']:.2f} m"
                        )

                        if not hydraulics_result["freeboard_ok"]:
                            st.error(
                                f"[{seg_label}] Freeboard tidak mencukupi pada debit rencana ini — "
                                "tinggi tanggul/channel perlu ditambah atau lebar dasar diperbesar "
                                "agar kedalaman normal turun."
                            )

                        st.caption(
                            f"R24 dipakai: {r24_mm:.1f} mm × Extreme Rainfall Factor {rain_factor:.1f} = "
                            f"{r24_mm_extreme:.1f} mm | C={runoff_c:.2f} | tc={tc_hours:.2f} jam | "
                            f"n={manning_n:.3f} | b={channel_b:.1f} m | z={channel_z:.1f} | "
                            f"S rata-rata={representative_slope:.4f}. Kecepatan & kedalaman manual "
                            "di atas kini DIGANTIKAN oleh hasil perhitungan ini."
                        )

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

            # D8 flow accumulation dihitung SEKALI di sini (dipakai baik untuk mode
            # Hujan/Uniform maupun sebagai proxy "debit" untuk profil ketebalan air
            # di sepanjang jalur pada mode Point Source) -- sebelumnya cuma dihitung
            # di cabang Hujan sehingga tidak tersedia saat trace_multiple_flow() perlu
            # tahu seberapa besar tangkapan air di tiap sel untuk merepresentasikan
            # aliran yang makin tebal ke arah hilir.
            with st.spinner(f"Menghitung D8 flow accumulation ({seg_label})..."):
                flow_acc_grid = compute_flow_accumulation_d8(grid_z, inside, dx, dy)
            flow_acc_grid[~inside] = 0.0

            if (
                source_type ==
                "Satu Titik (Point Source)"
            ) and is_sub_segment:

                # ---- PERBAIKAN ("sub-boundary terlihat terpisah dari aliran Main"):
                # sub-segmen TIDAK menelusuri jalur alirannya sendiri dari titik baru
                # (titik hulu Main hampir pasti di LUAR boundary kecil sub-segmen ini,
                # jadi kalau dipaksa trace ulang dari situ hampir selalu langsung
                # dianggap "di luar area"/gagal). Sebagai gantinya: jalur aliran Main
                # (ditelusuri di iterasi Main, sekarang memakai boundary GABUNGAN
                # Main + semua sub-boundary -- lihat cabang else di bawah) sudah
                # otomatis melewati/masuk ke boundary sub-segmen ini juga selama
                # memang dilalui alirannya. Di sini kita AMBIL BAGIAN jalur tsb yang
                # berada di dalam boundary sub-segmen ini saja, supaya flow_mask &
                # tampilan 3D-nya benar-benar tersambung/kontinu dengan Main, bukan
                # mensimulasikan sumber air baru yang berdiri sendiri. ----
                _main_sid_src2 = st.session_state["segments"][0]
                _main_res_src = st.session_state.get("segment_results", {}).get(_main_sid_src2)
                flow_paths = []
                if _main_res_src:
                    for _fp in _main_res_src.get("flow_paths", []):
                        _fpx, _fpy, _fpz, _fpd = _fp
                        if len(_fpx) == 0:
                            continue
                        _mask_in = np.array([
                            boundary.contains(Point(_x, _y)) for _x, _y in zip(_fpx, _fpy)
                        ])
                        if _mask_in.any():
                            _idx_in = np.where(_mask_in)[0]
                            flow_paths.append((
                                [_fpx[i] for i in _idx_in],
                                [_fpy[i] for i in _idx_in],
                                [_fpz[i] for i in _idx_in],
                                [_fpd[i] for i in _idx_in],
                            ))

                if flow_paths:
                    _arrive_depths = [p[3][0] for p in flow_paths if p[3]]
                    _arrive_v = st.session_state.get(
                        f"velocity_hulu_{_main_sid_src2}",
                        velocity_hulu
                    )
                    if _arrive_depths:
                        st.caption(
                            f"Aliran dari Segmen 1 (Main) memasuki sub-segmen ini "
                            f"({seg_label}) — estimasi kedalaman air ≈ "
                            f"{np.mean(_arrive_depths):.2f} m pada velocity hulu "
                            f"≈ {_arrive_v:.2f} m/s (mengikuti profil aliran dari Main, "
                            "bukan sumber air baru yang terpisah)."
                        )
                else:
                    st.warning(
                        f"Jalur aliran dari Segmen 1 (Main) belum melewati boundary "
                        f"sub-segmen '{seg_label}'. Cek apakah boundary sub-segmen ini "
                        "memang dilalui aliran dari titik hulu yang dipilih di Segmen 1 "
                        "(kalau tidak, sub-segmen ini secara fisik memang tidak menerima "
                        "aliran langsung dari titik tsb)."
                    )

            elif (
                source_type ==
                "Satu Titik (Point Source)"
            ):

                # ---- MODE KLIK-DI-PETA: peta desain sederhana (garis DXF + boundary) ----
                # untuk memilih titik aliran dengan klik kursor, sebagai alternatif ketik
                # koordinat manual. Ditaruh di sini (bukan di loop input awal) karena baru
                # di titik inilah `contours`, `boundary`, `grid_x`, `grid_y` sudah selesai
                # diproses dari DXF segmen ini.
                if point_method == "Klik di Peta Desain":
                    st.markdown(f"**Pilih Titik Aliran di Peta — {seg_label}**")
                    st.caption(
                        "Klik satu titik di peta di bawah ini untuk menentukan lokasi awal air "
                        "mulai mengalir (titik hulu / point source). Garis hitam = kontur/garis "
                        "DXF asli, garis magenta = boundary area kajian."
                    )

                    fig_pick = go.Figure()

                    # garis desain DXF (2D) sebagai konteks visual
                    _pick_legend_shown = False
                    for _c in contours:
                        if len(_c) < 2:
                            continue
                        fig_pick.add_trace(go.Scatter(
                            x=[p[0] for p in _c], y=[p[1] for p in _c],
                            mode="lines", line=dict(color="#222222", width=1),
                            name="Desain DXF", legendgroup="dxf",
                            showlegend=not _pick_legend_shown, hoverinfo="skip",
                        ))
                        _pick_legend_shown = True

                    # boundary area kajian
                    _pbx, _pby = boundary.exterior.xy
                    fig_pick.add_trace(go.Scatter(
                        x=list(_pbx), y=list(_pby), mode="lines",
                        line=dict(color="magenta", width=2), name="Boundary",
                        hoverinfo="skip",
                    ))

                    # grid titik tipis TAK-TERLIHAT (opacity ~0) sebagai target klik, supaya
                    # klik di mana pun dalam area kajian tetap menangkap koordinat (x,y) yang
                    # presisi mengikuti grid analisis -- bukan cuma titik vertex DXF.
                    _pick_step = max(1, grid_x.shape[0] // 60)
                    _gx_s = grid_x[::_pick_step, ::_pick_step]
                    _gy_s = grid_y[::_pick_step, ::_pick_step]
                    _inside_s = inside[::_pick_step, ::_pick_step]
                    fig_pick.add_trace(go.Scatter(
                        x=_gx_s[_inside_s].ravel(), y=_gy_s[_inside_s].ravel(),
                        mode="markers",
                        marker=dict(size=14, color="rgba(0,150,255,0.08)"),
                        name="(area klik)", showlegend=False,
                        hovertemplate="X=%{x:.2f}, Y=%{y:.2f}<extra></extra>",
                    ))

                    # tandai titik yang sedang terpilih (kalau ada) dengan marker mencolok
                    _prev_cx, _prev_cy = st.session_state.get(f"flow_click_xy_{sid}", (None, None))
                    if _prev_cx is not None:
                        fig_pick.add_trace(go.Scatter(
                            x=[_prev_cx], y=[_prev_cy], mode="markers",
                            marker=dict(size=16, color="yellow", symbol="star",
                                        line=dict(color="black", width=1.5)),
                            name="Titik terpilih", hoverinfo="skip",
                        ))

                    fig_pick.update_layout(
                        height=520, dragmode="pan",
                        xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                        yaxis=dict(scaleanchor="x", scaleratio=1),
                        margin=dict(l=10, r=10, t=10, b=10),
                        clickmode="event+select",
                    )

                    _pick_key = f"flow_click_map_{sid}"
                    try:
                        _ev = st.plotly_chart(
                            fig_pick, use_container_width=True,
                            on_select="rerun", selection_mode="points",
                            key=_pick_key,
                        )
                        _sel_points = (_ev or {}).get("selection", {}).get("points", [])
                        if _sel_points:
                            _clicked_x = _sel_points[-1].get("x")
                            _clicked_y = _sel_points[-1].get("y")
                            if _clicked_x is not None and _clicked_y is not None:
                                st.session_state[f"flow_click_xy_{sid}"] = (float(_clicked_x), float(_clicked_y))
                    except TypeError:
                        # fallback untuk versi Streamlit lama yang belum mendukung on_select
                        # di st.plotly_chart -- tetap tampilkan peta (statis, tanpa klik),
                        # dan beri tahu user cara mengaktifkan fitur klik.
                        st.plotly_chart(fig_pick, use_container_width=True)
                        st.warning(
                            "Versi Streamlit di server ini belum mendukung klik-pilih pada grafik "
                            "(butuh Streamlit >= 1.35 untuk parameter on_select). Update Streamlit "
                            "untuk mengaktifkan fitur klik-di-peta, atau gunakan mode ketik koordinat manual."
                        )

                    _clicked_xy = st.session_state.get(f"flow_click_xy_{sid}")
                    if _clicked_xy is None:
                        st.warning(
                            f"Belum ada titik yang diklik untuk {seg_label} — klik salah satu titik "
                            "di peta di atas dulu untuk melanjutkan simulasi aliran segmen ini."
                        )
                        continue
                    point_x, point_y = _clicked_xy
                    st.success(f"Titik aliran terpilih: X = {point_x:.3f}, Y = {point_y:.3f}")

                if not boundary.contains(
                    Point(point_x, point_y)
                ):
                    st.error(
                        "Point Source berada di luar area kontur"
                    )
                    continue

                # ---- PERBAIKAN ("aliran Main terlihat berhenti sebelum sub-boundary"):
                # kalau proyek ini punya sub-segmen lain (memakai kontur/DXF yang SAMA
                # dengan Main), jalur aliran ditelusuri memakai GABUNGAN boundary Main +
                # semua sub-boundary (masing-masing dibuffer tipis), bukan cuma boundary
                # Main sendiri. Sebelumnya trace berhenti begitu keluar boundary Main
                # walau area sub-segmen lain masih ada di depannya/di sekitarnya --
                # membuat aliran terlihat berhenti di tengah jalan padahal datanya
                # masih lanjut, dan boundary sub-segmen jadi terlihat terpisah/tidak
                # terhubung dari aliran Main. ----
                _trace_boundary = boundary
                _trace_inside = inside
                _other_sids_for_trace = [
                    s for s in st.session_state["segments"] if s != sid
                ]
                if _other_sids_for_trace:
                    _union_polys = [boundary]
                    for _osid in _other_sids_for_trace:
                        _obdxf = st.session_state.get(f"boundary_dxf_{_osid}")
                        if _obdxf is None:
                            continue
                        try:
                            _opoly = read_boundary_polygon(_obdxf)
                            _opoly = fix_geom(_opoly)
                            if _opoly is not None and not _opoly.is_empty:
                                _union_polys.append(_opoly.buffer(0.5))
                        except Exception:
                            continue
                    if len(_union_polys) > 1:
                        try:
                            _trace_boundary = unary_union(_union_polys)
                            # PENTING: `trace_multiple_flow` sekarang mengecek stop
                            # trace lewat raster `inside` (bukan lagi lewat polygon
                            # `boundary` langsung -- lihat catatan di dalam fungsi
                            # tsb), jadi union polygon di atas HARUS ikut dirasterkan
                            # ke grid yang sama supaya union-nya benar2 berlaku,
                            # bukan cuma dilewatkan sebagai parameter `boundary` yang
                            # tidak lagi dipakai untuk keputusan stop per-langkah.
                            _trace_inside = inside | vectorized.contains(
                                _trace_boundary, grid_x, grid_y
                            )
                        except Exception:
                            _trace_boundary = boundary
                            _trace_inside = inside

                flow_paths = trace_multiple_flow(
                    point_x,
                    point_y,
                    grid_x,
                    grid_y,
                    dz_dx,
                    dz_dy,
                    grid_z,
                    _trace_boundary,
                    inside=_trace_inside,
                    depth0=st.session_state.get(f"point_depth_{sid}", 0.2),
                    flow_acc=flow_acc_grid
                )

            # ================= FLOW MASK & FLOW ACCUMULATION (D8, seluruh grid) =================

            if source_type == "Hujan (Uniform)":

                flow_mask = ~np.isnan(grid_z)

                # D8 flow accumulation sudah dihitung sebelumnya (flow_acc_grid) —
                # dipakai ulang di sini, tidak perlu dihitung dua kali.
                flow_count = flow_acc_grid

            else:

                flow_mask = np.zeros_like(grid_z)

                for px, py, _, _ in flow_paths:

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

            # log1p menekan rentang dinamis akumulasi (sel outlet bisa punya nilai jauh lebih
            # besar dari sel hulu) sehingga pola jaringan aliran (channel network) tetap terlihat
            # setelah smoothing, bukan hilang menjadi gumpalan halus tak berpola.
            flow_density = gaussian_filter(
                np.log1p(flow_count),
                sigma=2
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
                        grain_size,
                        flow_depth
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

                    flatshading=False,

                    # PERBAIKAN: tambah lighting/shadowing supaya mesh 3D tidak
                    # terlihat "flat/rata" -- terrain sekarang punya kesan bayangan
                    # (shading) mengikuti bentuk permukaan asli, bukan cuma warna
                    # rainbow polos tanpa efek kedalaman.
                    lighting=dict(
                        ambient=0.55,
                        diffuse=0.9,
                        specular=0.25,
                        roughness=0.85,
                        fresnel=0.15,
                    ),
                    lightposition=dict(x=200, y=200, z=400),
                )
            )

            # ================= OVERLAY GARIS DXF ASLI DI ATAS SURFACE 3D =================
            # Menggambar ulang tiap polyline/entity DXF mentah (variabel `contours`, HASIL
            # read_contours() sebelum diinterpolasi ke grid) sebagai garis 3D di atas mesh
            # permukaan. Tujuannya supaya engineer bisa cek visual apakah surface hasil
            # interpolasi benar-benar menempel pada data DXF aslinya (bukan cuma percaya
            # hasil interpolasi tanpa verifikasi).
            if show_dxf_overlay_3d and contours:
                _MAX_DXF_OVERLAY_ENTITIES = 400  # batasi supaya render tidak terlalu berat
                _overlay_contours = contours
                if len(_overlay_contours) > _MAX_DXF_OVERLAY_ENTITIES:
                    _step = max(1, len(_overlay_contours) // _MAX_DXF_OVERLAY_ENTITIES)
                    _overlay_contours = _overlay_contours[::_step]

                _dxf_overlay_legend_shown = False
                for _c in _overlay_contours:
                    if len(_c) < 1:
                        continue
                    _cx = [p[0] for p in _c]
                    _cy = [p[1] for p in _c]
                    _cz = [p[2] * vertical_exaggeration for p in _c]

                    if len(_c) == 1:
                        # entity berupa POINT tunggal -> gambar sebagai marker, bukan garis
                        fig.add_trace(
                            go.Scatter3d(
                                x=_cx, y=_cy, z=_cz,
                                mode="markers",
                                marker=dict(size=3, color="#111111", symbol="circle"),
                                name="DXF Asli (overlay)",
                                legendgroup="dxf_overlay",
                                showlegend=not _dxf_overlay_legend_shown,
                                hoverinfo="skip",
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=_cx, y=_cy, z=_cz,
                                mode="lines",
                                line=dict(color="#111111", width=2),
                                name="DXF Asli (overlay)",
                                legendgroup="dxf_overlay",
                                showlegend=not _dxf_overlay_legend_shown,
                                hoverinfo="skip",
                            )
                        )
                    _dxf_overlay_legend_shown = True

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
                ],

                "RiskScore": zone_map[
                    critical_y,
                    critical_x
                ],

                "Velocity_ms": velocity_field[
                    critical_y,
                    critical_x
                ]
            })

            df_overflow = df_overflow.sort_values(
                "FlowDensity",
                ascending=False
            )

            # ---- klasifikasi jenis risiko dominan per titik kritis (setara Tabel 8 makalah) ----
            _slope_at_pts = slope[critical_y, critical_x]
            _sediment_at_pts = sediment_map[critical_y, critical_x]
            _flow_density_norm = df_overflow["FlowDensity"] / (np.nanmax(flow_density) + 1e-9)
            _steep_threshold = np.nanpercentile(slope[inside], 70) if np.any(inside) else 0.15

            def _classify_dominant_risk(sed_val, flow_norm_val, risk_val, slope_val):
                if sed_val is not None and sed_val > 0.6:
                    return "Sedimentasi"
                if flow_norm_val > 0.8 and risk_val > 1.0:
                    return "Overflow"
                if slope_val > _steep_threshold:
                    return "Erosi Tebing"
                return "Erosi Dasar"

            _RISK_TYPE_RECOMMENDATION = {
                "Sedimentasi": "Sediment trap + normalisasi rutin",
                "Overflow": "Peninggian tanggul + monitoring intensif",
                "Erosi Tebing": "Perkuatan tebing (riprap/revetment)",
                "Erosi Dasar": "Check dam + monitoring berkala",
            }

            df_overflow = df_overflow.assign(
                Slope=_slope_at_pts,
                SedimentProb=_sediment_at_pts,
            )
            df_overflow["FlowDensityNorm"] = _flow_density_norm.values
            df_overflow["JenisRisikoDominan"] = [
                _classify_dominant_risk(sv, fv, rv, slv)
                for sv, fv, rv, slv in zip(
                    df_overflow["SedimentProb"], df_overflow["FlowDensityNorm"],
                    df_overflow["RiskScore"], df_overflow["Slope"]
                )
            ]
            df_overflow["Rekomendasi"] = df_overflow["JenisRisikoDominan"].map(_RISK_TYPE_RECOMMENDATION)

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

            _seg_prefix = "".join([c for c in seg_label.upper() if c.isalnum()])[:4] or "EW"
            top10.insert(1, "ID_Titik", [f"{_seg_prefix}-{i:02d}" for i in top10["Rank"]])
            top10.insert(2, "Segmen", seg_label)

            st.markdown("**Sepuluh Titik Prioritas Mitigasi (setara Tabel 8 makalah acuan)**")
            st.caption(
                "Jenis risiko dominan diklasifikasi otomatis dari kombinasi: probabilitas deposisi "
                "(Sedimentasi), flow density × risk score (Overflow), dan kemiringan lokal (Erosi "
                "Tebing vs Erosi Dasar). Ambang kemiringan dihitung adaptif dari persentil ke-70 "
                "kemiringan area ini."
            )
            _display_cols = ["Rank", "ID_Titik", "Segmen", "X", "Y", "RL",
                              "JenisRisikoDominan", "RiskScore", "Rekomendasi"]
            st.dataframe(
                top10[_display_cols].round(3),
                use_container_width=True
            )

            _risk_type_colors = {
                "Erosi Tebing": "#C0392B", "Sedimentasi": "#2E86C1",
                "Overflow": "#2471A3", "Erosi Dasar": "#B7950B",
            }
            fig_priority = go.Figure()
            for rtype in top10["JenisRisikoDominan"].unique():
                sub = top10[top10["JenisRisikoDominan"] == rtype]
                fig_priority.add_trace(go.Bar(
                    x=sub["RiskScore"], y=sub["ID_Titik"], orientation="h",
                    name=rtype, marker_color=_risk_type_colors.get(rtype, "#888"),
                    text=sub["RiskScore"].round(2), textposition="outside"
                ))
            fig_priority.update_layout(
                title="Sebaran Risk Index — 10 Titik Prioritas Mitigasi",
                xaxis_title="Risk Index", height=420, barmode="overlay",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_priority, use_container_width=True)

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

            for px, py, pz, pdepth in flow_paths:

                pz_arr = np.array(pz)
                pdepth_arr = np.array(pdepth)
                # permukaan air digambar SEDIKIT DI ATAS permukaan tanah, setinggi
                # ketebalan air di titik tsb (pdepth), supaya terlihat sebagai lapisan
                # air yang punya volume/ketebalan -- bukan cuma garis tipis menempel
                # tanah seperti sebelumnya. Ukuran marker juga ikut membesar mengikuti
                # ketebalan (lebih tebal ke arah hilir kalau flow_acc tersedia).
                water_z = (pz_arr + pdepth_arr) * vertical_exaggeration
                marker_sizes = np.clip(4 + pdepth_arr * 25, 4, 22)

                fig.add_trace(
                    go.Scatter3d(
                        x=px,
                        y=py,
                        z=water_z,
                        mode="lines+markers",
                        line=dict(
                            color="#00C8FF",
                            width=5
                        ),
                        marker=dict(
                            size=marker_sizes,
                            color="#00C8FF",
                            opacity=0.85
                        ),
                        customdata=pdepth_arr,
                        hovertemplate="Ketebalan air ≈ %{customdata:.2f} m<extra></extra>",
                        name="Aliran Air",
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
        
            # PERBAIKAN: sebelumnya chart 3D ini dirender per-segmen TANPA syarat --
            # kalau ada 2 segmen (Main + 1 sub-boundary), user melihat 3 output 3D
            # sekaligus (chart Main di sini, chart Segmen-2 di sini juga saat loop
            # jalan utk segmen ke-2, DITAMBAH "Scene 3D Gabungan" terpisah di bawah
            # setelah loop selesai) -- padahal maksudnya cukup SATU output analisa
            # per proyek. Sekarang chart per-segmen ini HANYA ditampilkan kalau
            # total segmen = 1 (tidak ada sub-boundary tambahan); begitu ada 2+
            # segmen, satu-satunya output 3D adalah "Scene 3D Gabungan" di bawah
            # (yang sekarang sudah menggabungkan mesh rainbow tiap segmen jadi satu
            # scene, bukan cuma terrain polos + garis outline seperti sebelumnya).
            _total_segments_now = len(st.session_state["segments"])
            if _total_segments_now == 1:
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
            else:
                st.caption(
                    f"Chart 3D individual segmen '{seg_label}' digabung ke dalam "
                    "**Scene 3D Gabungan** di bagian bawah halaman (karena proyek ini "
                    f"punya {_total_segments_now} segmen) — bukan ditampilkan terpisah, "
                    "supaya hanya ada satu output analisa 3D per proyek."
                )

            # ================= SIMULASI ALIRAN AIR (ANIMASI 3D) =================
            st.markdown("#### Simulasi Aliran Air (Animasi)")
            st.caption(
                "Animasi ini memutar ulang jalur aliran (streamline steepest-descent) "
                "yang sama dengan hasil 'Numerical Modelling' di atas — bukan simulasi "
                "hidrolik baru — divisualisasikan sebagai pergerakan titik air dari hulu "
                "ke hilir mengikuti topografi hasil DXF, seperti video, langsung di browser."
            )

            run_animation = st.checkbox(
                f"Buat animasi untuk {seg_label}",
                key=f"anim_toggle_{sid}",
                value=False,
                help="Nonaktif secara default supaya RUN ANALYSIS tetap cepat. Centang untuk membangun animasi."
            )

            if run_animation and len(flow_paths) > 0:

                n_frames = st.slider(
                    "Jumlah frame animasi", 15, 80, 30, 5,
                    key=f"anim_frames_{sid}",
                    help="Makin banyak frame = animasi makin halus, tapi makin berat dibangun."
                )

                with st.spinner("Membangun frame animasi..."):

                    # Samakan panjang progres semua path terhadap path TERPANJANG,
                    # supaya path yang lebih pendek berhenti duluan (lebih realistis
                    # dibanding semua path "selesai" bersamaan).
                    max_len = max(len(px) for px, py, pz, pd in flow_paths)
                    max_len = max(max_len, 2)

                    frame_indices = np.linspace(1, max_len, n_frames, dtype=int)

                    anim_base_traces = list(fig.data[:-len(flow_paths)]) if len(flow_paths) > 0 else list(fig.data)

                    frames = []
                    for f_i, cut in enumerate(frame_indices):
                        frame_traces = []
                        for px, py, pz, pdep in flow_paths:
                            c = min(cut, len(px))
                            if c < 1:
                                continue
                            trail_x = px[:c]
                            trail_y = py[:c]
                            trail_pd = pdep[:c]
                            trail_z = ((np.array(pz[:c]) + np.array(trail_pd)) * vertical_exaggeration).tolist()

                            # jejak air yang sudah dilalui (permukaan air, bukan tanah)
                            frame_traces.append(
                                go.Scatter3d(
                                    x=trail_x, y=trail_y, z=trail_z,
                                    mode="lines",
                                    line=dict(color="rgba(0,200,255,0.55)", width=3),
                                    showlegend=False
                                )
                            )
                            # "partikel air" di ujung jejak (bagian terdepan aliran), ukuran
                            # mengikuti ketebalan air di titik itu supaya makin tebal makin
                            # besar terlihat -- lebih mirip aliran air sungguhan (bukan cuma
                            # titik seragam)
                            head_size = float(np.clip(6 + trail_pd[-1] * 25, 6, 24))
                            frame_traces.append(
                                go.Scatter3d(
                                    x=[trail_x[-1]], y=[trail_y[-1]], z=[trail_z[-1]],
                                    mode="markers",
                                    marker=dict(size=head_size, color="#00E5FF", symbol="circle"),
                                    showlegend=False
                                )
                            )
                        frames.append(go.Frame(data=anim_base_traces + frame_traces, name=str(f_i)))

                    fig_anim = go.Figure(
                        data=anim_base_traces + frames[0].data[len(anim_base_traces):],
                        frames=frames
                    )

                    fig_anim.update_layout(
                        title=dict(
                            text=f"Simulasi Aliran Air — {seg_label} ({analysis_method})",
                            x=0.5
                        ),
                        height=850,
                        scene=dict(aspectmode="data"),
                        updatemenus=[dict(
                            type="buttons",
                            showactive=False,
                            y=1,
                            x=0.05,
                            xanchor="left",
                            yanchor="top",
                            buttons=[
                                dict(
                                    label="▶ Play",
                                    method="animate",
                                    args=[None, dict(
                                        frame=dict(duration=140, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=0)
                                    )]
                                ),
                                dict(
                                    label="Pause",
                                    method="animate",
                                    args=[[None], dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode="immediate"
                                    )]
                                )
                            ]
                        )],
                        sliders=[dict(
                            steps=[
                                dict(
                                    method="animate",
                                    args=[[str(k)], dict(
                                        frame=dict(duration=0, redraw=True),
                                        mode="immediate"
                                    )],
                                    label=str(k)
                                ) for k in range(len(frames))
                            ],
                            x=0.1, y=0, len=0.85
                        )]
                    )

                    st.plotly_chart(fig_anim, use_container_width=True)
                    st.caption(
                        "Tekan ▶ Play untuk menjalankan animasi, atau geser slider untuk "
                        "melihat progres aliran pada frame tertentu."
                    )

            # ================= SIMULASI HUJAN (ANIMASI: TETES JATUH + ALIRAN MULTI-TITIK) =================
            st.markdown("#### Simulasi Hujan (Animasi Tetes Jatuh + Aliran Multi-Titik)")
            st.caption(
                "Berbeda dari animasi di atas (yang cuma mengikuti 1 titik hulu), simulasi ini "
                "mengambil beberapa titik ACAK tersebar di seluruh area kajian untuk mewakili "
                "hujan yang jatuh merata, lalu menganimasikan tetes air jatuh dari atas ke "
                "permukaan tanah, dilanjutkan air tersebut mengalir ke hilir mengikuti topografi "
                "-- seperti video simulasi hujan turun dan alirannya."
            )

            run_rain_animation = st.checkbox(
                f"Buat animasi hujan untuk {seg_label}",
                key=f"rain_anim_toggle_{sid}",
                value=False,
                help="Nonaktif secara default supaya RUN ANALYSIS tetap cepat. Centang untuk membangun animasi hujan."
            )

            if run_rain_animation:

                colra1, colra2, colra3 = st.columns(3)

                n_rain_points = colra1.slider(
                    "Jumlah titik hujan", 5, 40, 15, 1,
                    key=f"rain_n_points_{sid}",
                    help="Makin banyak titik = representasi hujan makin rapat, tapi makin berat dibangun."
                )
                n_frames_rain = colra2.slider(
                    "Jumlah frame animasi", 15, 80, 35, 5,
                    key=f"rain_anim_frames_{sid}"
                )
                rain_fall_height = colra3.number_input(
                    "Tinggi jatuh tetes (m)", 1.0, 200.0, 20.0, 1.0,
                    key=f"rain_fall_height_{sid}",
                    help="Cuma untuk efek visual tetes air jatuh dari langit sebelum mulai mengalir, bukan parameter hidrologi."
                )

                with st.spinner("Membangun animasi hujan (titik acak + streamline + tetes jatuh)..."):

                    rng = np.random.default_rng(42 + idx)
                    valid_iy, valid_ix = np.where(inside)

                    if len(valid_iy) == 0:
                        st.warning("Tidak ada sel valid di dalam boundary untuk mensimulasikan hujan.")
                    else:
                        n_pick = min(n_rain_points, len(valid_iy))
                        pick_idx = rng.choice(len(valid_iy), size=n_pick, replace=False)

                        rain_paths = []
                        for k in pick_idx:
                            rr, cc = int(valid_iy[k]), int(valid_ix[k])
                            rx0 = float(grid_x[rr, 0])
                            ry0 = float(grid_y[0, cc])
                            _paths = trace_multiple_flow(
                                rx0, ry0, grid_x, grid_y, dz_dx, dz_dy, grid_z, boundary,
                                n_stream=1, inside=inside,
                                depth0=0.1, flow_acc=flow_acc_grid
                            )
                            if _paths:
                                rain_paths.append(_paths[0])

                        if not rain_paths:
                            st.warning("Gagal membangun streamline hujan untuk segmen ini.")
                        else:
                            base_traces_rain = (
                                list(fig.data[:-len(flow_paths)])
                                if len(flow_paths) > 0 else list(fig.data)
                            )

                            max_len_rain = max(len(p[0]) for p in rain_paths)
                            max_len_rain = max(max_len_rain, 2)

                            n_fall_frames = max(4, n_frames_rain // 4)
                            n_flow_frames = max(4, n_frames_rain - n_fall_frames)
                            frame_indices_rain = np.linspace(1, max_len_rain, n_flow_frames, dtype=int)

                            frames_rain = []

                            # -- Fase 1: tetes hujan jatuh dari atas menuju permukaan tanah --
                            for f_i in range(n_fall_frames):
                                t = (f_i + 1) / n_fall_frames
                                frame_traces = []
                                for (px, py, pz, pdep) in rain_paths:
                                    z_ground = pz[0] * vertical_exaggeration
                                    z_start = z_ground + rain_fall_height * vertical_exaggeration
                                    z_now = z_start + (z_ground - z_start) * t
                                    frame_traces.append(
                                        go.Scatter3d(
                                            x=[px[0]], y=[py[0]], z=[z_now],
                                            mode="markers",
                                            marker=dict(size=5, color="#7FD8FF", symbol="circle", opacity=0.85),
                                            showlegend=False
                                        )
                                    )
                                frames_rain.append(
                                    go.Frame(data=base_traces_rain + frame_traces, name=f"fall_{f_i}")
                                )

                            # -- Fase 2: air mengalir ke hilir mengikuti masing-masing streamline --
                            for f_i, cut in enumerate(frame_indices_rain):
                                frame_traces = []
                                for (px, py, pz, pdep) in rain_paths:
                                    c = min(cut, len(px))
                                    if c < 1:
                                        continue
                                    trail_x = px[:c]
                                    trail_y = py[:c]
                                    trail_pd = pdep[:c]
                                    trail_z = (
                                        (np.array(pz[:c]) + np.array(trail_pd)) * vertical_exaggeration
                                    ).tolist()
                                    frame_traces.append(
                                        go.Scatter3d(
                                            x=trail_x, y=trail_y, z=trail_z,
                                            mode="lines",
                                            line=dict(color="rgba(120,210,255,0.5)", width=2),
                                            showlegend=False
                                        )
                                    )
                                    head_size = float(np.clip(4 + trail_pd[-1] * 20, 4, 18))
                                    frame_traces.append(
                                        go.Scatter3d(
                                            x=[trail_x[-1]], y=[trail_y[-1]], z=[trail_z[-1]],
                                            mode="markers",
                                            marker=dict(size=head_size, color="#7FD8FF", symbol="circle"),
                                            showlegend=False
                                        )
                                    )
                                frames_rain.append(
                                    go.Frame(data=base_traces_rain + frame_traces, name=f"flow_{f_i}")
                                )

                            fig_rain_anim = go.Figure(
                                data=base_traces_rain + frames_rain[0].data[len(base_traces_rain):],
                                frames=frames_rain
                            )

                            fig_rain_anim.update_layout(
                                title=dict(text=f"Simulasi Hujan — {seg_label}", x=0.5),
                                height=850,
                                scene=dict(aspectmode="data"),
                                updatemenus=[dict(
                                    type="buttons", showactive=False, y=1, x=0.05,
                                    xanchor="left", yanchor="top",
                                    buttons=[
                                        dict(label="▶ Play", method="animate", args=[None, dict(
                                            frame=dict(duration=110, redraw=True),
                                            fromcurrent=True, transition=dict(duration=0)
                                        )]),
                                        dict(label="Pause", method="animate", args=[[None], dict(
                                            frame=dict(duration=0, redraw=False), mode="immediate"
                                        )]),
                                    ]
                                )],
                                sliders=[dict(
                                    steps=[
                                        dict(
                                            method="animate",
                                            args=[[fr.name], dict(
                                                frame=dict(duration=0, redraw=True), mode="immediate"
                                            )],
                                            label=str(k)
                                        ) for k, fr in enumerate(frames_rain)
                                    ],
                                    x=0.1, y=0, len=0.85
                                )]
                            )

                            st.plotly_chart(fig_rain_anim, use_container_width=True)
                            st.caption(
                                "Fase awal: tetes air hujan 'jatuh' dari atas ke permukaan tanah di "
                                "beberapa titik acak. Fase berikutnya: air tsb mengalir ke hilir mengikuti "
                                "topografi. Tekan ▶ Play atau geser slider."
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
                        color="rgba(0, 195, 255, 0.35)"
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

            # ================= PETA RISIKO + ORTHOPHOTO OVERLAY (OPSIONAL) =================
            if orthophoto_data is not None:
                show_ortho_overlay = st.checkbox(
                    f"Tampilkan peta rainbow erosi/sedimentasi + garis DXF teroverlay di atas Orthophoto — {seg_label}",
                    value=True,
                    key=f"show_ortho_overlay_{sid}",
                )
                if show_ortho_overlay:
                    with st.spinner(f"[{seg_label}] Menyusun overlay orthophoto..."):
                        _fig_ortho, _ax_ortho = plt.subplots(figsize=(12, 9))

                        # 1) Orthophoto sebagai latar paling belakang
                        _ax_ortho.imshow(
                            orthophoto_data["rgb"],
                            extent=orthophoto_data["extent"],
                            origin="upper",
                            zorder=1,
                        )

                        # 2) Rainbow kontur risiko erosi (RdYlGn_r: hijau=aman, merah=erosi tinggi)
                        #    + sedimentasi (Blues), transparan supaya orthophoto tetap kelihatan.
                        _ax_ortho.contourf(
                            grid_x, grid_y, zone_map, levels=15,
                            cmap="RdYlGn_r", alpha=0.55, zorder=2,
                        )
                        _ax_ortho.contourf(
                            grid_x, grid_y, sediment_map, levels=[0.5, 0.7, 0.85, 1],
                            cmap="Blues", alpha=0.40, zorder=3,
                        )

                        # 3) Garis DXF asli (kontur) teroverlay paling atas
                        _plot_dxf_overlay_2d(_ax_ortho, contours, color="black", linewidth=0.5, alpha=0.85)

                        # 4) Boundary area kajian
                        _bx_o, _by_o = boundary.exterior.xy
                        _ax_ortho.plot(_bx_o, _by_o, color="magenta", linewidth=2.0, zorder=6, label="Boundary")

                        _ax_ortho.set_title(f"Peta Risiko Erosi/Sedimentasi + DXF di atas Orthophoto — {seg_label}",
                                             fontsize=11, fontweight="bold")
                        _ax_ortho.set_xlabel("Easting (m)")
                        _ax_ortho.set_ylabel("Northing (m)")
                        _ax_ortho.set_aspect("equal", adjustable="box")
                        _ax_ortho.legend(loc="upper right", fontsize=8)

                        # batasi tampilan ke extent area kajian (bukan seluruh orthophoto kalau jauh lebih luas)
                        _pad_x = (x_all.max() - x_all.min()) * 0.05
                        _pad_y = (y_all.max() - y_all.min()) * 0.05
                        _ax_ortho.set_xlim(x_all.min() - _pad_x, x_all.max() + _pad_x)
                        _ax_ortho.set_ylim(y_all.min() - _pad_y, y_all.max() + _pad_y)

                        st.pyplot(_fig_ortho, use_container_width=True)
                        plt.close(_fig_ortho)

                    st.caption(
                        "Orthophoto tidak di-reproject otomatis — pastikan sistem koordinatnya sama "
                        "dengan DXF. Kalau posisi orthophoto terlihat bergeser/tidak pas dengan garis "
                        "DXF, kemungkinan orthophoto dan DXF memakai datum/zona koordinat yang berbeda."
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

            st.session_state["analysis_method"] = analysis_method

            overflow_count = int(len(df_overflow))
            convergence_count = int(np.sum(convergence_zone & inside))
            boundary_area_ha = float(boundary.area) / 10000.0

            rekomendasi = generate_recommendation(
                design_type=design_type,
                seg_label=seg_label,
                analysis_method=analysis_method,
                erosion_ratio=float(erosion_ratio),
                erosion_area=float(erosion_area),
                sedimentation_area=float(sedimentation_area),
                max_zone=float(np.nanmax(zone_map)),
                overflow_count=overflow_count,
                convergence_count=convergence_count,
                boundary_area_ha=boundary_area_ha
            )

            hydro_text_for_ai = ""
            if hydraulics_result:
                hydro_text_for_ai = (
                    f"- Debit rencana (Rational Method): {hydraulics_result['q_design_m3s']:.3f} m3/s\n"
                    f"- Kedalaman & kecepatan normal (Manning's): {hydraulics_result['h_normal_m']:.2f} m, "
                    f"{hydraulics_result['v_normal_ms']:.2f} m/s\n"
                    f"- Rezim aliran: {hydraulics_result['flow_regime']}\n"
                    f"- Status freeboard: {'cukup' if hydraulics_result['freeboard_ok'] else 'KURANG'} "
                    f"({hydraulics_result['freeboard_m']:.2f} m tersedia, min. disarankan "
                    f"{hydraulics_result['min_freeboard_m']:.2f} m)"
                )

            ai_reco = generate_ai_recommendation(
                context={
                    "seg_label": seg_label,
                    "design_type": design_type,
                    "analysis_method": analysis_method,
                    "boundary_area_ha": boundary_area_ha,
                    "erosion_ratio": float(erosion_ratio),
                    "sedimentation_area": float(sedimentation_area),
                    "max_zone": float(np.nanmax(zone_map)),
                    "overflow_count": overflow_count,
                    "convergence_count": convergence_count,
                    "velocity_hulu": float(velocity_hulu),
                    "grain_size": float(grain_size),
                    "surface_rmse": "N/A (validasi RMSE permukaan belum tersedia di modul ini)",
                    "hydraulics_text": hydro_text_for_ai
                },
                rule_based=rekomendasi
            )

            st.markdown("#### Rekayasa & Rekomendasi Geoteknik")

            badge_color = {
                "red": "#4d0000",
                "orange": "#4d3800",
                "green": "#003d1a"
            }.get(rekomendasi["color"], "#222222")

            border_color = {
                "red": "#ff4d4d",
                "orange": "#ffb84d",
                "green": "#4dff88"
            }.get(rekomendasi["color"], "#888888")

            st.markdown(
                f"""
                <div style="
                    background:{badge_color};
                    border:1px solid {border_color};
                    border-radius:12px;
                    padding:16px 20px;
                    margin-bottom:10px;
                ">
                <div style="font-size:20px;font-weight:800;color:{border_color};">
                    {rekomendasi['status']}
                </div>
                <div style="margin-top:8px;color:#E6F4F1;font-size:15px;line-height:1.6;">
                    {ai_reco['narrative']}
                </div>
                <div style="margin-top:10px;font-size:11px;color:#9fb8b3;">
                    Sumber narasi: {ai_reco['source']}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Kalau narasi di atas masih rule-based (belum AI), tampilkan langkah
            # aktivasi secara JELAS & actionable di dalam app -- bukan cuma catatan
            # kecil di footer kartu, supaya gampang ditindaklanjuti tanpa harus buka
            # kode. Context yang dikirim ke AI di atas SUDAH berisi angka hasil
            # running nyata segmen ini (kecepatan, overflow, konvergensi, hasil
            # Manning's/Rational Method) -- begitu API key terpasang, narasi AI
            # otomatis "terkoneksi" dengan angka-angka tsb, tidak perlu ubah apa pun
            # lagi di sini.
            if ai_reco["source"].startswith("rule-based"):
                with st.expander("⚙ Aktifkan narasi AI (Claude) untuk rekomendasi ini", expanded=False):
                    st.markdown(
                        "Rekomendasi di atas masih dari mesin **rule-based** (aturan ambang deterministik). "
                        "Untuk mendapat narasi AI yang menyesuaikan angka hasil running tiap segmen "
                        "(kecepatan aliran, overflow, konvergensi, hasil Manning's/Rational Method — "
                        "semuanya sudah otomatis dikirim ke AI, tidak perlu setup tambahan selain API key), "
                        "aktifkan dengan langkah berikut:"
                    )
                    st.markdown(
                        "1. Buat/ambil API key di [console.anthropic.com](https://console.anthropic.com) "
                        "(butuh akun Anthropic + saldo/billing aktif).\n"
                        "2. Kalau app di-deploy di **Streamlit Community Cloud**: buka menu app → "
                        "**Settings → Secrets**, lalu tambahkan baris:\n"
                    )
                    st.code('anthropic_api_key = "sk-ant-xxxxxxxxxxxxxxxx"', language="toml")
                    st.markdown(
                        "3. Kalau dijalankan lokal: buat file `.streamlit/secrets.toml` di folder project "
                        "berisi baris yang sama seperti di atas.\n"
                        "4. Simpan, lalu **reboot/rerun app** — tidak perlu ubah kode apa pun, narasi AI "
                        "otomatis aktif begitu `st.secrets['anthropic_api_key']` terbaca."
                    )
                    st.caption(
                        "Jangan commit API key ke repository Git — selalu lewat Secrets/environment variable. "
                        "Kalau API key sudah dipasang tapi masih muncul rule-based, cek pesan error di "
                        "'[Catatan: narasi AI gagal diambil ...]' pada narasi di atas untuk detail sebabnya "
                        "(mis. saldo habis, model tidak tersedia, rate limit)."
                    )

            if rekomendasi["notes"]:
                with st.expander("Lihat indikator teknis pendukung"):
                    for n in rekomendasi["notes"]:
                        st.markdown(f"- {n}")

            st.markdown("**Rekomendasi rekayasa:**")
            for r in ai_reco["recommendations"]:
                st.markdown(f"- {r}")

            top10_overflow = df_overflow.head(10).copy()
            top10_overflow.insert(0, "Rank", np.arange(1, len(top10_overflow) + 1))
            top10_overflow.insert(1, "ID_Titik", [f"{_seg_prefix}-{i:02d}" for i in top10_overflow["Rank"]])
            top10_overflow.insert(2, "Segmen", seg_label)

            st.session_state["segment_results"][sid] = {
                "label": seg_label,
                "design_type": design_type,
                "analysis_method": analysis_method,
                "erosion_area": float(erosion_area),
                "sedimentation_area": float(sedimentation_area),
                "max_zone": float(np.nanmax(zone_map)),
                "erosion_ratio": float(erosion_ratio),
                "overflow_count": overflow_count,
                "convergence_count": convergence_count,
                "boundary_area_ha": boundary_area_ha,
                "top10_overflow": top10_overflow,
                "recommendation": rekomendasi,
                "ai_recommendation": ai_reco,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
                "zone_map": zone_map,
                "sediment_map": sediment_map,
                "boundary": boundary,
                "cell_area": cell_area,
                "inside": inside,
                "contour_diag": contour_diag,
                "grain_size_mm": float(grain_size),
                "velocity_hulu": float(velocity_hulu),
                "velocity_field": velocity_field,
                "tau_critical": float(tau_critical),
                "rain_factor": float(rain_factor),
                "online_rainfall": st.session_state.get("online_rainfall"),
                "hydraulics_result": hydraulics_result,
                "use_hydraulics": use_hydraulics,
                "online_rainfall_meta": st.session_state.get("online_rainfall_meta"),
                "slope": slope,
                "flow_density": flow_density,
                "rho_water": float(rho_water),
                "rho_soil": float(rho_soil),
                "erodibility_M": float(erodibility_M),
                "flow_weight": float(flow_weight),
                "flow_depth": float(flow_depth),
                "valid_mask": valid,
                "orthophoto": orthophoto_data,
                "contours": contours,
                "flow_paths": flow_paths,
                "is_sub_segment": is_sub_segment,
                "x_mesh": x_mesh,
                "y_mesh": y_mesh,
                "z_mesh": z_mesh,
                "valid_triangles": valid_triangles,
                "vertical_exaggeration": float(vertical_exaggeration),
            }

        # ================= SCENE 3D GABUNGAN (SEMUA SEGMEN) =================
        # Sub-segmen sekarang berbagi terrain (kontur DXF) yang sama dengan Segmen 1
        # (Main) -- lihat perubahan di bagian upload DXF & fallback kontur_dxf di
        # atas. Karena semuanya berada di sistem koordinat yang SAMA, di sini semua
        # segmen digambar BERSAMA dalam SATU scene 3D (bukan satu scene per segmen
        # seperti sebelumnya, yang membuat Main & sub-boundary terlihat seperti dua
        # simulasi terpisah/nyambung asal-asalan).
        #
        # PERBAIKAN: sebelumnya hanya terrain Main yang digambar (colorscale polos
        # "Earth", tanpa mewakili tingkat risiko erosi sama sekali) + garis outline
        # boundary tiap segmen -- jadi kontur rainbow (risiko erosi) segmen ke-2 dst
        # TIDAK pernah benar-benar tampil di scene gabungan ini, cuma di chart
        # individualnya yang terpisah (padahal chart individual itu sekarang
        # disembunyikan begitu ada >1 segmen, lihat catatan di atas). Sekarang tiap
        # segmen (termasuk Main) digambar sebagai Mesh3d rainbow-nya SENDIRI
        # (x_mesh/y_mesh/z_mesh/valid_triangles/zone_map hasil analisis segmen
        # tsb, disimpan di segment_results) -- jadi kontur rainbow-nya benar-benar
        # MENYATU dalam satu scene, bukan cuma outline+terrain polos.
        _seg_results_now = st.session_state.get("segment_results", {})

        # PERBAIKAN: sebelumnya syaratnya ">= 1" -- artinya Scene 3D Gabungan
        # SELALU muncul walau proyek cuma punya 1 segmen (Main saja, tanpa
        # sub-boundary tambahan), padahal maksud desainnya (lihat catatan di
        # atas & di bagian chart individual) adalah: kalau cuma 1 segmen,
        # cukup chart individualnya saja yang tampil -- Scene 3D Gabungan
        # hanya relevan/perlu kalau memang ada 2 segmen atau lebih untuk
        # digabung.
        if len(_seg_results_now) >= 2:

            st.markdown("---")
            st.subheader("Scene 3D Gabungan — Semua Segmen (Main + Sub-Boundary)")
            st.caption(
                "Semua segmen (Main & sub-boundary) digabung jadi SATU output analisa 3D "
                "yang sama — mesh rainbow (risiko erosi) tiap segmen digambar dalam satu "
                "scene, bukan simulasi terpisah-pisah. Urutan tampil: Segmen 1/Main "
                "digambar dulu, lalu tiap sub-boundary berikutnya (Segmen 2, 3, dst) "
                "MENIMPA area yang tumpang-tindih dengan segmen sebelumnya — jadi kalau "
                "sub-boundary berada di dalam area Main, yang tampil di situ murni hasil "
                "sub-boundary tsb, sama persis seperti tampilannya di chart individual "
                "(bukan campuran warna dua mesh transparan yang saling tembus pandang)."
            )

            _combo_palette = ["#00C8FF", "#FF7A00", "#B3FF00", "#FF3D9A", "#FFD400", "#8A2BE2", "#00FFA3"]

            fig_combo = go.Figure()

            _combo_colorscale = [[0, "green"], [0.5, "yellow"], [1, "red"]]
            # PERBAIKAN: cmin sebelumnya di-hardcode ke 0, padahal chart individual
            # (di atas) TIDAK memberi cmin/cmax sama sekali -- Plotly otomatis
            # menyesuaikan skala warna (green->yellow->red) ke rentang nilai
            # zone_map segmen itu sendiri (min aktual -> max aktual). Kalau nilai
            # minimum data sebenarnya bukan 0, cmin=0 di sini membuat pemetaan
            # warnanya bergeser dibanding chart individual -- itu sebabnya mesh
            # yang datanya identik terlihat beda warna antara chart individual &
            # Scene 3D Gabungan. Sekarang cmin/cmax gabungan dihitung dari
            # min/max aktual seluruh segmen, supaya konsisten dengan cara chart
            # individual menormalisasi warnanya.
            _combo_zmin = min(
                (float(np.nanmin(_r["zone_map"])) for _r in _seg_results_now.values()
                 if np.isfinite(_r["zone_map"]).any()),
                default=0.0
            )
            _combo_zmax = max(
                (float(np.nanmax(_r["zone_map"])) for _r in _seg_results_now.values()
                 if np.isfinite(_r["zone_map"]).any()),
                default=1.0
            )

            _seg_items_ordered = sorted(
                _seg_results_now.items(),
                key=lambda kv: kv[1].get("is_sub_segment", False)
            )

            def _sanitize_polygonal(geom):
                """Pastikan geometry boundary benar2 Polygon/MultiPolygon yang valid
                sebelum dipakai buat uji containment massal (vectorized.contains).
                PERBAIKAN ("Segmen 2 kelihatannya tidak benar2 menimpa Main di Scene
                3D Gabungan, hasilnya malah kayak Main doang"): boundary hasil
                auto-alpha-shape/`make_valid()` kadang jadi GeometryCollection
                campuran (bukan murni Polygon) kalau geometrinya rumit/self-
                intersecting -- `vectorized.contains()` bisa gagal diam-diam pada
                geometry semacam ini (exception ketangkep try/except lalu di-skip),
                sehingga triangle Main di area situ TIDAK JADI terpotong & tetap
                tumpang-tindih dengan mesh Segmen 2 di ketinggian z yang sama persis
                -> z-fighting, terlihat seolah cuma Main yang tampil. `.buffer(0)`
                + filter hanya bagian Polygon/MultiPolygon-nya menghilangkan celah
                ini di hampir semua kasus."""
                if geom is None:
                    return None
                try:
                    geom = geom.buffer(0)
                except Exception:
                    return geom
                if geom.is_empty:
                    return None
                if geom.geom_type in ("Polygon", "MultiPolygon"):
                    return geom
                if geom.geom_type == "GeometryCollection":
                    polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                    if not polys:
                        return None
                    try:
                        return unary_union(polys)
                    except Exception:
                        return polys[0]
                return None

            # Kontrol lebar transisi warna antar segmen (0 = potongan tegas seperti
            # sebelumnya). Ditaruh di sini (bukan di form input tiap segmen) karena
            # ini murni pengaturan TAMPILAN Scene 3D Gabungan, bukan parameter
            # analisis segmen manapun.
            _transition_width = st.slider(
                "Lebar transisi warna antar segmen di Scene 3D Gabungan (m)",
                0.0, 20.0, 3.0, 0.5,
                key="combo_color_transition_width",
                help=(
                    "Supaya batas antar Main & sub-segmen tidak terlihat terlalu tegas/"
                    "kotak, warna rainbow Main di sekitar TEPI LUAR boundary sub-segmen "
                    "dibaurkan secara gradasi menuju warna sub-segmen tsb (bukan lompatan "
                    "warna tiba-tiba). Set ke 0 untuk kembali ke potongan tegas apa adanya."
                )
            )

            # PERBAIKAN TOTAL ("hasil individual vs gabungan beda, & rainbow sub-boundary
            # ketutupan rainbow Main"): sebelumnya overlap antar segmen "diakalin" pakai
            # kombinasi z-lift (mengangkat mesh sub-boundary sedikit di atas Main) +
            # opacity Main diturunkan supaya sub-boundary "kelihatan lebih menonjol" --
            # ini bikin warnanya jadi campuran/beda dari chart individual (yang
            # opacity-nya 1.0 apa adanya), dan tetap rawan terlihat aneh tergantung
            # sudut kamera (z-fighting parsial, warna nge-blend, dst).
            #
            # Sekarang diganti pendekatan yang lebih benar & simpel: segmen yang digambar
            # LEBIH DULU (Main), pada bagian mesh yang beririsan dengan boundary segmen
            # yang digambar BELAKANGAN (sub-boundary), TRIANGLE MESH-nya di situ DIBUANG
            # sama sekali sebelum dirender (bukan cuma ditumpuk transparan di baliknya) --
            # jadi Main benar-benar "ditimpa" oleh Segmen 2, dst, PERSIS seperti yang
            # diminta. Tidak ada lagi overlap/z-fighting sama sekali, opacity semua mesh
            # kembali ke 1.0 (identik dgn chart individual), dan z-lift buatan sudah
            # tidak diperlukan lagi.
            for _i_seg, (_sid_k, _res) in enumerate(_seg_items_ordered):
                _ve = _res.get("vertical_exaggeration", vertical_exaggeration)

                _tri = _res.get("valid_triangles")
                if _tri is None or len(_tri) == 0:
                    continue
                _tri = np.asarray(_tri)

                # segmen (bukan cuma boundary-nya) yang digambar SETELAH segmen ini di
                # urutan combo -> area itu yang akan "menimpa" segmen ini
                _later_items = [
                    (_sanitize_polygonal(_res_later.get("boundary")), _res_later)
                    for _j, (_sid_later, _res_later) in enumerate(_seg_items_ordered)
                    if _j > _i_seg and _res_later.get("boundary") is not None
                ]
                _later_items = [(b, r) for b, r in _later_items if b is not None]
                _later_boundaries = [b for b, _r in _later_items]

                _vx = _res["x_mesh"]
                _vy = _res["y_mesh"]
                # .copy() -- ini array warna TAMPILAN lokal utk chart gabungan saja,
                # jangan sampai menimpa zone_map asli tersimpan (dipakai statistik/
                # ekspor & chart individual di tempat lain).
                _intensity = _res["zone_map"][_res["inside"]].copy()

                if _later_boundaries:
                    _tcx = (_vx[_tri[:, 0]] + _vx[_tri[:, 1]] + _vx[_tri[:, 2]]) / 3.0
                    _tcy = (_vy[_tri[:, 0]] + _vy[_tri[:, 1]] + _vy[_tri[:, 2]]) / 3.0
                    _covered = np.zeros(len(_tri), dtype=bool)
                    for _lb in _later_boundaries:
                        try:
                            _covered |= vectorized.contains(_lb, _tcx, _tcy)
                        except Exception:
                            # Fallback terakhir kalau vectorized.contains tetap gagal
                            # (geometry masih bermasalah walau sudah disanitasi) --
                            # uji manual per-titik pakai Shapely biasa. Lebih lambat,
                            # tapi memastikan potongan tetap terjadi (bukan diam-diam
                            # dilewati seperti sebelumnya, yang bikin Main & Segmen 2
                            # numpuk di ketinggian sama & terlihat seperti Main saja
                            # yang tampil).
                            try:
                                _covered |= np.array([
                                    _lb.contains(Point(_px, _py))
                                    for _px, _py in zip(_tcx, _tcy)
                                ])
                            except Exception:
                                st.warning(
                                    f"Gagal memotong overlap mesh untuk {_res['label']} "
                                    "terhadap salah satu segmen berikutnya — kemungkinan "
                                    "boundary segmen tsb geometrinya tidak valid. Overlap "
                                    "di area itu mungkin masih terlihat tumpang-tindih."
                                )
                    _tri = _tri[~_covered]

                    # ---- transisi warna gradasi di TEPI LUAR boundary segmen belakangan
                    # (mis. di sekitar tepi luar boundary Segmen 2, warna Main dibaurkan
                    # perlahan menuju warna Segmen 2), supaya batasnya tidak kelihatan
                    # kotak/tegas. Dikerjakan dgn beberapa "cincin" buffer bertingkat dari
                    # boundary segmen belakangan keluar sejauh _transition_width, tiap
                    # cincin membaurkan intensity vertex Main sedikit lebih kuat makin
                    # dekat ke boundary -- mendekati gradasi kontinu tanpa perlu hitungan
                    # jarak presisi per titik (yang jauh lebih berat dihitung). ----
                    if _transition_width > 0:
                        _n_rings = 6
                        for _lb, _res_later in _later_items:
                            _lvx = _res_later["x_mesh"]
                            _lvy = _res_later["y_mesh"]
                            _later_zone = _res_later["zone_map"][_res_later["inside"]]
                            if len(_later_zone) == 0:
                                continue
                            try:
                                _inner_ring = _lb.buffer(-max(_transition_width * 0.4, 0.1))
                                _edge_mask = vectorized.contains(_lb, _lvx, _lvy) & (
                                    ~vectorized.contains(_inner_ring, _lvx, _lvy)
                                )
                            except Exception:
                                _edge_mask = np.zeros_like(_lvx, dtype=bool)
                            if _edge_mask.any():
                                _target_val = float(np.nanmean(_later_zone[_edge_mask]))
                            else:
                                _target_val = float(np.nanmean(_later_zone))
                            if not np.isfinite(_target_val):
                                continue

                            for _k in range(_n_rings, 0, -1):
                                _frac_out = _k / _n_rings
                                _buf = _transition_width * _frac_out
                                try:
                                    _ring_poly = _lb.buffer(_buf)
                                    _in_ring_verts = vectorized.contains(_ring_poly, _vx, _vy)
                                except Exception:
                                    continue
                                _blend = 1.0 - _frac_out
                                _intensity[_in_ring_verts] = (
                                    _intensity[_in_ring_verts] * (1 - _blend)
                                    + _target_val * _blend
                                )

                if len(_tri) == 0:
                    continue

                # mesh rainbow (risiko erosi) segmen ini -- data & triangulasi PERSIS
                # sama dengan chart individualnya (cuma triangle yang ketutupan segmen
                # belakangan yang dibuang, & warna vertex di pita transisi dibaurkan),
                # digambar bersama di sini
                fig_combo.add_trace(
                    go.Mesh3d(
                        x=_res["x_mesh"],
                        y=_res["y_mesh"],
                        z=_res["z_mesh"] * _ve,
                        i=_tri[:, 0],
                        j=_tri[:, 1],
                        k=_tri[:, 2],
                        intensity=_intensity,
                        colorscale=_combo_colorscale,
                        cmin=_combo_zmin,
                        cmax=_combo_zmax,
                        opacity=1.0,
                        showscale=(_i_seg == 0),
                        colorbar=dict(title="Indeks Risiko Erosi") if _i_seg == 0 else None,
                        flatshading=False,
                        lighting=dict(
                            ambient=0.55,
                            diffuse=0.9,
                            specular=0.25,
                            roughness=0.85,
                            fresnel=0.15,
                        ),
                        lightposition=dict(x=200, y=200, z=400),
                        name=f"Mesh — {_res['label']}",
                        hovertemplate=f"{_res['label']}<br>x=%{{x:.1f}} y=%{{y:.1f}} z=%{{z:.1f}}<extra></extra>",
                    )
                )

            # ---- overlay garis DXF desain asli, SATU KALI (kontur dibagikan bersama
            # oleh semua segmen dari Segmen 1/Main), supaya tetap teroverlay di scene
            # gabungan -- sebelumnya overlay ini cuma muncul di chart individual per
            # segmen yang sudah disembunyikan begitu proyek punya >1 segmen. ----
            _main_res_for_overlay = next(
                (_r for _r in _seg_results_now.values() if not _r.get("is_sub_segment", False)),
                None
            )
            if _main_res_for_overlay is not None:
                _main_sid_for_overlay = st.session_state["segments"][0]
                _show_overlay_combo = st.session_state.get(f"dxf_overlay_3d_{_main_sid_for_overlay}", True)
                _overlay_contours_combo = _main_res_for_overlay.get("contours") or []
                if _show_overlay_combo and _overlay_contours_combo:
                    _ve_overlay = _main_res_for_overlay.get("vertical_exaggeration", vertical_exaggeration)
                    _MAX_DXF_OVERLAY_ENTITIES = 400
                    if len(_overlay_contours_combo) > _MAX_DXF_OVERLAY_ENTITIES:
                        _step = max(1, len(_overlay_contours_combo) // _MAX_DXF_OVERLAY_ENTITIES)
                        _overlay_contours_combo = _overlay_contours_combo[::_step]

                    _dxf_overlay_legend_shown_combo = False
                    for _c in _overlay_contours_combo:
                        if len(_c) < 1:
                            continue
                        _cx = [p[0] for p in _c]
                        _cy = [p[1] for p in _c]
                        _cz = [p[2] * _ve_overlay for p in _c]

                        if len(_c) == 1:
                            fig_combo.add_trace(
                                go.Scatter3d(
                                    x=_cx, y=_cy, z=_cz,
                                    mode="markers",
                                    marker=dict(size=3, color="#111111", symbol="circle"),
                                    name="DXF Desain Asli (overlay)",
                                    legendgroup="dxf_overlay_combo",
                                    showlegend=not _dxf_overlay_legend_shown_combo,
                                    hoverinfo="skip",
                                )
                            )
                        else:
                            fig_combo.add_trace(
                                go.Scatter3d(
                                    x=_cx, y=_cy, z=_cz,
                                    mode="lines",
                                    line=dict(color="#111111", width=2),
                                    name="DXF Desain Asli (overlay)",
                                    legendgroup="dxf_overlay_combo",
                                    showlegend=not _dxf_overlay_legend_shown_combo,
                                    hoverinfo="skip",
                                )
                            )
                        _dxf_overlay_legend_shown_combo = True

            for _i_seg, (_sid_k, _res) in enumerate(_seg_items_ordered):
                _ve = _res.get("vertical_exaggeration", vertical_exaggeration)
                _color = _combo_palette[_i_seg % len(_combo_palette)]

                _bx, _by = _res["boundary"].exterior.xy
                _bz_ref = float(np.nanmax(_res["grid_z"])) * _ve
                fig_combo.add_trace(
                    go.Scatter3d(
                        x=list(_bx), y=list(_by),
                        z=[_bz_ref] * len(_bx),
                        mode="lines",
                        line=dict(color=_color, width=4),
                        name=f"Boundary — {_res['label']}",
                    )
                )

                for _pi, _fp in enumerate(_res.get("flow_paths", [])):
                    _px, _py, _pz, _pd = _fp
                    if len(_px) < 2:
                        continue
                    _pz_arr = np.array(_pz)
                    _pd_arr = np.array(_pd)
                    _water_z = (_pz_arr + _pd_arr) * _ve
                    fig_combo.add_trace(
                        go.Scatter3d(
                            x=_px, y=_py, z=_water_z,
                            mode="lines",
                            line=dict(color=_color, width=4),
                            showlegend=(_pi == 0),
                            name=f"Aliran — {_res['label']}",
                        )
                    )

            fig_combo.update_layout(
                height=850,
                scene=dict(aspectmode="data"),
                # PERBAIKAN: sebelumnya legend dibiarkan di posisi default Plotly
                # (kanan-atas), TEPAT bertumpuk dengan colorbar rainbow (yang juga
                # nempel di kanan-atas) -- tulisan "Boundary — Segmen X" / "Aliran —
                # Segmen X" jadi saling tumpuk & sulit dibaca. Sekarang legend
                # dipindah ke kiri-atas (menjauh dari colorbar di kanan), dibuat
                # horizontal-wrap dgn background semi-transparan supaya tetap
                # terbaca di atas mesh 3D, dan font sedikit dikecilkan supaya tidak
                # makan banyak tempat kalau segmennya banyak.
                legend=dict(
                    itemsizing="constant",
                    x=0.01,
                    y=0.99,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(10,15,20,0.65)",
                    bordercolor="rgba(255,255,255,0.25)",
                    borderwidth=1,
                    font=dict(size=11),
                    tracegroupgap=2,
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )

            st.plotly_chart(fig_combo, use_container_width=True)

            # ---- ringkasan gabungan singkat (pelengkap, bukan pengganti detail per-segmen di bawah) ----
            _n_seg = len(_seg_results_now)
            _total_erosion = sum(v["erosion_area"] for v in _seg_results_now.values())
            _total_sediment = sum(v["sedimentation_area"] for v in _seg_results_now.values())
            _total_overflow = sum(v["overflow_count"] for v in _seg_results_now.values())

            st.markdown("**Ringkasan Gabungan (Semua Segmen)**")
            colcm1, colcm2, colcm3, colcm4 = st.columns(4)
            colcm1.metric("Jumlah Segmen", _n_seg)
            colcm2.metric("Total Area Erosi", f"{_total_erosion:.1f} m²")
            colcm3.metric("Total Area Sedimentasi", f"{_total_sediment:.1f} m²")
            colcm4.metric("Total Titik Overflow Risk", int(_total_overflow))
            st.caption(
                "Detail lengkap per segmen (peta risiko 2D, hidrolika, rekomendasi rekayasa, "
                "PDF report) tetap tersedia di bawah, per bagian masing-masing segmen."
            )


    # =====================================================
# CROSS SECTION TOOL
# =====================================================

if st.session_state.get("analysis_done", False):

    st.markdown("---")
    st.subheader("Cross Section Analysis")

    seg_results = st.session_state.get("segment_results", {})

    if seg_results:
        seg_options = {v["label"]: k for k, v in seg_results.items()}
        picked_label = st.selectbox(
            "Pilih segmen untuk Cross Section & Report",
            list(seg_options.keys())
        )
        picked_sid = seg_options[picked_label]
        active = seg_results[picked_sid]

        grid_x = active["grid_x"]
        grid_y = active["grid_y"]
        grid_z = active["grid_z"]
        zone_map = active["zone_map"]
        sediment_map = active["sediment_map"]

        st.session_state["active_segment_label"] = picked_label
        st.session_state["grid_x"] = grid_x
        st.session_state["grid_y"] = grid_y
        st.session_state["grid_z"] = grid_z
        st.session_state["zone_map"] = zone_map
        st.session_state["sediment_map"] = sediment_map
        st.session_state["erosion_area"] = active["erosion_area"]
        st.session_state["sedimentation_area"] = active["sedimentation_area"]
        st.session_state["max_zone"] = active["max_zone"]
        st.session_state["boundary"] = active["boundary"]
        st.session_state["cell_area"] = active["cell_area"]
        st.session_state["inside"] = active["inside"]
        st.session_state["analysis_method"] = active["analysis_method"]
    else:
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

    # =========================================================
    # ================= VALIDASI LAPANGAN (GROUND-TRUTH) =================
    # =========================================================
    # Mengikuti Bab 3.5-3.6 makalah EroSlope PERHAPI: confusion matrix 3x3
    # (Rendah/Sedang/Tinggi-Ekstrem), overall accuracy, precision/recall/F1
    # per kelas, dan Cohen's Kappa (Landis & Koch, 1977) untuk mengukur
    # kesesuaian klasifikasi model terhadap titik sampel observasi lapangan.

    from sklearn.metrics import (
        confusion_matrix as _sk_confusion_matrix,
        cohen_kappa_score as _sk_cohen_kappa,
        precision_recall_fscore_support as _sk_prfs,
    )

    st.markdown("---")
    st.subheader("Validasi Lapangan (Ground-Truth)")
    st.caption(
        "Masukkan titik sampel observasi lapangan (survey/inspeksi visual/patok kontrol) untuk "
        "menguji kesesuaian klasifikasi risiko model terhadap kondisi aktual — mengikuti metodologi "
        "confusion matrix & Cohen's Kappa pada makalah acuan (Landis & Koch, 1977)."
    )

    RISK_CLASSES = ["Rendah", "Sedang", "Tinggi/Ekstrem"]

    def _classify_score(score, low_hi=0.5, mid_hi=1.0):
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return None
        if score < low_hi:
            return "Rendah"
        elif score < mid_hi:
            return "Sedang"
        else:
            return "Tinggi/Ekstrem"

    seg_results_val = st.session_state.get("segment_results", {})

    if not seg_results_val:
        st.info("Jalankan analisis segmen dulu sebelum melakukan validasi lapangan.")
    else:
        val_seg_options = {v["label"]: k for k, v in seg_results_val.items()}
        val_picked_label = st.selectbox(
            "Segmen yang divalidasi",
            list(val_seg_options.keys()),
            key="val_segment_picker"
        )
        val_sid = val_seg_options[val_picked_label]
        val_seg = seg_results_val[val_sid]

        st.caption(
            f"Ambang klasifikasi memakai skema TARP yang sama dengan peta risiko: "
            f"Rendah (skor < 0.5), Sedang (0.5-1.0), Tinggi/Ekstrem (≥ 1.0). "
            f"Metode segmen ini: **{val_seg['analysis_method']}**."
        )

        default_gt = pd.DataFrame({
            "ID_Titik": ["GT-01", "GT-02", "GT-03"],
            "X": [float(np.nanmin(val_seg["grid_x"])) + 10] * 3,
            "Y": [float(np.nanmin(val_seg["grid_y"])) + 10] * 3,
            "Kelas_Observasi": ["Rendah", "Sedang", "Tinggi/Ekstrem"],
        })

        gt_key = f"ground_truth_table_{val_sid}"
        if gt_key not in st.session_state:
            st.session_state[gt_key] = default_gt

        st.write(
            "**Tabel titik sampel lapangan** — isi koordinat (X, Y dalam sistem koordinat DXF yang sama) "
            "dan kelas hasil pengamatan/inspeksi lapangan aktual di lokasi itu:"
        )
        edited_gt = st.data_editor(
            st.session_state[gt_key],
            num_rows="dynamic",
            column_config={
                "Kelas_Observasi": st.column_config.SelectboxColumn(
                    "Kelas_Observasi", options=RISK_CLASSES, required=True
                )
            },
            key=f"gt_editor_{val_sid}",
            use_container_width=True,
        )
        st.session_state[gt_key] = edited_gt

        gt_csv_upload = st.file_uploader(
            "Atau upload CSV titik sampel (kolom: ID_Titik, X, Y, Kelas_Observasi)",
            type=["csv"], key=f"gt_csv_{val_sid}"
        )
        if gt_csv_upload is not None:
            try:
                uploaded_gt = pd.read_csv(gt_csv_upload)
                required_cols = {"ID_Titik", "X", "Y", "Kelas_Observasi"}
                if required_cols.issubset(set(uploaded_gt.columns)):
                    st.session_state[gt_key] = uploaded_gt
                    edited_gt = uploaded_gt
                    st.success(f"{len(uploaded_gt)} titik sampel dimuat dari CSV.")
                else:
                    st.error(f"CSV harus punya kolom: {required_cols}")
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")

        if st.button("Jalankan Validasi", key=f"run_validation_{val_sid}"):

            gx, gy = val_seg["grid_x"], val_seg["grid_y"]
            zmap = val_seg["zone_map"]

            gx_flat = gx.ravel()
            gy_flat = gy.ravel()
            z_flat = zmap.ravel()

            y_true = []
            y_pred = []
            detail_rows = []

            for _, row in edited_gt.iterrows():
                try:
                    px, py = float(row["X"]), float(row["Y"])
                    obs_class = row["Kelas_Observasi"]
                except (ValueError, TypeError, KeyError):
                    continue
                if obs_class not in RISK_CLASSES:
                    continue

                dist2 = (gx_flat - px) ** 2 + (gy_flat - py) ** 2
                nearest_idx = np.nanargmin(dist2)
                nearest_score = z_flat[nearest_idx]
                pred_class = _classify_score(nearest_score)

                if pred_class is None:
                    continue  # titik di luar boundary (NaN) -> tidak bisa divalidasi

                y_true.append(obs_class)
                y_pred.append(pred_class)
                detail_rows.append({
                    "ID_Titik": row.get("ID_Titik", "-"),
                    "X": px, "Y": py,
                    "Kelas_Observasi": obs_class,
                    "Skor_Model": round(float(nearest_score), 3),
                    "Kelas_Prediksi": pred_class,
                    "Cocok": "Cocok" if obs_class == pred_class else "Tidak",
                })

            if len(y_true) < 2:
                st.error(
                    "Minimal 2 titik sampel yang valid (berada di dalam boundary area, kelas terisi) "
                    "diperlukan untuk validasi."
                )
            else:
                cm = _sk_confusion_matrix(y_true, y_pred, labels=RISK_CLASSES)
                overall_acc = float(np.trace(cm)) / float(np.sum(cm))
                kappa = _sk_cohen_kappa(y_true, y_pred, labels=RISK_CLASSES)
                precision, recall, f1, support = _sk_prfs(
                    y_true, y_pred, labels=RISK_CLASSES, zero_division=0
                )

                def _kappa_interpretation(k):
                    if k < 0:
                        return "Poor (lebih buruk dari acak)"
                    elif k < 0.20:
                        return "Slight"
                    elif k < 0.40:
                        return "Fair"
                    elif k < 0.60:
                        return "Moderate"
                    elif k < 0.80:
                        return "Substantial"
                    else:
                        return "Almost perfect agreement"

                st.markdown("#### Hasil Validasi")
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric("Overall Accuracy", f"{overall_acc*100:.1f}%")
                vcol2.metric("Cohen's Kappa (κ)", f"{kappa:.2f}")
                vcol3.metric("Interpretasi (Landis & Koch, 1977)", _kappa_interpretation(kappa))

                st.markdown("**Confusion Matrix** (baris = observasi lapangan, kolom = prediksi model)")
                cm_df = pd.DataFrame(cm, index=RISK_CLASSES, columns=RISK_CLASSES)
                cm_df["Total"] = cm_df.sum(axis=1)
                st.dataframe(cm_df, use_container_width=True)

                st.markdown("**Precision, Recall, F1-Score per kelas**")
                prfs_df = pd.DataFrame({
                    "Kelas": RISK_CLASSES,
                    "Precision": np.round(precision, 3),
                    "Recall": np.round(recall, 3),
                    "F1-Score": np.round(f1, 3),
                    "Jumlah Sampel": support,
                })
                st.dataframe(prfs_df, use_container_width=True)

                st.markdown("**Detail per titik sampel**")
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

                if kappa < 0.60:
                    st.warning(
                        "Kappa < 0.60 menunjukkan kesesuaian model terhadap lapangan masih lemah/moderate. "
                        "Pertimbangkan kalibrasi ulang parameter (grain_size, tau_critical, erodibility_M) "
                        "atau tambah titik sampel lapangan sebelum dipakai sebagai dasar keputusan operasional."
                    )

                # simpan hasil ke segment untuk disertakan di report PDF
                st.session_state["segment_results"][val_sid]["validation"] = {
                    "overall_accuracy": overall_acc,
                    "kappa": float(kappa),
                    "kappa_interpretation": _kappa_interpretation(kappa),
                    "confusion_matrix": cm_df,
                    "prfs": prfs_df,
                    "detail": pd.DataFrame(detail_rows),
                    "n_samples": len(y_true),
                }
                st.success("Hasil validasi tersimpan dan akan otomatis disertakan di Executive Report.")

        # ================= BANDINGKAN KETIGA METODE (Hjulström vs Shields vs Partheniades) =================
        st.markdown("---")
        st.markdown("#### Bandingkan Ketiga Metode Erosion Assessment")
        st.caption(
            "Menjalankan Hjulström, Shields, dan Partheniades+Flow Accumulation sekaligus pada segmen "
            "yang sama, lalu membandingkan akurasi klasifikasi masing-masing terhadap titik sampel "
            "lapangan yang sama — mengikuti struktur Tabel 5 & Gambar 8 pada makalah acuan."
        )

        if st.button("Jalankan Perbandingan 3 Metode", key=f"run_compare3_{val_sid}"):

            gt_table_cmp = st.session_state.get(gt_key, default_gt)
            valid_gt_rows = [
                r for _, r in gt_table_cmp.iterrows()
                if r.get("Kelas_Observasi") in RISK_CLASSES
            ]

            if len(valid_gt_rows) < 2:
                st.error("Isi minimal 2 titik sampel lapangan (di bagian atas) sebelum membandingkan metode.")
            else:
                gx_c, gy_c = val_seg["grid_x"], val_seg["grid_y"]
                slope_c = val_seg["slope"]
                flow_density_c = val_seg["flow_density"]
                velocity_field_c = val_seg["velocity_field"]
                valid_mask_c = val_seg["valid_mask"]
                grain_size_c = val_seg["grain_size_mm"]
                rho_water_c = val_seg["rho_water"]
                rho_soil_c = val_seg["rho_soil"]
                tau_critical_c = val_seg["tau_critical"]
                erodibility_M_c = val_seg["erodibility_M"]
                flow_weight_c = val_seg["flow_weight"]
                flow_depth_c = val_seg["flow_depth"]

                methods_to_run = ["Hjulstrom Diagram", "Shields Diagram", "Partheniades + Flow Accumulation"]
                comparison_rows = []
                zone_maps_by_method = {}

                for method_name in methods_to_run:
                    t0 = time.time()
                    zmap_c = np.full(gx_c.shape, np.nan)

                    if method_name == "Hjulstrom Diagram":
                        zmap_c[valid_mask_c] = np.vectorize(hjulstrom_zone, otypes=[float])(
                            velocity_field_c[valid_mask_c], grain_size_c
                        )
                    elif method_name == "Shields Diagram":
                        zmap_c[valid_mask_c] = np.vectorize(shields_zone, otypes=[float])(
                            velocity_field_c[valid_mask_c], slope_c[valid_mask_c],
                            rho_water_c, rho_soil_c, grain_size_c
                        )
                    else:
                        risk_map_c = partheniades_erosion(
                            slope=slope_c, flow_density=flow_density_c, rho_water=rho_water_c,
                            flow_depth=flow_depth_c, tau_critical=tau_critical_c,
                            erodibility_M=erodibility_M_c, flow_weight=flow_weight_c
                        )
                        zmap_c = risk_map_c.copy()
                        zmap_c = zmap_c / (np.nanpercentile(zmap_c, 99) + 1e-9)
                        zmap_c = np.clip(zmap_c, 0, 2)

                    zmap_c[~valid_mask_c] = np.nan
                    elapsed = time.time() - t0
                    zone_maps_by_method[method_name] = zmap_c

                    # validasi metode ini terhadap titik sampel yang sama
                    gx_flat_c = gx_c.ravel()
                    gy_flat_c = gy_c.ravel()
                    z_flat_c = zmap_c.ravel()
                    yt, yp = [], []
                    for r in valid_gt_rows:
                        try:
                            px, py = float(r["X"]), float(r["Y"])
                        except (ValueError, TypeError):
                            continue
                        dist2 = (gx_flat_c - px) ** 2 + (gy_flat_c - py) ** 2
                        nidx = np.nanargmin(dist2)
                        pcls = _classify_score(z_flat_c[nidx])
                        if pcls is None:
                            continue
                        yt.append(r["Kelas_Observasi"])
                        yp.append(pcls)

                    if len(yt) >= 2:
                        acc_m = float(np.mean([a == b for a, b in zip(yt, yp)]))
                        kappa_m = _sk_cohen_kappa(yt, yp, labels=RISK_CLASSES)
                    else:
                        acc_m, kappa_m = float("nan"), float("nan")

                    comparison_rows.append({
                        "Metode": method_name,
                        "Akurasi vs Sampel Lapangan (%)": round(acc_m * 100, 1) if not np.isnan(acc_m) else "-",
                        "Cohen's Kappa": round(kappa_m, 3) if not np.isnan(kappa_m) else "-",
                        "Skor Risiko Maks": round(float(np.nanmax(zmap_c)), 3),
                        "Rata-rata Skor Risiko": round(float(np.nanmean(zmap_c)), 3),
                        "Waktu Komputasi (detik)": round(elapsed, 3),
                    })

                comp_df = pd.DataFrame(comparison_rows)
                st.markdown("**Tabel Perbandingan (Tabel 5-equivalent)**")
                st.dataframe(comp_df, use_container_width=True)

                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    x=comp_df["Metode"],
                    y=[v if isinstance(v, (int, float)) else 0 for v in comp_df["Akurasi vs Sampel Lapangan (%)"]],
                    text=comp_df["Akurasi vs Sampel Lapangan (%)"],
                    textposition="outside",
                    marker_color=["#4C78A8", "#54A24B", "#E45756"],
                ))
                fig_cmp.update_layout(
                    title="Perbandingan Akurasi Klasifikasi Risiko 3 Metode (vs sampel lapangan Anda)",
                    yaxis_title="Akurasi (%)", yaxis_range=[0, 100], height=420
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                best_method = comp_df.loc[
                    comp_df["Akurasi vs Sampel Lapangan (%)"].apply(lambda v: v if isinstance(v, (int, float)) else -1).idxmax()
                ]
                st.info(
                    f"Metode dengan akurasi tertinggi terhadap sampel lapangan Anda saat ini: "
                    f"**{best_method['Metode']}** ({best_method['Akurasi vs Sampel Lapangan (%)']}%). "
                    "Catatan: hasil ini bergantung penuh pada jumlah & sebaran titik sampel yang diinput — "
                    "tambah titik sampel untuk kesimpulan yang lebih andal."
                )

                st.session_state["segment_results"][val_sid]["method_comparison"] = comp_df

    # ================= HELPER: render formula sebagai gambar (mathtext) =================
    def _render_formula_png(latex_expr, out_path, fontsize=15):
        f = plt.figure(figsize=(6.2, 0.9))
        f.text(0.02, 0.5, f"${latex_expr}$", fontsize=fontsize, va="center", ha="left")
        plt.axis("off")
        plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08, transparent=False)
        plt.close(f)
        return out_path

    # ================= HELPER: tabel TARP (Trigger Action Response Plan) =================
    def _tarp_rows(current_score):
        levels = [
            ("HIJAU (Normal)", 0.0, 0.5,
             "Skor risiko < 0.5 — kondisi stabil",
             "Inspeksi rutin bulanan, pantau curah hujan",
             "Pertahankan geometri drainase, tanpa aksi khusus",
             "Tim O&M lapangan"),
            ("KUNING (Waspada)", 0.5, 1.0,
             "Skor risiko 0.5-1.0 — erosi awal terdeteksi",
             "Inspeksi 2 minggu sekali, pasang patok monitoring",
             "Revegetasi, surface protection mat, perbaikan berm",
             "Pengawas lapangan + Geoteknik"),
            ("ORANYE (Siaga)", 1.0, 2.0,
             "Skor risiko 1.0-2.0 — erosi sedang, potensi berkembang",
             "Inspeksi mingguan, pasang alat ukur kecepatan aliran",
             "Riprap D50 150-300 mm, check dam, drop structure, turunkan kecepatan < 1 m/s",
             "Engineer Geoteknik + Manajer Proyek"),
            ("MERAH (Kritis)", 2.0, float("inf"),
             "Skor risiko > 2.0 — erosi ekstrem, risiko scouring/headcut/kegagalan drainase",
             "Monitoring harian/real-time, evakuasi area jika perlu",
             "Redesain drainase, concrete lining/reno mattress, detention pond, "
             "turunkan slope < 5% dan kecepatan < 1 m/s SEGERA",
             "Manajer Proyek + Ahli Geoteknik Bersertifikat"),
        ]
        rows = [["Level TARP", "Kriteria Trigger", "Aksi Monitoring", "Aksi Respons", "Penanggung Jawab"]]
        active_level_idx = None
        for i, (name, lo, hi, crit, monitor, action, pic) in enumerate(levels):
            mark = ""
            if lo <= current_score < hi:
                mark = "  ← KONDISI SAAT INI"
                active_level_idx = i
            rows.append([name + mark, crit, monitor, action, pic])
        return rows, active_level_idx

    # ================= HELPER: bangun peta erosi/sedimentasi per segmen =================
    def _build_segment_map_png(seg, out_path):
        """Peta risiko erosi/sedimentasi bergaya kartografi standar (mirip layout
        ArcGIS/QGIS): hillshade + kontur elevasi, overlay risiko, boundary,
        scale bar (sebelumnya diimpor tapi TIDAK PERNAH dipasang di peta),
        north arrow, graticule koordinat, neatline, dan title block."""

        gx, gy, gz = seg["grid_x"], seg["grid_y"], seg["grid_z"]
        zmap, smap, bnd = seg["zone_map"], seg["sediment_map"], seg["boundary"]
        ortho = seg.get("orthophoto")
        seg_contours = seg.get("contours")

        fig_s, ax_s = plt.subplots(figsize=(12, 8.5))
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(gz, vert_exag=3)

        extent = [np.nanmin(gx), np.nanmax(gx), np.nanmin(gy), np.nanmax(gy)]

        if ortho is not None:
            # Orthophoto sebagai latar; hillshade tidak dipakai lagi supaya foto asli
            # tidak tertutup abu-abu, dan risk map dibuat lebih transparan.
            ax_s.imshow(
                ortho["rgb"], extent=ortho["extent"], origin="upper", zorder=1
            )
            _risk_alpha = 0.55
            _sed_alpha = 0.40
        else:
            ax_s.imshow(
                hillshade, cmap="gray", alpha=0.45,
                extent=extent, origin="lower"
            )
            ax_s.contour(gx, gy, gz, levels=20, colors="black", linewidths=0.35, alpha=0.55)
            _risk_alpha = 0.65
            _sed_alpha = 0.50

        erosion_plot = ax_s.contourf(gx, gy, zmap, levels=15, cmap="RdYlGn_r", alpha=_risk_alpha)
        sedim_plot = ax_s.contourf(
            gx, gy, smap, levels=[0.5, 0.7, 0.85, 1], cmap="Blues", alpha=_sed_alpha
        )
        if ortho is not None and seg_contours:
            # Garis DXF asli digambar ulang sebagai overlay di atas orthophoto + risk map,
            # supaya bentuk kontur/desain DXF tetap kelihatan jelas di atas foto udara.
            _plot_dxf_overlay_2d(ax_s, seg_contours, color="black", linewidth=0.4, alpha=0.85)
        bx_s, by_s = bnd.exterior.xy
        ax_s.plot(bx_s, by_s, color="black" if ortho is None else "magenta", linewidth=2.2, label="Boundary Area")

        # ---- Graticule koordinat (garis bantu Easting/Northing bergaya peta teknik) ----
        ax_s.grid(True, which="major", linestyle="--", linewidth=0.4, color="grey", alpha=0.5)
        ax_s.tick_params(labelsize=8)
        ax_s.ticklabel_format(style="plain", axis="both")
        for label in ax_s.get_xticklabels():
            label.set_rotation(30)

        # ---- Colorbar ----
        cbar = plt.colorbar(erosion_plot, ax=ax_s, shrink=0.65, pad=0.02)
        cbar.set_label("Erosion Risk Index", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # ---- Legend (boundary + kategori sedimentasi) ----
        legend_handles = [
            Patch(facecolor="none", edgecolor="black", linewidth=2.2, label="Batas Area (Boundary)"),
            Patch(facecolor="#4d94ff", alpha=0.5, label="Potensi Sedimentasi"),
        ]
        ax_s.legend(
            handles=legend_handles, loc="upper left", fontsize=8,
            framealpha=0.9, edgecolor="black"
        )

        # ---- Scale Bar (sebelumnya diimpor tapi tidak pernah dipasang) ----
        try:
            scalebar = ScaleBar(
                1, units="m", location="lower right",
                box_alpha=0.8, color="black", box_color="white",
                font_properties={"size": 8}
            )
            ax_s.add_artist(scalebar)
        except Exception:
            pass

        # ---- North Arrow ----
        arrow_x = extent[1] - (extent[1] - extent[0]) * 0.06
        arrow_y_base = extent[2] + (extent[3] - extent[2]) * 0.08
        arrow_len = (extent[3] - extent[2]) * 0.08
        ax_s.annotate(
            "N",
            xy=(arrow_x, arrow_y_base + arrow_len),
            xytext=(arrow_x, arrow_y_base),
            ha="center", va="center", fontsize=12, fontweight="bold",
            arrowprops=dict(facecolor="black", edgecolor="black", width=3, headwidth=10, headlength=10)
        )

        # ---- Neatline (bingkai peta) ----
        for spine in ax_s.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.4)

        ax_s.set_title(
            f"Peta Risiko Erosi & Sedimentasi — {seg['label']}",
            fontsize=14, fontweight="bold", pad=14
        )
        ax_s.set_xlabel("Easting (m)", fontsize=9)
        ax_s.set_ylabel("Northing (m)", fontsize=9)
        ax_s.set_aspect("equal", adjustable="box")

        # ---- Title block kecil di bawah peta (koordinat sistem, sumber, tanggal) ----
        fig_s.text(
            0.5, 0.01,
            "Sistem Koordinat: mengikuti DXF sumber (proyeksi lokal/UTM sesuai input)  |  "
            "Sumber: Analisis Geoteknik Otomatis  |  Dibuat: " + pd.Timestamp.now().strftime("%d %b %Y"),
            ha="center", fontsize=7.5, color="#333333"
        )

        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig_s)
        return out_path

    # ================= HELPER: peta komposit gaya "figure ilmiah" (klasifikasi + grafik + 3D) =================
    def _build_segment_composite_map_png(seg, out_path):
        """Layout komposit satu halaman mengikuti gaya figure laporan geoteknik/tambang:
        - Panel besar kiri: peta klasifikasi risiko (diskrit, dengan kotak Keterangan/legend)
          + north arrow + scale bar + boundary.
        - Panel kanan atas: grafik luasan per kategori risiko (Ha).
        - Dua panel kanan tengah: tampilan hillshade & zoom area kritis (pengganti foto lapangan,
          karena foto drone/lapangan asli tidak tersedia di pipeline ini).
        - Panel kanan bawah: render 3D model permukaan (Model 3D) berwarna sesuai indeks risiko.
        Dibuat landscape (lebar > tinggi) supaya proporsinya pas dipasang di halaman landscape."""

        from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registrasi proyeksi 3d)

        gx, gy, gz = seg["grid_x"], seg["grid_y"], seg["grid_z"]
        zmap, smap, bnd = seg["zone_map"], seg["sediment_map"], seg["boundary"]
        inside_mask = seg.get("inside")

        # ---- Klasifikasi diskrit 4 kelas mengikuti ambang TARP (konsisten dgn seluruh laporan) ----
        class_bounds = [0, 0.5, 1.0, 2.0, np.inf]
        class_colors = ["#4CAF50", "#FFD54F", "#FF9800", "#D32F2F"]
        class_labels = ["Stabil (Hijau)", "Waspada (Kuning)", "Siaga (Oranye)", "Kritis (Merah)"]
        cmap_cls = ListedColormap(class_colors)
        norm_cls = BoundaryNorm(class_bounds, cmap_cls.N)

        fig = plt.figure(figsize=(15.5, 9.2))
        gspec = fig.add_gridspec(
            3, 3,
            width_ratios=[2.0, 0.9, 0.9],
            height_ratios=[1.0, 1.0, 1.0],
            wspace=0.28, hspace=0.42
        )

        # ============ PANEL UTAMA: PETA KLASIFIKASI ============
        ax_main = fig.add_subplot(gspec[:, 0])
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(gz, vert_exag=3)
        extent = [np.nanmin(gx), np.nanmax(gx), np.nanmin(gy), np.nanmax(gy)]

        ax_main.imshow(hillshade, cmap="gray", alpha=0.35, extent=extent, origin="lower")
        ax_main.contourf(gx, gy, zmap, levels=class_bounds, cmap=cmap_cls, norm=norm_cls, alpha=0.75)
        ax_main.contour(gx, gy, gz, levels=15, colors="black", linewidths=0.3, alpha=0.4)

        bx, by = bnd.exterior.xy
        ax_main.plot(bx, by, color="black", linewidth=2.0)

        # kotak "Keterangan" (legend) khas peta tematik ArcGIS, dengan judul
        legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=l)
                           for c, l in zip(class_colors, class_labels)]
        legend_handles.append(Patch(facecolor="none", edgecolor="black", linewidth=2.0, label="Batas Area"))
        leg = ax_main.legend(
            handles=legend_handles, loc="upper left", fontsize=8.5,
            title="Keterangan", title_fontsize=9.5, framealpha=0.95, edgecolor="black"
        )
        leg.get_frame().set_facecolor("white")

        try:
            scalebar = ScaleBar(1, units="m", location="lower right",
                                 box_alpha=0.85, color="black", box_color="white",
                                 font_properties={"size": 7.5})
            ax_main.add_artist(scalebar)
        except Exception:
            pass

        arrow_x = extent[1] - (extent[1] - extent[0]) * 0.07
        arrow_y0 = extent[2] + (extent[3] - extent[2]) * 0.06
        arrow_len = (extent[3] - extent[2]) * 0.07
        ax_main.annotate(
            "N", xy=(arrow_x, arrow_y0 + arrow_len), xytext=(arrow_x, arrow_y0),
            ha="center", va="center", fontsize=11, fontweight="bold",
            arrowprops=dict(facecolor="black", edgecolor="black", width=2.5, headwidth=9, headlength=9)
        )
        for spine in ax_main.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.3)

        ax_main.set_title(f"Peta Sebaran Potensi Erosi & Sedimentasi\n{seg['label']}", fontsize=12, fontweight="bold")
        ax_main.set_xlabel("Easting (m)", fontsize=8.5)
        ax_main.set_ylabel("Northing (m)", fontsize=8.5)
        ax_main.tick_params(labelsize=7.5)
        ax_main.set_aspect("equal", adjustable="box")

        # ============ PANEL KANAN ATAS: GRAFIK LUAS PER KATEGORI ============
        ax_bar = fig.add_subplot(gspec[0, 1:])
        cell_area = seg.get("cell_area", 1.0)
        if inside_mask is not None:
            vals = zmap[inside_mask]
        else:
            vals = zmap[~np.isnan(zmap)]
        areas_ha = []
        for lo, hi in zip(class_bounds[:-1], class_bounds[1:]):
            n_cell = np.sum((vals >= lo) & (vals < hi))
            areas_ha.append(n_cell * cell_area / 10000.0)
        ax_bar.bar(class_labels, areas_ha, color=class_colors, edgecolor="black", linewidth=0.6)
        ax_bar.set_title("Luas per Kategori Risiko (Ha)", fontsize=9.5, fontweight="bold")
        ax_bar.tick_params(axis="x", labelsize=6.5, rotation=18)
        ax_bar.tick_params(axis="y", labelsize=7)
        ax_bar.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(areas_ha):
            ax_bar.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

        # ============ 2 PANEL KANAN TENGAH: HILLSHADE & ZOOM TITIK KRITIS ============
        ax_hs = fig.add_subplot(gspec[1, 1])
        ax_hs.imshow(hillshade, cmap="gist_earth", extent=extent, origin="lower")
        ax_hs.plot(bx, by, color="black", linewidth=1.0)
        ax_hs.set_title("Topografi (Hillshade)", fontsize=8, fontweight="bold")
        ax_hs.set_xticks([]); ax_hs.set_yticks([])
        for spine in ax_hs.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(0.8)

        ax_zoom = fig.add_subplot(gspec[1, 2])
        top10 = seg.get("top10_overflow")
        if top10 is not None and len(top10) > 0:
            cx_, cy_ = float(top10.iloc[0]["X"]), float(top10.iloc[0]["Y"])
            zoom_span = (extent[1] - extent[0]) * 0.12
            ax_zoom.imshow(hillshade, cmap="gray", extent=extent, origin="lower", alpha=0.5)
            ax_zoom.contourf(gx, gy, zmap, levels=class_bounds, cmap=cmap_cls, norm=norm_cls, alpha=0.8)
            ax_zoom.plot(cx_, cy_, marker="*", color="black", markersize=14, markeredgecolor="white")
            ax_zoom.set_xlim(cx_ - zoom_span, cx_ + zoom_span)
            ax_zoom.set_ylim(cy_ - zoom_span, cy_ + zoom_span)
        ax_zoom.set_title("Zoom Titik Kritis", fontsize=8, fontweight="bold")
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(0.8)

        # ============ PANEL KANAN BAWAH: MODEL 3D ============
        ax3d = fig.add_subplot(gspec[2, 1:], projection="3d")
        step = max(1, gx.shape[0] // 80)
        gx_s, gy_s, gz_s, zmap_s = gx[::step, ::step], gy[::step, ::step], gz[::step, ::step], zmap[::step, ::step]
        face_colors = cmap_cls(norm_cls(np.nan_to_num(zmap_s, nan=0)))
        ax3d.plot_surface(
            gx_s, gy_s, gz_s, facecolors=face_colors,
            rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True
        )
        ax3d.set_title("Model 3D", fontsize=9, fontweight="bold")
        ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
        ax3d.view_init(elev=48, azim=-60)
        try:
            ax3d.set_box_aspect((1, 1, 0.35))
        except Exception:
            pass

        fig.suptitle(
            f"Analisis Sebaran Potensi Erosi & Sedimentasi — {seg['label']} "
            f"({pd.Timestamp.now().strftime('%d %B %Y')})",
            fontsize=11, fontweight="bold", y=0.995
        )

        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    if st.session_state.get("analysis_done", False) and st.button("GENERATE EXECUTIVE REPORT"):

        seg_results_all = st.session_state.get("segment_results", {})

        if not seg_results_all:
            st.error("Belum ada hasil analisis segmen yang tersimpan. Jalankan analisis dulu.")
        else:
            st.info(f"Menyusun laporan teknis untuk {len(seg_results_all)} segmen...")

            # ================= PAGE / STYLE SETUP (format artikel jurnal, A4; peta = landscape) =================
            PAGE_W, PAGE_H = A4
            LAND_W, LAND_H = landscape(A4)
            MARGIN = 2.1 * cm
            CONTENT_W = PAGE_W - 2 * MARGIN
            LAND_CONTENT_W = LAND_W - 2 * MARGIN

            pdf_file = "Executive_Report.pdf"

            frame_portrait = Frame(
                MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN - 0.4 * cm,
                id="portrait_frame", topPadding=0.4 * cm
            )
            frame_landscape = Frame(
                MARGIN, MARGIN, LAND_W - 2 * MARGIN, LAND_H - 2 * MARGIN - 0.4 * cm,
                id="landscape_frame", topPadding=0.4 * cm
            )

            def _header_footer_portrait(canvas, doc_):
                _draw_header_footer(canvas, doc_, PAGE_W, PAGE_H)

            def _header_footer_landscape(canvas, doc_):
                _draw_header_footer(canvas, doc_, LAND_W, LAND_H)

            doc = BaseDocTemplate(
                pdf_file,
                pagesize=A4,
                leftMargin=MARGIN, rightMargin=MARGIN,
                topMargin=2.3 * cm, bottomMargin=2.2 * cm,
                title="Laporan Teknis Analisis Erosi & Sedimentasi",
                author="Geotechnical Intelligence Platform"
            )
            doc.addPageTemplates([
                PageTemplate(id="Portrait", frames=[frame_portrait], pagesize=A4, onPage=_header_footer_portrait),
                PageTemplate(id="Landscape", frames=[frame_landscape], pagesize=landscape(A4), onPage=_header_footer_landscape),
            ])

            base = getSampleStyleSheet()
            styles = {
                "Title": ParagraphStyle(
                    "ArtTitle", parent=base["Title"], fontName="Helvetica-Bold",
                    fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=4,
                    textColor=colors.HexColor("#0B3D2E")
                ),
                "Subtitle": ParagraphStyle(
                    "ArtSubtitle", parent=base["Normal"], fontName="Helvetica",
                    fontSize=9.5, leading=13, alignment=TA_CENTER,
                    textColor=colors.HexColor("#555555")
                ),
                "H1": ParagraphStyle(
                    "ArtH1", parent=base["Heading1"], fontName="Helvetica-Bold",
                    fontSize=13.5, leading=17, spaceBefore=16, spaceAfter=6,
                    textColor=colors.HexColor("#0B3D2E")
                ),
                "H2": ParagraphStyle(
                    "ArtH2", parent=base["Heading2"], fontName="Helvetica-Bold",
                    fontSize=10.5, leading=14, spaceBefore=10, spaceAfter=5,
                    textColor=colors.HexColor("#12523D")
                ),
                "Body": ParagraphStyle(
                    "ArtBody", parent=base["BodyText"], fontName="Helvetica",
                    fontSize=9.3, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6
                ),
                "BodyItalic": ParagraphStyle(
                    "ArtBodyItalic", parent=base["BodyText"], fontName="Helvetica-Oblique",
                    fontSize=8.8, leading=12.5, alignment=TA_JUSTIFY,
                    textColor=colors.HexColor("#444444"), spaceAfter=6
                ),
                "Caption": ParagraphStyle(
                    "ArtCaption", parent=base["Normal"], fontName="Helvetica-Oblique",
                    fontSize=8, leading=11, alignment=TA_CENTER,
                    textColor=colors.HexColor("#666666"), spaceBefore=3, spaceAfter=10
                ),
                "Bullet": ParagraphStyle(
                    "ArtBullet", parent=base["BodyText"], fontName="Helvetica",
                    fontSize=9.1, leading=13, leftIndent=10, spaceAfter=3
                ),
                "TblHeader": ParagraphStyle(
                    "TblHeader", parent=base["Normal"], fontName="Helvetica-Bold",
                    fontSize=7.6, leading=9.5, textColor=colors.white, alignment=TA_LEFT
                ),
                "TblCell": ParagraphStyle(
                    "TblCell", parent=base["Normal"], fontName="Helvetica",
                    fontSize=7.4, leading=9.6, alignment=TA_LEFT
                ),
                "TblCellBold": ParagraphStyle(
                    "TblCellBold", parent=base["Normal"], fontName="Helvetica-Bold",
                    fontSize=7.4, leading=9.6, alignment=TA_LEFT
                ),
            }

            def _p(text, style="Body"):
                return Paragraph(text, styles[style])

            def _divider():
                return HRFlowable(
                    width="100%", thickness=0.8, color=colors.HexColor("#0B3D2E"),
                    spaceBefore=2, spaceAfter=10
                )

            def _fit_image(path, max_width=CONTENT_W, max_height=None):
                """Skala gambar proporsional (tanpa distorsi & tanpa sisa ruang kosong
                dari rasio yang dipaksakan) berdasarkan dimensi asli file."""
                try:
                    with PILImage.open(path) as im:
                        iw, ih = im.size
                except Exception:
                    return Image(path, width=max_width, height=max_width * 0.6)

                ratio = ih / iw if iw else 0.6
                w = max_width
                h = w * ratio
                if max_height and h > max_height:
                    h = max_height
                    w = h / ratio if ratio else max_width
                return Image(path, width=w, height=h)

            def _wrap_row(row, header=False, bold_cols=None):
                bold_cols = bold_cols or []
                out = []
                for ci, cell in enumerate(row):
                    txt = str(cell)
                    if header:
                        out.append(Paragraph(txt, styles["TblHeader"]))
                    elif ci in bold_cols:
                        out.append(Paragraph(txt, styles["TblCellBold"]))
                    else:
                        out.append(Paragraph(txt, styles["TblCell"]))
                return out

            def _build_table(rows, col_widths, bold_cols=None, highlight_row=None):
                """Tabel standar artikel: header hijau tua, isi ter-wrap rapi (tidak overflow),
                baris selang-seling, opsional highlight 1 baris (mis. status TARP aktif)."""
                wrapped = [_wrap_row(rows[0], header=True)]
                for r in rows[1:]:
                    wrapped.append(_wrap_row(r, bold_cols=bold_cols))

                t = Table(wrapped, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
                cmds = [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D2E")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F4")]),
                ]
                if highlight_row is not None:
                    cmds.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#FFE3B0")))
                t.setStyle(TableStyle(cmds))
                return t

            def _draw_header_footer(canvas, doc_, page_w, page_h):
                canvas.saveState()
                canvas.setStrokeColor(colors.HexColor("#0B3D2E"))
                canvas.setLineWidth(0.6)
                canvas.line(MARGIN, page_h - 1.5 * cm, page_w - MARGIN, page_h - 1.5 * cm)
                canvas.setFont("Helvetica", 7.5)
                canvas.setFillColor(colors.HexColor("#666666"))
                canvas.drawString(MARGIN, page_h - 1.3 * cm, "Laporan Teknis Analisis Erosi & Sedimentasi")
                canvas.drawRightString(
                    page_w - MARGIN, page_h - 1.3 * cm,
                    pd.Timestamp.now().strftime("%d %B %Y")
                )
                canvas.line(MARGIN, 1.5 * cm, page_w - MARGIN, 1.5 * cm)
                canvas.drawString(MARGIN, 1.15 * cm, "Geotechnical Intelligence Platform — Dokumen Hasil Analisis Otomatis")
                canvas.drawRightString(page_w - MARGIN, 1.15 * cm, f"Halaman {doc_.page}")
                canvas.restoreState()

            story = []

            rain_meta_global = st.session_state.get("online_rainfall_meta")
            rain_val_global = st.session_state.get("online_rainfall")

            # ===================== COVER / RINGKASAN EKSEKUTIF =====================
            story.append(Paragraph("LAPORAN TEKNIS ANALISIS EROSI & SEDIMENTASI", styles["Title"]))
            story.append(Paragraph("Evaluasi Geoteknik Sekat/Channel Berbasis Model Numerik DXF", styles["Subtitle"]))
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"Diterbitkan {pd.Timestamp.now().strftime('%d %B %Y, %H:%M')} WIB &nbsp;|&nbsp; "
                f"{len(seg_results_all)} segmen dianalisis &nbsp;|&nbsp; "
                f"Sumber hujan: {(rain_meta_global or {}).get('source', 'tidak tersedia')}",
                styles["Subtitle"]
            ))
            story.append(Spacer(1, 10))
            story.append(_divider())

            # --- ringkasan eksekutif naratif (abstrak) ---
            n_seg = len(seg_results_all)
            worst_seg = max(seg_results_all.values(), key=lambda s: s["max_zone"])
            avg_erosion_ratio = np.mean([s["erosion_ratio"] for s in seg_results_all.values()]) * 100
            n_reject = sum(
                1 for s in seg_results_all.values()
                if "REJECT" in (s.get("recommendation") or {}).get("status", "")
                or "KRITIS" in (s.get("recommendation") or {}).get("status", "")
            )
            story.append(Paragraph("RINGKASAN EKSEKUTIF", styles["H1"]))
            story.append(Paragraph(
                f"Laporan ini merangkum hasil evaluasi geoteknik otomatis terhadap {n_seg} segmen "
                f"sekat/channel berdasarkan pemodelan permukaan 3D dari data DXF, dikombinasikan dengan "
                f"analisis hidrologi-hidrolika dan salah satu dari tiga metode erosi (Hjulström, Shields, "
                f"atau Partheniades). Rata-rata area berpotensi erosi di seluruh segmen adalah "
                f"<b>{avg_erosion_ratio:.1f}%</b>. Segmen dengan indeks risiko tertinggi adalah "
                f"<b>{worst_seg['label']}</b> (indeks {worst_seg['max_zone']:.2f} dari skala 0-2). "
                f"Dari seluruh segmen, <b>{n_reject}</b> segmen berada pada status kritis/tidak "
                f"direkomendasikan dan memerlukan tindak lanjut prioritas. Detail metodologi, kalkulasi "
                f"titik kritis, dan rekomendasi rekayasa per segmen disajikan pada bagian berikut.",
                styles["Body"]
            ))
            story.append(Spacer(1, 6))

            # tabel ringkasan seluruh segmen (perbandingan)
            summary_header = ["Segmen", "Metode", "Erosi\n(Ha)", "Sedimentasi\n(Ha)", "Indeks\nRisiko Maks", "Level TARP"]
            summary_rows = [summary_header]
            for sid, seg in seg_results_all.items():
                tarp_rows_tmp, active_idx_tmp = _tarp_rows(seg["max_zone"])
                level_name = tarp_rows_tmp[active_idx_tmp + 1][0].split("  ←")[0] if active_idx_tmp is not None else "-"
                summary_rows.append([
                    seg["label"], seg["analysis_method"],
                    f"{seg['erosion_area']:.2f}", f"{seg['sedimentation_area']:.2f}",
                    f"{seg['max_zone']:.2f}", level_name
                ])

            story.append(Paragraph("Tabel 1. Ringkasan Perbandingan Antar Segmen", styles["H2"]))
            story.append(_build_table(
                summary_rows,
                col_widths=[CONTENT_W*0.22, CONTENT_W*0.26, CONTENT_W*0.13, CONTENT_W*0.15, CONTENT_W*0.13, CONTENT_W*0.11]
            ))
            story.append(Paragraph(
                "Sumber: hasil RUN ANALYSIS pada platform, tanggal sebagaimana tercantum pada header laporan.",
                styles["Caption"]
            ))
            story.append(PageBreak())

            # ===================== CROSS SECTION (jika ada) =====================
            if "section_distance" in st.session_state:
                fig_sec, ax_sec = plt.subplots(figsize=(9, 3.6))
                ax_sec.plot(st.session_state["section_distance"], st.session_state["section_elevation"], linewidth=1.8, color="#0B3D2E")
                ax_sec.fill_between(st.session_state["section_distance"], st.session_state["section_elevation"],
                                     alpha=0.12, color="#0B3D2E")
                ax_sec.set_title("Cross Section A-A'", fontsize=11, fontweight="bold")
                ax_sec.set_xlabel("Jarak (m)", fontsize=9)
                ax_sec.set_ylabel("Elevasi (m)", fontsize=9)
                ax_sec.grid(alpha=0.3, linestyle="--", linewidth=0.5)
                ax_sec.tick_params(labelsize=8)
                plt.tight_layout()
                plt.savefig("CrossSection.png", dpi=220)
                plt.close(fig_sec)

                story.append(Paragraph("PENAMPANG MELINTANG (CROSS SECTION A-A')", styles["H1"]))
                story.append(_divider())
                story.append(_fit_image("CrossSection.png", max_width=CONTENT_W))
                story.append(Paragraph("Gambar 1. Profil elevasi sepanjang garis penampang A-A' yang dipilih pada alat Cross Section.", styles["Caption"]))
                story.append(PageBreak())

            # ===================== PER-SEGMEN =====================
            for seg_i, (sid, seg) in enumerate(seg_results_all.items()):

                story.append(Paragraph(f"{seg_i + 1}. SEGMEN: {seg['label'].upper()}", styles["H1"]))
                story.append(_divider())

                # --- data & parameter input ---
                param_rows = [
                    ["Parameter", "Nilai"],
                    ["Jenis kondisi", seg["design_type"]],
                    ["Metode analisis", seg["analysis_method"]],
                    ["Luas area (Ha)", f"{seg['boundary_area_ha']:.2f}"],
                    ["Ukuran butir (grain size)", f"{seg['grain_size_mm']:.3f} mm"],
                    ["Kecepatan aliran representatif", f"{seg['velocity_hulu']:.3f} m/s"],
                    ["Tegangan geser kritis (tau_critical)", f"{seg['tau_critical']:.3f}"],
                    ["Faktor hujan ekstrem", f"{seg['rain_factor']:.2f}x"],
                    ["Hujan desain terpakai", f"{seg['online_rainfall']:.2f} mm/hari" if seg.get("online_rainfall") else "tidak tersedia"],
                    ["Sumber hujan", (seg.get('online_rainfall_meta') or {}).get('source', '-')],
                ]
                cd = seg.get("contour_diag") or {}
                if cd:
                    param_rows.append(["Titik kontur DXF terbaca", f"{cd.get('n_raw_points', '-')}"])
                    param_rows.append(["Titik blunder terdeteksi", f"{cd.get('n_blunder_points', 0)}"])

                story.append(Paragraph("Data & Parameter Input", styles["H2"]))
                story.append(_build_table(param_rows, col_widths=[CONTENT_W*0.45, CONTENT_W*0.55]))

                if cd.get("all_z_zero"):
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(
                        "<b>PERINGATAN:</b> Semua titik kontur input pada segmen ini memiliki elevasi Z=0. "
                        "Hasil surface 3D dan analisis pada segmen ini berpotensi TIDAK MEWAKILI kondisi "
                        "topografi aktual. Disarankan verifikasi ulang data DXF sumber.",
                        styles["Body"]
                    ))
                story.append(Spacer(1, 6))

                # --- hidrologi & hidrolika (jika diaktifkan) ---
                hydro = seg.get("hydraulics_result")
                if seg.get("use_hydraulics") and hydro:
                    story.append(Paragraph("Hidrologi &amp; Hidrolika (Rational Method + Manning's Equation)", styles["H2"]))
                    hydro_rows = [
                        ["Parameter", "Nilai"],
                        ["Intensitas hujan (Mononobe)", f"{hydro['intensity_mm_hr']:.1f} mm/jam"],
                        ["Debit rencana Q (Rational Method)", f"{hydro['q_design_m3s']:.3f} m3/s"],
                        ["Kedalaman normal h (Manning's)", f"{hydro['h_normal_m']:.3f} m"],
                        ["Kecepatan normal V (Manning's)", f"{hydro['v_normal_ms']:.3f} m/s"],
                        ["Froude Number", f"{hydro['froude']:.2f} - {hydro['flow_regime']}" if hydro.get("froude") else "-"],
                        ["Freeboard tersedia", f"{hydro['freeboard_m']:.2f} m"],
                        ["Freeboard minimum disarankan", f"{hydro['min_freeboard_m']:.2f} m"],
                        ["Status freeboard", "CUKUP" if hydro["freeboard_ok"] else "KURANG — perlu redesain tinggi/lebar channel"],
                    ]
                    story.append(_build_table(
                        hydro_rows, col_widths=[CONTENT_W*0.5, CONTENT_W*0.5],
                        highlight_row=8
                    ))
                    story.append(Paragraph(
                        "Kecepatan aliran &amp; kedalaman pada analisis erosi segmen ini dihitung dari hasil "
                        "Manning's Equation di atas (bukan input manual), diturunkan dari debit rencana Metode "
                        "Rasional dengan intensitas hujan estimasi rumus Mononobe.",
                        styles["BodyItalic"]
                    ))
                    story.append(Spacer(1, 6))

                # --- peta risiko (ringkas, portrait) ---
                map_path = f"Erosion_Map_{sid}.png"
                _build_segment_map_png(seg, map_path)
                story.append(Paragraph("Peta Risiko Erosi & Sedimentasi", styles["H2"]))
                story.append(_fit_image(map_path, max_width=CONTENT_W))
                story.append(Paragraph(
                    f"Gambar {seg_i + 2}. Peta risiko erosi (kontur merah-hijau), potensi sedimentasi (biru), "
                    f"hillshade topografi, dan boundary area untuk {seg['label']}. Versi peta komposit "
                    f"lengkap (klasifikasi + grafik + model 3D) disajikan pada lembar landscape berikut.",
                    styles["Caption"]
                ))

                # --- peta komposit (LANDSCAPE, mengikuti gaya figure referensi pengguna) ---
                composite_path = f"Composite_Map_{sid}.png"
                _build_segment_composite_map_png(seg, composite_path)

                story.append(NextPageTemplate("Landscape"))
                story.append(PageBreak())
                story.append(Paragraph(
                    f"LEMBAR PETA — SEBARAN POTENSI EROSI & SEDIMENTASI: {seg['label'].upper()}",
                    styles["H1"]
                ))
                story.append(_divider())
                story.append(_fit_image(composite_path, max_width=LAND_CONTENT_W, max_height=LAND_H - 5.5 * cm))
                story.append(Paragraph(
                    f"Gambar {seg_i + 2}a. Peta komposit {seg['label']}: klasifikasi risiko erosi/sedimentasi "
                    "4 kelas (kiri), grafik luas per kategori (kanan atas), topografi hillshade & zoom titik "
                    "kritis (kanan tengah), dan model permukaan 3D berwarna indeks risiko (kanan bawah).",
                    styles["Caption"]
                ))
                story.append(NextPageTemplate("Portrait"))
                story.append(PageBreak())

                # --- metodologi & formula ---
                story.append(Paragraph("Metodologi & Formula", styles["H2"]))
                if seg["analysis_method"] == "Hjulstrom Diagram":
                    story.append(Paragraph(
                        "Klasifikasi risiko erosi/deposisi menggunakan pendekatan diagram Hjulström, yang "
                        "membandingkan kecepatan aliran aktual terhadap ambang kecepatan erosi dan deposisi "
                        "sebagai fungsi diameter butir sedimen (d, dalam meter). <i>Catatan keterbatasan: "
                        "pendekatan power-law ini tidak menangkap efek kohesi pada material lempung/lanau "
                        "halus (&lt;0,1 mm), di mana kurva Hjulström asli menunjukkan partikel sangat halus "
                        "justru lebih tahan erosi karena kohesi.</i>",
                        styles["Body"]
                    ))
                    f1_path = f"formula_verosion_{sid}.png"
                    f2_path = f"formula_vdep_{sid}.png"
                    f3_path = f"formula_score_{sid}.png"
                    _render_formula_png(r"v_{erosion} = 0.1 \times d^{-0.4}", f1_path)
                    _render_formula_png(r"v_{deposition} = 0.01 \times d^{-0.2}", f2_path)
                    _render_formula_png(
                        r"Skor = clip\left(2 \times \frac{v - v_{deposition}}{v_{erosion} - v_{deposition}},\ 0,\ 2\right)",
                        f3_path
                    )
                    story.append(_fit_image(f1_path, max_width=CONTENT_W*0.55))
                    story.append(_fit_image(f2_path, max_width=CONTENT_W*0.55))
                    story.append(_fit_image(f3_path, max_width=CONTENT_W*0.85))
                    story.append(Paragraph(
                        "Skor 0 = zona deposisi dominan, skor 2 = zona erosi ekstrem. Nilai d dikonversi "
                        "dari mm ke meter sebelum dihitung.",
                        styles["Body"]
                    ))
                elif seg["analysis_method"] == "Shields Diagram":
                    story.append(Paragraph(
                        "Klasifikasi mobilitas sedimen menggunakan parameter Shields tak berdimensi (θ), yang "
                        "membandingkan gaya penggerak aliran terhadap gaya penahan berat butiran terendam "
                        "(Shields, 1936). Tegangan geser dasar dihitung memakai depth-slope product (dengan "
                        "kedalaman aliran h), bukan hanya kemiringan S — koreksi ini penting karena tanpa "
                        "suku h, τ₀ tidak bersatuan Pascal yang valid secara dimensional.",
                        styles["Body"]
                    ))
                    fs1_path = f"formula_tau0_shields_{sid}.png"
                    fs2_path = f"formula_theta_{sid}.png"
                    _render_formula_png(r"\tau_0 = \rho_w \times g \times h \times S", fs1_path)
                    _render_formula_png(
                        r"\theta = \frac{\tau_0}{(\rho_s - \rho_w) \times g \times D_{50}}",
                        fs2_path
                    )
                    story.append(_fit_image(fs1_path, max_width=CONTENT_W*0.55))
                    story.append(_fit_image(fs2_path, max_width=CONTENT_W*0.65))
                    story.append(Paragraph(
                        f"dengan τ₀ tegangan geser dasar aktual (N/m²), h = kedalaman aliran = "
                        f"{seg['flow_depth']:.2f} m, ρw dan ρs densitas air dan sedimen (kg/m³), g = 9,81 m/s², "
                        f"dan D50 diameter butiran median (mm). Klasifikasi: θ &lt; 0,03 → stabil/deposisi "
                        "(skor 0); 0,03 ≤ θ &lt; 0,06 → transisi (skor 1); θ ≥ 0,06 → erosi aktif (skor 2).",
                        styles["Body"]
                    ))
                elif seg["analysis_method"] == "Partheniades + Flow Accumulation":
                    story.append(Paragraph(
                        "Laju erosi dihitung sebagai fungsi linear selisih tegangan geser aktual terhadap "
                        "tegangan geser kritis material (Partheniades, 1965), dikombinasikan dengan bobot "
                        "flow accumulation ternormalisasi untuk merepresentasikan efek konsentrasi aliran.",
                        styles["Body"]
                    ))
                    fp1_path = f"formula_tau0_parth_{sid}.png"
                    fp2_path = f"formula_erate_{sid}.png"
                    fp3_path = f"formula_riskidx_{sid}.png"
                    _render_formula_png(r"\tau_0 = \rho_w \times g \times h \times S", fp1_path)
                    _render_formula_png(
                        r"E = M \times \left(\frac{\tau_0}{\tau_c} - 1\right),\ \ untuk\ \tau_0 > \tau_c",
                        fp2_path
                    )
                    _render_formula_png(
                        r"RiskIndex = clip\left(\frac{E \times FlowAccum_{norm}}{P_{99}(E \times FlowAccum_{norm})},\ 0,\ 2\right)",
                        fp3_path, fontsize=12
                    )
                    story.append(_fit_image(fp1_path, max_width=CONTENT_W*0.55))
                    story.append(_fit_image(fp2_path, max_width=CONTENT_W*0.75))
                    story.append(_fit_image(fp3_path, max_width=CONTENT_W*0.9))
                    story.append(Paragraph(
                        f"dengan h = kedalaman aliran = {seg['flow_depth']:.2f} m, M = koefisien erodibilitas "
                        f"= {seg['erodibility_M']:.4f} kg/m²·s, τc = tegangan geser kritis = "
                        f"{seg['tau_critical']:.3f} N/m². Apabila τ₀ ≤ τc, laju erosi E diasumsikan nol (zona "
                        "stabil/deposisi). RiskIndex dinormalisasi terhadap persentil ke-99 hasil kali laju "
                        "erosi dengan flow accumulation ternormalisasi, lalu di-clip ke rentang [0,2].",
                        styles["Body"]
                    ))
                else:
                    story.append(Paragraph(
                        f"Metode analisis yang digunakan pada segmen ini: <b>{seg['analysis_method']}</b>. "
                        "Formula detail mengikuti definisi standar metode tersebut sebagaimana diterapkan pada modul.",
                        styles["Body"]
                    ))

                # --- perhitungan titik paling kritis ---
                story.append(Paragraph("Perhitungan pada Titik Paling Kritis", styles["H2"]))
                top10 = seg["top10_overflow"]
                if top10 is not None and len(top10) > 0:
                    crit = top10.iloc[0]
                    if seg["analysis_method"] == "Hjulstrom Diagram":
                        d_m = seg["grain_size_mm"] / 1000.0 + 1e-6
                        v_ero_crit = 0.1 * (d_m ** -0.4)
                        v_dep_crit = 0.01 * (d_m ** -0.2)
                        v_actual = crit.get("Velocity_ms", float("nan"))
                        story.append(Paragraph(
                            f"Titik kritis berada pada koordinat X={crit['X']:.2f}, Y={crit['Y']:.2f}, "
                            f"elevasi (RL)={crit['RL']:.2f} m, dengan skor risiko = {crit.get('RiskScore', float('nan')):.2f}. "
                            f"Diameter butir d = {seg['grain_size_mm']:.3f} mm = {d_m:.6f} m. "
                            f"v_erosion = 0,1 × ({d_m:.6f})^-0,4 = <b>{v_ero_crit:.4f} m/s</b>. "
                            f"v_deposition = 0,01 × ({d_m:.6f})^-0,2 = <b>{v_dep_crit:.4f} m/s</b>. "
                            f"Kecepatan aliran aktual pada titik ini = <b>{v_actual:.4f} m/s</b>. "
                            + (
                                "Kecepatan aktual MELEBIHI ambang erosi → titik ini aktif tererosi."
                                if v_actual >= v_ero_crit else
                                "Kecepatan aktual berada di bawah ambang deposisi → titik ini cenderung deposisi."
                                if v_actual <= v_dep_crit else
                                "Kecepatan aktual berada di zona transisi (antara ambang deposisi dan erosi)."
                            ),
                            styles["Body"]
                        ))
                    elif seg["analysis_method"] == "Shields Diagram":
                        g_grav = 9.81
                        d_m = seg["grain_size_mm"] / 1000.0 + 1e-9
                        slope_at_crit = float(crit.get("Slope", 0.0))
                        h_for_crit = seg.get("flow_depth", 0.3)
                        tau0_crit = seg["rho_water"] * g_grav * h_for_crit * slope_at_crit
                        theta_crit = tau0_crit / (
                            (seg["rho_soil"] - seg["rho_water"]) * g_grav * d_m + 1e-9
                        )
                        theta_class = (
                            "stabil/deposisi (θ &lt; 0,03)" if theta_crit < 0.03 else
                            "transisi (0,03 ≤ θ &lt; 0,06)" if theta_crit < 0.06 else
                            "erosi aktif (θ ≥ 0,06)"
                        )
                        story.append(Paragraph(
                            f"Titik kritis pada koordinat X={crit['X']:.2f}, Y={crit['Y']:.2f}, "
                            f"RL={crit['RL']:.2f} m, kemiringan lokal S={slope_at_crit:.4f} m/m, "
                            f"kedalaman aliran h={h_for_crit:.2f} m. "
                            f"τ₀ = {seg['rho_water']:.0f} × 9,81 × {h_for_crit:.2f} × {slope_at_crit:.4f} = "
                            f"<b>{tau0_crit:.4f} N/m²</b>. "
                            f"θ = {tau0_crit:.4f} / [({seg['rho_soil']:.0f} − {seg['rho_water']:.0f}) × 9,81 × "
                            f"{d_m:.6f}] = <b>{theta_crit:.4f}</b>. "
                            f"Klasifikasi: <b>{theta_class}</b>.",
                            styles["Body"]
                        ))
                    elif seg["analysis_method"] == "Partheniades + Flow Accumulation":
                        g_grav = 9.81
                        slope_at_crit = float(crit.get("Slope", 0.0))
                        tau0_crit = seg["rho_water"] * g_grav * seg["flow_depth"] * slope_at_crit
                        tau_c = seg["tau_critical"]
                        if tau0_crit > tau_c:
                            e_rate = seg["erodibility_M"] * (tau0_crit / tau_c - 1.0)
                            e_text = f"<b>{e_rate:.5f} kg/m²·s</b> (aktif tererosi, τ₀ &gt; τc)"
                        else:
                            e_text = "0 kg/m²·s (τ₀ ≤ τc → zona stabil/deposisi)"
                        story.append(Paragraph(
                            f"Titik kritis pada koordinat X={crit['X']:.2f}, Y={crit['Y']:.2f}, "
                            f"RL={crit['RL']:.2f} m, kemiringan lokal S={slope_at_crit:.4f} m/m, "
                            f"kepadatan aliran={crit.get('FlowDensity', float('nan')):.3f}. "
                            f"τ₀ = {seg['rho_water']:.0f} × 9,81 × {seg['flow_depth']:.2f} × "
                            f"{slope_at_crit:.4f} = <b>{tau0_crit:.4f} N/m²</b> (τc = {tau_c:.3f} N/m²). "
                            f"Laju erosi E = {e_text}. "
                            f"Skor risiko akhir (setelah bobot flow accumulation & normalisasi) = "
                            f"<b>{crit.get('RiskScore', float('nan')):.2f}</b>.",
                            styles["Body"]
                        ))
                    else:
                        story.append(Paragraph(
                            f"Titik kritis berada pada koordinat X={crit['X']:.2f}, Y={crit['Y']:.2f}, "
                            f"elevasi (RL)={crit['RL']:.2f} m, kepadatan aliran={crit['FlowDensity']:.3f}.",
                            styles["Body"]
                        ))

                    # tabel 10 titik prioritas mitigasi
                    table_cols = [c for c in
                                  ["Rank", "ID_Titik", "X", "Y", "RL", "JenisRisikoDominan",
                                   "RiskScore", "Rekomendasi"]
                                  if c in top10.columns]
                    crit_rows = [table_cols] + top10[table_cols].head(10).round(3).astype(str).values.tolist()
                    n_cols = len(table_cols)
                    # lebar proporsional: kolom teks panjang (Rekomendasi) dapat porsi terbesar
                    base_w = {
                        "Rank": 0.05, "ID_Titik": 0.09, "X": 0.09, "Y": 0.09, "RL": 0.08,
                        "JenisRisikoDominan": 0.15, "RiskScore": 0.08, "Rekomendasi": 0.37
                    }
                    col_w = [CONTENT_W * base_w.get(c, 1.0/n_cols) for c in table_cols]

                    story.append(Paragraph(
                        "Sepuluh titik prioritas mitigasi, diurutkan berdasar kepadatan aliran, dengan "
                        "klasifikasi jenis risiko dominan dan rekomendasi spesifik per titik:",
                        styles["Body"]
                    ))
                    story.append(_build_table(crit_rows, col_widths=col_w))
                    story.append(Paragraph(
                        f"Tabel {seg_i + 2}. Titik prioritas mitigasi — {seg['label']}.",
                        styles["Caption"]
                    ))
                else:
                    story.append(Paragraph("Tidak ada titik overflow/kritis signifikan terdeteksi pada segmen ini.", styles["Body"]))

                # --- TARP ---
                story.append(Paragraph("TARP (Trigger Action Response Plan)", styles["H2"]))
                tarp_rows, active_idx = _tarp_rows(seg["max_zone"])
                tarp_col_w = [CONTENT_W*0.15, CONTENT_W*0.22, CONTENT_W*0.20, CONTENT_W*0.28, CONTENT_W*0.15]
                story.append(_build_table(
                    tarp_rows, col_widths=tarp_col_w,
                    highlight_row=(active_idx + 1) if active_idx is not None else None
                ))

                # --- validasi lapangan ---
                validation = seg.get("validation")
                if validation:
                    story.append(Paragraph("Validasi Lapangan (Ground-Truth)", styles["H2"]))
                    story.append(Paragraph(
                        f"Validasi terhadap {validation['n_samples']} titik sampel observasi lapangan: "
                        f"Overall Accuracy = <b>{validation['overall_accuracy']*100:.1f}%</b>, "
                        f"Cohen's Kappa (κ) = <b>{validation['kappa']:.2f}</b> "
                        f"(kategori: {validation['kappa_interpretation']}, mengikuti skala Landis & Koch, 1977).",
                        styles["Body"]
                    ))
                    cm_df_r = validation["confusion_matrix"].reset_index().rename(columns={"index": "Observasi\\Prediksi"})
                    cm_rows_r = [list(cm_df_r.columns)] + cm_df_r.astype(str).values.tolist()
                    n_cm_cols = len(cm_rows_r[0])
                    story.append(_build_table(cm_rows_r, col_widths=[CONTENT_W/n_cm_cols]*n_cm_cols))
                    story.append(Spacer(1, 6))

                    prfs_df_r = validation["prfs"]
                    prfs_rows_r = [list(prfs_df_r.columns)] + prfs_df_r.astype(str).values.tolist()
                    n_pr_cols = len(prfs_rows_r[0])
                    story.append(Paragraph("Precision, Recall, F1-Score per kelas:", styles["Body"]))
                    story.append(_build_table(prfs_rows_r, col_widths=[CONTENT_W/n_pr_cols]*n_pr_cols))
                else:
                    story.append(Paragraph(
                        "Validasi lapangan belum dijalankan untuk segmen ini. Disarankan menjalankan modul "
                        "Validasi Lapangan dengan minimal beberapa titik sampel ground-truth sebelum hasil ini "
                        "dipakai sebagai dasar keputusan operasional final.",
                        styles["BodyItalic"]
                    ))

                # --- perbandingan 3 metode ---
                method_comparison = seg.get("method_comparison")
                if method_comparison is not None and len(method_comparison) > 0:
                    story.append(Paragraph("Perbandingan 3 Metode Erosion Assessment", styles["H2"]))
                    mc_rows = [list(method_comparison.columns)] + method_comparison.astype(str).values.tolist()
                    n_mc_cols = len(mc_rows[0])
                    story.append(_build_table(mc_rows, col_widths=[CONTENT_W/n_mc_cols]*n_mc_cols))

                # --- rekomendasi (AI + rule-based status) ---
                story.append(Paragraph("Rekomendasi Rekayasa (Analisis Berbasis AI)", styles["H2"]))
                rekom = seg.get("recommendation") or {}
                ai_rekom = seg.get("ai_recommendation") or {}
                if rekom.get("status"):
                    story.append(Paragraph(f"<b>Status Klasifikasi (TARP):</b> {rekom['status']}", styles["Body"]))
                if ai_rekom.get("narrative"):
                    story.append(Paragraph(ai_rekom["narrative"].replace("\n", "<br/>"), styles["Body"]))
                for rec in ai_rekom.get("recommendations", []):
                    story.append(Paragraph(f"• {rec}", styles["Bullet"]))
                story.append(Paragraph(
                    f"<i>Sumber narasi: {ai_rekom.get('source', 'rule-based')}</i>",
                    styles["Caption"]
                ))

                story.append(PageBreak())

            # ===================== DISCLAIMER =====================
            story.append(Paragraph("CATATAN & KETERBATASAN", styles["H1"]))
            story.append(_divider())
            story.append(Paragraph(
                "Laporan ini dihasilkan otomatis dari model numerik berbasis data topografi (DXF) dan data "
                "hujan yang diinput/diambil secara online pada tanggal pembuatan laporan. Narasi rekomendasi "
                "pada tiap segmen dapat dihasilkan oleh model AI (Claude) berdasarkan angka hasil running "
                "aktual segmen tersebut; klasifikasi status/TARP tetap mengikuti aturan deterministik agar "
                "dapat diaudit. Akurasi hasil bergantung langsung pada kelengkapan dan ketelitian data survei "
                "sumber. Verifikasi lapangan oleh Ahli Geoteknik/Sumber Daya Air bersertifikat tetap "
                "diperlukan sebelum keputusan desain final diambil, khususnya untuk segmen yang mendapat "
                "peringatan kualitas data pada bagian Data & Parameter Input.",
                styles["Body"]
            ))

            story.append(Paragraph("REFERENSI METODOLOGI", styles["H2"]))
            for ref in [
                "Hjulström, F. (1935). Studies of the morphological activity of rivers as illustrated by the "
                "River Fyris. Bulletin of the Geological Institute of Uppsala, 25, 221-527.",
                "Shields, A. (1936). Anwendung der Ähnlichkeitsmechanik und der Turbulenzforschung auf die "
                "Geschiebebewegung. Mitteilungen der Preußischen Versuchsanstalt für Wasserbau und Schiffbau, Berlin.",
                "Partheniades, E. (1965). Erosion and deposition of cohesive soils. Journal of the Hydraulics "
                "Division, ASCE, 91(1), 105-139.",
                "Chow, V.T. (1959). Open-Channel Hydraulics. McGraw-Hill, New York. (persamaan Manning)",
                "Mononobe (dalam Suripin, 2004). Sistem Drainase Perkotaan yang Berkelanjutan — rumus "
                "intensitas hujan dari data hujan harian.",
                "Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological "
                "Measurement, 20(1), 37-46.",
                "Landis, J.R., & Koch, G.G. (1977). The measurement of observer agreement for categorical data. "
                "Biometrics, 33(1), 159-174.",
            ]:
                story.append(Paragraph(f"• {ref}", styles["Body"]))

            doc.build(story)

            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="Download Laporan Teknis Lengkap (PDF)",
                    data=f,
                    file_name="Laporan_Teknis_Erosi_Sedimentasi.pdf",
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



# =========================================================
# =============== HELPER: SLOPE MONITORING PDF PARSER =====
# =========================================================
import re as _re_smp
import tempfile as _tempfile_smp
import os as _os_smp

try:
    import fitz as _fitz_smp  # PyMuPDF
    _SMP_FITZ_OK = True
    _SMP_FITZ_ERR = None
except Exception as _e_fitz_smp:
    _SMP_FITZ_OK = False
    _SMP_FITZ_ERR = str(_e_fitz_smp)

_SMP_STATUS_MAP = [
    (_re_smp.compile(r'evakuasi', _re_smp.I), 'Evakuasi'),
    (_re_smp.compile(r'waspada', _re_smp.I), 'Waspada'),
    (_re_smp.compile(r'hati[\s-]?hati', _re_smp.I), 'Hati-hati'),
    (_re_smp.compile(r'stabi', _re_smp.I), 'Stabil'),
    (_re_smp.compile(r'tidak\s*terbaca', _re_smp.I), 'Tidak terbaca'),
]
_SMP_STATUS_RANK = {'Stabil': 0, 'Hati-hati': 1, 'Waspada': 2, 'Evakuasi': 3, 'Tidak terbaca': None}
_SMP_ID_FULL_RE = _re_smp.compile(r'^(PRS|PRW|GPS|RTS|SSRXT|FG)[-_ ]?[A-Za-z0-9]{1,8}$', _re_smp.IGNORECASE)

_SMP_STATUS_COLOR = {
    "Naik Drastis (>=2 level)": "#C0392B",
    "Naik 1 Level": "#E67E22",
    "Meningkat (level sama)": "#F1C40F",
    "Tetap / Stabil": "#27AE60",
    "Membaik": "#1F9D8A",
    "Berubah Keterbacaan": "#7F8C8D",
    "Titik Baru": "#8E44AD",
    "Titik Hilang": "#95A5A6",
}


def _smp_pdf_to_bbox_words(pdf_path):
    """Ekstrak kata + posisi (koordinat titik PDF, asal kiri-atas halaman) pakai PyMuPDF."""
    doc = _fitz_smp.open(pdf_path)
    try:
        page = doc[0]
        words = []
        for w in page.get_text("words"):
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            if text.strip() == "":
                continue
            words.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "text": text})
        return words
    finally:
        doc.close()


def _smp_get_page_size(pdf_path):
    doc = _fitz_smp.open(pdf_path)
    try:
        r = doc[0].rect
        return float(r.width), float(r.height)
    finally:
        doc.close()


def _smp_merge_tight_tokens(words, gap_thresh=1.6, y_thresh=1.2):
    words = sorted(words, key=lambda w: (round((w["y0"] + w["y1"]) / 2, 1), w["x0"]))
    merged = []
    cur = None
    for w in words:
        yc = (w["y0"] + w["y1"]) / 2
        if cur is not None:
            cur_yc = (cur["y0"] + cur["y1"]) / 2
            gap = w["x0"] - cur["x1"]
            if abs(yc - cur_yc) <= y_thresh and 0 <= gap <= gap_thresh:
                cur["text"] += w["text"]
                cur["x1"] = w["x1"]
                cur["y0"] = min(cur["y0"], w["y0"])
                cur["y1"] = max(cur["y1"], w["y1"])
                continue
        if cur is not None:
            merged.append(cur)
        cur = dict(w)
    if cur is not None:
        merged.append(cur)
    return merged


def _smp_normalize_id(raw_id):
    s = raw_id.upper().strip()
    s = _re_smp.sub(r'[\s_]+', '-', s)
    s = _re_smp.sub(r'-+', '-', s)
    return s.strip('-')


def _smp_classify_status(text):
    for pat, label in _SMP_STATUS_MAP:
        if pat.search(text):
            return label
    return None


def _smp_extract_points(pdf_path):
    words = _smp_pdf_to_bbox_words(pdf_path)
    words = _smp_merge_tight_tokens(words)
    points = {}
    for w in words:
        if not _SMP_ID_FULL_RE.match(w["text"]):
            continue
        yc = (w["y0"] + w["y1"]) / 2
        neighbors = [
            w2 for w2 in words
            if w2["x0"] >= w["x0"] and w2["x0"] <= w["x1"] + 320
            and abs((w2["y0"] + w2["y1"]) / 2 - yc) <= 3.2
        ]
        neighbors.sort(key=lambda w2: w2["x0"])
        local_text = " ".join(n["text"] for n in neighbors)
        m = _re_smp.search(r'([\-\d.]+)\s*mm\s*/?\s*day', local_text, _re_smp.IGNORECASE)
        if not m:
            continue
        val_raw = m.group(1)
        try:
            value = float(val_raw)
        except ValueError:
            value = None
        status = _smp_classify_status(local_text[m.end():m.end() + 40])
        if status is None:
            continue
        pid = _smp_normalize_id(w["text"])
        if not _re_smp.search(r'\d', pid):
            continue
        entry = {
            "ID Titik": pid, "Nilai (mm/day)": value, "Status": status,
            "_rank": _SMP_STATUS_RANK[status],
            "_x": round((w["x0"] + w["x1"]) / 2, 1), "_y": round(yc, 1),
        }
        if pid not in points:
            points[pid] = entry
    return points


# =========================================================
# ===== KALIBRASI SUMBU GRID PETA + KOORDINAT "Block N/E" =====
# =========================================================
# Masalah: kotak callout "AREA GEOTECHNICAL ISSUE" dihubungkan ke titik yang
# sebenarnya lewat leader line panjang -> posisi TEKS callout != posisi titik
# di peta. Solusinya: pakai koordinat "Block N/E" yang tertulis di dalam kotak
# callout itu sendiri, lalu konversi N/E -> piksel pakai kalibrasi sumbu grid
# peta (angka 0..21000 / -3000..12000 di pinggir peta), BUKAN posisi label ID.

def _smp_debug_dump_blocks(pdf_path, max_blocks=40):
    """Helper diagnostik: tampilkan teks mentah per blok PDF (dgn koordinatnya)
    supaya kamu bisa cek pola teks asli (format 'Block N/E', label ID, dst)
    dan menyesuaikan regex _RE_BLOCK_NE / _SMP_ID_FULL_RE kalau perlu."""
    doc = _fitz_smp.open(pdf_path)
    try:
        page = doc[0]
        blocks = page.get_text("blocks")
    finally:
        doc.close()
    out = []
    for (x0, y0, x1, y1, text, bno, btype) in blocks[:max_blocks]:
        out.append(f"--- block {bno}  bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) ---\n{text}")
    return "\n".join(out)


def _smp_calibrate_grid_axes(words):
    """
    Cari label angka sumbu grid peta (mis. 0,1000,2000,...,21000 di baris atas/
    bawah peta = Easting; -3000,...,12000 di kolom kiri/kanan = Northing) lalu
    fit transformasi linear piksel <-> koordinat grid: px = a*value + b.

    Return {"e_to_px": fn, "n_to_py": fn} kalau berhasil (fit-nya cukup lurus),
    atau None kalau gagal (mis. teks sumbu tidak terbaca / bukan kelipatan rapi).
    """
    import numpy as _np_cal
    from collections import defaultdict

    cands = []
    for w in words:
        t = w["text"].strip().replace(",", "")
        if _re_smp.fullmatch(r'-?\d{3,6}', t):
            v = int(t)
            cx = (w["x0"] + w["x1"]) / 2
            cy = (w["y0"] + w["y1"]) / 2
            cands.append((v, cx, cy))

    if len(cands) < 8:
        return None

    # kelompokkan kandidat jadi "baris" (utk sumbu-X, label di atas/bawah, y hampir sama)
    # dan "kolom" (utk sumbu-Y, label di kiri/kanan, x hampir sama)
    rows = defaultdict(list)
    cols = defaultdict(list)
    for v, cx, cy in cands:
        rows[round(cy / 20)].append((v, cx, cy))
        cols[round(cx / 20)].append((v, cx, cy))

    def _best_axis(groups, use_x_as_pos):
        best = None
        for pts in groups.values():
            if len(pts) < 5:
                continue
            vals = _np_cal.array([p[0] for p in pts], dtype=float)
            pos = _np_cal.array([p[1] if use_x_as_pos else p[2] for p in pts], dtype=float)
            if _np_cal.std(vals) < 1e-6:
                continue
            A = _np_cal.vstack([vals, _np_cal.ones_like(vals)]).T
            (a, b), *_ = _np_cal.linalg.lstsq(A, pos, rcond=None)
            pred = a * vals + b
            ss_res = _np_cal.sum((pos - pred) ** 2)
            ss_tot = _np_cal.sum((pos - pos.mean()) ** 2) + 1e-9
            r2 = 1 - ss_res / ss_tot
            if r2 > 0.995 and (best is None or len(pts) > best[0]):
                best = (len(pts), a, b, r2)
        return best

    x_axis = _best_axis(rows, use_x_as_pos=True)   # sumbu Easting/X
    y_axis = _best_axis(cols, use_x_as_pos=False)  # sumbu Northing/Y

    if x_axis is None or y_axis is None:
        return None

    _, a_x, b_x, _ = x_axis
    _, a_y, b_y, _ = y_axis

    return {
        "e_to_px": (lambda e_val, _a=a_x, _b=b_x: _a * e_val + _b),
        "n_to_py": (lambda n_val, _a=a_y, _b=b_y: _a * n_val + _b),
    }


# Pola default utk "Block N: <angka>, E: <angka>" di dalam kotak callout.
# SESUAIKAN kalau format asli di PDF-mu beda (cek pakai _smp_debug_dump_blocks).
_RE_SMP_BLOCK_NE = _re_smp.compile(
    r'Block\s*[:\-]?\s*N[:\s]*(-?[\d.]+)[,\s/]+E[:\s]*(-?[\d.]+)',
    _re_smp.IGNORECASE
)


def _smp_extract_block_ne(pdf_path):
    """
    Baca teks per BLOK PDF (bukan per kata lepas) supaya urutan baris
    'ID titik' -> 'Block N/E' di dalam kotak callout yang sama tetap terjaga.
    Return dict {ID Titik: (northing, easting)}.
    """
    doc = _fitz_smp.open(pdf_path)
    try:
        page = doc[0]
        blocks = page.get_text("blocks")
    finally:
        doc.close()

    result = {}
    for (x0, y0, x1, y1, text, bno, btype) in blocks:
        if not _RE_SMP_BLOCK_NE.search(text):
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        id_lines, ne_lines = [], []
        for i, ln in enumerate(lines):
            m_id = _SMP_ID_FULL_RE.match(ln.strip())
            if m_id:
                id_lines.append((i, _smp_normalize_id(ln.strip())))
            m_ne = _RE_SMP_BLOCK_NE.search(ln)
            if m_ne:
                try:
                    n_val = float(m_ne.group(1))
                    e_val = float(m_ne.group(2))
                    ne_lines.append((i, n_val, e_val))
                except ValueError:
                    pass
        if not id_lines or not ne_lines:
            continue
        # pasangkan tiap ID dengan baris Block N/E TERDEKAT dalam blok yang sama
        # (asumsi: tiap titik diikuti/didahului langsung oleh baris Block N/E-nya)
        for idx, pid in id_lines:
            _, n_val, e_val = min(ne_lines, key=lambda t: abs(t[0] - idx))
            result[pid] = (n_val, e_val)

    return result


def _smp_apply_block_ne_positions(points, block_ne_map, calib):
    """Timpa posisi (_x, _y) titik yang punya data Block N/E + kalibrasi valid,
    supaya posisi plot pakai lokasi titik SEBENARNYA, bukan posisi label callout.
    Return (points, n_corrected)."""
    n_corrected = 0
    if calib is None:
        return points, 0
    for pid, entry in points.items():
        if pid in block_ne_map:
            n_val, e_val = block_ne_map[pid]
            entry["_x"] = round(calib["e_to_px"](e_val), 1)
            entry["_y"] = round(calib["n_to_py"](n_val), 1)
            entry["_pos_source"] = "block_ne"
            n_corrected += 1
        else:
            entry.setdefault("_pos_source", "label")
    return points, n_corrected


def _smp_render_page_png(pdf_path, dpi=130):
    doc = _fitz_smp.open(pdf_path)
    try:
        page = doc[0]
        zoom = dpi / 72.0
        mat = _fitz_smp.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        out_path = pdf_path + "_render.png"
        pix.save(out_path)
        return out_path
    finally:
        doc.close()


# =========================================================
# ===== DETEKSI MARKER DARI GAMBAR PETA (before vs after) =====
# =========================================================
# Sumber deviasi SEKARANG dari perbandingan MARKER (ikon kotak/bulat/segitiga)
# yang tergambar langsung di badan peta, before vs after -- BUKAN dari teks di
# kotak callout kuning (itu cuma dipakai untuk tabel data mentah, terpisah).
# Karena posisi marker dideteksi langsung dari piksel gambar, tidak perlu lagi
# kalibrasi grid / Block-N-E / khawatir soal leader-line yang meleset.

def _smp_find_callout_boxes(img_rgb):
    """Deteksi kotak callout kuning pucat ('AREA GEOTECHNICAL ISSUE' dst) di
    render peta, supaya marker DI DALAM kotak itu (bagian dari tabel/legenda,
    bukan titik di badan peta) bisa diabaikan. Return list bbox (x0,y0,x1,y1)."""
    import numpy as _np_cb
    from scipy.ndimage import label as _cc_label, find_objects as _cc_objs

    arr = _np_cb.array(img_rgb.convert("RGB")).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # kuning pucat khas highlight box: R & G tinggi, B jelas lebih rendah
    yellow_mask = (r > 230) & (g > 225) & (b < 205) & (b > 110)

    lbl, _ = _cc_label(yellow_mask)
    boxes = []
    for sl in _cc_objs(lbl):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if w * h > 2500 and w > 35 and h > 35:   # kotak callout biasanya cukup besar
            boxes.append((x0, y0, x1, y1))
    return boxes


def _smp_point_in_boxes(px, py, boxes, margin=4):
    for (x0, y0, x1, y1) in boxes:
        if (x0 - margin) <= px <= (x1 + margin) and (y0 - margin) <= py <= (y1 + margin):
            return True
    return False


def _smp_color_dist(c1, c2):
    return sum((float(a) - float(b)) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _smp_detect_markers(img_rgb, exclude_boxes, min_area=4, max_area=260,
                         sat_thresh=55, val_thresh=60, top_left_exclude=None):
    """
    Deteksi blob marker kecil berwarna solid (kotak/bulat/segitiga instrumen
    monitoring) di badan peta, KECUALI yang ada di dalam kotak callout
    (exclude_boxes) atau panel kop-peta/legenda (top_left_exclude).
    Return list of dict {x, y, color(R,G,B), area, w, h}.

    Kalau hasil deteksi kebanyakan/kesedikitan, atur min_area/max_area/sat_thresh
    lewat slider di UI (parameternya dibuat bisa di-tuning dari tab3).
    """
    import numpy as _np_dm
    from scipy.ndimage import label as _cc_label, find_objects as _cc_objs

    arr = _np_dm.array(img_rgb.convert("RGB")).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    mx = _np_dm.maximum(_np_dm.maximum(r, g), b)
    mn = _np_dm.minimum(_np_dm.minimum(r, g), b)
    sat = mx - mn
    colorful_mask = (sat > sat_thresh) & (mx > val_thresh)

    lbl, _ = _cc_label(colorful_mask)
    markers = []
    for sl in _cc_objs(lbl):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        area = h * w
        if not (min_area <= area <= max_area) or w == 0 or h == 0:
            continue
        aspect = w / h
        if aspect < 0.4 or aspect > 2.5:
            continue  # marker instrumen biasanya proporsional, bukan garis/teks panjang
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if _smp_point_in_boxes(cx, cy, exclude_boxes):
            continue
        if top_left_exclude is not None:
            ex0, ey0, ex1, ey1 = top_left_exclude
            if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                continue
        patch = arr[y0:y1, x0:x1].reshape(-1, 3)
        color = tuple(int(v) for v in _np_dm.median(patch, axis=0))
        markers.append({"x": cx, "y": cy, "color": color, "area": area, "w": w, "h": h})
    return markers


def _smp_match_markers(markers_before, markers_after, max_dist_px=14, color_change_thresh=40):
    """
    Pasangkan marker before<->after berdasar posisi TERDEKAT (asumsi kedua render
    PDF pakai base-map & skala identik, cuma marker instrumennya yang beda/berubah).
    status: 'changed' (posisi sama, warna beda cukup jauh), 'same' (posisi & warna
    sama -> tidak perlu digambar), 'new' (cuma ada di after), 'removed' (cuma ada
    di before).
    """
    used_after = set()
    pairs = []
    for mb in markers_before:
        best_j, best_d = None, None
        for j, ma in enumerate(markers_after):
            if j in used_after:
                continue
            d = ((mb["x"] - ma["x"]) ** 2 + (mb["y"] - ma["y"]) ** 2) ** 0.5
            if d <= max_dist_px and (best_d is None or d < best_d):
                best_d, best_j = d, j
        if best_j is not None:
            used_after.add(best_j)
            ma = markers_after[best_j]
            cdist = _smp_color_dist(mb["color"], ma["color"])
            pairs.append({
                "x": (mb["x"] + ma["x"]) / 2.0, "y": (mb["y"] + ma["y"]) / 2.0,
                "before_color": mb["color"], "after_color": ma["color"],
                "color_dist": cdist,
                "status": "changed" if cdist > color_change_thresh else "same",
            })
        else:
            pairs.append({
                "x": mb["x"], "y": mb["y"],
                "before_color": mb["color"], "after_color": None,
                "color_dist": None, "status": "removed",
            })
    for j, ma in enumerate(markers_after):
        if j not in used_after:
            pairs.append({
                "x": ma["x"], "y": ma["y"],
                "before_color": None, "after_color": ma["color"],
                "color_dist": None, "status": "new",
            })
    return pairs


def _smp_debug_marker_overlay(img_rgb, markers, exclude_boxes):
    """Gambar overlay diagnostik: kotak merah = callout box yg diabaikan,
    titik cyan = marker yg terdeteksi & ikut dibandingkan. Buat verifikasi visual
    sebelum percaya hasil deteksinya."""
    from PIL import ImageDraw
    im = img_rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    for (x0, y0, x1, y1) in exclude_boxes:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    for m in markers:
        x, y = m["x"], m["y"]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 255, 255), width=2)
    return im


def _smp_classify_deviation(before_entry, after_entry):
    if before_entry is None and after_entry is not None:
        return "Titik Baru"
    if after_entry is None and before_entry is not None:
        return "Titik Hilang"
    br, ar = before_entry["_rank"], after_entry["_rank"]
    if br is None or ar is None:
        if br != ar:
            return "Berubah Keterbacaan"
        return "Tetap / Stabil"
    dr = ar - br
    if dr >= 2:
        return "Naik Drastis (>=2 level)"
    if dr == 1:
        return "Naik 1 Level"
    if dr < 0:
        return "Membaik"
    bv, av = before_entry["Nilai (mm/day)"], after_entry["Nilai (mm/day)"]
    if bv is not None and av is not None and av > bv + 1.0:
        return "Meningkat (level sama)"
    return "Tetap / Stabil"


def _smp_deviation_score(before_entry, after_entry):
    """Skor numerik seberapa besar deviasi titik ini (0 = tidak berubah, tidak perlu diinterpolasi)."""
    cat = _smp_classify_deviation(before_entry, after_entry)
    if before_entry is None or after_entry is None:
        return 2.5, cat  # titik baru / hilang tetap dianggap deviasi
    br, ar = before_entry["_rank"], after_entry["_rank"]
    if br is None or ar is None:
        return (2.0 if br != ar else 0.0), cat
    dr = ar - br
    bv, av = before_entry["Nilai (mm/day)"], after_entry["Nilai (mm/day)"]
    dv = (av - bv) if (av is not None and bv is not None) else 0.0
    score = max(dr, 0) * 3.0 + max(dv, 0) * 0.4
    return score, cat


# =========================================================
# ============ TAB 3: MONITORING DEVIATION =================
# =========================================================
with tab3:

    st.subheader("Monitoring Deviation — Peta Kontur Deviasi")
    st.caption(
        "Upload 2 PDF Slope Monitoring Map (before & after). Hanya titik yang "
        "benar-benar berubah status/nilai yang akan diinterpolasi jadi kontur "
        "rainbow di atas peta — titik yang tetap sama tidak ikut diinterpolasi."
    )

    if not _SMP_FITZ_OK:
        st.error(
            "Fitur ini butuh library **PyMuPDF**. Jalankan `pip install pymupdf` "
            "lalu restart aplikasi.\n\nDetail error: " + str(_SMP_FITZ_ERR)
        )
        st.stop()

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        md_before_file = st.file_uploader("PDF Sebelum (Before)", type=["pdf"], key="md_before_pdf")
    with col_up2:
        md_after_file = st.file_uploader("PDF Sesudah (After)", type=["pdf"], key="md_after_pdf")

    if md_before_file and md_after_file:
        if st.button("Proses Peta Deviasi", key="md_process_btn"):
            with st.spinner("Membaca PDF & mengekstrak titik pemantauan..."):
                tmp_dir = _tempfile_smp.mkdtemp()
                before_path = _os_smp.path.join(tmp_dir, "before.pdf")
                after_path = _os_smp.path.join(tmp_dir, "after.pdf")
                with open(before_path, "wb") as f:
                    f.write(md_before_file.getbuffer())
                with open(after_path, "wb") as f:
                    f.write(md_after_file.getbuffer())

                try:
                    before_pts = _smp_extract_points(before_path)
                    after_pts = _smp_extract_points(after_path)
                except Exception as e:
                    st.error(f"Gagal membaca PDF: {e}")
                    before_pts, after_pts = {}, {}

                # --- koreksi posisi: pakai koordinat "Block N/E" di kotak callout,
                # bukan posisi teks label ID (yang sering jauh dari titik asli
                # karena dihubungkan lewat leader line panjang) ---
                n_corrected_before = n_corrected_after = 0
                calib_ok = False
                try:
                    words_before = _smp_pdf_to_bbox_words(before_path)
                    calib = _smp_calibrate_grid_axes(words_before)
                    calib_ok = calib is not None
                    if calib_ok:
                        block_ne_before = _smp_extract_block_ne(before_path)
                        block_ne_after = _smp_extract_block_ne(after_path)
                        before_pts, n_corrected_before = _smp_apply_block_ne_positions(
                            before_pts, block_ne_before, calib
                        )
                        after_pts, n_corrected_after = _smp_apply_block_ne_positions(
                            after_pts, block_ne_after, calib
                        )
                except Exception as e:
                    st.warning(f"Kalibrasi grid / koordinat Block N-E gagal, pakai posisi label: {e}")

                st.session_state["md_before_pts"] = before_pts
                st.session_state["md_after_pts"] = after_pts
                st.session_state["md_before_path"] = before_path
                st.session_state["md_after_path"] = after_path
                st.session_state["md_calib_ok"] = calib_ok
                st.session_state["md_n_corrected"] = (n_corrected_before, n_corrected_after)
                st.session_state.pop("md_before_override", None)
                st.session_state.pop("md_after_override", None)

                if not calib_ok:
                    st.warning(
                        "Kalibrasi sumbu grid peta (angka 0..21000 / -3000..12000) tidak "
                        "berhasil dibaca otomatis — posisi titik yang leader-line-nya panjang "
                        "kemungkinan masih meleset (pakai posisi label). Jalankan "
                        "`_smp_debug_dump_blocks(pdf_path)` untuk cek pola teks aslinya."
                    )
                elif n_corrected_before == 0 and n_corrected_after == 0:
                    st.warning(
                        "Sumbu grid berhasil dikalibrasi, tapi tidak ada teks 'Block N/E' yang "
                        "cocok dengan pola regex di kotak callout — cek format teks aslinya "
                        "lewat `_smp_debug_dump_blocks(pdf_path)` dan sesuaikan _RE_SMP_BLOCK_NE."
                    )
                else:
                    st.success(
                        f"Posisi {n_corrected_before} titik (before) & {n_corrected_after} titik "
                        f"(after) dikoreksi pakai koordinat Block N/E asli (bukan posisi label)."
                    )

    if st.session_state.get("md_before_pts"):

        before_pts = st.session_state["md_before_pts"]
        after_pts = st.session_state["md_after_pts"]

        # override dari koreksi manual (kalau ada), tetap pakai posisi hasil ekstraksi asli
        before_ov = st.session_state.get("md_before_override", {})
        after_ov = st.session_state.get("md_after_override", {})

        before_use = {pid: {**v, **before_ov.get(pid, {})} for pid, v in before_pts.items()}
        after_use = {pid: {**v, **after_ov.get(pid, {})} for pid, v in after_pts.items()}

        mode = st.radio(
            "Sumber deviasi untuk peta kontur",
            ["Marker di gambar peta (bandingkan ikon before vs after)", "Teks kotak callout (cara lama)"],
            index=0, key="md_source_mode",
            help="Tabel data mentah di bawah selalu pakai hasil ekstraksi teks, "
                 "terlepas dari mode yang dipilih di sini.",
        )
        use_image_mode = mode.startswith("Marker")

        # n_naik dihitung dari data teks (tabel), independen dari mode peta kontur
        n_naik = 0
        for pid in sorted(set(before_use) | set(after_use)):
            _, cat_tmp = _smp_deviation_score(before_use.get(pid), after_use.get(pid))
            if cat_tmp in ("Naik Drastis (>=2 level)", "Naik 1 Level"):
                n_naik += 1

        if use_image_mode:
            # ================= MODE BARU: deteksi marker dari gambar peta =================
            only_color_change = st.checkbox(
                "Hanya tampilkan marker yang WARNANYA berubah di posisi yang sama "
                "(sembunyikan marker 'baru'/'hilang' — biasanya cuma gagal cocok posisi, bukan perubahan asli)",
                value=True, key="md_only_color_change",
            )
            with st.expander("Parameter deteksi marker (atur kalau hasil kurang pas)"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    min_area = st.number_input("Luas minimal marker (px^2)", 1, 500, 8, key="md_min_area")
                    max_area = st.number_input("Luas maksimal marker (px^2)", 5, 2000, 260, key="md_max_area")
                with c2:
                    sat_thresh = st.slider("Ambang saturasi warna", 0, 150, 55, key="md_sat_thresh")
                    val_thresh = st.slider("Ambang kecerahan minimal", 0, 200, 60, key="md_val_thresh")
                with c3:
                    max_dist_px = st.slider("Toleransi jarak pencocokan (px)", 2, 50, 25, key="md_max_dist_px")
                    color_change_thresh = st.slider("Ambang 'warna dianggap berubah'", 5, 150, 45, key="md_color_thresh")
                show_debug = st.checkbox("Tampilkan overlay deteksi (buat verifikasi)", value=False, key="md_show_debug")

            try:
                before_png = _smp_render_page_png(st.session_state["md_before_path"], dpi=130)
                after_png = _smp_render_page_png(st.session_state["md_after_path"], dpi=130)
                before_img = PILImage.open(before_png)
                after_img = PILImage.open(after_png)

                boxes_before = _smp_find_callout_boxes(before_img)
                boxes_after = _smp_find_callout_boxes(after_img)

                markers_before = _smp_detect_markers(before_img, boxes_before, min_area, max_area,
                                                      sat_thresh, val_thresh)
                markers_after = _smp_detect_markers(after_img, boxes_after, min_area, max_area,
                                                     sat_thresh, val_thresh)

                pairs = _smp_match_markers(markers_before, markers_after, max_dist_px, color_change_thresh)
                if only_color_change:
                    changed_pairs = [p for p in pairs if p["status"] == "changed"]
                else:
                    changed_pairs = [p for p in pairs if p["status"] in ("changed", "new", "removed")]

                n_changed_only = len([p for p in pairs if p["status"] == "changed"])
                n_new = len([p for p in pairs if p["status"] == "new"])
                n_removed = len([p for p in pairs if p["status"] == "removed"])
                st.caption(
                    f"{len(markers_before)} marker terdeteksi di peta before, {len(markers_after)} di after — "
                    f"**{n_changed_only} warna berubah**"
                    + ("" if only_color_change else f", {n_new} baru, {n_removed} hilang")
                    + "."
                )
                if not only_color_change and (n_new + n_removed) > n_changed_only * 3:
                    st.warning(
                        "Jumlah 'baru'/'hilang' jauh lebih banyak dari 'warna berubah' — kemungkinan besar "
                        "itu bukan perubahan asli, cuma gagal dicocokkan (coba naikkan 'Toleransi jarak "
                        "pencocokan' di parameter). Checkbox di atas (default aktif) menyembunyikan ini."
                    )
                if show_debug:
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.image(_smp_debug_marker_overlay(before_img, markers_before, boxes_before),
                                 caption="Deteksi - Before (kotak merah = callout diabaikan, titik cyan = marker)")
                    with dcol2:
                        st.image(_smp_debug_marker_overlay(after_img, markers_after, boxes_after),
                                 caption="Deteksi - After")

                W, H = before_img.width, before_img.height
                fig, ax = plt.subplots(figsize=(16, 11))
                ax.imshow(before_img)
                ax.axis("off")

                if len(changed_pairs) > 0:
                    import numpy as _np_smp
                    from scipy.spatial import cKDTree as _KDTree_smp

                    res = 3
                    gx = _np_smp.linspace(0, W, max(W // res, 2))
                    gy = _np_smp.linspace(0, H, max(H // res, 2))
                    GX, GY = _np_smp.meshgrid(gx, gy)
                    grid_pts = _np_smp.column_stack([GX.ravel(), GY.ravel()])

                    sigma_px = 30
                    cutoff_px = 2.2 * sigma_px

                    colorable = [p for p in changed_pairs if p["status"] in ("changed", "new")]
                    removed_pts = [p for p in changed_pairs if p["status"] == "removed"]

                    if colorable:
                        pts_xy = _np_smp.array([[p["x"], p["y"]] for p in colorable])
                        tree = _KDTree_smp(pts_xy)
                        dist, idx = tree.query(grid_pts, k=1)
                        dist = dist.reshape(GX.shape)
                        idx = idx.reshape(GX.shape)

                        alpha = _np_smp.clip(1 - dist / cutoff_px, 0, 1) ** 1.4 * 0.75
                        alpha[dist > cutoff_px] = 0.0

                        colors_arr = _np_smp.array(
                            [[c / 255.0 for c in (p["after_color"] or p["before_color"])] for p in colorable]
                        )
                        rgba = _np_smp.zeros((*GX.shape, 4))
                        rgba[..., :3] = colors_arr[idx]
                        rgba[..., 3] = alpha
                        ax.imshow(rgba, extent=(0, W, H, 0), origin="upper", zorder=2)

                    for p in colorable:
                        color01 = tuple(c / 255.0 for c in (p["after_color"] or p["before_color"]))
                        edge = "blue" if p["status"] == "new" else "black"
                        lw = 1.4 if p["status"] == "new" else 0.6
                        ax.scatter(p["x"], p["y"], s=32, facecolor=color01, edgecolor=edge,
                                   linewidth=lw, zorder=4)

                    for p in removed_pts:
                        ax.scatter(p["x"], p["y"], s=45, facecolor="none", edgecolor="black",
                                   linewidth=1.3, marker="x", zorder=4)

                    legend_handles = []
                    if colorable:
                        label_txt = "Warna marker berubah (isi = warna after)"
                        legend_handles.append(Patch(facecolor="#888888", edgecolor="black", label=label_txt))
                        if not only_color_change and any(p["status"] == "new" for p in colorable):
                            legend_handles.append(Patch(facecolor="none", edgecolor="blue", label="Marker baru"))
                    if removed_pts:
                        legend_handles.append(Patch(facecolor="none", edgecolor="black", label="Marker hilang (x)"))
                    if legend_handles:
                        ax.legend(handles=legend_handles, loc="lower left", fontsize=7, framealpha=0.9)

                    ax.set_title("Kontur Deviasi - Perbandingan Marker Peta (Before vs After)", fontsize=14)
                else:
                    ax.set_title("Tidak ada marker yang berbeda antara before & after", fontsize=14)

                st.pyplot(fig)
                out_path = "/tmp/monitoring_deviation.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                with open(out_path, "rb") as f:
                    st.download_button("Download Peta Deviasi (PNG)", f, file_name="monitoring_deviation.png",
                                        key="md_dl_image_mode")

            except Exception as e:
                st.error(f"Gagal membuat peta deviasi (mode marker gambar): {e}")

        else:
            # ================= MODE LAMA: teks kotak callout =================
            only_calibrated = st.checkbox(
                "Hanya tampilkan/konturkan titik yang posisinya berhasil dikalibrasi ke "
                "koordinat Block N/E asli (rekomendasi — hindari kontur nongol di kotak callout)",
                value=True, key="md_only_calibrated",
            )

            all_ids = sorted(set(before_use) | set(after_use))
            deviation_points = []
            n_skipped_uncalibrated = 0
            for pid in all_ids:
                b = before_use.get(pid)
                a = after_use.get(pid)
                score, cat = _smp_deviation_score(b, a)
                if cat == "Tetap / Stabil":
                    continue
                pos_src = a or b
                is_calibrated = pos_src.get("_pos_source") == "block_ne"
                if only_calibrated and not is_calibrated:
                    n_skipped_uncalibrated += 1
                    continue
                magnitude = score if score > 0 else 1.0
                deviation_points.append({
                    "id": pid, "score": magnitude, "cat": cat,
                    "x": pos_src["_x"], "y": pos_src["_y"],
                    "calibrated": is_calibrated,
                })

            st.caption(
                f"{len(before_pts)} titik (before) / {len(after_pts)} titik (after) terbaca — "
                f"**{len(deviation_points)} titik mengalami perubahan status** dari total {len(all_ids)} titik gabungan."
            )
            if n_skipped_uncalibrated > 0:
                calib_ok_state = st.session_state.get("md_calib_ok", None)
                if calib_ok_state is False:
                    st.warning(
                        f"Kalibrasi sumbu grid peta gagal total, jadi SEMUA {n_skipped_uncalibrated} "
                        f"titik disembunyikan (posisinya cuma dari label callout, pasti tidak akurat). "
                        f"Cek `_smp_debug_dump_blocks(pdf_path)` untuk lihat teks sumbu grid aslinya."
                    )
                else:
                    st.info(
                        f"{n_skipped_uncalibrated} titik deviasi disembunyikan dari peta kontur karena "
                        f"posisinya belum berhasil dikalibrasi ke koordinat Block N/E asli (masih posisi "
                        f"label callout) — matikan checkbox di atas kalau tetap mau menampilkannya "
                        f"(dengan risiko posisi meleset ke kotak callout)."
                    )

            try:
                bg_png = _smp_render_page_png(st.session_state["md_before_path"], dpi=130)
                bg_img = PILImage.open(bg_png)
                page_w_pts, page_h_pts = _smp_get_page_size(st.session_state["md_before_path"])
                scale_x = bg_img.width / page_w_pts
                scale_y = bg_img.height / page_h_pts
                W, H = bg_img.width, bg_img.height

                fig, ax = plt.subplots(figsize=(16, 11))
                ax.imshow(bg_img)
                ax.axis("off")

                if len(deviation_points) > 0:
                    import numpy as _np_smp
                    import matplotlib.colors as _mcolors_smp

                    res = 3  # downsample grid demi kecepatan
                    gx = _np_smp.linspace(0, W, max(W // res, 2))
                    gy = _np_smp.linspace(0, H, max(H // res, 2))
                    GX, GY = _np_smp.meshgrid(gx, gy)

                    sigma_px = 30               # radius pengaruh tiap titik (px)
                    cutoff_px = 2.2 * sigma_px  # HARD cutoff

                    cats_present = []
                    for pt in deviation_points:
                        if pt["cat"] not in cats_present:
                            cats_present.append(pt["cat"])

                    for cat_name in cats_present:
                        pts_in_cat = [p for p in deviation_points if p["cat"] == cat_name]
                        field = _np_smp.zeros_like(GX, dtype=float)
                        min_dist = _np_smp.full_like(GX, _np_smp.inf)
                        for pt in pts_in_cat:
                            px, py = pt["x"] * scale_x, pt["y"] * scale_y
                            d2 = (GX - px) ** 2 + (GY - py) ** 2
                            d = _np_smp.sqrt(d2)
                            contrib = pt["score"] * _np_smp.exp(-d2 / (2 * sigma_px ** 2))
                            contrib[d > cutoff_px] = 0.0
                            field += contrib
                            min_dist = _np_smp.minimum(min_dist, d)

                        field = gaussian_filter(field, sigma=1.0)
                        if field.max() <= 0:
                            continue
                        mask = (min_dist > cutoff_px) | (field < field.max() * 0.18)
                        norm_field = _np_smp.clip(field / field.max(), 0, 1)

                        color_rgb = _mcolors_smp.to_rgb(_SMP_STATUS_COLOR.get(cat_name, "#999999"))
                        rgba = _np_smp.zeros((*field.shape, 4))
                        rgba[..., 0] = color_rgb[0]
                        rgba[..., 1] = color_rgb[1]
                        rgba[..., 2] = color_rgb[2]
                        rgba[..., 3] = _np_smp.where(mask, 0.0, norm_field * 0.72)

                        ax.imshow(rgba, extent=(0, W, H, 0), origin="upper", zorder=2)
                        field_masked = _np_smp.ma.masked_where(mask, field)
                        ax.contour(GX, GY, field_masked, levels=4,
                                   colors=[_SMP_STATUS_COLOR.get(cat_name, "#999999")],
                                   linewidths=0.5, alpha=0.55, zorder=3)

                    legend_handles = [
                        Patch(facecolor=_SMP_STATUS_COLOR.get(c, "#999999"), edgecolor="black", label=c)
                        for c in cats_present
                    ]
                    ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
                              framealpha=0.9, title="Perubahan Status (Before -> After)")

                    for pt in deviation_points:
                        px, py = pt["x"] * scale_x, pt["y"] * scale_y
                        face = _SMP_STATUS_COLOR.get(pt["cat"], "#999999")
                        if pt.get("calibrated", False):
                            edge, lw = "black", 0.6
                        else:
                            edge, lw = "#FF00FF", 1.3
                        ax.scatter(px, py, s=30, facecolor=face, edgecolor=edge,
                                   linewidth=lw, zorder=4)
                        label_txt = pt["id"] + ("" if pt.get("calibrated", False) else " (?)")
                        ax.annotate(label_txt, (px, py), fontsize=6, color="black",
                                    xytext=(3, 3), textcoords="offset points", zorder=4)

                    ax.set_title("Kontur Deviasi Monitoring (Before -> After)", fontsize=14)
                else:
                    ax.set_title("Tidak ada titik yang mengalami deviasi", fontsize=14)

                st.pyplot(fig)

                out_path = "/tmp/monitoring_deviation.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                with open(out_path, "rb") as f:
                    st.download_button("Download Peta Deviasi (PNG)", f, file_name="monitoring_deviation.png",
                                        key="md_dl_text_mode")

            except Exception as e:
                st.error(f"Gagal membuat peta deviasi: {e}")

        if n_naik > 0:
            st.error(f"{n_naik} titik menunjukkan kenaikan status pergerakan — perlu perhatian.")
        else:
            st.success("Tidak ada titik yang naik status pergerakan secara signifikan.")

        # --- data mentah & koreksi manual, disembunyikan biar peta jadi fokus utama ---
        with st.expander("Data mentah hasil ekstraksi & koreksi manual (opsional)"):
            st.caption(
                "Peta ini padat teks sehingga sesekali ada salah baca. Kalau ada nilai "
                "yang keliru, koreksi di sini lalu klik 'Update Peta dari Koreksi'."
            )
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Titik Before**")
                df_before_edit = st.data_editor(
                    pd.DataFrame(
                        [{"ID Titik": v["ID Titik"], "Nilai (mm/day)": v["Nilai (mm/day)"], "Status": v["Status"]}
                         for v in before_use.values()]
                    ).sort_values("ID Titik"),
                    num_rows="fixed", use_container_width=True, key="md_before_editor",
                    column_config={"Status": st.column_config.SelectboxColumn(
                        options=["Stabil", "Hati-hati", "Waspada", "Evakuasi", "Tidak terbaca"])}
                )
            with col_t2:
                st.markdown("**Titik After**")
                df_after_edit = st.data_editor(
                    pd.DataFrame(
                        [{"ID Titik": v["ID Titik"], "Nilai (mm/day)": v["Nilai (mm/day)"], "Status": v["Status"]}
                         for v in after_use.values()]
                    ).sort_values("ID Titik"),
                    num_rows="fixed", use_container_width=True, key="md_after_editor",
                    column_config={"Status": st.column_config.SelectboxColumn(
                        options=["Stabil", "Hati-hati", "Waspada", "Evakuasi", "Tidak terbaca"])}
                )

            if st.button("Update Peta dari Koreksi", key="md_update_btn"):
                new_before_ov = {}
                for _, row in df_before_edit.iterrows():
                    new_before_ov[row["ID Titik"]] = {
                        "Nilai (mm/day)": row["Nilai (mm/day)"], "Status": row["Status"],
                        "_rank": _SMP_STATUS_RANK.get(row["Status"]),
                    }
                new_after_ov = {}
                for _, row in df_after_edit.iterrows():
                    new_after_ov[row["ID Titik"]] = {
                        "Nilai (mm/day)": row["Nilai (mm/day)"], "Status": row["Status"],
                        "_rank": _SMP_STATUS_RANK.get(row["Status"]),
                    }
                st.session_state["md_before_override"] = new_before_ov
                st.session_state["md_after_override"] = new_after_ov
                st.rerun()

    else:
        st.info("Upload kedua PDF (before & after), lalu klik 'Proses Peta Deviasi' untuk memulai.")

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

# (judul "MQG AI Assistant (FAISS Mode)" sekarang dirender di dalam panel
# melayang saja -- lihat bagian CHAT di bawah -- bukan selalu tampil di sini)

# ================= LOAD DOKUMEN =================
# PERBAIKAN PERFORMA: di-cache supaya file di folder docs/ (txt/csv/xlsx/pdf/
# docx) tidak dibaca & di-parse ulang dari disk pada SETIAP rerun (sebelumnya
# baris ini dipanggil polos di level module, jadi ikut jalan ulang tiap ada
# interaksi widget apa pun di seluruh app, bukan cuma saat buka chat).
@st.cache_data
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
        return f"ANALISA FS:\n{ml_info}"

    elif "piping" in prompt:
        return "Piping → kontrol hydraulic gradient & drainase"

    elif "erosi" in prompt:
        return "Erosi → riprap & vegetasi"

    else:
        results = semantic_search(prompt, doc_chunks, faiss_index)

        context = "\n\n---\n\n".join(results)

        return f"""
HASIL PALING RELEVAN:

{context}

ML:
{ml_info}
"""

# ================= CHAT (tombol emoji melayang + bubble popup) =================
# PERBAIKAN: dulu section ini (riwayat chat + input) dirender polos & selalu
# tampil di bawah semua tab (di luar with tab1/tab2/tab3), jadi kelihatan
# 'muncul terus' apa pun tab yang lagi aktif. Sekarang disembunyikan jadi
# tombol emoji di pojok kanan-bawah -- diklik sekali untuk buka panel chat
# (bubble popup), diklik lagi (ikon jadi "✖") untuk menutup & balik jadi
# emoji. Logika AI/FAISS di atas (load_documents, build_faiss_index,
# geo_ai_response, dst) TIDAK berubah sama sekali, cuma bagian tampilannya.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mqg_chat_open" not in st.session_state:
    st.session_state["mqg_chat_open"] = False

with st.container(key="mqg_chat_fab"):
    _fab_icon = "✖" if st.session_state["mqg_chat_open"] else "🤖"
    if st.button(_fab_icon, key="mqg_chat_fab_btn", help="Tanya AI Geoteknik"):
        st.session_state["mqg_chat_open"] = not st.session_state["mqg_chat_open"]
        # PERBAIKAN PERFORMA: st.rerun() manual dihapus -- st.button yang
        # diklik SUDAH otomatis memicu rerun dari Streamlit, jadi memanggil
        # st.rerun() lagi di sini membuat seluruh script (~8000 baris, semua
        # tab & analisis) dieksekusi DUA KALI untuk setiap satu klik tombol.

if st.session_state["mqg_chat_open"]:
    with st.container(key="mqg_chat_panel"):
        st.markdown("**MQG AI Assistant (FAISS Mode)**")

        for msg in st.session_state.messages:
            avatar = "audience.png" if msg["role"] == "user" else "ai_bot.png"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # st.chat_input diganti st.form + text_input supaya input pasti tetap
        # berada DI DALAM panel melayang ini -- st.chat_input punya perilaku
        # khusus "menempel di dasar viewport" di Streamlit yang bisa lolos dari
        # container CSS fixed-position seperti punya kita di sini.
        with st.form(key="mqg_chat_form", clear_on_submit=True):
            prompt = st.text_input(
                "Tanya geoteknik...",
                label_visibility="collapsed",
                placeholder="Tanya geoteknik..."
            )
            submitted = st.form_submit_button("Kirim")

        df_pred = st.session_state.get("df_pred", None)

        if submitted and prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            result = geo_ai_response(prompt, knowledge_base, df_pred)

            st.session_state.messages.append({"role": "assistant", "content": result})
            st.rerun()
