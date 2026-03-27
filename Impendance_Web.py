import streamlit as st
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# ================= Page Config & Theme =================
st.set_page_config(page_title="RF/SI Impedance Calculator", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ================= Defaults & Session State =================
defaults = {
    'unit': 'mil',
    'ms_w': 17.0, 'ms_h': 26.2, 'ms_t': 1.65, 'ms_s': 8.0, 'ms_er': 4.2, 
    'ms_sm_h': 1.0, 'ms_sm_er': 3.4, 'ms_target_z': 50.0, 'ms_solve_target': 'W',
    'ms_is_diff': False, 'ms_use_sm': False,
    
    'cpw_w_bot': 5.16, 'cpw_w_top': 4.16, 'cpw_h': 3.91, 'cpw_g': 6.0, 'cpw_s': 8.0, 
    'cpw_t': 1.38, 'cpw_er': 4.1, 'cpw_sm_c1': 0.8, 'cpw_sm_c2': 0.5, 'cpw_sm_er': 3.5,
    'cpw_target_z': 100.0, 'cpw_solve_target': 'W',
    'cpw_is_cpwg': True, 'cpw_is_diff': True, 'cpw_use_sm': True
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= Core Math Engine =================
def get_ellip_ratio(k):
    if k >= 0.9999: return 50.0 
    if k <= 0.0001: return 0.02
    k_prime = math.sqrt(1 - k**2)
    if 0 <= k <= 0.707:
        return math.pi / math.log(2 * (1 + math.sqrt(k_prime)) / (1 - math.sqrt(k_prime)))
    else:
        return (1 / math.pi) * math.log(2 * (1 + math.sqrt(k)) / (1 - math.sqrt(k)))

def get_microstrip_impedance(w, h, t, s, er, is_diff, use_sm, sm_h, sm_er):
    if w <= 0 or h <= 0: return 9999.0
    w_eff = w + (t / math.pi) * (1 + math.log(2 * h / t)) if t > 0 else w
    ratio = w_eff / h
    e_eff_bare = (er + 1) / 2 + ((er - 1) / 2) * (1 / math.sqrt(1 + 12 / ratio))
    if ratio <= 1:
        z0_bare = (60 / math.sqrt(e_eff_bare)) * math.log(8 / ratio + 0.25 * ratio)
    else:
        z0_bare = (120 * math.pi) / (math.sqrt(e_eff_bare) * (ratio + 1.393 + 0.667 * math.log(ratio + 1.444)))
    
    final_z0 = z0_bare
    if use_sm and sm_h > 0:
        fill_factor = 0.5 * math.tanh(2 * sm_h / w)
        e_eff_coated = e_eff_bare + (sm_er - 1) * fill_factor
        final_z0 = z0_bare * math.sqrt(e_eff_bare / e_eff_coated)
        
    if is_diff:
        if s <= 0: return 9999.0
        coupling_factor = 1 - 0.48 * math.exp(-0.96 * s / h)
        return 2 * final_z0 * coupling_factor
    return final_z0

def calc_cpw_half_admittance(w_eff, g_eff, h, t, er, is_cpwg, use_sm, c1, c2, sm_er):
    a, b = w_eff, w_eff + 2 * g_eff
    k1 = a / b
    q1 = get_ellip_ratio(k1)
    if is_cpwg: k3 = math.tanh(math.pi * a / (4 * h)) / math.tanh(math.pi * b / (4 * h))
    else: k3 = math.sinh(math.pi * a / (4 * h)) / math.sinh(math.pi * b / (4 * h))
    q3 = get_ellip_ratio(k3)
    denominator = q1 + q3
    numerator = (1.0 * q1) + (er * q3)
    if use_sm and c1 > 0:
        k_fill = math.sinh(math.pi * a / (4 * c1)) / math.sinh(math.pi * b / (4 * c1))
        q_fill = get_ellip_ratio(k_fill)
        numerator += (sm_er - 1.0) * q_fill
    e_eff = numerator / denominator
    z0_air = (60 * math.pi) / denominator
    z0_full = z0_air / math.sqrt(e_eff)
    return (1 / z0_full) / 2

def get_cpw_impedance(w_bot, w_top, h, g_bot, s_bot, t, er, is_cpwg, is_diff, use_sm, c1, c2, sm_er):
    w_avg = (w_bot + w_top) / 2.0
    side_slope = (w_bot - w_top) / 2.0
    g_avg = g_bot + side_slope
    trap_factor = 0.75 if abs(w_bot - w_top) > 0.1 else 1.0
    raw_delta = (1.25 * t / math.pi) * (1 + math.log(4 * math.pi * w_avg / t)) if t > 0 else 0
    final_delta = raw_delta * trap_factor

    def get_admittance_with_fixed_delta(w_geom, g_geom, delta_val):
        if delta_val > 0.45 * g_geom: delta_val = 0.45 * g_geom
        return calc_cpw_half_admittance(w_geom + delta_val, g_geom - delta_val, h, t, er, is_cpwg, use_sm, c1, c2, sm_er)

    if is_diff:
        s_avg = s_bot + side_slope
        y_outer = get_admittance_with_fixed_delta(w_avg, g_avg, final_delta)
        y_inner = get_admittance_with_fixed_delta(w_avg, s_avg/2.0, final_delta)
        return 2 * (1 / (y_outer + y_inner))
    else:
        y_half = get_admittance_with_fixed_delta(w_avg, g_avg, final_delta)
        return 1 / (2 * y_half)

# ================= Goal Seek Callbacks =================
def ms_goal_seek_callback():
    low, high = 0.5, 200.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if st.session_state.ms_solve_target == "W":
            z_calc = get_microstrip_impedance(mid, st.session_state.ms_h, st.session_state.ms_t, st.session_state.ms_s, st.session_state.ms_er, st.session_state.ms_is_diff, st.session_state.ms_use_sm, st.session_state.ms_sm_h, st.session_state.ms_sm_er)
            if z_calc > st.session_state.ms_target_z: low = mid
            else: high = mid
        else:
            z_calc = get_microstrip_impedance(st.session_state.ms_w, st.session_state.ms_h, st.session_state.ms_t, mid, st.session_state.ms_er, st.session_state.ms_is_diff, st.session_state.ms_use_sm, st.session_state.ms_sm_h, st.session_state.ms_sm_er)
            if z_calc < st.session_state.ms_target_z: low = mid
            else: high = mid
    
    best_val = (low + high) / 2.0
    if st.session_state.ms_solve_target == "W":
        st.session_state.ms_w = float(best_val)
    else:
        st.session_state.ms_s = float(best_val)

def cpw_goal_seek_callback():
    low, high = 0.5, 200.0
    w_diff = st.session_state.cpw_w_bot - st.session_state.cpw_w_top
    for _ in range(60):
        mid = (low + high) / 2.0
        if st.session_state.cpw_solve_target == "W":
            test_w_bot = mid + w_diff/2
            test_w_top = max(0.1, mid - w_diff/2)
            z_calc = get_cpw_impedance(test_w_bot, test_w_top, st.session_state.cpw_h, st.session_state.cpw_g, st.session_state.cpw_s, st.session_state.cpw_t, st.session_state.cpw_er, st.session_state.cpw_is_cpwg, st.session_state.cpw_is_diff, st.session_state.cpw_use_sm, st.session_state.cpw_sm_c1, st.session_state.cpw_sm_c2, st.session_state.cpw_sm_er)
            if z_calc > st.session_state.cpw_target_z: low = mid
            else: high = mid
        else:
            z_calc = get_cpw_impedance(st.session_state.cpw_w_bot, st.session_state.cpw_w_top, st.session_state.cpw_h, st.session_state.cpw_g, mid, st.session_state.cpw_t, st.session_state.cpw_er, st.session_state.cpw_is_cpwg, st.session_state.cpw_is_diff, st.session_state.cpw_use_sm, st.session_state.cpw_sm_c1, st.session_state.cpw_sm_c2, st.session_state.cpw_sm_er)
            if z_calc < st.session_state.cpw_target_z: low = mid
            else: high = mid
    
    best_val = (low + high) / 2.0
    if st.session_state.cpw_solve_target == "W":
        st.session_state.cpw_w_bot = float(best_val + w_diff/2)
        st.session_state.cpw_w_top = float(max(0.1, best_val - w_diff/2))
    else:
        st.session_state.cpw_s = float(best_val)


# ================= Matplotlib UI Drawing Engine =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = 'none'

def draw_cross_section(mode="CPW", is_diff=True, is_cpwg=True, has_sm=True):
    bg_color_ui = '#0e1117'    
    col_label_gold = '#FFD700' 
    col_cop_metal = '#B87333'  
    col_diel_fr4 = '#607D8B'   
    col_gnd_metal = '#A9A9A9'  
    col_sm_ink = '#228B22'     
    
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=bg_color_ui)
    ax.axis('off')
    ax.set_facecolor(bg_color_ui)
    
    total_w = 120
    h = 25
    thick = 5
    sm_h = 5
    w = 12
    s = 10
    g = 15

    ax.set_xlim(-total_w/2 - 20, total_w/2 + 20)
    ax.set_ylim(-h - 15, thick + sm_h + 15)

    ax.add_patch(patches.Rectangle((-total_w/2, -h), total_w, h, facecolor=col_diel_fr4, edgecolor='none', linewidth=1))
    ax.text(total_w/2 + 5, -h/2, "Er", fontsize=11, fontweight='bold', va='center', color=col_label_gold)

    if is_cpwg or mode == "Microstrip":
        ax.add_patch(patches.Rectangle((-total_w/2, -h-3), total_w, 3, facecolor=col_gnd_metal, edgecolor='none', linewidth=1))

    def draw_sm_ink(areas):
        if not has_sm: return
        for (xs, xe, r_type) in areas:
            y_bot = 0 if r_type == 'gap' else thick
            ax.add_patch(patches.Rectangle((xs, y_bot), xe-xs, thick+sm_h-y_bot, facecolor=col_sm_ink, edgecolor='none', alpha=0.65))
        t = ax.text(total_w/2, thick + sm_h + 4, "SM", color=col_label_gold, fontweight='bold', fontsize=10, ha='center')
        t.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground=bg_color_ui)])

    if mode == "Microstrip":
        sm_areas = []
        if is_diff:
            x1, x2 = -s/2 - w, s/2
            ax.add_patch(patches.Rectangle((x1, 0), w, thick, facecolor=col_cop_metal, edgecolor='none'))
            ax.add_patch(patches.Rectangle((x2, 0), w, thick, facecolor=col_cop_metal, edgecolor='none'))
            sm_areas = [(-total_w/2, x1, 'gap'), (x1, x1+w, 'metal'), (x1+w, x2, 'gap'), (x2, x2+w, 'metal'), (x2+w, total_w/2, 'gap')]
            ax.annotate('', xy=(-s/2, thick+2), xytext=(s/2, thick+2), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5))
            ax.text(0, thick+3, "S", color=col_label_gold, ha='center', va='bottom', fontweight='bold')
            ax.annotate('', xy=(x1, -3), xytext=(x1+w, -3), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5))
            ax.text(x1+w/2, -5, "W", color=col_label_gold, ha='center', va='top', fontweight='bold')
        else:
            ax.add_patch(patches.Rectangle((-w/2, 0), w, thick, facecolor=col_cop_metal, edgecolor='none'))
            sm_areas = [(-total_w/2, -w/2, 'gap'), (-w/2, w/2, 'metal'), (w/2, total_w/2, 'gap')]
            ax.annotate('', xy=(-w/2, thick+2), xytext=(w/2, thick+2), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5))
            ax.text(0, thick+3, "W", color=col_label_gold, ha='center', va='bottom', fontweight='bold')

        draw_sm_ink(sm_areas)
        ax.annotate('', xy=(-total_w/2-3, 0), xytext=(-total_w/2-3, -h), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.2))
        ax.text(-total_w/2-6, -h/2, "H", ha='right', va='center', fontweight='bold', color=col_label_gold)

    elif mode == "CPW":
        trap_offset = 2
        def draw_trap_metal(xc):
            x1, x2 = xc - w/2, xc + w/2
            x3, x4 = xc + w/2 - trap_offset, xc - w/2 + trap_offset
            poly = patches.Polygon([[x1, 0], [x2, 0], [x3, thick], [x4, thick]], facecolor=col_cop_metal, edgecolor='none')
            ax.add_patch(poly)
            return x1, x2

        sm_areas = []
        if is_diff:
            xc1, xc2 = -s/2 - w/2, s/2 + w/2
            t1_l, t1_r = draw_trap_metal(xc1)
            t2_l, t2_r = draw_trap_metal(xc2)
            lgnd_x, rgnd_x = t1_l - g, t2_r + g
            ax.add_patch(patches.Rectangle((-total_w/2, 0), total_w/2 + lgnd_x, thick, facecolor=col_cop_metal, edgecolor='none'))
            ax.add_patch(patches.Rectangle((rgnd_x, 0), total_w/2 - rgnd_x, thick, facecolor=col_cop_metal, edgecolor='none'))
            sm_areas = [(-total_w/2, lgnd_x, 'metal'), (lgnd_x, t1_l, 'gap'), (t1_l, t1_r, 'metal'), (t1_r, t2_l, 'gap'), (t2_l, t2_r, 'metal'), (t2_r, rgnd_x, 'gap'), (rgnd_x, total_w/2, 'metal')]
            
            ax.annotate('', xy=(xc1, thick+2), xytext=(xc2, thick+2), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5))
            ax.text(0, thick+3, "S (avg)", color=col_label_gold, ha='center', va='bottom', fontweight='bold')
            ax.annotate('', xy=(lgnd_x, thick+2), xytext=(t1_l, thick+2), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5, ls='--'))
            ax.text((lgnd_x+t1_l)/2, thick+3, "G", color=col_label_gold, ha='center', va='bottom', fontweight='bold')
            ax.annotate('', xy=(t2_l, -3), xytext=(t2_r, -3), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5))
            ax.text(xc2, -5, "W", color=col_label_gold, ha='center', va='top', fontweight='bold')
        else:
            t_l, t_r = draw_trap_metal(0)
            lgnd_x, rgnd_x = t_l - g, t_r + g
            ax.add_patch(patches.Rectangle((-total_w/2, 0), total_w/2 + lgnd_x, thick, facecolor=col_cop_metal, edgecolor='none'))
            ax.add_patch(patches.Rectangle((rgnd_x, 0), total_w/2 - rgnd_x, thick, facecolor=col_cop_metal, edgecolor='none'))
            sm_areas = [(-total_w/2, lgnd_x, 'metal'), (lgnd_x, t_l, 'gap'), (t_l, t_r, 'metal'), (t_r, rgnd_x, 'gap'), (rgnd_x, total_w/2, 'metal')]

            ax.annotate('', xy=(lgnd_x, thick+2), xytext=(t_l, thick+2), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5, ls='--'))
            ax.text((lgnd_x+t_l)/2, thick+3, "G", color=col_label_gold, ha='center', va='bottom', fontweight='bold')
            ax.annotate('', xy=(t_l, -3), xytext=(t_r, -3), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.5))
            ax.text(0, -5, "W", color=col_label_gold, ha='center', va='top', fontweight='bold')

        draw_sm_ink(sm_areas)
        ax.annotate('', xy=(-total_w/2-3, 0), xytext=(-total_w/2-3, -h), arrowprops=dict(arrowstyle='<->', color=col_label_gold, lw=1.2))
        ax.text(-total_w/2-6, -h/2, "H", ha='right', va='center', fontweight='bold', color=col_label_gold)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig

# ================= UI Layout =================

st.title("📡 RF Impedance Calculator")

col_unit, col_preset, _ = st.columns([1.5, 2.5, 4])
with col_unit:
    st.session_state.unit = st.radio("📐 Unit:", ["mil", "mm"], horizontal=True)

with col_preset:
    presets = {
        "Custom Input": None,
        "JLCPCB JLC04161H-3313 (4-Layer 1.6mm)": {"H": 3.91, "T": 1.38, "Er": 4.1, "C1": 0.8, "C2": 0.5, "CEr": 3.5},
        "JLCPCB JLC04161H-7628 (4-Layer 1.6mm)": {"H": 7.8, "T": 1.38, "Er": 4.6, "C1": 0.8, "C2": 0.5, "CEr": 3.5}
    }
    selected_preset = st.selectbox("📂 Quick Load Stackup Presets:", list(presets.keys()))
    if selected_preset != "Custom Input":
        p = presets[selected_preset]
        for key, val in p.items():
            if f'cpw_{key.lower()}' in st.session_state: st.session_state[f'cpw_{key.lower()}'] = val
            if key in ["H", "T", "Er", "C1", "CEr"]:
                k_ms = 'ms_h' if key=="H" else 'ms_t' if key=="T" else 'ms_er' if key=="Er" else 'ms_sm_h' if key=="C1" else 'ms_sm_er'
                st.session_state[k_ms] = val

tab1, tab2 = st.tabs(["📝 Coplanar Waveguide (CPW/CPWG)", "📝 Microstrip"])

# === CPW Tab ===
with tab1:
    st.subheader("CPW / CPWG Configuration")
    col_img, col_opt = st.columns([2, 1])
    
    with col_opt:
        is_cpwg = st.checkbox("Ground Backed (CPWG)", key="cpw_is_cpwg")
        is_cpw_diff = st.checkbox("Differential Pair (Diff)", key="cpw_is_diff")
        use_cpw_sm = st.checkbox("Add Solder Mask", key="cpw_use_sm")

    with col_img:
        st.pyplot(draw_cross_section("CPW", is_cpw_diff, is_cpwg, use_cpw_sm))

    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Bottom Width W_bot", key='cpw_w_bot', format="%.4f")
        st.number_input("Top Width W_top", key='cpw_w_top', format="%.4f")
        st.number_input("Substrate Height H", key='cpw_h', format="%.4f")
    with c2:
        st.number_input("Trace Spacing S", key='cpw_s', format="%.4f", disabled=not is_cpw_diff)
        st.number_input("GND Spacing G", key='cpw_g', format="%.4f")
        st.number_input("Copper Thickness T", key='cpw_t', format="%.4f")
    with c3:
        st.number_input("Substrate Er", key='cpw_er', format="%.4f")
        if use_cpw_sm:
            st.number_input("Mask over Substrate C1", key='cpw_sm_c1', format="%.4f")
            st.number_input("Mask over Trace C2", key='cpw_sm_c2', format="%.4f")
            st.number_input("Mask Er", key='cpw_sm_er', format="%.4f")

    st.markdown("### 🎯 Goal Seek (Synthesis)")
    g1, g2, g3 = st.columns([1, 2, 1])
    with g1:
        st.number_input("Target Impedance (Ω)", key='cpw_target_z')
    with g2:
        st.radio("Parameter to Solve:", ["W", "S"] if is_cpw_diff else ["W"], horizontal=True, key='cpw_solve_target')
    with g3:
        st.write("") 
        st.button("🚀 Solve for Optimal Dimension", key="btn_solve_cpw", use_container_width=True, on_click=cpw_goal_seek_callback)

    z_res = get_cpw_impedance(st.session_state.cpw_w_bot, st.session_state.cpw_w_top, st.session_state.cpw_h, st.session_state.cpw_g, st.session_state.cpw_s, st.session_state.cpw_t, st.session_state.cpw_er, is_cpwg, is_cpw_diff, use_cpw_sm, st.session_state.cpw_sm_c1, st.session_state.cpw_sm_c2, st.session_state.cpw_sm_er)
    
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #001F3F; border-radius: 10px;'>
            <h2 style='color: #00FFFF;'>{'Differential' if is_cpw_diff else 'Single-Ended'} Impedance: {z_res:.2f} Ω</h2>
        </div>
    """, unsafe_allow_html=True)


# === Microstrip Tab ===
with tab2:
    st.subheader("Microstrip Configuration")
    col_img2, col_opt2 = st.columns([2, 1])
    with col_opt2:
        is_ms_diff = st.checkbox("Differential Pair (Diff)", key="ms_is_diff")
        use_ms_sm = st.checkbox("Add Solder Mask", key="ms_use_sm")
    with col_img2:
        st.pyplot(draw_cross_section("Microstrip", is_ms_diff, True, use_ms_sm))

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.number_input("Trace Width W", key='ms_w', format="%.4f")
        st.number_input("Substrate Height H", key='ms_h', format="%.4f")
    with m2:
        st.number_input("Trace Spacing S", key='ms_s', format="%.4f", disabled=not is_ms_diff)
        st.number_input("Copper Thickness T", key='ms_t', format="%.4f")
    with m3:
        st.number_input("Substrate Er", key='ms_er', format="%.4f")
        if use_ms_sm:
            st.number_input("Mask Thickness H_sm", key='ms_sm_h', format="%.4f")
            st.number_input("Mask Er", key='ms_sm_er', format="%.4f")

    st.markdown("### 🎯 Goal Seek (Synthesis)")
    gm1, gm2, gm3 = st.columns([1, 2, 1])
    with gm1:
        st.number_input("Target Impedance (Ω)", key='ms_target_z')
    with gm2:
        st.radio("Parameter to Solve:", ["W", "S"] if is_ms_diff else ["W"], horizontal=True, key='ms_solve_target')
    with gm3:
        st.write("")
        st.button("🚀 Solve for Optimal Dimension", key="btn_solve_ms", use_container_width=True, on_click=ms_goal_seek_callback)

    z_res_ms = get_microstrip_impedance(st.session_state.ms_w, st.session_state.ms_h, st.session_state.ms_t, st.session_state.ms_s, st.session_state.ms_er, is_ms_diff, use_ms_sm, st.session_state.ms_sm_h, st.session_state.ms_sm_er)
    
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #001F3F; border-radius: 10px;'>
            <h2 style='color: #00FFFF;'>{'Differential' if is_ms_diff else 'Single-Ended'} Impedance: {z_res_ms:.2f} Ω</h2>
        </div>
    """, unsafe_allow_html=True)