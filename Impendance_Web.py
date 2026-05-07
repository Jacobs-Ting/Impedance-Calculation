import streamlit as st
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# ================= Page Config & Theme =================
st.set_page_config(page_title="RF Impedance Calculator", page_icon="📡", layout="wide")

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
    'cpw_is_cpwg': True, 'cpw_is_diff': True, 'cpw_use_sm': True,

    'sl_w': 5.0, 'sl_h1': 4.0, 'sl_h2': 4.0, 'sl_t': 1.38, 'sl_er': 4.1, 'sl_s': 8.0,
    'sl_target_z': 50.0, 'sl_solve_target': 'W', 'sl_is_diff': False,

    'em_w': 5.5, 'em_hp': 4.0, 'em_h': 6.0, 'em_t': 0.6, 'em_er': 4.2, 'em_s': 8.0,
    'em_target_z': 50.0, 'em_solve_target': 'W', 'em_is_diff': False
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
        final_z0 = z0_bare * math.sqrt(e_eff_bare / (e_eff_bare + (sm_er - 1) * fill_factor))
    if is_diff:
        if s <= 0: return 9999.0
        return 2 * final_z0 * (1 - 0.48 * math.exp(-0.96 * s / h))
    return final_z0

def get_cpw_impedance(w_bot, w_top, h, g_bot, s_bot, t, er, is_cpwg, is_diff, use_sm, c1, c2, sm_er):
    w_avg = (w_bot + w_top) / 2.0
    side_slope = (w_bot - w_top) / 2.0
    g_avg = g_bot + side_slope
    trap_factor = 0.75 if abs(w_bot - w_top) > 0.1 else 1.0
    raw_delta = (1.25 * t / math.pi) * (1 + math.log(4 * math.pi * w_avg / t)) if t > 0 else 0
    final_delta = raw_delta * trap_factor
    
    def calc_admittance(w_eff, g_eff):
        a, b = w_eff, w_eff + 2 * g_eff
        k1 = a / b
        q1 = get_ellip_ratio(k1)
        k3 = math.tanh(math.pi * a / (4 * h)) / math.tanh(math.pi * b / (4 * h)) if is_cpwg else math.sinh(math.pi * a / (4 * h)) / math.sinh(math.pi * b / (4 * h))
        q3 = get_ellip_ratio(k3)
        num = (1.0 * q1) + (er * q3)
        if use_sm and c1 > 0:
            k_f = math.sinh(math.pi * a / (4 * c1)) / math.sinh(math.pi * b / (4 * c1))
            num += (sm_er - 1.0) * get_ellip_ratio(k_f)
        z0 = ((60 * math.pi) / (q1 + q3)) / math.sqrt(num / (q1 + q3))
        return (1/z0)/2

    if is_diff:
        y_out = calc_admittance(w_avg + final_delta, g_avg - final_delta)
        y_in = calc_admittance(w_avg + final_delta, (s_bot + side_slope)/2.0 - final_delta)
        return 2 * (1 / (y_out + y_in))
    return 1 / (2 * calc_admittance(w_avg + final_delta, g_avg - final_delta))

def get_stripline_impedance(w, h1, h2, t, s, er, is_diff):
    if w <= 0 or h1 <= 0 or h2 <= 0: return 9999.0
    h_near = min(h1, h2)
    h_far = max(h1, h2)
    b = h1 + h2 + t
    z0 = (80 / math.sqrt(er)) * math.log(1.9 * (2*h_near + t) / (0.8*w + t)) * (1 - (h_near / (4 * h_far)))
    if is_diff:
        return 2 * z0 * (1 - 0.347 * math.exp(-2.9 * s / b))
    return z0

def get_embedded_microstrip_impedance(w, hp, h, t, s, er, is_diff):
    if w <= 0 or hp <= 0 or h <= 0 or hp > h: return 9999.0
    e_rp = er * (1 - math.exp(-1.55 * (h / hp)))
    if e_rp <= 0: return 9999.0
    z0 = (60 / math.sqrt(e_rp)) * math.log((5.98 * hp) / (0.8 * w + t))
    if is_diff:
        if s <= 0: return 9999.0
        return 2 * z0 * (1 - 0.48 * math.exp(-0.96 * s / hp))
    return z0

# ================= Goal Seek Callbacks =================
def ms_goal_seek_callback():
    low, high = 0.5, 200.0
    for _ in range(60):
        mid = (low + high) / 2.0
        z = get_microstrip_impedance(mid if st.session_state.ms_solve_target=="W" else st.session_state.ms_w, st.session_state.ms_h, st.session_state.ms_t, mid if st.session_state.ms_solve_target=="S" else st.session_state.ms_s, st.session_state.ms_er, st.session_state.ms_is_diff, st.session_state.ms_use_sm, st.session_state.ms_sm_h, st.session_state.ms_sm_er)
        if st.session_state.ms_solve_target == "W":
            if z > st.session_state.ms_target_z: low = mid
            else: high = mid
        else:
            if z < st.session_state.ms_target_z: low = mid
            else: high = mid
    if st.session_state.ms_solve_target == "W": st.session_state.ms_w = float((low+high)/2)
    else: st.session_state.ms_s = float((low+high)/2)

def cpw_goal_seek_callback():
    low, high = 0.5, 200.0
    w_diff = st.session_state.cpw_w_bot - st.session_state.cpw_w_top
    for _ in range(60):
        mid = (low+high)/2.0
        z = get_cpw_impedance(mid + w_diff/2 if st.session_state.cpw_solve_target=="W" else st.session_state.cpw_w_bot, mid - w_diff/2 if st.session_state.cpw_solve_target=="W" else st.session_state.cpw_w_top, st.session_state.cpw_h, st.session_state.cpw_g, mid if st.session_state.cpw_solve_target=="S" else st.session_state.cpw_s, st.session_state.cpw_t, st.session_state.cpw_er, st.session_state.cpw_is_cpwg, st.session_state.cpw_is_diff, st.session_state.cpw_use_sm, st.session_state.cpw_sm_c1, st.session_state.cpw_sm_c2, st.session_state.cpw_sm_er)
        if st.session_state.cpw_solve_target == "W":
            if z > st.session_state.cpw_target_z: low = mid
            else: high = mid
        else:
            if z < st.session_state.cpw_target_z: low = mid
            else: high = mid
    if st.session_state.cpw_solve_target == "W":
        st.session_state.cpw_w_bot = float((low+high)/2 + w_diff/2)
        st.session_state.cpw_w_top = float(max(0.1, (low+high)/2 - w_diff/2))
    else: st.session_state.cpw_s = float((low+high)/2)

def sl_goal_seek_callback():
    low, high = 0.5, 200.0
    for _ in range(60):
        mid = (low + high) / 2.0
        z = get_stripline_impedance(mid if st.session_state.sl_solve_target=="W" else st.session_state.sl_w, st.session_state.sl_h1, st.session_state.sl_h2, st.session_state.sl_t, mid if st.session_state.sl_solve_target=="S" else st.session_state.sl_s, st.session_state.sl_er, st.session_state.sl_is_diff)
        if st.session_state.sl_solve_target == "W":
            if z > st.session_state.sl_target_z: low = mid
            else: high = mid
        else:
            if z < st.session_state.sl_target_z: low = mid
            else: high = mid
    if st.session_state.sl_solve_target == "W": st.session_state.sl_w = float((low+high)/2)
    else: st.session_state.sl_s = float((low+high)/2)

def em_goal_seek_callback():
    low, high = 0.5, 200.0
    for _ in range(60):
        mid = (low + high) / 2.0
        z = get_embedded_microstrip_impedance(mid if st.session_state.em_solve_target=="W" else st.session_state.em_w, st.session_state.em_hp, st.session_state.em_h, st.session_state.em_t, mid if st.session_state.em_solve_target=="S" else st.session_state.em_s, st.session_state.em_er, st.session_state.em_is_diff)
        if st.session_state.em_solve_target == "W":
            if z > st.session_state.em_target_z: low = mid
            else: high = mid
        else:
            if z < st.session_state.em_target_z: low = mid
            else: high = mid
    if st.session_state.em_solve_target == "W": st.session_state.em_w = float((low+high)/2)
    else: st.session_state.em_s = float((low+high)/2)

# ================= Matplotlib UI Drawing Engine =================
plt.rcParams['axes.edgecolor'] = 'none'

def draw_cross_section(mode="CPW", is_diff=True, is_cpwg=True, has_sm=True, h1=10, h2=10):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    col_gold = '#FFD700'; col_cop = '#B87333'; col_diel = '#607D8B'; col_gnd = '#A9A9A9'; col_sm = '#228B22'
    tw, thick, sm_h, w, s, g = 120, 5, 5, 12, 10, 15
    
    if mode == "Stripline":
        # Safe Visual Scaling (防止因為參數過小導致圖形擠成一團)
        v_h1 = max(h1, 8)
        v_h2 = max(h2, 8)
        h_total = v_h1 + v_h2 + thick
        
        ax.set_xlim(-tw/2 - 20, tw/2 + 20)
        ax.set_ylim(-h_total - 10, 15)
        ax.add_patch(patches.Rectangle((-tw/2, 0), tw, 3, facecolor=col_gnd)) 
        ax.add_patch(patches.Rectangle((-tw/2, -h_total), tw, 3, facecolor=col_gnd)) 
        ax.add_patch(patches.Rectangle((-tw/2, -h_total), tw, h_total, facecolor=col_diel, alpha=0.6))
        
        y_trace = -v_h1 - thick/2
        if is_diff:
            ax.add_patch(patches.Rectangle((-s/2 - w, y_trace), w, thick, facecolor=col_cop))
            ax.add_patch(patches.Rectangle((s/2, y_trace), w, thick, facecolor=col_cop))
            ax.annotate('', xy=(-s/2, y_trace+thick+2), xytext=(s/2, y_trace+thick+2), arrowprops=dict(arrowstyle='<->', color=col_gold))
            ax.text(0, y_trace+thick+3, "S", color=col_gold, ha='center', fontweight='bold')
        else:
            ax.add_patch(patches.Rectangle((-w/2, y_trace), w, thick, facecolor=col_cop))
            
        ax.annotate('', xy=(-tw/2-5, 0), xytext=(-tw/2-5, y_trace+thick), arrowprops=dict(arrowstyle='<->', color=col_gold))
        ax.text(-tw/2-7, (y_trace+thick)/2, "H1", ha='right', color=col_gold, fontweight='bold')
        ax.annotate('', xy=(-tw/2-5, y_trace), xytext=(-tw/2-5, -h_total), arrowprops=dict(arrowstyle='<->', color=col_gold))
        ax.text(-tw/2-7, (y_trace - h_total)/2, "H2", ha='right', color=col_gold, fontweight='bold')
        ax.text(tw/2 + 5, -v_h1, "Er", color=col_gold, fontweight='bold')

    elif mode == "EmbeddedMicrostrip":
        # Safe Visual Scaling (強制讓上方介質高度大於銅箔厚度，完美包覆走線)
        v_hp = max(h1, 10) 
        v_h = max(h2, v_hp + thick + 8) 
        
        ax.set_xlim(-tw/2 - 20, tw/2 + 20)
        ax.set_ylim(-v_hp - 10, v_h - v_hp + 15)
        ax.add_patch(patches.Rectangle((-tw/2, -v_hp), tw, v_h, facecolor=col_diel, alpha=0.8)) 
        ax.add_patch(patches.Rectangle((-tw/2, -v_hp-3), tw, 3, facecolor=col_gnd)) 
        
        y_trace = 0
        if is_diff:
            ax.add_patch(patches.Rectangle((-s/2 - w, y_trace), w, thick, facecolor=col_cop))
            ax.add_patch(patches.Rectangle((s/2, y_trace), w, thick, facecolor=col_cop))
            ax.annotate('', xy=(-s/2, y_trace+thick+2), xytext=(s/2, y_trace+thick+2), arrowprops=dict(arrowstyle='<->', color=col_gold))
            ax.text(0, y_trace+thick+3, "S", color=col_gold, ha='center', fontweight='bold')
        else:
            ax.add_patch(patches.Rectangle((-w/2, y_trace), w, thick, facecolor=col_cop))
        
        ax.annotate('', xy=(-tw/2-5, y_trace), xytext=(-tw/2-5, -v_hp), arrowprops=dict(arrowstyle='<->', color=col_gold))
        ax.text(-tw/2-7, -v_hp/2, "hp", ha='right', color=col_gold, fontweight='bold')
        ax.annotate('', xy=(tw/2+5, -v_hp+v_h), xytext=(tw/2+5, -v_hp), arrowprops=dict(arrowstyle='<->', color=col_gold))
        ax.text(tw/2+7, -v_hp+v_h/2, "h", ha='left', color=col_gold, fontweight='bold')
        ax.text(tw/4, -v_hp/2, "Er", color=col_gold, fontweight='bold')

    else:
        v_h1 = max(h1, 10)
        ax.set_xlim(-tw/2 - 20, tw/2 + 20)
        ax.set_ylim(-v_h1 - 10, thick + sm_h + 15)
        ax.add_patch(patches.Rectangle((-tw/2, -v_h1), tw, v_h1, facecolor=col_diel))
        ax.text(tw/2 + 5, -v_h1/2, "Er", fontweight='bold', color=col_gold)
        ax.add_patch(patches.Rectangle((-tw/2, -v_h1-3), tw, 3, facecolor=col_gnd))
        
        if mode == "Microstrip":
            if is_diff:
                ax.add_patch(patches.Rectangle((-s/2 - w, 0), w, thick, facecolor=col_cop))
                ax.add_patch(patches.Rectangle((s/2, 0), w, thick, facecolor=col_cop))
            else:
                ax.add_patch(patches.Rectangle((-w/2, 0), w, thick, facecolor=col_cop))
            if has_sm: ax.add_patch(patches.Rectangle((-tw/2, 0), tw, sm_h, facecolor=col_sm, alpha=0.5))
            
        elif mode == "CPW":
            if is_diff:
                ax.add_patch(patches.Rectangle((-s/2 - w, 0), w, thick, facecolor=col_cop))
                ax.add_patch(patches.Rectangle((s/2, 0), w, thick, facecolor=col_cop))
            else:
                ax.add_patch(patches.Rectangle((-w/2, 0), w, thick, facecolor=col_cop))
            ax.add_patch(patches.Rectangle((-tw/2, 0), tw/2-w-g, thick, facecolor=col_cop))
            ax.add_patch(patches.Rectangle((w+g, 0), tw/2-w-g, thick, facecolor=col_cop))
            if has_sm: ax.add_patch(patches.Rectangle((-tw/2, 0), tw, thick+sm_h, facecolor=col_sm, alpha=0.5))

    return fig

# ================= UI Layout =================
st.title("📡 RF Impedance Calculator")

col_u, col_p, _ = st.columns([1.5, 2.5, 4])
with col_u: st.session_state.unit = st.radio("📐 Unit:", ["mil", "mm"], horizontal=True)
with col_p:
    presets = {
        "Custom Input": None,
        "JLCPCB JLC04161H-3313 (4-Layer 1.6mm)": {"H": 3.91, "T": 1.38, "Er": 4.1, "C1": 0.8, "C2": 0.5, "CEr": 3.5},
    }
    sel = st.selectbox("📂 Quick Load Stackup Presets:", list(presets.keys()))
    if sel != "Custom Input":
        p = presets[sel]
        for k_val in ["ms_h", "cpw_h", "sl_h1", "em_hp"]: st.session_state[k_val] = p["H"]
        st.session_state.sl_h2 = p["H"]
        st.session_state.em_h = p["H"] * 2
        for k_val in ["ms_t", "cpw_t", "sl_t", "em_t"]: st.session_state[k_val] = p["T"]
        for k_val in ["ms_er", "cpw_er", "sl_er", "em_er"]: st.session_state[k_val] = p["Er"]

t1, t2, t3, t4 = st.tabs(["📝 CPW/CPWG", "📝 Microstrip", "📝 Stripline", "📝 Embedded Microstrip"])

# === Tab 4: Embedded Microstrip ===
with t4:
    st.subheader("Embedded Microstrip Configuration")
    c_img4, c_opt4 = st.columns([2, 1])
    with c_opt4:
        is_em_diff = st.checkbox("Differential Pair (Diff)", key="em_is_diff")
    with c_img4:
        st.pyplot(draw_cross_section("EmbeddedMicrostrip", is_em_diff, True, False, st.session_state.em_hp, st.session_state.em_h))

    st.markdown("---")
    
    ratio = st.session_state.em_h / st.session_state.em_hp if st.session_state.em_hp > 0 else 0
    if ratio < 1.2:
        st.warning(f"⚠️ Warning: For optimal accuracy using the IPC formula, the ratio of Total Height (h) to Trace Height (hp) should be greater than 1.2. (Current ratio: {ratio:.2f})")

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.number_input("Trace Width W", key='em_w', format="%.4f")
        st.number_input("Trace Height hp", key='em_hp', format="%.4f")
        st.number_input("Total Height h", key='em_h', format="%.4f")
    with ec2:
        st.number_input("Trace Spacing S", key='em_s', format="%.4f", disabled=not is_em_diff)
        st.number_input("Copper Thickness T", key='em_t', format="%.4f")
    with ec3:
        st.number_input("Substrate Er", key='em_er', format="%.4f")

    st.markdown("### 🎯 Goal Seek (Synthesis)")
    eg1, eg2, eg3 = st.columns([1, 2, 1])
    with eg1: st.number_input("Target Impedance (Ω)", key='em_target_z', value=50.0)
    with eg2: st.radio("Parameter to Solve:", ["W", "S"] if is_em_diff else ["W"], horizontal=True, key='em_solve_target')
    with eg3: 
        st.write("")
        st.button("🚀 Solve for Embedded", key="btn_solve_em", use_container_width=True, on_click=em_goal_seek_callback)

    z_em = get_embedded_microstrip_impedance(st.session_state.em_w, st.session_state.em_hp, st.session_state.em_h, st.session_state.em_t, st.session_state.em_s, st.session_state.em_er, is_em_diff)
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #001F3F; border-radius: 10px;'><h2 style='color: #00FFFF;'>{'Differential' if is_em_diff else 'Single-Ended'} Embedded: {z_em:.2f} Ω</h2></div>", unsafe_allow_html=True)


# === Tab 3: Stripline ===
with t3:
    st.subheader("Asymmetric Stripline Configuration")
    c_img, c_opt = st.columns([2, 1])
    with c_opt:
        is_sl_diff = st.checkbox("Differential Pair (Diff)", key="sl_is_diff")
    with c_img:
        st.pyplot(draw_cross_section("Stripline", is_sl_diff, True, False, st.session_state.sl_h1, st.session_state.sl_h2))

    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.number_input("Trace Width W", key='sl_w', format="%.4f")
        st.number_input("Top Height H1", key='sl_h1', format="%.4f")
        st.number_input("Bottom Height H2", key='sl_h2', format="%.4f")
    with sc2:
        st.number_input("Trace Spacing S", key='sl_s', format="%.4f", disabled=not is_sl_diff)
        st.number_input("Copper Thickness T", key='sl_t', format="%.4f")
    with sc3:
        st.number_input("Substrate Er", key='sl_er', format="%.4f")

    st.markdown("### 🎯 Goal Seek (Synthesis)")
    sg1, sg2, sg3 = st.columns([1, 2, 1])
    with sg1: st.number_input("Target Impedance (Ω)", key='sl_target_z', value=50.0)
    with sg2: st.radio("Parameter to Solve:", ["W", "S"] if is_sl_diff else ["W"], horizontal=True, key='sl_solve_target')
    with sg3: 
        st.write("")
        st.button("🚀 Solve for Stripline", key="btn_solve_sl", use_container_width=True, on_click=sl_goal_seek_callback)

    z_sl = get_stripline_impedance(st.session_state.sl_w, st.session_state.sl_h1, st.session_state.sl_h2, st.session_state.sl_t, st.session_state.sl_s, st.session_state.sl_er, is_sl_diff)
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #001F3F; border-radius: 10px;'><h2 style='color: #00FFFF;'>{'Differential' if is_sl_diff else 'Single-Ended'} Stripline: {z_sl:.2f} Ω</h2></div>", unsafe_allow_html=True)

# === Tab 1 & Tab 2 ===
with t1:
    st.subheader("CPW / CPWG Configuration")
    col_img, col_opt = st.columns([2, 1])
    with col_opt:
        is_cpwg = st.checkbox("Ground Backed (CPWG)", key="cpw_is_cpwg")
        is_cpw_diff = st.checkbox("Differential Pair (Diff)", key="cpw_is_diff")
        use_cpw_sm = st.checkbox("Add Solder Mask", key="cpw_use_sm")
    with col_img: st.pyplot(draw_cross_section("CPW", is_cpw_diff, is_cpwg, use_cpw_sm, st.session_state.cpw_h))
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
    with g1: st.number_input("Target Impedance (Ω)", key='cpw_target_z', value=100.0)
    with g2: st.radio("Parameter to Solve:", ["W", "S"] if is_cpw_diff else ["W"], horizontal=True, key='cpw_solve_target')
    with g3: 
        st.write("")
        st.button("🚀 Solve for CPW", key="btn_solve_cpw", use_container_width=True, on_click=cpw_goal_seek_callback)
    z_res = get_cpw_impedance(st.session_state.cpw_w_bot, st.session_state.cpw_w_top, st.session_state.cpw_h, st.session_state.cpw_g, st.session_state.cpw_s, st.session_state.cpw_t, st.session_state.cpw_er, is_cpwg, is_cpw_diff, use_cpw_sm, st.session_state.cpw_sm_c1, st.session_state.cpw_sm_c2, st.session_state.cpw_sm_er)
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #001F3F; border-radius: 10px;'><h2 style='color: #00FFFF;'>{'Differential' if is_cpw_diff else 'Single-Ended'} Impedance: {z_res:.2f} Ω</h2></div>", unsafe_allow_html=True)

with t2:
    st.subheader("Microstrip Configuration")
    col_img2, col_opt2 = st.columns([2, 1])
    with col_opt2:
        is_ms_diff = st.checkbox("Differential Pair (Diff)", key="ms_is_diff")
        use_ms_sm = st.checkbox("Add Solder Mask", key="ms_use_sm")
    with col_img2: st.pyplot(draw_cross_section("Microstrip", is_ms_diff, True, use_ms_sm, st.session_state.ms_h))
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
    with gm1: st.number_input("Target Impedance (Ω)", key='ms_target_z', value=50.0)
    with gm2: st.radio("Parameter to Solve:", ["W", "S"] if is_ms_diff else ["W"], horizontal=True, key='ms_solve_target')
    with gm3: 
        st.write("")
        st.button("🚀 Solve for Microstrip", key="btn_solve_ms", use_container_width=True, on_click=ms_goal_seek_callback)
    z_res_ms = get_microstrip_impedance(st.session_state.ms_w, st.session_state.ms_h, st.session_state.ms_t, st.session_state.ms_s, st.session_state.ms_er, is_ms_diff, use_ms_sm, st.session_state.ms_sm_h, st.session_state.ms_sm_er)
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #001F3F; border-radius: 10px;'><h2 style='color: #00FFFF;'>{'Differential' if is_ms_diff else 'Single-Ended'} Impedance: {z_res_ms:.2f} Ω</h2></div>", unsafe_allow_html=True)