"""
ScallopVision — light-preference analysis demo for Coonamessett Farm Foundation.

Sections:
  1. Hero - what this is, in plain English, so it's self-explanatory without a call.
  2. Live detection reels - the model running frame-by-frame on real footage,
     so a viewer can see it actually working, not just read a heatmap.
  3. Tank session results - precomputed demo sessions (heatmaps + stats + AI narration).
  4. Upload your own footage - runs the live pipeline on a fresh pair of videos.

Schema note: the source footage is a multi-day timelapse, not one continuous
session, so there's no per-scallop tracking or single "session light color" -
see src/pipeline.py's docstring. Results are aggregated per-frame observations
bucketed by that frame's own light color, with one heatmap per color plus an
"overall" combined view.

Experimental design (confirmed via client email + diagram, 2026-07-13): two
independent tanks (top/bottom), each with its own fixed scallop population.
Each tank runs one test light (green or blue) on one side at a time, rotated
to a different corner every ~2 weeks. Red is the ambient night cycle, not a
test color. Hypothesis: does a colored light attract scallops toward it?
Neither the raw per-color detection rate (confounded by brightness/visibility)
nor the positional wall-split (nearly identical across colors including
red/night) currently isolates a light-driven effect on their own - see the
note-box in the app for what's still open.
"""

import json
import os
import sys
import tempfile

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# pipeline.py pulls in cv2 + ultralytics, which depend on native system
# libraries (libGL, glib, etc.) that can be missing/misconfigured on a given
# deploy environment. Import it lazily (only when "Upload your own footage"
# is actually used) so a native-library hiccup there can't take down the
# whole app - the precomputed demo sessions don't need it at all.
from narration import generate_ai_summary, COLOR_LABELS

load_dotenv()

ROOT_DIR = os.path.dirname(__file__)
DEMO_SESSIONS_DIR = os.path.join(ROOT_DIR, "data", "demo_sessions")
REELS_DIR = os.path.join(ROOT_DIR, "data", "reels")

COLOR_HEX = {
    "green": "#3fb27f",
    "blue": "#4d8fd6",
    "red": "#d1665a",
    "unknown": "#7c8896",
}
ACCENT = "#3fb2a6"

st.set_page_config(page_title="ScallopVision", page_icon="🐚", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, footer, header {{visibility: hidden;}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(ellipse 900px 500px at 12% -8%, rgba(63,178,166,0.10), transparent 60%),
        radial-gradient(ellipse 700px 500px at 100% 15%, rgba(77,143,214,0.07), transparent 55%),
        radial-gradient(ellipse 1200px 800px at 50% 110%, rgba(63,178,127,0.06), transparent 60%),
        #0a0e13;
}}

h1, h2, h3, .section-title, .hero h1 {{
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    letter-spacing: -0.01em;
}}

.hero {{
    position: relative;
    padding: 2rem 2.2rem;
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01));
    border: 1px solid #1c2530;
    margin-bottom: 1.5rem;
    overflow: hidden;
}}
.hero::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {ACCENT}, #4d8fd6, transparent);
    opacity: 0.7;
}}
.hero-eyebrow {{
    color: {ACCENT};
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}}
.hero h1 {{
    font-size: 2rem;
    font-weight: 700;
    color: #f4f6f7;
    margin-bottom: 0.6rem;
}}
.hero p {{
    color: #93a1ac;
    font-size: 0.98rem;
    max-width: 800px;
    line-height: 1.6;
}}

.section-card {{
    position: relative;
    background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.006));
    border: 1px solid #1c2530;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.18);
}}
.section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #eef1f3;
    margin-bottom: 0.25rem;
    letter-spacing: -0.005em;
}}
.section-sub {{
    color: #7c8896;
    font-size: 0.88rem;
    margin-bottom: 1rem;
    line-height: 1.5;
}}

.color-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-weight: 600;
    font-size: 0.85rem;
}}
.color-dot {{
    width: 9px; height: 9px; border-radius: 3px;
    box-shadow: 0 0 8px currentColor;
}}

.pct-bar-track {{
    background: #151b22;
    border-radius: 5px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
    border: 1px solid rgba(255,255,255,0.03);
}}
.pct-bar-fill {{
    height: 100%;
    border-radius: 5px;
}}

.big-stat {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #eef1f3;
}}
.small-label {{
    color: #74828d;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.note-box {{
    background: linear-gradient(180deg, rgba(63,178,166,0.06), rgba(63,178,166,0.02));
    border: 1px solid rgba(63,178,166,0.15);
    border-left: 3px solid {ACCENT};
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    color: #9aa5b1;
    font-size: 0.86rem;
    line-height: 1.55;
    margin: 0.9rem 0;
}}

video[data-testid="stVideo"] {{
    border-radius: 10px;
    border: 1px solid #1c2530;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}}

div[data-testid="stImageContainer"] img {{
    border-radius: 8px;
    border: 1px solid #1c2530;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px 6px 0 0;
}}

hr {{ border-color: #1c2530; }}

.feature-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.3rem;
}}
@media (max-width: 900px) {{
    .feature-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
.feature-card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.006));
    border: 1px solid #1c2530;
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
.feature-card:hover {{
    border-color: rgba(63,178,166,0.35);
    transform: translateY(-1px);
}}
.feature-icon {{
    width: 34px; height: 34px;
    border-radius: 9px;
    background: rgba(63,178,166,0.12);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.7rem;
}}
.feature-icon svg {{ width: 18px; height: 18px; }}
.feature-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: #eef1f3;
    margin-bottom: 0.35rem;
}}
.feature-desc {{
    color: #8894a0;
    font-size: 0.85rem;
    line-height: 1.5;
}}
</style>
""", unsafe_allow_html=True)


def color_chip(color):
    hex_color = COLOR_HEX.get(color, "#7c8896")
    label = COLOR_LABELS.get(color, color).title()
    return (
        f'<span class="color-chip" style="color:{hex_color};">'
        f'<span class="color-dot" style="background:{hex_color};"></span>{label}</span>'
    )


def avg_bar(color, avg_value, max_value):
    """Bar scaled to average detections per frame OF THAT COLOR - NOT a
    percentage of total observations. The scallop population is fixed and
    enclosed, so raw observation totals are skewed by how many frames of
    each color happened to get sampled; this per-frame average is the fair,
    apples-to-apples comparison across colors."""
    hex_color = COLOR_HEX.get(color, "#7c8896")
    width_pct = (avg_value / max_value * 100) if max_value else 0
    return (
        f'<div style="margin-bottom:10px;">{color_chip(color)} '
        f'<span style="float:right; color:#e6e9ec; font-weight:700;">{avg_value:.1f}/frame</span>'
        f'<div class="pct-bar-track"><div class="pct-bar-fill" style="width:{width_pct}%; background:{hex_color};"></div></div></div>'
    )


def _dark_layout(fig, height=280, yaxis_title=None):
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.18, x=0),
        yaxis_title=yaxis_title,
        font=dict(color="#9aa5b1"),
        colorway=["#4d8fd6", "#3fb27f"],
    )
    return fig


def camera_breakdown_chart(cam_a, cam_b, colors):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=cam_a.get("name", "Camera A"),
        x=[COLOR_LABELS.get(c, c).title() for c in colors],
        y=[cam_a.get("light_color_frame_counts", {}).get(c, 0) for c in colors],
    ))
    fig.add_trace(go.Bar(
        name=cam_b.get("name", "Camera B"),
        x=[COLOR_LABELS.get(c, c).title() for c in colors],
        y=[cam_b.get("light_color_frame_counts", {}).get(c, 0) for c in colors],
    ))
    return _dark_layout(fig, yaxis_title="Frames sampled")


def daynight_chart(cam_a, cam_b):
    fig = go.Figure()
    for cam in (cam_a, cam_b):
        dn = cam.get("day_night_frame_counts", {})
        fig.add_trace(go.Bar(name=cam.get("name", ""), x=["Day", "Night"],
                              y=[dn.get("day", 0), dn.get("night", 0)]))
    return _dark_layout(fig, height=240, yaxis_title="Frames sampled")


def motion_chart(motion_score):
    by_color = motion_score.get("avg_shift_by_color", {})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[COLOR_LABELS.get(c, c).title() for c in by_color],
        y=list(by_color.values()),
        marker_color=[COLOR_HEX.get(c, "#7c8896") for c in by_color],
    ))
    return _dark_layout(fig, height=220, yaxis_title="Avg. positional shift")


st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Coonamessett Farm Foundation — Scallop Light Study</div>
  <h1>ScallopVision</h1>
  <p>Two cameras watch each tank around the clock. A custom-trained detector finds every scallop
  in every sampled frame, and each frame's own light color is read automatically — no manual
  logging. The result is a data-backed read on which light color scallops actually favor, built
  from a multi-day recording rather than a single short observation window.</p>
</div>
""", unsafe_allow_html=True)


REEL_ORDER = ["top_left", "top_right"]


def list_reels():
    if not os.path.isdir(REELS_DIR):
        return []
    found = {f.replace("_reel.mp4", ""): f for f in os.listdir(REELS_DIR) if f.endswith(".mp4")}
    return [(name, found[name]) for name in REEL_ORDER if name in found]


reels = list_reels()
if reels:
    st.markdown("""
    <div class="section-card">
      <div class="section-title">Detection preview</div>
      <div class="section-sub">Real tank footage, processed frame-by-frame by the current detector at full resolution — every box is a live scallop detection, not a mockup. The source recordings are compressed multi-day timelapses, so each clip below covers roughly half a day of real time — watch the burned-in timestamp (bottom right) and the light color shifting as it cycles through the rotation.</div>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(len(reels))
    for (name, reel), col in zip(reels, cols):
        with col:
            label = name.replace("_", " ").title()
            st.video(os.path.join(REELS_DIR, reel), autoplay=True, loop=True, muted=True)
            st.caption(f"{label} camera")


FEATURES = [
    (
        '<path d="M3 7l6-3 6 3 6-3v13l-6 3-6-3-6 3V7z"/><path d="M9 4v13M15 7v13"/>',
        "Overhead 2D tank mapping",
        "Fuses both camera views into one top-down map of the tank floor, so every scallop's position is shown relative to the whole tank — not just to whichever camera happened to see it.",
    ),
    (
        '<path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/>',
        "Automatic light-color reading",
        "Reads each frame's actual light color directly off the footage — green, blue, or red/night — with no manual logging required.",
    ),
    (
        '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.6" fill="currentColor"/>',
        "Every scallop, every frame",
        "A custom-trained detector locates every visible scallop in every sampled frame at full resolution, not a downsampled approximation.",
    ),
    (
        '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
        "Multi-day pattern tracking",
        "Built to analyze recordings spanning many days, sampling across every light-color rotation instead of one short snapshot.",
    ),
    (
        '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
        "Day/night aware",
        "Automatically separates the ambient day/night cycle from the colored test lights, since red is functionally night for scallops.",
    ),
    (
        '<path d="M7 20V10M12 20V4M17 20v-7"/>',
        "Cross-tank comparison",
        "Puts independent tanks side by side on the same metrics, so you can see whether a pattern actually holds up or not.",
    ),
    (
        '<path d="M4 12h4l3 8 4-16 3 8h4"/>',
        "Motion score",
        "Matches detected scallops between consecutive snapshots to measure how much repositioning is happening — near zero when the tank is still, higher when scallops are actively moving around.",
    ),
]

with st.expander("What ScallopVision does", expanded=False):
    feature_html = '<div class="feature-grid">'
    for icon, title, desc in FEATURES:
        feature_html += f'''
        <div class="feature-card">
          <div class="feature-icon" style="color:{ACCENT};">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{icon}</svg>
          </div>
          <div class="feature-title">{title}</div>
          <div class="feature-desc">{desc}</div>
        </div>'''
    feature_html += '</div>'
    st.markdown(feature_html, unsafe_allow_html=True)


with st.expander("About the experiment & what this data can/can't tell you yet", expanded=False):
    st.markdown("""
    <p style="color:#9aa5b1; line-height:1.6; margin-top:0.2rem;">
    Two independent tanks (top and bottom), each with its own fixed set of scallops that stayed in
    that tank for the full study. Each tank has two cameras mounted on opposite walls, facing each
    other, and at any given time runs a single test light — green or blue — on one side, rotated to
    a different corner roughly every two weeks so each color spent time in every position. Red is
    part of the normal day/night cycle, not a test color: scallops can't see red, so it's the
    functional equivalent of night. The underlying question: does a colored light attract scallops
    toward it?
    </p>
    <p style="color:#9aa5b1; line-height:1.6;">
    Each camera's raw footage is perspective-corrected ("rectified") into a straight-down view of
    just the floor it can see, then the two cameras' rectified views are stitched into one shared
    top-down map — one camera's view is mirrored so "near this camera's wall" lines up consistently
    on both sides. Every scallop the detector finds, in every sampled frame, gets plotted onto that
    map, tagged with whichever light was active in that frame.
    </p>
    <div class="note-box">
    <b>What this demo can and can't tell you yet:</b> the detector and automatic light-reading both
    work reliably — that part is solid. But turning that into "scallops prefer green" requires knowing
    which physical corner had the colored light at each point in time, and matching that against where
    scallops sat. Light scatters through the whole tank, so we can't reliably recover "which corner was
    lit" from color alone — that needs the actual placement log. The two candidate metrics below
    (detection rate by color, and positional split by wall) are shown for transparency, but neither
    cleanly isolates a light-driven effect on their own: detection rate is confounded by how much easier
    scallops are to see under brighter colors, and the positional split turns out nearly identical
    regardless of which color is active — including red/night — suggesting it reflects a standing wall
    preference more than a light response. Worth resolving with the actual corner-assignment schedule.
    </div>
    """, unsafe_allow_html=True)


def list_demo_sessions():
    if not os.path.isdir(DEMO_SESSIONS_DIR):
        return []
    sessions = []
    for name in sorted(os.listdir(DEMO_SESSIONS_DIR)):
        session_dir = os.path.join(DEMO_SESSIONS_DIR, name)
        stats_path = os.path.join(session_dir, "stats.json")
        if os.path.isfile(stats_path):
            sessions.append((name, session_dir))
    return sessions


def render_results(stats: dict, heatmap_paths: dict, session_label: str):
    cam_a = stats.get("camera_a", {})
    cam_b = stats.get("camera_b", {})
    counts = stats.get("light_color_observation_counts", {})
    avg_per_color = stats.get("light_color_avg_per_frame", {})
    frames_total = cam_a.get("frames_sampled", 0) + cam_b.get("frames_sampled", 0)
    obs_total = stats.get("total_floor_observations", 0)
    avg_per_frame = (obs_total / frames_total) if frames_total else 0
    max_avg = max(avg_per_color.values()) if avg_per_color else 1

    tab_overview, tab_motion, tab_summary = st.tabs(["Overview", "Motion & timing", "Summary"])

    with tab_overview:
        st.markdown(f"""
        <div class="note-box">
        <b>Reading these numbers:</b> the tank's scallops are a fixed, enclosed population — they don't
        enter or leave. What varies is how many are visible/positioned favorably in a given frame, so
        "{obs_total:.0f} observations" means {frames_total} independent snapshots × ~{avg_per_frame:.0f}
        scallops visible per snapshot on average, not {obs_total:.0f} distinct animals. The chart on the
        right shows <b>average scallops detected per frame of each color</b> — not a percentage of the
        total, since a color sampled in more frames would otherwise look artificially "preferred" just by
        accumulating a bigger raw total.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.markdown('<div class="section-title">Top-down floor map</div>', unsafe_allow_html=True)
            st.caption(f"Both cameras' rectified views combined · {cam_a.get('name','camera A')} + {cam_b.get('name','camera B')}")
            colors_present = [c for c in heatmap_paths if c != "overall" and os.path.isfile(heatmap_paths[c])]
            tab_labels = ["Overall"] + [COLOR_LABELS.get(c, c).title() for c in colors_present]
            map_tabs = st.tabs(tab_labels)
            with map_tabs[0]:
                if os.path.isfile(heatmap_paths.get("overall", "")):
                    st.image(heatmap_paths["overall"], width="stretch",
                              caption="Every sampled frame combined, regardless of light color — brighter = more scallops detected there")
            for tab, color in zip(map_tabs[1:], colors_present):
                with tab:
                    avg_val = avg_per_color.get(color, 0)
                    st.image(heatmap_paths[color], width="stretch",
                              caption=f"Only frames where {COLOR_LABELS.get(color, color)} was active — {counts.get(color, 0):.0f} total detections, {avg_val:.1f} avg/frame")

        with col2:
            st.markdown('<div class="section-title">Light-color preference</div>', unsafe_allow_html=True)
            st.caption("Average scallops detected per frame, by the light color active in that frame")
            bars_html = "".join(avg_bar(c, v, max_avg) for c, v in sorted(avg_per_color.items(), key=lambda kv: kv[1], reverse=True))
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="small-label">Avg / frame</div>'
                            f'<div class="big-stat">{avg_per_frame:.0f}</div>'
                            f'<div class="small-label">scallops detected</div>',
                            unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="small-label">{cam_a.get("name","Camera A")}</div>'
                            f'<div class="big-stat">{cam_a.get("frames_sampled",0)}</div>'
                            f'<div class="small-label">frames sampled</div>',
                            unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="small-label">{cam_b.get("name","Camera B")}</div>'
                            f'<div class="big-stat">{cam_b.get("frames_sampled",0)}</div>'
                            f'<div class="small-label">frames sampled</div>',
                            unsafe_allow_html=True)

        with st.expander("Raw stats JSON"):
            st.json(stats)

    with tab_motion:
        motion_score = stats.get("motion_score", {})
        if motion_score.get("frame_pairs_matched"):
            st.markdown('<div class="section-title">Motion score</div>', unsafe_allow_html=True)
            st.caption(
                "How much scallops repositioned between consecutive sampled frames — near zero means "
                "the tank was mostly still, higher means more movement. Not a calibrated speed (sample "
                "spacing in real time is uneven across this timelapse), so read it as relative, not absolute."
            )
            mc1, mc2 = st.columns([1, 2])
            with mc1:
                st.markdown(f'<div class="small-label">Overall</div>'
                            f'<div class="big-stat">{motion_score.get("overall_avg_shift", 0):.0f}</div>'
                            f'<div class="small-label">avg. shift / snapshot pair</div>',
                            unsafe_allow_html=True)
            with mc2:
                st.plotly_chart(motion_chart(motion_score), width='stretch', config={"displayModeBar": False})
        else:
            st.caption("Not enough consecutive-frame matches were available to compute a motion score for this session.")

        st.markdown('<div class="section-title" style="margin-top:1.2rem;">Per-camera breakdown</div>', unsafe_allow_html=True)
        st.caption("How many sampled frames each camera saw under each light color")
        colors_all = list(avg_per_color.keys())
        st.plotly_chart(camera_breakdown_chart(cam_a, cam_b, colors_all), width='stretch',
                         config={"displayModeBar": False})

        st.markdown('<div class="section-title" style="margin-top:1.2rem;">Day vs. night frames sampled</div>', unsafe_allow_html=True)
        st.plotly_chart(daynight_chart(cam_a, cam_b), width='stretch', config={"displayModeBar": False})

    with tab_summary:
        with st.spinner("Generating summary..."):
            narration = generate_ai_summary(stats, session_label)
        st.markdown(narration)


sessions = list_demo_sessions()
if sessions:
    st.markdown("""
    <div class="section-card">
      <div class="section-title">Session results</div>
      <div class="section-sub">Pick a session to see where scallops were detected, broken down by the light color active at the time.</div>
    """, unsafe_allow_html=True)
    names = [n for n, _ in sessions]
    selected = st.selectbox("Choose a session", names, label_visibility="collapsed")
    session_dir = dict(sessions)[selected]
    with open(os.path.join(session_dir, "stats.json")) as f:
        stats = json.load(f)
    heatmap_paths = {"overall": os.path.join(session_dir, "heatmap_overall.png")}
    for color in stats.get("light_color_avg_per_frame", {}):
        heatmap_paths[color] = os.path.join(session_dir, f"heatmap_{color}.png")
    render_results(stats, heatmap_paths, selected)
    st.markdown("</div>", unsafe_allow_html=True)

    # Cross-session comparison - uses every session's own stats.json directly,
    # independent of which one is picked above. Uses avg-per-frame (not raw
    # totals/fractions) since the population is fixed - see render_results.
    tank_avgs = {}
    for _, session_dir in sessions:
        with open(os.path.join(session_dir, "stats.json")) as f:
            s = json.load(f)
        tank_avgs[s.get("tank", session_dir)] = s.get("light_color_avg_per_frame", {})

    if len(tank_avgs) >= 2:
        st.markdown("""
        <div class="section-card">
          <div class="section-title">Session comparison</div>
          <div class="section-sub">Average scallops detected per frame, by color, across the two independent tanks.</div>
        """, unsafe_allow_html=True)
        all_colors = sorted(
            {c for f in tank_avgs.values() for c in f},
            key=lambda c: -max(f.get(c, 0) for f in tank_avgs.values()),
        )
        fig = go.Figure()
        for tank, avgs in tank_avgs.items():
            fig.add_trace(go.Bar(
                name=f"{tank.title()} session",
                x=[COLOR_LABELS.get(c, c).title() for c in all_colors],
                y=[avgs.get(c, 0) for c in all_colors],
            ))
        fig.update_layout(colorway=[ACCENT, "#d1a35a"])
        st.plotly_chart(_dark_layout(fig, height=320, yaxis_title="Avg scallops / frame"),
                         width='stretch', config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No precomputed demo sessions found yet in data/demo_sessions/. Run src/pipeline.py first.")


with st.expander("Upload your own footage (runs the live pipeline)"):
    st.write("Upload two camera videos from the same tank (the ones facing each other).")
    tank_name = st.text_input("Tank name", value="bottom")
    col_a, col_b = st.columns(2)
    with col_a:
        cam_a_name = st.text_input("Camera A name", value="bottom_left")
        cam_a_file = st.file_uploader("Camera A video", type=["mp4"], key="cam_a")
    with col_b:
        cam_b_name = st.text_input("Camera B name", value="bottom_right")
        cam_b_file = st.file_uploader("Camera B video", type=["mp4"], key="cam_b")

    if st.button("Run analysis", disabled=not (cam_a_file and cam_b_file)):
        try:
            from pipeline import run_session
        except ImportError as e:
            st.error(
                f"Couldn't load the detection pipeline in this environment (missing native "
                f"library: {e}). The precomputed demo sessions above are unaffected - this only "
                f"blocks live analysis of freshly uploaded footage."
            )
            st.stop()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cam_a_path = os.path.join(tmp_dir, "cam_a.mp4")
            cam_b_path = os.path.join(tmp_dir, "cam_b.mp4")
            with open(cam_a_path, "wb") as f:
                f.write(cam_a_file.read())
            with open(cam_b_path, "wb") as f:
                f.write(cam_b_file.read())

            out_dir = os.path.join(tmp_dir, "out")
            with st.spinner("Running per-frame detection and light classification — this can take a few minutes on longer videos..."):
                stats, heatmap_paths = run_session(tank_name, cam_a_path, cam_a_name, cam_b_path, cam_b_name, out_dir)
                heatmap_bytes = {}
                for color, path in heatmap_paths.items():
                    with open(path, "rb") as f:
                        heatmap_bytes[color] = f.read()

            display_paths = {}
            for color, data in heatmap_bytes.items():
                p = os.path.join(tempfile.gettempdir(), f"scallopvision_last_heatmap_{color}.png")
                with open(p, "wb") as f:
                    f.write(data)
                display_paths[color] = p

            render_results(stats, display_paths, "your uploaded footage")
