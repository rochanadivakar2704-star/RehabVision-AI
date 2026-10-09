from pathlib import Path
import tempfile
import html
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from src.pose_features import extract_video_features

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
HISTORY = RESULTS / "session_history.csv"
METRICS_FILE = RESULTS / "model_metrics.csv"
PREDICTIONS_FILE = RESULTS / "out_of_fold_predictions.csv"
SEGMENTATION_FILE = ROOT / "Segmentation.csv"

st.set_page_config(
    page_title="RehabVision AI | Movement Intelligence",
    page_icon="🦿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
if "rv_theme" not in st.session_state:
    st.session_state["rv_theme"] = "Light"
theme = st.sidebar.selectbox("Appearance", ["Light", "Dark"], index=0, key="rv_theme_select")
if theme == "Light":
    app_bg, sidebar_bg, panel_bg, line = "#f4f8fc", "#eaf4f8", "#ffffff", "#d7e5ee"
    text_primary, text_secondary, accent = "#16324a", "#526b7e", "#16a995"
    hero_bg, card_bg = "linear-gradient(120deg,#e4f7f3,#e8f3ff)", "linear-gradient(145deg,#ffffff,#f5fbff)"
else:
    app_bg, sidebar_bg, panel_bg, line = "#0b1220", "#101b2d", "#121d30", "#293b54"
    text_primary, text_secondary, accent = "#f2f7ff", "#c7d3e5", "#35d0ba"
    hero_bg, card_bg = "linear-gradient(120deg,#1a334c,#101d30)", "linear-gradient(145deg,#162438,#101b2d)"

st.markdown(f"""
<style>
.stApp {{ background:{app_bg}; color:{text_primary}; }}
[data-testid="stHeader"] {{ background:{app_bg}; }}
[data-testid="stSidebar"] {{ background:{sidebar_bg}; border-right:1px solid {line}; }}
.block-container {{ padding-top:1.5rem; padding-bottom:3rem; max-width:1500px; }}
h1,h2,h3,h4 {{ color:{text_primary} !important; letter-spacing:-.02em; }}
p,li,label,small {{ color:{text_secondary}; }}
.rv-eyebrow {{ color:{accent}; text-transform:uppercase; font-size:.74rem; font-weight:800; letter-spacing:.16em; margin-bottom:.35rem; }}
.rv-hero {{ padding:1.5rem 1.65rem; border:1px solid {line}; border-radius:22px; background:{hero_bg}; margin-bottom:1.2rem; box-shadow:0 12px 32px rgba(20,60,90,.08); }}
.rv-hero p {{ color:{text_secondary}; margin-bottom:0; }}
.rv-chip {{ display:inline-block; padding:.28rem .65rem; margin:.2rem .25rem .2rem 0; border-radius:999px; border:1px solid {line}; background:{panel_bg}; color:{accent}; font-size:.76rem; font-weight:700; }}
.rv-card {{ background:{card_bg}; border:1px solid {line}; border-radius:17px; padding:1rem 1.05rem; min-height:115px; box-shadow:0 8px 24px rgba(25,65,90,.05); }}
.rv-card-label {{ color:{text_secondary}; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
.rv-card-value {{ color:{text_primary}; font-size:1.8rem; font-weight:800; margin:.35rem 0 .1rem; animation:rv-pop .45s ease-out; }}
.rv-card-note {{ color:{accent}; font-size:.78rem; }}
@keyframes rv-pop {{ from {{ opacity:.35; transform:translateY(5px); }} to {{ opacity:1; transform:translateY(0); }} }}
div[data-testid="stMetric"] {{ background:{panel_bg}; border:1px solid {line}; padding:14px 16px; border-radius:15px; }}
div[data-testid="stMetricLabel"] {{ color:{text_secondary}; }}
div[data-testid="stMetricValue"] {{ color:{text_primary}; }}
.stButton>button,.stDownloadButton>button {{ border-radius:11px; border:1px solid {line}; font-weight:700; }}
.stButton>button[kind="primary"] {{ background:linear-gradient(90deg,{accent},#75bfff); border:0; color:#062b35; }}
div[data-testid="stDataFrame"] {{ border:1px solid {line}; border-radius:12px; overflow:hidden; }}
div[data-testid="stAlert"] {{ border-radius:12px; }}
hr {{ border-color:{line}; }}
</style>
""", unsafe_allow_html=True)


def hero(kicker, title, subtitle):
    st.markdown(
        f'<div class="rv-hero"><div class="rv-eyebrow">{html.escape(kicker)}</div>'
        f'<h1 style="margin:.1rem 0 .45rem">{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="rv-card"><div class="rv-card-label">{html.escape(str(label))}</div>'
        f'<div class="rv-card-value">{html.escape(str(value))}</div>'
        f'<div class="rv-card-note">{html.escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def load_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def load_history():
    return load_csv(HISTORY)


def save_history(row):
    new = pd.DataFrame([row])
    old = load_history()
    if not old.empty:
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(HISTORY, index=False)


def load_model_bundle():
    bundles = list(MODELS.glob("*.joblib")) if MODELS.exists() else []
    if not bundles:
        return None, None
    preferred = [p for p in bundles if "random_forest" in p.name.lower()]
    chosen = preferred[0] if preferred else bundles[0]
    try:
        return joblib.load(chosen), None
    except Exception as exc:
        return None, f"Model could not be loaded: {exc}"


def fmt_percent(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"



def build_session_pdf(row):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch,
                            topMargin=0.65*inch, bottomMargin=0.65*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RVTitle", parent=styles["Title"], textColor=colors.HexColor("#147d83"), spaceAfter=10))
    story = [Paragraph("RehabVision AI", styles["RVTitle"]),
             Paragraph("Movement Assessment Summary", styles["Heading2"]), Spacer(1, 8),
             Paragraph("Educational prototype report. Not a diagnosis or a substitute for a physiotherapist.", styles["BodyText"]),
             Spacer(1, 12)]
    rows = [["Field", "Recorded value"]]
    fields = ["timestamp", "exercise", "filename", "status", "predicted_label", "model_confidence",
              "duration_sec", "pose_detection_rate", "mean_visible_landmark_fraction"]
    for key in fields:
        if key in row:
            value = row.get(key, "")
            if key in ("pose_detection_rate", "mean_visible_landmark_fraction"):
                try: value = f"{float(value)*100:.1f}%"
                except Exception: pass
            elif key == "model_confidence":
                try: value = f"{float(value):.3f}" if pd.notna(value) else "Not available"
                except Exception: pass
            rows.append([key.replace("_", " ").title(), str(value)])
    table = Table(rows, colWidths=[2.25*inch, 4.55*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dff6f1")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#16324a")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#d5e3ea")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.extend([table, Spacer(1, 14), Paragraph("Interpretation", styles["Heading3"]),
                  Paragraph("Pose detection and joint-angle values are estimates from video landmarks. Camera angle, lighting, occlusion and tracking quality can affect them. Session trends are descriptive and do not establish clinical recovery.", styles["BodyText"])])
    doc.build(story)
    return buffer.getvalue()


bundle, model_error = load_model_bundle()
history = load_history()
metrics = load_csv(METRICS_FILE)
predictions = load_csv(PREDICTIONS_FILE)
segmentation = load_csv(SEGMENTATION_FILE)
if not segmentation.empty and len(segmentation.columns) == 1 and ";" in str(segmentation.columns[0]):
    try:
        segmentation = pd.read_csv(SEGMENTATION_FILE, sep=";")
    except Exception:
        pass

with st.sidebar:
    st.markdown("## 🦿 RehabVision **AI**")
    st.caption("MOVEMENT INTELLIGENCE PLATFORM")
    st.markdown('<span class="rv-chip">PROTOTYPE</span><span class="rv-chip">COMPUTER VISION</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Movement Lab",
            "Progress Tracker",
            "AI Model Centre",
            "Exercise Library",
            "Assessment Reports",
            "Session History",
            "About & Help",
        ],
        label_visibility="visible",
    )
    st.divider()
    demo_mode = st.toggle("Demo display (500 synthetic sessions)", value=True, help="Shows clearly labelled illustrative demo volume on the Overview page; these are not real assessments.")
    st.caption("For educational demonstration only.")
    st.caption("Not a diagnostic or treatment tool.")

# ---------- Overview ----------
if page == "Overview":
    hero("REHABILITATION ANALYTICS • 02.0", "Movement insights, made visible.", "A camera-based movement-feature prototype for reviewing exercise videos, joint-angle trends and model evaluation.")
    actual_sessions = len(history)
    # Deterministic synthetic data is used only to demonstrate the interface.
    # It is never written into the real session-history file or used as model evidence.
    demo_exercises = ["Squat", "Arm abduction", "Knee extension", "Shoulder flexion", "Heel raise", "Sit-to-stand"]
    rng_demo = np.random.default_rng(20261009)
    demo_history = pd.DataFrame({
        "session_id": [f"DEMO-{i:04d}" for i in range(1, 501)],
        "timestamp": pd.date_range(end=pd.Timestamp.now().normalize(), periods=500, freq="D"),
        "exercise": [demo_exercises[i % len(demo_exercises)] for i in range(500)],
        "pose_detection_rate": np.clip(rng_demo.normal(0.94, 0.035, 500), 0.78, 0.995),
        "mean_visible_landmark_fraction": np.clip(rng_demo.normal(0.91, 0.045, 500), 0.70, 0.99),
        "left_knee_angle_range": np.clip(rng_demo.normal(52, 9, 500), 25, 85),
        "right_knee_angle_range": np.clip(rng_demo.normal(51, 9, 500), 25, 85),
        "duration_sec": np.clip(rng_demo.normal(28, 6, 500), 10, 60).round(1),
        "status": "Synthetic demo"
    })
    display_history = demo_history if demo_mode else history.copy()
    exercise_count = int(display_history["exercise"].nunique()) if "exercise" in display_history.columns and not display_history.empty else 0
    pose_rate = pd.to_numeric(display_history.get("pose_detection_rate", pd.Series(dtype=float)), errors="coerce")
    mean_pose = fmt_percent(pose_rate.mean()) if not pose_rate.dropna().empty else "—"
    model_status = "Bundle loaded" if bundle is not None else "Evaluation only"
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Assessment sessions", len(display_history), "Synthetic demo records" if demo_mode else "Saved locally")
    with c2: metric_card("Exercise categories", exercise_count, "Illustrative exercise mix" if demo_mode else "Across saved sessions")
    with c3: metric_card("Mean pose detection", mean_pose, "Synthetic demo estimate" if demo_mode else "Recorded sessions only")
    with c4: metric_card("Model status", model_status, "Classifier bundle checked separately")
    if demo_mode:
        st.info("DEMO MODE: 500 deterministic synthetic records are generated for interface demonstration only. They are not real assessments, patient data, or model-validation results. Disable Demo display to view saved sessions.")
    st.markdown('<div class="rv-section"><h3>Explore the platform</h3></div>', unsafe_allow_html=True)
    cards = [
        ("🎥", "Movement Lab", "Upload a clip and inspect pose landmarks, estimated joint angles and video-quality indicators.", "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=900&q=80"),
        ("📈", "Progress Tracker", "Explore saved session features over time. Trends are descriptive, not clinical recovery scores.", "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?auto=format&fit=crop&w=900&q=80"),
        ("🧠", "AI Model Centre", "Review saved cross-validation metrics, out-of-fold predictions and the confusion matrix.", "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=80"),
        ("📚", "Exercise Library", "Review recording guidance and general exercise-assessment notes.", "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80"),
    ]
    cols = st.columns(2)
    for i, (emoji, title, desc, photo_url) in enumerate(cards):
        with cols[i % 2]:
            st.image(photo_url, use_container_width=True)
            st.markdown(f'<div class="rv-card" style="min-height:135px"><div style="font-size:1.5rem">{emoji}</div><h3 style="margin:.35rem 0">{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="rv-section"><h3>Recent activity</h3></div>', unsafe_allow_html=True)
    if history.empty:
        st.info("No saved sessions yet. Open Movement Lab to analyse your first exercise video.")
    else:
        show_cols = [c for c in ["timestamp", "exercise", "filename", "status", "pose_detection_rate"] if c in history.columns]
        st.dataframe(history[show_cols].tail(5).iloc[::-1], use_container_width=True, hide_index=True)
    st.info("Safety note: joint angles are estimated from 2D video landmarks. Camera position, occlusion, lighting and tracking errors can affect results.")

# ---------- Movement Lab ----------
elif page == "Movement Lab":
    hero("VIDEO ANALYSIS", "Movement Lab", "Read uploaded exercise videos, detect body-pose landmarks, estimate joint angles, and save completed assessment measurements.")
    with st.container(border=True):
        left, right = st.columns([1.15, 1])
        with left:
            exercise = st.selectbox("Exercise category", ["Arm abduction", "Squat", "Other / exploratory"])
            up = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "m4v"], help="Use a short clip with the person fully visible.")
            st.caption("Tip: record in good lighting, keep the full body in frame, and avoid camera movement.")
        with right:
            st.markdown("#### Before you analyse")
            st.markdown("- Keep the camera steady.\n- Avoid people or objects blocking the movement.\n- Wear clothing that helps distinguish body landmarks.\n- Use the same view when comparing sessions.")
            st.warning("This tool provides experimental measurements, not a clinical judgement.")
    if up is not None and st.button("▶ Analyse video", type="primary", use_container_width=True):
        video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix or ".mp4") as tmp:
                tmp.write(up.getbuffer())
                video_path = tmp.name
            with st.spinner("Extracting pose landmarks and movement features..."):
                features, overlay, ts = extract_video_features(video_path, make_overlay=True)
            pose_rate = float(features.get("pose_detection_rate", np.nan))
            visible = float(features.get("mean_visible_landmark_fraction", np.nan))
            duration = float(features.get("duration_sec", np.nan))
            st.session_state["rv_last_features"] = features
            st.session_state["rv_last_ts"] = ts
            st.session_state["rv_last_overlay"] = overlay
            st.session_state["rv_last_file"] = up.name
            st.session_state["rv_last_exercise"] = exercise
            a, b, c = st.columns(3)
            a.metric("Pose detection rate", fmt_percent(pose_rate))
            b.metric("Visible-landmark fraction", fmt_percent(visible))
            c.metric("Clip duration", f"{duration:.1f} s" if np.isfinite(duration) else "—")
            if overlay is not None:
                st.markdown("#### Pose overlay")
                st.image(overlay, channels="RGB", use_container_width=True)
            else:
                st.error("No pose was detected in the sampled frames.")
            reliable = np.isfinite(pose_rate) and np.isfinite(visible) and pose_rate >= .60 and visible >= .35
            predicted, confidence, status = "", np.nan, "features_only"
            st.markdown("#### Assessment status")
            if not reliable:
                st.error("Low video/pose quality: interpret extracted features cautiously. No reliable assessment status can be assigned.")
                status = "abstain_low_video_quality"
            elif bundle is None:
                st.info("Feature-analysis mode: no compatible trained classifier bundle is installed in the models folder.")
            else:
                try:
                    cols = bundle["feature_cols"]
                    X = pd.DataFrame([{k: features.get(k, np.nan) for k in cols}], columns=cols)
                    model = bundle["pipeline"]
                    pred = int(model.predict(X)[0])
                    predicted = str(bundle["label_encoder"].inverse_transform([pred])[0])
                    try:
                        confidence = float(np.max(model.predict_proba(X)[0]))
                    except Exception:
                        confidence = np.nan
                    if np.isfinite(confidence) and confidence < .60:
                        st.warning(f"Prediction withheld because model confidence is low ({confidence:.2f}).")
                        status = "abstain_low_model_confidence"
                    else:
                        st.success(f"Predicted dataset label: {predicted}")
                        if np.isfinite(confidence):
                            st.caption(f"Model confidence (not a guarantee): {confidence:.3f}")
                        status = "assessed"
                except Exception as exc:
                    st.error(f"The available model bundle could not process these features: {exc}")
                    status = "model_incompatible"
            st.markdown("#### Joint-angle summaries")
            vals = {k: round(float(v), 2) for k, v in features.items()
                    if ("_angle_mean" in k or "_angle_range" in k) and np.isfinite(v)}
            if vals:
                st.dataframe(pd.DataFrame([{"Feature": k, "Estimated value": v} for k, v in vals.items()]), hide_index=True, use_container_width=True)
            else:
                st.caption("No joint-angle summaries were available for this clip.")
            angle_cols = [k for k in ts.columns if k.endswith("_angle") and pd.to_numeric(ts[k], errors="coerce").notna().sum() > 2]
            if angle_cols:
                plot_df = ts[["time_sec"] + angle_cols[:8]].melt(id_vars="time_sec", var_name="Joint angle", value_name="Estimated angle (degrees)")
                plot_df["Joint angle"] = plot_df["Joint angle"].str.replace("_", " ").str.title()
                fig = px.line(plot_df, x="time_sec", y="Estimated angle (degrees)", color="Joint angle",
                              title="Interactive joint-angle timeline", template="plotly_white" if theme == "Light" else "plotly_dark")
                fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10),
                                  xaxis_title="Time (seconds)", yaxis_title="Estimated 2D angle (degrees)")
                st.plotly_chart(fig, use_container_width=True)
            row = {
                "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                "exercise": exercise, "filename": up.name, "status": status,
                "predicted_label": predicted, "model_confidence": confidence, **features,
            }
            save_history(row)
            st.success("Assessment summary saved to local session history.")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("⬇ Download assessment CSV", pd.DataFrame([row]).to_csv(index=False), "rehabvision_assessment.csv", "text/csv", use_container_width=True)
            with d2:
                st.download_button("⬇ Download sampled time series", ts.to_csv(index=False), "movement_timeseries.csv", "text/csv", use_container_width=True)
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
        finally:
            if video_path:
                try:
                    Path(video_path).unlink(missing_ok=True)
                except Exception:
                    pass
    elif up is None:
        st.caption("Your analysis output will appear here after you upload a video and click Analyse video.")

# ---------- Progress Tracker ----------
elif page == "Progress Tracker":
    hero("LONGITUDINAL VIEW", "Progress Tracker", "Explore trends across saved sessions or clearly labelled synthetic demonstration records. These are descriptive features, not validated clinical recovery scores.")
    if demo_mode:
        rng_progress = np.random.default_rng(20261009)
        progress_df = pd.DataFrame({
            "session_id": [f"DEMO-{i:04d}" for i in range(1, 501)],
            "timestamp": pd.date_range(end=pd.Timestamp.now().normalize(), periods=500, freq="D"),
            "exercise": [["Squat", "Arm abduction", "Knee extension", "Shoulder flexion", "Heel raise", "Sit-to-stand"][i % 6] for i in range(500)],
            "pose_detection_rate": np.clip(rng_progress.normal(0.94, 0.035, 500), 0.78, 0.995),
            "mean_visible_landmark_fraction": np.clip(rng_progress.normal(0.91, 0.045, 500), 0.70, 0.99),
            "left_knee_angle_range": np.clip(rng_progress.normal(52, 9, 500), 25, 85),
            "right_knee_angle_range": np.clip(rng_progress.normal(51, 9, 500), 25, 85),
            "duration_sec": np.clip(rng_progress.normal(28, 6, 500), 10, 60).round(1),
            "status": "Synthetic demo"
        })
        st.warning("DEMO DATA: all 500 records on this page are synthetic and intended to demonstrate charts, filters and exports. They must not be reported as patient results.")
    else:
        progress_df = history.copy()
    if progress_df.empty:
        st.info("No saved sessions found. Analyse a video in Movement Lab first, or enable Demo display to explore the dashboard with synthetic data.")
    else:
        hist = progress_df.copy()
        if "timestamp" in hist.columns:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            exercises = sorted(hist["exercise"].dropna().astype(str).unique()) if "exercise" in hist.columns else []
            chosen_ex = st.multiselect("Filter by exercise", exercises, default=exercises)
        with c2:
            if "timestamp" in hist.columns and hist["timestamp"].notna().any():
                min_date = hist["timestamp"].min().date()
                max_date = hist["timestamp"].max().date()
                date_range = st.date_input("Date range", value=(min_date, max_date))
            else:
                date_range = None
        if "exercise" in hist.columns and chosen_ex:
            hist = hist[hist["exercise"].astype(str).isin(chosen_ex)]
        if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and "timestamp" in hist.columns:
            hist = hist[hist["timestamp"].dt.date.between(date_range[0], date_range[1])]
        feature_options = [c for c in ["pose_detection_rate", "mean_visible_landmark_fraction", "left_knee_angle_range", "right_knee_angle_range", "duration_sec"] if c in hist.columns]
        if not hist.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Records in view", len(hist))
            m2.metric("Exercise types", hist["exercise"].nunique() if "exercise" in hist.columns else "—")
            m3.metric("Mean pose detection", fmt_percent(pd.to_numeric(hist.get("pose_detection_rate"), errors="coerce").mean()) if "pose_detection_rate" in hist.columns else "—")
            if feature_options:
                feature = st.selectbox("Feature to plot", feature_options)
                hist = hist.sort_values("timestamp") if "timestamp" in hist.columns else hist
                plot_df = hist.copy()
                plot_df["Feature value"] = pd.to_numeric(plot_df[feature], errors="coerce")
                plot_df["Session"] = plot_df["timestamp"].dt.strftime("%d %b") if "timestamp" in plot_df.columns else np.arange(1, len(plot_df)+1)
                fig = px.line(plot_df, x="Session", y="Feature value", color="exercise" if "exercise" in plot_df.columns else None,
                              title=f"{feature.replace('_', ' ').title()} by record", markers=False,
                              template="plotly_white" if theme == "Light" else "plotly_dark")
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Record date", yaxis_title=feature.replace("_", " ").title())
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist.tail(100).iloc[::-1], use_container_width=True, hide_index=True)
            st.caption("The table displays up to the latest 100 filtered records; CSV export includes all filtered records.")
            st.download_button("⬇ Export filtered records", hist.to_csv(index=False), "rehabvision_demo_or_saved_history.csv", "text/csv")
        else:
            st.warning("No records match these filters.")

# ---------- AI Model Centre ----------
elif page == "AI Model Centre":
    hero("MODEL EVALUATION", "AI Model Centre", "Transparent reporting of the saved REHAB24-6 Random Forest evaluation. These metrics are not a clinical validation.")
    if not metrics.empty:
        row = metrics.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, key, label in [
            (c1, "accuracy", "Accuracy"), (c2, "balanced_accuracy", "Balanced accuracy"),
            (c3, "precision_macro", "Macro precision"), (c4, "recall_macro", "Macro recall"),
            (c5, "f1_macro", "Macro F1"),
        ]:
            with col:
                metric_card(label, fmt_percent(row.get(key)), "5-fold participant-grouped CV")
        st.markdown('<div class="rv-section"><h3>Confusion matrix</h3></div>', unsafe_allow_html=True)
        cm_path = FIGURES / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Saved out-of-fold confusion matrix", use_container_width=True)
        else:
            st.info("Confusion matrix image not found in results/figures.")
        pm_path = FIGURES / "performance_metrics.png"
        if pm_path.exists():
            st.image(str(pm_path), caption="Saved model performance metrics", use_container_width=True)
        st.markdown('<div class="rv-section"><h3>Evaluation summary</h3></div>', unsafe_allow_html=True)
        summary = {
            "Repetitions used": int(row.get("n_repetitions", 0)),
            "Participants": int(row.get("n_participants", 0)),
            "Cross-validation folds": int(row.get("n_splits", 0)),
        }
        cols = st.columns(3)
        for col, (label, value) in zip(cols, summary.items()):
            with col:
                metric_card(label, value)
        if not predictions.empty:
            st.markdown("#### Out-of-fold predictions")
            st.caption("Each row is a prediction generated for a held-out fold. Identifiers are shown for auditability.")
            st.dataframe(predictions.head(100), use_container_width=True, hide_index=True)
            st.download_button("⬇ Download all predictions", predictions.to_csv(index=False), "out_of_fold_predictions.csv", "text/csv")
        st.warning("Preliminary result: frame-index alignment has not been fully verified for every annotation. The evaluation dataset contains only 10 participants. Do not interpret these scores as clinical performance.")
    else:
        st.info("No saved model_metrics.csv found. Train/evaluate the dataset model to populate this page.")
    if bundle is None:
        st.info("No compatible deployable classifier bundle was found in models/. The evaluation metrics shown above are from offline cross-validation and are not automatically used to predict uploaded videos.")
    else:
        st.success("A model bundle was found in models/. Movement Lab will attempt to use it only if its feature schema matches the extracted video features.")
    if model_error:
        st.error(model_error)
    st.markdown("#### Dataset label definition")
    st.markdown("- `0` = incorrect repetition\n- `1` = correct repetition\n\nThese are dataset annotations, not a medical diagnosis.")

# ---------- Exercise Library ----------
elif page == "Exercise Library":
    hero("MOVEMENT REFERENCE", "Exercise Library", "A small educational guide to recording and reviewing common movement clips. This is not a prescribed exercise programme.")
    search_exercise = st.text_input("Search exercise library", placeholder="Search squat, arm, recording...")
    exercise_info = [
        {"title":"Squat", "category":"Lower-body movement", "description":"Record from a stable side or front-oblique view with hips, knees and ankles visible.", "image":"https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=1000&q=85", "tips":["Keep the full body in frame.","Avoid a camera angle that hides the knees or hips.","Use a comfortable, self-selected range of motion.","Stop if you experience pain, dizziness or unusual symptoms."]},
        {"title":"Arm abduction", "category":"Upper-body movement", "description":"Record from the front with both shoulders, elbows and wrists visible.", "image":"https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=1000&q=85", "tips":["Keep the camera at a stable height.","Avoid cropping the hands or shoulders.","Use consistent lighting and distance.","Do not force the arm through a painful range."]},
        {"title":"Video recording checklist", "category":"Capture quality", "description":"Good recording conditions make computer-vision features easier to interpret.", "image":"https://images.unsplash.com/photo-1534438327276-14e5300c3a48?auto=format&fit=crop&w=1000&q=85", "tips":["Use a stable camera and uncluttered background.","Ensure adequate lighting.","Keep relevant joints visible.","Avoid occlusion by people or objects."]},
    ]
    filtered_exercises = [item for item in exercise_info if search_exercise.lower() in (item["title"] + " " + item["category"] + " " + item["description"]).lower()]
    cards = st.columns(3)
    for idx, item in enumerate(filtered_exercises):
        with cards[idx % 3]:
            st.image(item["image"], use_container_width=True)
            st.markdown(f'<div class="rv-card"><div class="rv-eyebrow">{html.escape(item["category"])}</div><h3>{html.escape(item["title"])}</h3><p>{html.escape(item["description"])}</p></div>', unsafe_allow_html=True)
            with st.expander("View recording tips"):
                for tip in item["tips"]:
                    st.markdown(f"- {tip}")
    st.markdown("#### Simplified movement illustrations")
    ill1, ill2 = st.columns(2)
    with ill1:
        st.markdown("**Squat — key joints to keep visible**")
        st.markdown('<svg viewBox="0 0 220 180" width="100%" role="img" aria-label="Simplified squat pose illustration"><circle cx="110" cy="22" r="12" fill="#8ecae6"/><path d="M110 34 L105 74 L78 98 L62 132 M105 74 L133 94 L151 130 M105 46 L75 63 L60 80 M105 46 L133 58 L151 79" stroke="#18a999" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" fill="none"/><circle cx="105" cy="74" r="5" fill="#168aad"/><circle cx="78" cy="98" r="5" fill="#168aad"/><circle cx="133" cy="94" r="5" fill="#168aad"/><path d="M50 138 H170" stroke="#b9d7e8" stroke-width="3" stroke-linecap="round"/></svg>', unsafe_allow_html=True)
    with ill2:
        st.markdown("**Arm abduction — shoulder movement**")
        st.markdown('<svg viewBox="0 0 220 180" width="100%" role="img" aria-label="Simplified arm abduction illustration"><circle cx="110" cy="22" r="12" fill="#8ecae6"/><path d="M110 34 L110 92 M110 49 L76 30 L42 30 M110 49 L144 30 L178 30 M110 92 L88 136 L86 158 M110 92 L132 136 L134 158" stroke="#18a999" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" fill="none"/><circle cx="110" cy="49" r="5" fill="#168aad"/><circle cx="76" cy="30" r="5" fill="#168aad"/><circle cx="144" cy="30" r="5" fill="#168aad"/><path d="M35 167 H185" stroke="#b9d7e8" stroke-width="3" stroke-linecap="round"/></svg>', unsafe_allow_html=True)
    st.info("Illustrations are simplified educational diagrams, not anatomical reference images or clinical exercise prescriptions.")

# ---------- Assessment Reports ----------
elif page == "Assessment Reports":
    hero("EXPORT CENTRE", "Assessment Reports", "Download session records, movement time-series data and model evaluation outputs for review.")
    if history.empty:
        st.info("No assessment records are available yet. Analyse a video in Movement Lab to create a report.")
    else:
        st.markdown("#### Session records")
        st.dataframe(history.tail(100).iloc[::-1], use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ Download complete session history", history.to_csv(index=False), "rehabvision_session_history.csv", "text/csv", use_container_width=True)
        latest_row = history.iloc[-1].to_dict()
        st.download_button("⬇ Download latest assessment as PDF", build_session_pdf(latest_row), "rehabvision_assessment_report.pdf", "application/pdf", use_container_width=True)
        with c2:
            if not predictions.empty:
                st.download_button("⬇ Download model predictions", predictions.to_csv(index=False), "rehabvision_model_predictions.csv", "text/csv", use_container_width=True)
    st.markdown("#### Evaluation files")
    for path, label in [
        (METRICS_FILE, "Model metrics"),
        (PREDICTIONS_FILE, "Out-of-fold predictions"),
        (RESULTS / "repetition_features.csv", "Repetition-level features"),
        (RESULTS / "skipped_repetitions.csv", "Skipped repetitions"),
    ]:
        if path.exists():
            st.write(f"**{label}:** `{path.relative_to(ROOT)}`")
            try:
                st.download_button(f"Download {label}", path.read_bytes(), path.name, "text/csv", key=f"download_{path.name}")
            except Exception:
                st.caption("Could not prepare this file for download.")

# ---------- Session History ----------
elif page == "Session History":
    hero("LOCAL RECORDS", "Session History", "Review and export assessments saved on this machine.")
    if history.empty:
        st.info("No sessions saved yet. Start in Movement Lab.")
    else:
        hist = history.copy()
        c1, c2 = st.columns([1, 1])
        with c1:
            query = st.text_input("Search filename or exercise", placeholder="e.g. squat_clip.mp4")
        with c2:
            statuses = sorted(hist["status"].dropna().astype(str).unique()) if "status" in hist.columns else []
            selected_status = st.multiselect("Filter status", statuses, default=statuses)
        if query.strip():
            mask = pd.Series(False, index=hist.index)
            for col in ["filename", "exercise"]:
                if col in hist.columns:
                    mask |= hist[col].astype(str).str.contains(query.strip(), case=False, na=False)
            hist = hist[mask]
        if "status" in hist.columns and selected_status:
            hist = hist[hist["status"].astype(str).isin(selected_status)]
        st.write(f"**{len(hist)}** matching session(s)")
        st.dataframe(hist.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("⬇ Export displayed sessions", hist.to_csv(index=False), "rehabvision_session_history_filtered.csv", "text/csv")
        with st.expander("Delete local session history"):
            st.warning("This permanently deletes the local CSV history file. It does not delete the original exercise videos.")
            confirm = st.checkbox("I understand and want to delete the session history file.")
            if st.button("Delete session history", disabled=not confirm):
                try:
                    HISTORY.unlink(missing_ok=True)
                    st.success("Session history deleted. Refresh the page to update the view.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete history: {exc}")

# ---------- About & Help ----------
elif page == "About & Help":
    hero("PROJECT INFORMATION", "About RehabVision AI", "A student-built prototype exploring computer vision and predictive analytics for physiotherapy movement review.")
    st.markdown("### What the prototype does")
    st.markdown("- Accepts short exercise videos.\n- Uses the existing pose-feature extraction pipeline to estimate landmarks and movement-related features.\n- Displays pose-detection and landmark-visibility indicators.\n- Plots estimated 2D joint-angle time series when available.\n- Stores assessment summaries locally as CSV.\n- Displays offline REHAB24-6 model-evaluation results when the result files are present.")
    st.markdown("### What it does not do")
    st.markdown("- It is not a medical device or a diagnostic tool.\n- It does not prove recovery or recommend treatment.\n- Offline model evaluation is separate from uploaded-video predictions unless a compatible model bundle is installed.\n- The current evaluation is preliminary and uses only 10 participants.")
    st.markdown("### Technical stack")
    for item in ["Python", "Streamlit", "OpenCV", "MediaPipe pose-feature module", "NumPy / pandas", "scikit-learn / joblib", "Matplotlib"]:
        st.markdown(f'<span class="rv-chip">{html.escape(item)}</span>', unsafe_allow_html=True)
    st.markdown("### Troubleshooting")
    with st.expander("The app does not open"):
        st.write("Keep the terminal running and open http://localhost:8501 in your Windows browser.")
    with st.expander("No pose is detected"):
        st.write("Try a shorter clip, brighter lighting, a steady camera, and make sure the person is fully visible.")
    with st.expander("The AI model page has metrics but Movement Lab has no predictions"):
        st.write("The metrics come from offline cross-validation. Uploaded-video predictions require a compatible `.joblib` bundle in the models folder; the offline results file is not itself a deployable model.")
    st.caption("Educational research prototype. Do not use the output as a substitute for professional assessment.")
