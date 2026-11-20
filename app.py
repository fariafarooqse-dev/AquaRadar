import csv
import io
import sqlite3

import pandas as pd
import streamlit as st

import database as db
from model_service import analyze_audio_bytes

st.set_page_config(
    page_title="AquaRadar",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IMPORTANT:
# Analyze Audio shows live model output (model_service.analyze_audio_bytes) and saves each analysis.
# Dashboard and Alert Management are built from saved records in aquaradar.db (database.py).
# The Pipeline Map is still illustrative demo content and must stay labelled as such.
# Language toggle changes only what is shown on screen. Database values stay in English.

# ----------------------------------------------------------------------------
# Language support
# ----------------------------------------------------------------------------
LANG_OPTIONS = ["English", "اردو"]

with st.sidebar:
    lang = st.radio("Language / زبان", LANG_OPTIONS, key="lang", horizontal=True)

UR = lang == "اردو"


def tr(en: str, ur: str) -> str:
    """Return the Urdu text when Urdu is selected, otherwise English."""
    return ur if UR else en


if UR:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap');
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li,
        .stApp textarea, .stApp input,
        [data-testid="stMarkdownContainer"] *, [data-testid="stCaptionContainer"] *,
        [data-testid="stWidgetLabel"] *, [data-testid="stAlert"] * {
            font-family: 'Noto Naskh Arabic', 'Segoe UI', Tahoma, sans-serif !important;
        }
        /* Keep Streamlit's icon font, otherwise icons show as text (arrow_down, upload...) */
        .stApp span[data-testid="stIconMaterial"],
        .stApp span[class*="material-symbols"],
        .stApp span[class*="material-icons"],
        .stApp i[class*="material"] {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"],
        [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stCaptionContainer"],
        [data-testid="stMain"] [data-testid="stAlert"] {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Labels and translations
# ----------------------------------------------------------------------------
CLASS_LABELS_EN = {
    "leak": "Resembles leak examples",
    "no_leak": "Resembles no-leak examples",
    "noise": "Resembles environmental noise",
}
CLASS_LABELS_UR = {
    "leak": "لیک کی مثالوں جیسی آواز",
    "no_leak": "بغیر لیک کی مثالوں جیسی آواز",
    "noise": "ماحول کے شور جیسی آواز",
}

SHORT_LABELS = {
    "leak": ("Leak-like", "لیک جیسی"),
    "no_leak": ("No-leak-like", "بغیر لیک جیسی"),
    "noise": ("Noise", "شور"),
}

# Display-only translations for values stored in English in the database.
VALUE_UR = {
    # analysis status
    "completed": "مکمل",
    "uncertain": "غیر یقینی",
    "error": "خرابی",
    # alert statuses
    "New": "نیا",
    "Under Review": "زیرِ جائزہ",
    "Inspection Required": "معائنہ درکار",
    "Inspected": "معائنہ ہو چکا",
    "Resolved": "حل شدہ",
    # inspection outcomes
    "Leak found and repaired": "لیک ملا اور مرمت ہو گئی",
    "Leak not found": "لیک نہیں ملا",
    "Inconclusive": "نتیجہ واضح نہیں",
    "False alarm": "غلط الارم",
    # water supply sources
    "Municipal supply": "میونسپل سپلائی",
    "Tubewell / borehole": "ٹیوب ویل / بورنگ",
    "Tanker": "ٹینکر",
    "Mixed / other": "ملا جلا / دیگر",
}


def disp(value):
    """Display text for a stored English value (status, outcome...)."""
    if value is None:
        return "-"
    return VALUE_UR.get(value, value) if UR else value


def class_label(cls, fallback="-") -> str:
    labels = CLASS_LABELS_UR if UR else CLASS_LABELS_EN
    return labels.get(cls, fallback)


def building_picker(key: str):
    """Optional building dropdown. Returns a building id or None."""
    names = {b["id"]: b["name"] for b in db.list_buildings()}
    return st.selectbox(
        tr("Building (optional)", "عمارت (اختیاری)"),
        [None] + list(names),
        format_func=lambda i: tr("— none —", "— کوئی نہیں —") if i is None else names[i],
        key=key,
    )


# (background, accent colour, icon, English label, Urdu label, English hint, Urdu hint)
CARD_STYLE = {
    "leak": ("#fdecea", "#c62828", "🔴", "Resembles leak examples", "لیک کی مثالوں جیسی آواز",
             "Leak-like sound found: inspect this location",
             "لیک جیسی آواز ملی: اس جگہ کا معائنہ کریں"),
    "no_leak": ("#e8f5e9", "#2e7d32", "🟢", "Resembles no-leak examples", "بغیر لیک کی مثالوں جیسی آواز",
                "No leak-like sound found",
                "لیک جیسی آواز نہیں ملی"),
    "noise": ("#eceff1", "#455a64", "⚪", "Resembles environmental noise", "ماحول کے شور جیسی آواز",
              "Environmental noise only: record again",
              "صرف ماحول کا شور: دوبارہ ریکارڈ کریں"),
    "uncertain": ("#fff8e1", "#b26a00", "🟡", "Uncertain result", "نتیجہ غیر یقینی",
                  "Unclear result: record again",
                  "نتیجہ غیر یقینی: دوبارہ ریکارڈ کریں"),
}


def result_card(result) -> None:
    """Big colour-coded card so the result is clear at a glance."""
    key = result.predicted_class if result.status == "completed" else "uncertain"
    bg, fg, icon, en, ur, hint_en, hint_ur = CARD_STYLE.get(key, CARD_STYLE["uncertain"])
    title = ur if UR else en
    hint = hint_ur if UR else hint_en
    direction = "rtl" if UR else "ltr"
    align = "right" if UR else "left"
    st.markdown(
        f"""<div style="background:{bg};border-left:8px solid {fg};
        padding:16px 20px;border-radius:10px;margin-bottom:12px;
        direction:{direction};text-align:{align};">
        <div style="font-size:1.4rem;font-weight:700;color:{fg};">{icon} {title}</div>
        <div style="font-size:1.15rem;color:{fg};">{hint}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def show_result(result) -> None:
    """Render an AnalysisResult exactly as returned by the model service."""
    st.markdown("#### " + tr("Live model result", "ماڈل کا براہِ راست نتیجہ"))
    if result.status == "completed":
        result_card(result)
        st.caption(result.message)
        c1, c2, c3 = st.columns(3)
        pair = SHORT_LABELS.get(result.predicted_class, (str(result.predicted_class),) * 2)
        c1.caption(tr("Model result", "ماڈل کا نتیجہ"))
        c1.markdown("**" + tr(*pair) + "**")
        c2.caption(tr("Model score", "ماڈل اسکور"))
        c2.markdown("**" + (f"{result.confidence:.2f}" if result.confidence is not None else "n/a") + "**")
        c3.caption(tr("Model version", "ماڈل ورژن"))
        c3.markdown("**" + (result.model_version or "n/a") + "**")
    elif result.status == "uncertain":
        result_card(result)
        st.warning(result.message)
        if result.confidence is not None:
            st.metric(tr("Highest model score", "سب سے زیادہ ماڈل اسکور"), f"{result.confidence:.2f}")
    else:
        st.error(result.message)
    for w in result.warnings:
        st.caption(tr("Note: ", "نوٹ: ") + str(w))
    st.caption(tr(
        "Model score = the model's estimate for the top class, averaged over 5-second windows. "
        "It is not a calibrated probability that a leak exists.",
        "ماڈل اسکور سے مراد سب سے اوپر والی کلاس کے لیے ماڈل کا اندازہ ہے، جس کا اوسط 5 سیکنڈ کی ونڈوز پر نکالا جاتا ہے۔ "
        "یہ اس بات کا درست امکان نہیں کہ لیک موجود ہے۔",
    ))


def flash(message: str) -> None:
    st.session_state["flash"] = message


def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


def default_reason(result) -> str:
    if result.status == "completed" and result.predicted_class == "leak":
        return "Potential leak-like audio (model screening result)"
    if result.status == "uncertain":
        return "Uncertain model result - manual review requested"
    return "Operator-requested inspection"


def score_text(value) -> str:
    return f"{value:.2f}" if value is not None else "-"


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
st.title("💧 AquaRadar")
st.caption(tr(
    "AI-assisted acoustic screening and maintenance coordination — prototype",
    "مصنوعی ذہانت کی مدد سے آواز کے ذریعے لیک کی ابتدائی جانچ اور مرمت کی رابطہ کاری — پروٹو ٹائپ",
))

PAGES = ["Dashboard", "Buildings", "Analyze Audio", "Pipeline Map", "Alert Management", "Reports", "About"]
PAGE_NAMES = {
    "Dashboard": ("Dashboard", "ڈیش بورڈ"),
    "Buildings": ("Building Profiles", "عمارت کے پروفائل"),
    "Analyze Audio": ("Analyze Audio", "آڈیو کا تجزیہ"),
    "Pipeline Map": ("Pipeline Map", "پائپ لائن نقشہ"),
    "Alert Management": ("Alert Management", "الرٹ مینجمنٹ"),
    "Reports": ("Reports", "رپورٹس"),
    "About": ("About & Limitations", "تعارف اور حدود"),
}

with st.sidebar:
    st.header(tr("Navigation", "نیویگیشن"))
    page = st.radio(
        tr("Open page", "صفحہ کھولیں"),
        PAGES,
        format_func=lambda p: tr(*PAGE_NAMES[p]),
        key="page",
    )
    st.divider()
    st.caption(tr(
        "⚠️ Prototype: Pipeline Map is illustrative demo content. "
        "Other pages use saved records and the trained model.",
        "⚠️ پروٹو ٹائپ: پائپ لائن نقشہ صرف مثال کے طور پر ہے۔ "
        "باقی صفحات محفوظ شدہ ریکارڈ اور تربیت یافتہ ماڈل استعمال کرتے ہیں۔",
    ))

show_flash()

# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
if page == "Dashboard":
    st.subheader(tr("Overview", "خلاصہ"))
    counts = db.summary_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Analyses saved", "محفوظ تجزیے"), counts["analyses_total"])
    c2.metric(tr("Open alerts", "کھلے الرٹس"), counts["alerts_open"])
    c3.metric(tr("Awaiting inspection", "معائنے کے منتظر"), counts["alerts_by_status"]["Inspection Required"])
    c4.metric(tr("Resolved alerts", "حل شدہ الرٹس"), counts["alerts_by_status"]["Resolved"])
    by_status = ", ".join(f"{disp(k)}: {v}" for k, v in sorted(counts["analyses_by_status"].items()))
    st.caption(tr(
        f"Analyses by status: {by_status or 'none yet'}. Computed from records saved in aquaradar.db; times are UTC.",
        f"حالت کے لحاظ سے تجزیے: {by_status or 'ابھی کوئی نہیں'}۔ یہ aquaradar.db میں محفوظ ریکارڈ سے نکالے گئے ہیں؛ وقت UTC میں ہے۔",
    ))

    all_rows = db.list_analyses(limit=1000)
    if all_rows:
        st.subheader(tr("Analyses by model result", "ماڈل کے نتیجے کے لحاظ سے تجزیے"))
        tally = {}
        for r in all_rows:
            pair = SHORT_LABELS.get(r["predicted_class"], ("Uncertain / error", "غیر یقینی / خرابی"))
            label = tr(*pair)
            tally[label] = tally.get(label, 0) + 1
        st.bar_chart(pd.Series(tally, name=tr("Analyses", "تجزیے")), horizontal=True)

    by_b = db.building_summary()
    if by_b:
        st.subheader(tr("By building", "عمارت کے لحاظ سے"))
        st.dataframe(
            [
                {
                    tr("Building", "عمارت"): x["name"],
                    tr("Analyses", "تجزیے"): x["analyses"],
                    tr("Open alerts", "کھلے الرٹس"): x["open_alerts"],
                }
                for x in by_b
            ],
            width="stretch",
            hide_index=True,
        )

    st.subheader(tr("Recent analyses", "حالیہ تجزیے"))
    recent = db.list_analyses(limit=10)
    if not recent:
        st.info(tr("No analyses saved yet. Upload a recording on the Analyze Audio page.",
                   "ابھی کوئی تجزیہ محفوظ نہیں۔ آڈیو کا تجزیہ صفحے پر ریکارڈنگ اپ لوڈ کریں۔"))
    else:
        st.dataframe(
            [
                {
                    tr("Time (UTC)", "وقت (UTC)"): r["created_at"],
                    tr("File", "فائل"): r["filename"],
                    tr("Building", "عمارت"): r.get("building_name") or "-",
                    tr("Location", "مقام"): r["location_label"] or "-",
                    tr("Status", "حالت"): disp(r["status"]),
                    tr("Model result", "ماڈل کا نتیجہ"): class_label(r["predicted_class"]),
                    tr("Model score", "ماڈل اسکور"): score_text(r["confidence"]),
                }
                for r in recent
            ],
            width="stretch",
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# Building profiles
# ----------------------------------------------------------------------------
elif page == "Buildings":
    st.subheader(tr("Building profiles", "عمارت کے پروفائل"))
    st.caption(tr(
        "A digital water-infrastructure record for each building, even when no pipe blueprints exist. "
        "Enter what is known and leave unknown fields empty.",
        "ہر عمارت کا پانی کے نظام کا ڈیجیٹل ریکارڈ، چاہے پائپ کے نقشے موجود نہ ہوں۔ "
        "جو معلوم ہو وہ لکھیں اور نامعلوم خانے خالی چھوڑ دیں۔",
    ))
    existing = db.list_buildings()
    with st.expander(tr("Add a building", "نئی عمارت شامل کریں"), expanded=not existing):
        with st.form("building_form", clear_on_submit=True):
            b_name = st.text_input(tr("Building name (required)", "عمارت کا نام (لازمی)"))
            b_address = st.text_input(tr("Address / area", "پتہ / علاقہ"))
            f1, f2 = st.columns(2)
            b_age = f1.number_input(tr("Approx. age in years (0 = unknown)", "تقریباً عمر سالوں میں (0 = معلوم نہیں)"),
                                    min_value=0, max_value=300, value=0, step=1)
            b_floors = f2.number_input(tr("Number of floors (0 = unknown)", "منزلوں کی تعداد (0 = معلوم نہیں)"),
                                       min_value=0, max_value=100, value=0, step=1)
            b_source = st.selectbox(tr("Water supply source", "پانی کا ذریعہ"), db.WATER_SOURCES, format_func=disp)
            b_tank = st.text_input(tr("Tank location", "ٹینک کی جگہ"),
                                   placeholder=tr("e.g., roof, north side", "مثلاً: چھت، شمالی طرف"))
            b_pump = st.text_input(tr("Pump location", "پمپ کی جگہ"),
                                   placeholder=tr("e.g., ground floor utility room", "مثلاً: گراؤنڈ فلور کا یوٹیلیٹی روم"))
            b_routes = st.text_area(tr("Known pipe routes", "معلوم پائپ کے راستے"),
                                    placeholder=tr("e.g., main line runs along the east wall",
                                                   "مثلاً: مرکزی لائن مشرقی دیوار کے ساتھ جاتی ہے"))
            b_issues = st.text_area(tr("Previous maintenance issues", "پچھلے مرمت کے مسائل"))
            submitted = st.form_submit_button(tr("Save building", "عمارت محفوظ کریں"), type="primary")
        if submitted:
            try:
                new_id = db.create_building(
                    b_name, address=b_address, age_years=int(b_age) or None, floors=int(b_floors) or None,
                    water_source=b_source, tank_location=b_tank, pump_location=b_pump,
                    pipe_routes=b_routes, past_issues=b_issues,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(tr(f"Building #{new_id} saved.", f"عمارت #{new_id} محفوظ ہو گئی۔"))
                st.rerun()

    if not existing:
        st.info(tr("No buildings yet. Add your first building above.",
                   "ابھی کوئی عمارت نہیں۔ اوپر اپنی پہلی عمارت شامل کریں۔"))
    else:
        def _txt(v):
            return "-" if v is None else str(v)

        st.dataframe(
            [
                {
                    tr("Building", "عمارت"): b["name"],
                    tr("Address / area", "پتہ / علاقہ"): _txt(b["address"]),
                    tr("Age (years)", "عمر (سال)"): _txt(b["age_years"]),
                    tr("Floors", "منزلیں"): _txt(b["floors"]),
                    tr("Water source", "پانی کا ذریعہ"): disp(b["water_source"]),
                    tr("Tank", "ٹینک"): _txt(b["tank_location"]),
                    tr("Pump", "پمپ"): _txt(b["pump_location"]),
                    tr("Pipe routes", "پائپ کے راستے"): _txt(b["pipe_routes"]),
                    tr("Past issues", "پچھلے مسائل"): _txt(b["past_issues"]),
                }
                for b in existing
            ],
            width="stretch",
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# Analyze Audio
# ----------------------------------------------------------------------------
elif page == "Analyze Audio":
    st.subheader(tr("Analyze an audio recording", "آڈیو ریکارڈنگ کا تجزیہ کریں"))
    st.write(tr(
        "Upload a WAV recording (at least 5 seconds) and run the trained model. "
        "The result is a screening signal, not confirmation of a leak.",
        "WAV ریکارڈنگ (کم از کم 5 سیکنڈ) اپ لوڈ کریں اور تربیت یافتہ ماڈل چلائیں۔ "
        "نتیجہ صرف ابتدائی جانچ کا اشارہ ہے، لیک کی تصدیق نہیں۔",
    ))
    st.info(tr(
        "**3 steps:** 1) Place the sensor firmly on the pipe.  2) Record at least 5 seconds as a WAV file.  "
        "3) Upload it here and press Analyze.",
        "**3 قدم:** 1) سینسر کو پائپ پر مضبوطی سے لگائیں۔  2) کم از کم 5 سیکنڈ کی WAV ریکارڈنگ کریں۔  "
        "3) یہاں اپ لوڈ کریں اور تجزیہ دبائیں۔",
    ))
    audio = st.file_uploader(
        tr("Choose an audio file", "آڈیو فائل منتخب کریں"),
        type=["wav"],
        help=tr(
            "The pilot model was trained on 5-second windows from our own rig recordings. "
            "Longer recordings are analysed window by window (first 2 minutes).",
            "پائلٹ ماڈل ہماری اپنی رِگ کی ریکارڈنگز کی 5 سیکنڈ کی ونڈوز پر تربیت یافتہ ہے۔ "
            "لمبی ریکارڈنگز ونڈو بہ ونڈو جانچی جاتی ہیں (پہلے 2 منٹ)۔",
        ),
        key="audio_uploader",
    )
    building_id = building_picker("analyze_building")
    location = st.text_input(
        tr("Location / pipe segment label", "مقام / پائپ کے حصے کا نام"),
        placeholder=tr("e.g., Engineering Block - Segment A", "مثلاً: انجینئرنگ بلاک - حصہ A"),
        key="analyze_location",
    )
    notes = st.text_area(
        tr("Optional inspection notes", "معائنے کے اختیاری نوٹس"),
        placeholder=tr("Recording context, device, or environmental noise...",
                       "ریکارڈنگ کا پس منظر، آلہ، یا ماحول کا شور..."),
        key="analyze_notes",
    )
    if audio is None:
        st.caption(tr("No file uploaded yet.", "ابھی کوئی فائل اپ لوڈ نہیں ہوئی۔"))
    else:
        file_bytes = audio.getvalue()
        st.audio(audio)
        st.success(tr(f"File received: {audio.name}", f"فائل موصول ہو گئی: {audio.name}"))
        result_key = (audio.name, len(file_bytes))
        if st.button(tr("Analyze recording", "ریکارڈنگ کا تجزیہ کریں"), type="primary", key="analyze_btn"):
            with st.spinner(tr("Analyzing recording...", "ریکارڈنگ کا تجزیہ ہو رہا ہے...")):
                result = analyze_audio_bytes(file_bytes, audio.name)
                try:
                    analysis_id = db.save_analysis(result, audio.name, location, notes, building_id=building_id)
                except sqlite3.Error:
                    analysis_id = None
            st.session_state["analysis"] = {"key": result_key, "result": result, "analysis_id": analysis_id}
        saved = st.session_state.get("analysis")
        if saved and saved["key"] == result_key:
            result, analysis_id = saved["result"], saved["analysis_id"]
            show_result(result)
            if analysis_id is None:
                st.error(tr(
                    "The result could not be saved to the database, so an alert cannot be created from it.",
                    "نتیجہ ڈیٹا بیس میں محفوظ نہیں ہو سکا، اس لیے اس سے الرٹ نہیں بنایا جا سکتا۔",
                ))
            else:
                st.caption(tr(f"Saved as analysis #{analysis_id}.", f"تجزیہ نمبر #{analysis_id} کے طور پر محفوظ ہوا۔"))
                created = st.session_state.setdefault("alerts_created", {})
                if result.status == "error":
                    st.caption(tr(
                        "No alert can be created from a failed analysis. Use Alert Management to raise a manual alert.",
                        "ناکام تجزیے سے الرٹ نہیں بن سکتا۔ دستی الرٹ کے لیے الرٹ مینجمنٹ استعمال کریں۔",
                    ))
                elif analysis_id in created:
                    st.info(tr(
                        f"Alert #{created[analysis_id]} was created from this analysis. "
                        "Manage it on the Alert Management page.",
                        f"اس تجزیے سے الرٹ #{created[analysis_id]} بن چکا ہے۔ اسے الرٹ مینجمنٹ صفحے پر دیکھیں۔",
                    ))
                else:
                    st.markdown("#### " + tr("Create an inspection alert (optional)",
                                             "معائنے کا الرٹ بنائیں (اختیاری)"))
                    st.caption(tr(
                        "An alert is a maintenance record for follow-up. It does not confirm a leak. "
                        "The location label above is required.",
                        "الرٹ فالو اپ کے لیے مرمت کا ریکارڈ ہے۔ یہ لیک کی تصدیق نہیں کرتا۔ اوپر مقام کا نام لازمی ہے۔",
                    ))
                    reason = st.text_input(tr("Reason", "وجہ"), value=default_reason(result),
                                           key=f"reason_{analysis_id}")
                    if st.button(tr("Create alert", "الرٹ بنائیں"), key=f"create_{analysis_id}"):
                        try:
                            alert_id = db.create_alert(location, reason, analysis_id=analysis_id, notes=notes)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            created[analysis_id] = alert_id
                            flash(tr(f"Alert #{alert_id} created with status New.",
                                     f"الرٹ #{alert_id} نئی حالت کے ساتھ بن گیا۔"))
                            st.rerun()
        else:
            st.caption(tr("Press 'Analyze recording' to run the model.",
                          "ماڈل چلانے کے لیے 'ریکارڈنگ کا تجزیہ کریں' دبائیں۔"))

# ----------------------------------------------------------------------------
# Pipeline Map
# ----------------------------------------------------------------------------
elif page == "Pipeline Map":
    st.subheader(tr("Pipeline / campus map", "پائپ لائن / کیمپس کا نقشہ"))
    st.info(tr(
        "Map placeholder. Any locations shown in this starter should be treated as illustrative, not GPS-verified.",
        "نقشے کی جگہ۔ یہاں دکھائے گئے تمام مقامات صرف مثال کے طور پر ہیں، GPS سے تصدیق شدہ نہیں۔",
    ))
    st.markdown(tr(
        "- **Demo point A:** Engineering Block (illustrative)\n"
        "- **Demo point B:** Library (illustrative)\n"
        "- **Demo point C:** Hostel Block (illustrative)",
        "- **نمونہ نقطہ A:** انجینئرنگ بلاک (مثال)\n"
        "- **نمونہ نقطہ B:** لائبریری (مثال)\n"
        "- **نمونہ نقطہ C:** ہاسٹل بلاک (مثال)",
    ))
    st.caption(tr(
        "Do not imply precise leak localization. A location label is only as reliable as the data entered by an operator.",
        "لیک کی درست جگہ کا دعویٰ نہ کریں۔ مقام کا نام اتنا ہی قابلِ اعتماد ہے جتنا آپریٹر نے درج کیا۔",
    ))

# ----------------------------------------------------------------------------
# Alert Management
# ----------------------------------------------------------------------------
elif page == "Alert Management":
    st.subheader(tr("Alert management", "الرٹ مینجمنٹ"))
    st.caption(tr(
        "Alerts are maintenance records created from an analysis or by an operator. "
        "They are not proof that a leak exists. Times are UTC.",
        "الرٹس مرمت کے ریکارڈ ہیں جو کسی تجزیے سے یا آپریٹر کی طرف سے بنتے ہیں۔ "
        "یہ لیک کا ثبوت نہیں۔ وقت UTC میں ہے۔",
    ))
    status_filter = st.selectbox(
        tr("Show alerts", "الرٹس دکھائیں"),
        ["All"] + db.STATUSES,
        format_func=lambda s: tr("All", "سب") if s == "All" else disp(s),
        key="alert_filter",
    )
    shown = db.list_alerts(None if status_filter == "All" else status_filter)
    if not shown:
        st.info(tr("No alerts to show yet. Create one from an analysis or with the manual form below.",
                   "ابھی کوئی الرٹ نہیں۔ کسی تجزیے سے یا نیچے دستی فارم سے بنائیں۔"))
    else:
        st.dataframe(
            [
                {
                    tr("Alert", "الرٹ"): f"#{a['id']}",
                    tr("Created (UTC)", "بنایا گیا (UTC)"): a["created_at"],
                    tr("Building", "عمارت"): a.get("building_name") or "-",
                    tr("Location", "مقام"): a["location_label"],
                    tr("Reason", "وجہ"): a["reason"],
                    tr("Status", "حالت"): disp(a["status"]),
                    tr("Model result", "ماڈل کا نتیجہ"): (
                        tr("Manual alert", "دستی الرٹ") if a["analysis_id"] is None
                        else class_label(a["predicted_class"], tr("Uncertain / none", "غیر یقینی / کوئی نہیں"))
                    ),
                    tr("Model score", "ماڈل اسکور"): score_text(a["confidence"]),
                    tr("Inspection outcome", "معائنے کا نتیجہ"): disp(a["inspection_outcome"]),
                }
                for a in shown
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander(tr("Create a manual alert", "دستی الرٹ بنائیں")):
        m_location = st.text_input(tr("Location / pipe segment label", "مقام / پائپ کے حصے کا نام"),
                                   key="manual_location")
        m_building = building_picker("manual_building")
        m_reason = st.text_input(tr("Reason", "وجہ"), key="manual_reason",
                                 placeholder=tr("e.g., Operator heard hissing", "مثلاً: آپریٹر نے سیٹی جیسی آواز سنی"))
        m_notes = st.text_area(tr("Notes (optional)", "نوٹس (اختیاری)"), key="manual_notes")
        if st.button(tr("Create manual alert", "دستی الرٹ بنائیں"), key="manual_create"):
            try:
                alert_id = db.create_alert(m_location, m_reason, notes=m_notes, building_id=m_building)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(tr(f"Manual alert #{alert_id} created with status New.",
                         f"دستی الرٹ #{alert_id} نئی حالت کے ساتھ بن گیا۔"))
                st.rerun()

    open_alerts = [a for a in db.list_alerts() if a["status"] != "Resolved"]
    if open_alerts:
        st.markdown("#### " + tr("Update an alert", "الرٹ اپ ڈیٹ کریں"))
        by_id = {a["id"]: a for a in open_alerts}
        chosen = st.selectbox(
            tr("Alert", "الرٹ"), list(by_id),
            format_func=lambda i: f"#{i} - {by_id[i]['location_label']} - {disp(by_id[i]['status'])}",
            key="update_alert",
        )
        current = by_id[chosen]
        next_statuses = db.STATUSES[db.STATUSES.index(current["status"]) + 1:]
        new_status = st.selectbox(tr("New status", "نئی حالت"), next_statuses,
                                  format_func=disp, key="update_status")
        outcome = None
        if new_status == "Resolved":
            outcome = st.selectbox(tr("Inspection outcome", "معائنے کا نتیجہ"), db.OUTCOMES,
                                   format_func=disp, key="update_outcome")
        note = st.text_input(tr("Note (optional)", "نوٹ (اختیاری)"), key="update_note")
        if current["notes"]:
            st.caption(tr("Notes so far:", "اب تک کے نوٹس:"))
            st.text(current["notes"])
        st.caption(tr(
            "Status can only move forward. A resolved alert is closed and keeps its inspection outcome.",
            "حالت صرف آگے بڑھ سکتی ہے۔ حل شدہ الرٹ بند ہو جاتا ہے اور اس کا معائنے کا نتیجہ محفوظ رہتا ہے۔",
        ))
        if st.button(tr("Update alert", "الرٹ اپ ڈیٹ کریں"), key="update_btn"):
            try:
                db.update_alert_status(chosen, new_status, outcome, note)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(tr(f"Alert #{chosen} updated to {new_status}.",
                         f"الرٹ #{chosen} کی حالت '{disp(new_status)}' کر دی گئی۔"))
                st.rerun()

# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
elif page == "Reports":
    st.subheader(tr("Reports", "رپورٹس"))
    st.write(tr(
        "Download the saved analysis records. Fields include analysis date, location label, model version, "
        "model result, model score (the model's estimate, not a calibrated probability), and status. Times are UTC.",
        "محفوظ شدہ تجزیوں کا ریکارڈ ڈاؤن لوڈ کریں۔ اس میں تجزیے کی تاریخ، مقام، ماڈل ورژن، ماڈل کا نتیجہ، "
        "ماڈل اسکور (ماڈل کا اندازہ، درست امکان نہیں) اور حالت شامل ہیں۔ وقت UTC میں ہے۔",
    ))
    report_rows = db.list_analyses(limit=1000)
    if not report_rows:
        st.info(tr("No analyses saved yet. Upload a recording on the Analyze Audio page first.",
                   "ابھی کوئی تجزیہ محفوظ نہیں۔ پہلے آڈیو کا تجزیہ صفحے پر ریکارڈنگ اپ لوڈ کریں۔"))
    else:
        st.dataframe([dict(r) for r in report_rows], width="stretch", hide_index=True)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(dict(report_rows[0]).keys()))
        writer.writeheader()
        writer.writerows([dict(r) for r in report_rows])
        st.download_button(
            tr("Download analyses (CSV)", "تجزیے ڈاؤن لوڈ کریں (CSV)"),
            buf.getvalue(),
            file_name="aquaradar_analyses.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------
else:
    st.subheader(tr("About AquaRadar", "ایکوا ریڈار کے بارے میں"))
    st.write(tr(
        "AquaRadar is a prototype concept for AI-assisted screening of pipe-acoustic recordings "
        "and coordination of follow-up inspections.",
        "ایکوا ریڈار ایک پروٹو ٹائپ تصور ہے جو پائپ کی آواز کی ریکارڈنگز کی مصنوعی ذہانت سے ابتدائی جانچ "
        "اور فالو اپ معائنوں کی رابطہ کاری کے لیے ہے۔",
    ))
    st.warning(tr(
        "A model prediction is a screening signal, not confirmation of a leak. "
        "The prototype does not guarantee exact localization or quantify water loss.",
        "ماڈل کی پیش گوئی ابتدائی جانچ کا اشارہ ہے، لیک کی تصدیق نہیں۔ "
        "یہ پروٹو ٹائپ لیک کی درست جگہ یا پانی کے ضیاع کی مقدار کی ضمانت نہیں دیتا۔",
    ))
    st.markdown(tr(
        "- Spectrograms are visualizations, not proof of leakage.\n"
        "- Actual model outputs must come from a trained and evaluated model.\n"
        "- Demo values must remain clearly labelled.\n"
        "- Dataset licence, recording conditions, class balance, and held-out evaluation must be documented.",
        "- اسپیکٹروگرام صرف تصویری نمائش ہیں، لیکیج کا ثبوت نہیں۔\n"
        "- ماڈل کے اصل نتائج تربیت یافتہ اور جانچے گئے ماڈل ہی سے آنے چاہئیں۔\n"
        "- نمونے کی قدریں واضح طور پر لیبل شدہ رہنی چاہئیں۔\n"
        "- ڈیٹا سیٹ کا لائسنس، ریکارڈنگ کے حالات، کلاسز کا توازن اور الگ رکھے گئے ڈیٹا پر جانچ لکھی جانی چاہیے۔",
    ))
