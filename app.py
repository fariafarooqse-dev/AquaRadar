import sqlite3

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

CLASS_LABELS = {
    "leak": "Resembles leak examples",
    "no_leak": "Resembles no-leak examples",
    "noise": "Resembles environmental noise",
}


def show_result(result) -> None:
    """Render an AnalysisResult exactly as returned by the model service."""
    st.markdown("#### Live model result")
    if result.status == "completed":
        (st.warning if result.predicted_class == "leak" else st.info)(result.message)
        c1, c2, c3 = st.columns(3)
        c1.metric("Model result", CLASS_LABELS.get(result.predicted_class, str(result.predicted_class)))
        c2.metric("Model score", f"{result.confidence:.2f}" if result.confidence is not None else "n/a")
        c3.metric("Model version", result.model_version or "n/a")
    elif result.status == "uncertain":
        st.warning(result.message)
        if result.confidence is not None:
            st.metric("Highest model score", f"{result.confidence:.2f}")
    else:
        st.error(result.message)
    for w in result.warnings:
        st.caption(f"Note: {w}")
    st.caption(
        "Model score = share of the model's decision trees voting for the top class, averaged over "
        "1-second windows. It is not a probability that a leak exists."
    )


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


st.title("💧 AquaRadar")
st.caption("AI-assisted acoustic screening and maintenance coordination — prototype")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Open page",
        [
            "Dashboard",
            "Analyze Audio",
            "Pipeline Map",
            "Alert Management",
            "Reports",
            "About & Limitations",
        ],
    )
    st.divider()
    st.warning("Prototype mode: the Pipeline Map is illustrative demo content and Reports are not built yet. "
               "Dashboard, Analyze Audio and Alert Management use saved records and the trained model.")

show_flash()

if page == "Dashboard":
    st.subheader("Overview")
    counts = db.summary_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyses saved", counts["analyses_total"])
    c2.metric("Open alerts", counts["alerts_open"])
    c3.metric("Awaiting inspection", counts["alerts_by_status"]["Inspection Required"])
    c4.metric("Resolved alerts", counts["alerts_by_status"]["Resolved"])
    by_status = ", ".join(f"{k}: {v}" for k, v in sorted(counts["analyses_by_status"].items()))
    st.caption(f"Analyses by status: {by_status or 'none yet'}. "
               "Computed from records saved in aquaradar.db; times are UTC.")
    st.subheader("Recent analyses")
    recent = db.list_analyses(limit=10)
    if not recent:
        st.info("No analyses saved yet. Upload a recording on the Analyze Audio page.")
    else:
        st.dataframe(
            [
                {
                    "Time (UTC)": r["created_at"],
                    "File": r["filename"],
                    "Location": r["location_label"] or "-",
                    "Status": r["status"],
                    "Model result": CLASS_LABELS.get(r["predicted_class"], "-"),
                    "Model score": score_text(r["confidence"]),
                }
                for r in recent
            ],
            width="stretch",
            hide_index=True,
        )

elif page == "Analyze Audio":
    st.subheader("Analyze an audio recording")
    st.write(
        "Upload a recording (WAV, FLAC or OGG) and run the trained model. "
        "The result is a screening signal, not confirmation of a leak."
    )
    audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac", "ogg"],
        help="The model was trained on 1-second clips. Longer recordings are analysed in 1-second windows (first 60 s).",
    )
    location = st.text_input("Location / pipe segment label", placeholder="e.g., Engineering Block - Segment A",
                             key="analyze_location")
    notes = st.text_area("Optional inspection notes", placeholder="Recording context, device, or environmental noise...",
                         key="analyze_notes")
    if audio is None:
        st.caption("No file uploaded yet.")
    else:
        file_bytes = audio.getvalue()
        st.audio(audio)
        st.success(f"File received: {audio.name}")
        result_key = (audio.name, len(file_bytes))
        if st.button("Analyze recording", type="primary", key="analyze_btn"):
            with st.spinner("Analyzing recording..."):
                result = analyze_audio_bytes(file_bytes, audio.name)
                try:
                    analysis_id = db.save_analysis(result, audio.name, location, notes)
                except sqlite3.Error:
                    analysis_id = None
            st.session_state["analysis"] = {"key": result_key, "result": result, "analysis_id": analysis_id}
        saved = st.session_state.get("analysis")
        if saved and saved["key"] == result_key:
            result, analysis_id = saved["result"], saved["analysis_id"]
            show_result(result)
            if analysis_id is None:
                st.error("The result could not be saved to the database, so an alert cannot be created from it.")
            else:
                st.caption(f"Saved as analysis #{analysis_id}.")
                created = st.session_state.setdefault("alerts_created", {})
                if result.status == "error":
                    st.caption("No alert can be created from a failed analysis. "
                               "Use Alert Management to raise a manual alert.")
                elif analysis_id in created:
                    st.info(f"Alert #{created[analysis_id]} was created from this analysis. "
                            "Manage it on the Alert Management page.")
                else:
                    st.markdown("#### Create an inspection alert (optional)")
                    st.caption("An alert is a maintenance record for follow-up. It does not confirm a leak. "
                               "The location label above is required.")
                    reason = st.text_input("Reason", value=default_reason(result), key=f"reason_{analysis_id}")
                    if st.button("Create alert", key=f"create_{analysis_id}"):
                        try:
                            alert_id = db.create_alert(location, reason, analysis_id=analysis_id, notes=notes)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            created[analysis_id] = alert_id
                            flash(f"Alert #{alert_id} created with status New.")
                            st.rerun()
        else:
            st.caption("Press 'Analyze recording' to run the model.")

elif page == "Pipeline Map":
    st.subheader("Pipeline / campus map")
    st.info("Map placeholder. Any locations shown in this starter should be treated as illustrative, not GPS-verified.")
    st.markdown(
        "- **Demo point A:** Engineering Block (illustrative)\n"
        "- **Demo point B:** Library (illustrative)\n"
        "- **Demo point C:** Hostel Block (illustrative)"
    )
    st.caption("Do not imply precise leak localization. A location label is only as reliable as the data entered by an operator.")

elif page == "Alert Management":
    st.subheader("Alert management")
    st.caption("Alerts are maintenance records created from an analysis or by an operator. "
               "They are not proof that a leak exists. Times are UTC.")
    status_filter = st.selectbox("Show alerts", ["All"] + db.STATUSES, key="alert_filter")
    shown = db.list_alerts(None if status_filter == "All" else status_filter)
    if not shown:
        st.info("No alerts to show yet. Create one from an analysis or with the manual form below.")
    else:
        st.dataframe(
            [
                {
                    "Alert": f"#{a['id']}",
                    "Created (UTC)": a["created_at"],
                    "Location": a["location_label"],
                    "Reason": a["reason"],
                    "Status": a["status"],
                    "Model result": ("Manual alert" if a["analysis_id"] is None
                                     else CLASS_LABELS.get(a["predicted_class"], "Uncertain / none")),
                    "Model score": score_text(a["confidence"]),
                    "Inspection outcome": a["inspection_outcome"] or "-",
                }
                for a in shown
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("Create a manual alert"):
        m_location = st.text_input("Location / pipe segment label", key="manual_location")
        m_reason = st.text_input("Reason", key="manual_reason", placeholder="e.g., Operator heard hissing")
        m_notes = st.text_area("Notes (optional)", key="manual_notes")
        if st.button("Create manual alert", key="manual_create"):
            try:
                alert_id = db.create_alert(m_location, m_reason, notes=m_notes)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(f"Manual alert #{alert_id} created with status New.")
                st.rerun()

    open_alerts = [a for a in db.list_alerts() if a["status"] != "Resolved"]
    if open_alerts:
        st.markdown("#### Update an alert")
        by_id = {a["id"]: a for a in open_alerts}
        chosen = st.selectbox(
            "Alert", list(by_id),
            format_func=lambda i: f"#{i} - {by_id[i]['location_label']} - {by_id[i]['status']}",
            key="update_alert",
        )
        current = by_id[chosen]
        next_statuses = db.STATUSES[db.STATUSES.index(current["status"]) + 1:]
        new_status = st.selectbox("New status", next_statuses, key="update_status")
        outcome = None
        if new_status == "Resolved":
            outcome = st.selectbox("Inspection outcome", db.OUTCOMES, key="update_outcome")
        note = st.text_input("Note (optional)", key="update_note")
        if current["notes"]:
            st.caption("Notes so far:")
            st.text(current["notes"])
        st.caption("Status can only move forward. A resolved alert is closed and keeps its inspection outcome.")
        if st.button("Update alert", key="update_btn"):
            try:
                db.update_alert_status(chosen, new_status, outcome, note)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash(f"Alert #{chosen} updated to {new_status}.")
                st.rerun()

elif page == "Reports":
    st.subheader("Reports")
    st.info("Reporting will be implemented after real analysis and alert records are connected.")
    st.write("Potential report fields: analysis date, location label, model version, prediction, confidence (only if calibrated/meaningful), review status, and inspection outcome.")

else:
    st.subheader("About AquaRadar")
    st.write(
        "AquaRadar is a prototype concept for AI-assisted screening of pipe-acoustic recordings "
        "and coordination of follow-up inspections."
    )
    st.warning(
        "A model prediction is a screening signal, not confirmation of a leak. "
        "The prototype does not guarantee exact localization or quantify water loss."
    )
    st.markdown(
        "- Spectrograms are visualizations, not proof of leakage.\n"
        "- Actual model outputs must come from a trained and evaluated model.\n"
        "- Demo values must remain clearly labelled.\n"
        "- Dataset licence, recording conditions, class balance, and held-out evaluation must be documented."
    )
