import streamlit as st

st.set_page_config(
    page_title="AquaRadar",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IMPORTANT:
# This is a UI starter. Results shown here are simulated placeholders.
# Do not describe them as output from a trained model.
# Replace demo behavior only after the model input/output contract is agreed.

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
    st.warning("Prototype mode: sample values are simulated.")

if page == "Dashboard":
    st.subheader("Campus overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Analyses today", "12", help="Simulated demo value")
    c2.metric("Open alerts", "4", help="Simulated demo value")
    c3.metric("Awaiting inspection", "2", help="Simulated demo value")
    st.info("These metrics are illustrative demo data, not live campus measurements.")
    st.subheader("Recent activity")
    st.dataframe(
        [
            {"Time": "09:15", "Location": "Building A — demo", "Result": "Review recommended", "Status": "New"},
            {"Time": "10:40", "Location": "Library — demo", "Result": "No strong leak-like signal", "Status": "Under Review"},
            {"Time": "11:05", "Location": "Lab Block — demo", "Result": "Uncertain / review audio", "Status": "Inspection Required"},
        ],
        use_container_width=True,
        hide_index=True,
    )

elif page == "Analyze Audio":
    st.subheader("Analyze an audio recording")
    st.write("Upload a supported audio recording to begin the prototype workflow.")
    audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac", "ogg"],
        help="WAV is preferred for the initial prototype. Actual supported formats depend on the eventual audio pipeline.",
    )
    location = st.text_input("Location / pipe segment label", placeholder="e.g., Engineering Block — Segment A")
    notes = st.text_area("Optional inspection notes", placeholder="Recording context, device, or environmental noise...")
    if audio is not None:
        st.audio(audio)
        st.success(f"File received: {audio.name}")
        st.warning(
            "No trained model is connected yet. Do not interpret this screen as a real prediction. "
            "The model service and input validation will be connected after the team agrees on the contract."
        )
        st.write("Location entered:", location or "Not specified")
        if notes:
            st.write("Notes:", notes)
    else:
        st.caption("No file uploaded yet.")

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
    st.write("This starter shows the intended workflow. SQLite persistence is not connected yet.")
    st.dataframe(
        [
            {"Alert ID": "DEMO-001", "Location": "Building A — demo", "Reason": "Potential leak-like audio", "Status": "New"},
            {"Alert ID": "DEMO-002", "Location": "Library — demo", "Reason": "Manual review requested", "Status": "Under Review"},
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Suggested statuses: New → Under Review → Inspection Required → Resolved.")
    st.caption("A resolved alert should retain the inspection outcome: leak repaired, leak not found, inconclusive, or false alarm.")

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
