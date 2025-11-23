import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer

POLICY_DIR = "data"
POLICY_DESCRIPTION = {
    "cancer_care": "This policy is designed to provide financial support specifically for individuals diagnosed with cancer.",
    "lady_360": "The policy is designed specifically to meet the protection needs of women, offering coverage for various health-related events.",
    "life_secure": "This policy is designed to provide financial protection in the event of significant life challenges, focusing on three main areas: Total and Permanent Disability (TPD), Terminal Illness (TI), and death.",
    "private_car": "This policy is a private car insurance agreement between you, the policyholder, and the insurer. It outlines the coverage provided for your vehicle in exchange for the premiums you pay.",
    "silver_secure": "The Silver Secure policy is a non-participating, regular premium term plan designed to provide coverage as you age."
}

st.title("📑 View All Policies")
st.write("Browse and preview available policy documents below.")
st.markdown("---")
st.set_page_config(layout="wide")

if not os.path.exists(POLICY_DIR):
    st.error(f"Policy directory not found: {POLICY_DIR}")
    st.stop()

pdf_files = sorted([f for f in os.listdir(POLICY_DIR) if f.lower().endswith(".pdf")])

if not pdf_files:
    st.warning("No PDF policies found in /data.")
    st.stop()

for pdf in pdf_files:
    policy_name = pdf.replace(".pdf", "").replace("_", " ")
    pdf_path = os.path.join(POLICY_DIR, pdf)

    # each pdf details
    st.info(f"Policy name: {policy_name}", icon="📄")
    st.write(f"**Description:** {POLICY_DESCRIPTION.get(policy_name.replace(' ', '_'), 'No description available.')}")

    # --- Read once ---
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # --- Download button ---
    st.download_button(
        label="📥 Download PDF",
        data=pdf_bytes,
        file_name=pdf,
        mime="application/pdf",
        key=f"download_{pdf}",
    )

    # --- Preview (Cloud-safe) ---
    with st.expander("🔎 Preview PDF", expanded=False):
        pdf_viewer(pdf_bytes, height=600, width=1200)

    st.markdown("---")
