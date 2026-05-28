import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="EVOASTRA AI",
    page_icon="🧬",
    layout="wide"
)

# =====================================================
# SIMPLE STYLING
# =====================================================

st.markdown("""
<style>

body {
    font-family: Arial;
}

.stApp {
    background-color: #f4f7ff;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""

<div class="card" style="text-align:center;">

<h1 style="color:#2563eb;">
🧬 EVOASTRA AI
</h1>

<p>
Clinical Trial Recommendation System
</p>

</div>

""", unsafe_allow_html=True)

# =====================================================
# SAMPLE DATASET
# =====================================================

data = pd.DataFrame({

    "Trial": [

        "Breast Cancer Immunotherapy",
        "Diabetes Insulin Study",
        "Heart Disease Monitoring",
        "COVID Vaccine Trial",
        "Lung Cancer Precision Medicine"

    ],

    "Condition": [

        "breast cancer",
        "diabetes",
        "heart disease",
        "covid-19",
        "lung cancer"

    ],

    "Description": [

        "Immunotherapy treatment for breast cancer patients",

        "Insulin monitoring for diabetes patients",

        "Heart monitoring and cardiovascular treatment",

        "COVID vaccine effectiveness study",

        "Precision medicine treatment for lung cancer"

    ]
})

# =====================================================
# LOAD AI MODEL
# =====================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_model()

# =====================================================
# MAIN LAYOUT
# =====================================================

col1, col2 = st.columns(2)

# =====================================================
# LEFT PANEL
# =====================================================

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🩺 Patient Information")

    disease = st.selectbox(

        "Select Disease",

        [
            "breast cancer",
            "diabetes",
            "heart disease",
            "covid-19",
            "lung cancer"
        ]
    )

    notes = st.text_area(
        "Patient Notes"
    )

    find_btn = st.button(
        "🔍 Find Trials"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =====================================================
# RIGHT PANEL
# =====================================================

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 Healthcare AI")

    question = st.text_input(
        "Ask AI"
    )

    if st.button("⚡ Generate"):

        if "cancer" in question.lower():

            st.success(
                "Cancer trials study advanced therapies and immunotherapy."
            )

        elif "diabetes" in question.lower():

            st.success(
                "Diabetes trials focus on insulin and glucose monitoring."
            )

        else:

            st.success(
                "Healthcare AI assistant ready."
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =====================================================
# AI MATCHING
# =====================================================

if find_btn:

    patient_text = disease + " " + notes

    # Convert text into embeddings

    trial_embeddings = model.encode(
        data["Description"]
    )

    patient_embedding = model.encode(
        [patient_text]
    )

    # Similarity

    similarity = cosine_similarity(
        patient_embedding,
        trial_embeddings
    )[0]

    data["Similarity"] = similarity

    top_match = data.sort_values(
        by="Similarity",
        ascending=False
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "🎯 Recommended Trials"
    )

    st.dataframe(
        top_match,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )
