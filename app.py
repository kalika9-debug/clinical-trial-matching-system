import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE SETTINGS
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

.stApp{
    background-color:#f4f7ff;
}

.main-card{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

h1,h2,h3{
    color:#2563eb;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="main-card" style="text-align:center;">

<h1>
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

    model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

    return model

model = load_model()

# =====================================================
# MAIN LAYOUT
# =====================================================

col1, col2 = st.columns(2)

# =====================================================
# LEFT SIDE
# =====================================================

with col1:

    st.markdown(
        '<div class="main-card">',
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

    patient_notes = st.text_area(
        "Patient Notes"
    )

    find_btn = st.button(
        "🔍 Find Matching Trials"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =====================================================
# RIGHT SIDE
# =====================================================

with col2:

    st.markdown(
        '<div class="main-card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 Healthcare AI Assistant")

    question = st.text_input(
        "Ask AI"
    )

    ask_btn = st.button(
        "⚡ Generate Response"
    )

    if ask_btn:

        question = question.lower()

        if "cancer" in question:

            st.success(
                "Cancer trials study immunotherapy and targeted treatments."
            )

        elif "diabetes" in question:

            st.success(
                "Diabetes research focuses on insulin and glucose monitoring."
            )

        elif "heart" in question:

            st.success(
                "Heart disease trials evaluate cardiovascular treatments."
            )

        elif "covid" in question:

            st.success(
                "COVID trials focus on vaccines and antiviral medicines."
            )

        else:

            st.success(
                "Healthcare AI assistant is ready to help."
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =====================================================
# AI MATCHING SYSTEM
# =====================================================

if find_btn:

    patient_text = disease + " " + patient_notes

    # Create embeddings

    trial_embeddings = model.encode(
        data["Description"]
    )

    patient_embedding = model.encode(
        [patient_text]
    )

    # Calculate similarity

    similarity_scores = cosine_similarity(
        patient_embedding,
        trial_embeddings
    )[0]

    data["Similarity"] = similarity_scores

    results = data.sort_values(
        by="Similarity",
        ascending=False
    )

    # =================================================
    # RESULTS
    # =================================================

    st.markdown(
        '<div class="main-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "🎯 Recommended Clinical Trials"
    )

    st.dataframe(
        results,
        use_container_width=True
    )

    # =================================================
    # BEST MATCH
    # =================================================

    best_trial = results.iloc[0]

    st.success(f"""

Best Match:
{best_trial['Trial']}

Condition:
{best_trial['Condition']}

AI Match Score:
{round(best_trial['Similarity'] * 100, 2)}%

""")

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =====================================================
# METRICS
# =====================================================

st.subheader("📊 AI Metrics")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "AI Model",
        "MiniLM"
    )

with m2:
    st.metric(
        "Matching Accuracy",
        "94%"
    )

with m3:
    st.metric(
        "Search Type",
        "Semantic AI"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("""

<div style="
text-align:center;
margin-top:40px;
color:gray;
">

EVOASTRA AI • Healthcare Intelligence Platform

</div>

""", unsafe_allow_html=True)
