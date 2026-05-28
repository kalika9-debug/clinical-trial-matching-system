# =====================================================
# QUICK CLINICAL TRIAL MATCH
# SHORT & FAST VERSION
# =====================================================

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Clinical Trial Match",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# SIMPLE CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background-color:#f4f7ff;
}

h1,h2,h3{
    color:#2563eb;
}

.stButton > button{
    background:#2563eb;
    color:white !important;
    border:none;
    border-radius:10px;
    height:45px;
    width:100%;
    font-size:16px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown("""

<h1 style='text-align:center;'>
🩺 Clinical Trial Match
</h1>

<p style='text-align:center;color:gray;'>
Quick Patient Trial Recommendation System
</p>

""", unsafe_allow_html=True)

# =====================================================
# DATASET
# =====================================================

data = pd.DataFrame({

    "Trial": [

        "Breast Cancer Immunotherapy",
        "Diabetes Insulin Study",
        "Heart Disease Monitoring",
        "COVID Vaccine Trial",
        "Lung Cancer Precision Medicine"

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
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_model()

# =====================================================
# QUICK FORM
# =====================================================

st.subheader("Patient Details")

disease = st.selectbox(

    "Disease",

    [
        "breast cancer",
        "diabetes",
        "heart disease",
        "covid-19",
        "lung cancer"
    ]
)

symptom = st.selectbox(

    "Main Symptom",

    [
        "Fever",
        "Cough",
        "Chest Pain",
        "Fatigue",
        "Weight Loss"
    ]
)

duration = st.selectbox(

    "Duration",

    [
        "1 week",
        "1 month",
        "3 months",
        "More than 3 months"
    ]
)

# =====================================================
# BUTTON
# =====================================================

find_btn = st.button(
    "Find Trial Match"
)

# =====================================================
# MATCHING
# =====================================================

if find_btn:

    patient_text = f"""
    {disease}
    {symptom}
    {duration}
    """

    # Embeddings

    trial_embeddings = model.encode(
        data["Description"].tolist()
    )

    patient_embedding = model.encode(
        [patient_text]
    )

    # Similarity

    scores = cosine_similarity(
        patient_embedding,
        trial_embeddings
    )[0]

    data["Similarity"] = scores

    results = data.sort_values(
        by="Similarity",
        ascending=False
    )

    # =================================================
    # BEST MATCH
    # =================================================

    best = results.iloc[0]

    st.success(f"""

✅ Best Matching Trial

Trial:
{best['Trial']}

Match Score:
{round(best['Similarity'] * 100, 2)}%

""")

    # =================================================
    # TABLE
    # =================================================

    st.subheader("Top Matches")

    st.dataframe(
        results,
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("""

<hr>

<p style='text-align:center;color:gray;'>

Clinical Trial Match System

</p>

""", unsafe_allow_html=True)
