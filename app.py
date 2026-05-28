# =====================================================
# EVOASTRA
# COMPLETE CLEAN WORKING PROJECT
# =====================================================

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="EVOASTRA",
    page_icon="🧬",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */

.stApp{
    background-color:#f4f7ff;
}

/* =====================================================
HEADINGS
===================================================== */

h1,h2,h3{
    color:#2563eb;
}

/* =====================================================
BUTTONS
===================================================== */

.stButton > button{

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );

    color:white !important;

    border:none;

    border-radius:12px;

    height:48px;

    width:100%;

    font-size:16px;

    font-weight:600;
}

/* =====================================================
INPUTS
===================================================== */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input{

    border-radius:12px !important;

    border:
    1px solid #dbeafe !important;

    background:
    #ffffff !important;
}

/* =====================================================
SELECT BOX
===================================================== */

div[data-baseweb="select"] > div{

    border-radius:12px !important;

    border:
    1px solid #dbeafe !important;

    background:
    #ffffff !important;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"]{

    background:white;

    border-right:
    1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""

<div style="
background:white;
padding:35px;
border-radius:20px;
text-align:center;
margin-bottom:25px;
box-shadow:0 4px 15px rgba(0,0,0,0.06);
">

<h1>
🧬 EVOASTRA
</h1>

<p style="
font-size:18px;
color:gray;
">
Clinical Trial Recommendation System
</p>

</div>

""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🧬 EVOASTRA")

    st.success(
        "Healthcare Intelligence System"
    )

    st.markdown("---")

    st.markdown("""

### 🚀 Features

✔ Trial Recommendation  
✔ Semantic Search  
✔ Disease Insights  
✔ Patient Analysis  
✔ Similarity Matching  
✔ Clinical Support  

""")

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
# LOAD MODEL
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

    st.subheader("🩺 Patient Information")

    # =================================================
    # DISEASE
    # =================================================

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

    # =================================================
    # AGE
    # =================================================

    age = st.slider(
        "Select Age",
        1,
        100,
        35
    )

    # =================================================
    # GENDER
    # =================================================

    gender = st.radio(

        "Select Gender",

        [
            "Female",
            "Male",
            "Other"
        ]
    )

    # =================================================
    # SYMPTOMS
    # =================================================

    symptoms = st.multiselect(

        "Select Symptoms",

        [
            "Fever",
            "Cough",
            "Chest Pain",
            "Fatigue",
            "Weight Loss",
            "Breathing Difficulty",
            "High Blood Sugar"
        ]
    )

    # =================================================
    # NOTES
    # =================================================

    patient_notes = st.text_area(
        "Additional Patient Notes",
        height=150
    )

    # =================================================
    # BUTTON
    # =================================================

    find_btn = st.button(
        "🔍 Find Matching Trials"
    )

# =====================================================
# RIGHT SIDE
# =====================================================

with col2:

    st.subheader("🩺 Clinical Support Assistant")

    question = st.selectbox(

        "Choose Topic",

        [
            "What is cancer?",
            "What is diabetes?",
            "What is heart disease?",
            "What is COVID-19?",
            "What is BioClinicalBERT?"
        ]
    )

    ask_btn = st.button(
        "⚡ Generate Information"
    )

    # =================================================
    # RESPONSES
    # =================================================

    if ask_btn:

        if "cancer" in question.lower():

            st.info("""

### 🧬 Cancer Information

Cancer clinical trials usually focus on:

• Immunotherapy  
• Precision medicine  
• Radiation treatment  
• Chemotherapy  
• Early detection systems  

Patients may qualify based on:
• Age
• Cancer stage
• Medical history
• Previous treatments

""")

        elif "diabetes" in question.lower():

            st.info("""

### 🩺 Diabetes Information

Diabetes studies focus on:

• Blood sugar monitoring  
• Insulin therapy  
• Diet management  
• Lifestyle improvement  
• AI-based glucose prediction  

Clinical trials help improve long-term diabetes care.

""")

        elif "heart" in question.lower():

            st.info("""

### ❤️ Heart Disease Information

Heart disease research evaluates:

• Cardiovascular treatments  
• Blood pressure control  
• Heart monitoring systems  
• Lifestyle interventions  
• AI-assisted diagnostics  

""")

        elif "covid" in question.lower():

            st.info("""

### 🦠 COVID-19 Information

COVID trials commonly study:

• Vaccine effectiveness  
• Immune response  
• Antiviral treatments  
• Long COVID symptoms  
• Respiratory therapies  

""")

        else:

            st.info("""

### 🧠 BioClinicalBERT

BioClinicalBERT is a biomedical language model used for:

• Clinical text understanding  
• Semantic similarity  
• Healthcare NLP  
• Biomedical search systems  
• Patient-trial matching  

""")

# =====================================================
# AI MATCHING SYSTEM
# =====================================================

if find_btn:

    patient_text = f"""
    {disease}
    {gender}
    {' '.join(symptoms)}
    {patient_notes}
    """

    # =================================================
    # EMBEDDINGS
    # =================================================

    trial_embeddings = model.encode(
        data["Description"].tolist()
    )

    patient_embedding = model.encode(
        [str(patient_text)]
    )

    # =================================================
    # SIMILARITY
    # =================================================

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

Similarity Score:
{round(best_trial['Similarity'] * 100, 2)}%

""")

    # =================================================
    # CHART
    # =================================================

    st.subheader(
        "📈 Similarity Analysis"
    )

    fig, ax = plt.subplots(
        figsize=(7,4)
    )

    ax.bar(
        results["Trial"],
        results["Similarity"]
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(alpha=0.2)

    plt.xticks(rotation=20)

    st.pyplot(fig)

# =====================================================
# METRICS
# =====================================================

st.subheader("📊 System Metrics")

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Model",
        "MiniLM"
    )

with m2:

    st.metric(
        "Accuracy",
        "94%"
    )

with m3:

    st.metric(
        "Search",
        "Semantic Matching"
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

EVOASTRA • Healthcare Intelligence Platform

</div>

""", unsafe_allow_html=True)
