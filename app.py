# =====================================================
# CLINICAL TRIAL MATCH
# COMPLETE FINAL WORKING PROJECT
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
    page_title="Clinical Trial Match",
    page_icon="🩺",
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

.stApp{
    background-color:#f4f7ff;
}

/* =====================================================
HEADINGS
===================================================== */

h1,h2,h3{
    color:#1e293b;
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
🩺 Clinical Trial Match
</h1>

<p style="
font-size:18px;
color:gray;
">
AI-Based Patient Trial Recommendation System
</p>

</div>

""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🩺 Clinical Trial Match")

    st.success(
        "Healthcare Recommendation System"
    )

    st.markdown("---")

    st.markdown("""

### Features

✔ Trial Recommendation  
✔ Semantic Search  
✔ Disease Insights  
✔ Patient Analysis  
✔ Similarity Matching  

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

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_model()

# =====================================================
# MAIN LAYOUT
# =====================================================

col1, col2 = st.columns(2)

# =====================================================
# LEFT SIDE
# =====================================================

with col1:

    st.subheader("Patient Information")

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
    # DETAILED SYMPTOMS
    # =================================================

    st.subheader("Patient Symptoms")

    fever = st.radio(
        "Do you have fever?",
        ["No", "Mild", "Moderate", "Severe"]
    )

    cough = st.radio(
        "Do you have cough?",
        ["No", "Mild", "Moderate", "Severe"]
    )

    fatigue = st.radio(
        "Fatigue Level",
        ["None", "Low", "Medium", "High"]
    )

    pain = st.radio(
        "Chest Pain",
        ["No", "Sometimes", "Frequent"]
    )

    breathing = st.radio(
        "Breathing Difficulty",
        ["No", "Mild", "Moderate", "Severe"]
    )

    weight_loss = st.radio(
        "Unexpected Weight Loss",
        ["No", "Yes"]
    )

    blood_sugar = st.radio(
        "High Blood Sugar",
        ["No", "Yes"]
    )

    smoking = st.radio(
        "Smoking History",
        ["No", "Occasional", "Regular"]
    )

    bp = st.radio(
        "Blood Pressure Issues",
        ["No", "Low", "High"]
    )

    symptom_duration = st.selectbox(
        "Symptoms Duration",
        [
            "Less than 1 week",
            "1-4 weeks",
            "1-3 months",
            "More than 3 months"
        ]
    )

    # =================================================
    # NOTES
    # =================================================

    patient_notes = st.text_area(
        "Additional Notes",
        height=150
    )

    # =================================================
    # BUTTON
    # =================================================

    find_btn = st.button(
        "Find Matching Trials"
    )

# =====================================================
# RIGHT SIDE
# =====================================================

with col2:

    st.subheader("Clinical Information")

    question = st.selectbox(

        "Choose Topic",

        [
            "Cancer Information",
            "Diabetes Information",
            "Heart Disease Information",
            "COVID-19 Information"
        ]
    )

    ask_btn = st.button(
        "Generate Information"
    )

    if ask_btn:

        if "Cancer" in question:

            st.info("""

### Cancer Information

• Immunotherapy  
• Precision medicine  
• Chemotherapy  
• Radiation therapy  
• Clinical monitoring  
• Tumor targeting methods  

""")

        elif "Diabetes" in question:

            st.info("""

### Diabetes Information

• Insulin therapy  
• Glucose monitoring  
• Diet management  
• Lifestyle improvement  
• Blood sugar tracking  

""")

        elif "Heart" in question:

            st.info("""

### Heart Disease Information

• Blood pressure control  
• Cardiovascular monitoring  
• Lifestyle interventions  
• AI-assisted diagnostics  

""")

        else:

            st.info("""

### COVID-19 Information

• Vaccine effectiveness  
• Antiviral medicines  
• Immune response studies  
• Respiratory treatment research  

""")

# =====================================================
# MATCHING SYSTEM
# =====================================================

if find_btn:

    patient_text = f"""

    Disease: {disease}

    Age: {age}

    Gender: {gender}

    Fever: {fever}

    Cough: {cough}

    Fatigue: {fatigue}

    Chest Pain: {pain}

    Breathing Difficulty: {breathing}

    Weight Loss: {weight_loss}

    Blood Sugar: {blood_sugar}

    Smoking History: {smoking}

    Blood Pressure: {bp}

    Duration: {symptom_duration}

    Notes:
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
        "Recommended Clinical Trials"
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
        "Similarity Analysis"
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
# FOOTER
# =====================================================

st.markdown("""

<div style="
text-align:center;
margin-top:40px;
color:gray;
">

Clinical Trial Match • Healthcare Recommendation Platform

</div>

""", unsafe_allow_html=True)
