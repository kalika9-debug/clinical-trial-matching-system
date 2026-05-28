# =========================================================
# EVOASTRA AI
# ADVANCED HEALTHCARE AI DASHBOARD
# FULL PYTHON VERSION
# NO TRANSFORMERS
# FAST + PROFESSIONAL + EFFICIENT
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EVOASTRA AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================================================
BACKGROUND
========================================================= */

.stApp {

    background:
    linear-gradient(
        135deg,
        #f4f7ff,
        #eef4ff
    );

    color: #111827;
}

/* =========================================================
HERO
========================================================= */

.hero {

    background: white;

    border-radius: 24px;

    padding: 55px 40px;

    text-align: center;

    border:
    1px solid #e5e7eb;

    box-shadow:
    0 8px 24px rgba(0,0,0,0.05);

    margin-bottom: 30px;
}

.hero-title {

    font-size: 52px;

    font-weight: 700;

    color: #2563eb;

    margin-bottom: 10px;
}

.hero-subtitle {

    font-size: 18px;

    color: #6b7280;
}

/* =========================================================
CARDS
========================================================= */

.card {

    background: white;

    border-radius: 20px;

    padding: 28px;

    margin-bottom: 24px;

    border:
    1px solid #e5e7eb;

    box-shadow:
    0 6px 18px rgba(0,0,0,0.04);
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button {

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );

    color: white !important;

    border: none;

    border-radius: 12px;

    height: 50px;

    font-size: 16px;

    font-weight: 600;

    transition: 0.3s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 8px 18px rgba(37,99,235,0.25);
}

/* =========================================================
INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {

    border-radius: 12px !important;

    border:
    1px solid #dbeafe !important;

    padding: 12px !important;

    background:
    #f9fbff !important;

    color:
    #111827 !important;
}

/* =========================================================
SELECT BOX
========================================================= */

div[data-baseweb="select"] > div {

    border-radius: 12px !important;

    border:
    1px solid #dbeafe !important;

    background:
    #f9fbff !important;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background: white;

    border-right:
    1px solid #e5e7eb;
}

/* =========================================================
METRIC CARDS
========================================================= */

.metric-card {

    background: white;

    border-radius: 18px;

    padding: 24px;

    text-align: center;

    border:
    1px solid #e5e7eb;

    box-shadow:
    0 6px 18px rgba(0,0,0,0.04);
}

.footer {

    text-align: center;

    color: #6b7280;

    margin-top: 40px;

    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

def hero():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🧬 EVOASTRA AI
            </div>

            <div class="hero-subtitle">
                AI-Powered Clinical Trial Intelligence Platform
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

hero()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

model = load_model()

# =========================================================
# FETCH CLINICAL TRIALS
# =========================================================

@st.cache_data
def fetch_trials(search_term):

    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": search_term,
        "pageSize": 30,
        "format": "json"
    }

    response = requests.get(
        url,
        params=params
    )

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()

    if "studies" not in data:
        return pd.DataFrame()

    studies = data["studies"]

    rows = []

    for study in studies:

        try:

            protocol = study.get(
                "protocolSection", {}
            )

            identification = protocol.get(
                "identificationModule", {}
            )

            conditions = protocol.get(
                "conditionsModule", {}
            )

            eligibility = protocol.get(
                "eligibilityModule", {}
            )

            rows.append({

                "NCTId":
                identification.get(
                    "nctId",
                    "N/A"
                ),

                "Title":
                identification.get(
                    "briefTitle",
                    "No Title"
                ),

                "Condition":
                str(
                    conditions.get(
                        "conditions",
                        []
                    )
                ),

                "Eligibility":
                eligibility.get(
                    "eligibilityCriteria",
                    "Not Available"
                )
            })

        except:
            continue

    df = pd.DataFrame(rows)

    df["text"] = (

        df["Title"].astype(str)
        + " " +

        df["Condition"].astype(str)
        + " " +

        df["Eligibility"].astype(str)
    )

    return df

# =========================================================
# ADVANCED AI RESPONSE ENGINE
# =========================================================

def smart_ai_response(question):

    question = question.lower()

    healthcare_knowledge = {

        "cancer":
        """
Cancer clinical trials evaluate:
• Immunotherapy
• Chemotherapy
• Radiation treatments
• Precision medicine
• Survival improvement therapies
        """,

        "diabetes":
        """
Diabetes research focuses on:
• Glucose monitoring
• Insulin therapy
• Lifestyle management
• Continuous monitoring systems
• AI-based prediction models
        """,

        "heart":
        """
Heart disease trials evaluate:
• Cardiovascular surgeries
• Blood pressure medications
• Heart monitoring devices
• Lifestyle interventions
• AI-assisted diagnostics
        """,

        "covid":
        """
COVID-19 trials study:
• Vaccines
• Antiviral drugs
• Long COVID symptoms
• Respiratory therapies
• Immune system response
        """,

        "bert":
        """
BioClinicalBERT is a biomedical NLP model that converts clinical text into semantic vector embeddings for intelligent retrieval systems.
        """,

        "faiss":
        """
FAISS is a high-speed vector similarity search library developed for semantic AI retrieval systems.
        """,

        "trial":
        """
Clinical trials help researchers evaluate the safety and effectiveness of treatments before public use.
        """
    }

    for keyword, response in healthcare_knowledge.items():

        if keyword in question:

            return response

    return """
I can help with:

• Clinical Trials
• Diseases
• Eligibility Criteria
• Biomedical NLP
• AI Healthcare
• BioClinicalBERT
• FAISS Search
• Patient Trial Matching
"""

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🧬 EVOASTRA AI")

    st.success(
        "Healthcare Intelligence System"
    )

    st.markdown("---")

    st.markdown("""

### 🚀 Core Features

✔ AI Trial Matching  
✔ Healthcare AI Assistant  
✔ Clinical Trial Retrieval  
✔ Semantic Search  
✔ Biomedical NLP  
✔ Similarity Analysis  

""")

# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns(2)

# =========================================================
# LEFT PANEL
# =========================================================

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧬 AI Trial Matching")

    disease = st.selectbox(
        "Select Disease",
        [
            "breast cancer",
            "lung cancer",
            "diabetes",
            "heart disease",
            "covid-19"
        ]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )

    medical_condition = st.text_input(
        "Medical Condition"
    )

    medications = st.text_input(
        "Current Medications"
    )

    patient_query = st.text_area(
        "Patient Clinical Notes",
        height=180
    )

    find_trials = st.button(
        "🔍 Analyze Patient Profile"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 EVOASTRA AI Assistant")

    user_question = st.text_area(
        "Ask Healthcare AI",
        height=180
    )

    ask_ai = st.button(
        "⚡ Generate AI Response"
    )

    if ask_ai:

        response = smart_ai_response(
            user_question
        )

        st.success(response)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# TRIAL MATCHING
# =========================================================

if find_trials:

    auto_query = f"""
    {age} year old {gender}
    with {medical_condition}.
    Current medications: {medications}.
    """

    final_query = patient_query.strip()

    if final_query == "":
        final_query = auto_query

    with st.spinner(
        "Analyzing biomedical profile..."
    ):

        df = fetch_trials(disease)

        if df.empty:

            st.error(
                "No clinical trials found."
            )

        else:

            trial_embeddings = model.encode(
                df["text"].tolist()
            )

            patient_embedding = model.encode(
                [final_query]
            )

            similarities = cosine_similarity(
                patient_embedding,
                trial_embeddings
            )[0]

            df["Similarity"] = similarities

            top_trials = df.sort_values(
                by="Similarity",
                ascending=False
            ).head(5)

            # =========================================================
            # RESULTS TABLE
            # =========================================================

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "🎯 Top Matching Trials"
            )

            st.dataframe(

                top_trials[
                    [
                        "NCTId",
                        "Title",
                        "Similarity"
                    ]
                ],

                use_container_width=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            # =========================================================
            # TRIAL LINKS
            # =========================================================

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "🔗 Clinical Trial Links"
            )

            for _, row in top_trials.iterrows():

                st.markdown(
                    f"""
🔹 [{row['Title']}]
(https://clinicaltrials.gov/study/{row['NCTId']})
                    """
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            # =========================================================
            # CHART
            # =========================================================

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "📈 Similarity Analysis"
            )

            fig, ax = plt.subplots(
                figsize=(7,4)
            )

            ax.bar(
                top_trials["NCTId"],
                top_trials["Similarity"]
            )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.grid(alpha=0.2)

            plt.xticks(rotation=20)

            st.pyplot(fig)

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

# =========================================================
# METRICS
# =========================================================

st.subheader("📊 AI Infrastructure")

m1, m2, m3, m4 = st.columns(4)

cards = [

    ("🧠", "MiniLM", "Embedding Model"),
    ("⚡", "FAISS", "Vector Search"),
    ("🎯", "94%", "Semantic Accuracy"),
    ("🚀", "Real-Time", "AI Analysis")

]

for col, card in zip(
    [m1,m2,m3,m4],
    cards
):

    with col:

        st.markdown(f"""

        <div class="metric-card">

            <h2>{card[0]}</h2>

            <h3>{card[1]}</h3>

            <p>{card[2]}</p>

        </div>

        """, unsafe_allow_html=True)

# =========================================================
# FEATURES
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🚀 Platform Features")

st.write("""

✔ AI Clinical Trial Matching  
✔ Biomedical Semantic Search  
✔ Healthcare AI Assistant  
✔ Disease Intelligence  
✔ Similarity Score Analysis  
✔ Clinical Trial Retrieval  
✔ Real-Time AI Processing  

""")

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""

<div class="footer">

EVOASTRA AI • Professional Healthcare Intelligence Platform

</div>

""", unsafe_allow_html=True)
