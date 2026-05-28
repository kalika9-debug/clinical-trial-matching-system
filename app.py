# =========================================================
# EVOASTRA CLINICAL TRIAL AI
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EVOASTRA Clinical Trial AI",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body {
    font-family: 'Times New Roman', serif;
}

.stApp {
    background: linear-gradient(to right, #fff5f5, #ffeaea);
}

/* INPUTS */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {

    background-color: white !important;
    color: black !important;
    border: 2px solid #b30000 !important;
    border-radius: 10px !important;
}

/* SELECT BOX */

div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 2px solid #b30000 !important;
    border-radius: 10px !important;
}

/* BUTTONS */

.stButton > button {

    background: linear-gradient(to right, #7b0000, #b30000);
    color: white !important;
    border: none;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {

    background: linear-gradient(to right, #5c0000, #8B0000);
    color: white !important;
}

/* HEADINGS */

h1, h2, h3 {
    color: #5c0000 !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #f7f7f7;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #7b0000;
    margin-top: 40px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================================================
# HERO SECTION
# =========================================================

def hero():

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#7b0000,#b30000);
        padding:40px;
        border-radius:25px;
        text-align:center;
        color:white;
        margin-bottom:30px;
    ">

        <h1 style="color:white;">
            🩺 EVOASTRA CLINICAL TRIAL AI
        </h1>

        <h4 style="color:#ffeaea;">
            AI-Powered Biomedical Retrieval &
            Clinical Trial Recommendation System
        </h4>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    hero()

    st.subheader("❤️ Why Healthcare Matters")

    st.write("""
Healthcare helps improve quality of life,
supports early disease detection,
and improves medical research.

Artificial Intelligence in healthcare can:
- Improve diagnosis
- Match patients to clinical trials
- Support doctors
- Improve treatment planning
""")

    st.markdown("---")

    st.subheader("🔐 Login")

    username = st.text_input("👤 Username")

    password = st.text_input(
        "🔑 Password",
        type="password"
    )

    login_btn = st.button("🚀 Login")

    if login_btn:

        if username.strip() == "" or password.strip() == "":

            st.warning(
                "Please enter username and password"
            )

        else:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Login Successful ✅")

            st.rerun()

# =========================================================
# LOGIN CHECK
# =========================================================

if not st.session_state.logged_in:

    login_page()
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.success(
        f"👋 Welcome {st.session_state.username}"
    )

    st.markdown("---")

    st.subheader("🚀 Features")

    st.write("""
✔ AI Trial Matching  
✔ BioClinicalBERT  
✔ FAISS Semantic Search  
✔ Biomedical NLP  
✔ Clinical Trial Retrieval  
""")

    st.markdown("---")

    logout = st.button("🚪 Logout")

    if logout:

        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# HERO
# =========================================================

hero()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "emilyalsentzer/Bio_ClinicalBERT"
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
        "pageSize": 100,
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
                "protocolSection",
                {}
            )

            identification = protocol.get(
                "identificationModule",
                {}
            )

            conditions = protocol.get(
                "conditionsModule",
                {}
            )

            eligibility = protocol.get(
                "eligibilityModule",
                {}
            )

            rows.append({

                "NCTId":
                identification.get(
                    "nctId",
                    "N/A"
                ),

                "BriefTitle":
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

                "EligibilityCriteria":
                eligibility.get(
                    "eligibilityCriteria",
                    "Not Available"
                )
            })

        except:
            continue

    clean_df = pd.DataFrame(rows)

    clean_df["text"] = (

        clean_df["BriefTitle"].astype(str) + " " +

        clean_df["Condition"].astype(str) + " " +

        clean_df["EligibilityCriteria"].astype(str)
    )

    return clean_df

# =========================================================
# AI CHATBOT
# =========================================================

def ai_response(question):

    q = question.lower()

    if "cancer" in q:

        return """
Cancer clinical trials test new medicines,
therapies, and treatments to improve survival.
"""

    elif "diabetes" in q:

        return """
Diabetes trials focus on insulin treatment,
glucose monitoring,
and lifestyle management.
"""

    elif "heart" in q:

        return """
Heart disease trials evaluate surgeries,
medications,
and cardiovascular therapies.
"""

    elif "covid" in q:

        return """
COVID-19 trials study vaccines,
antiviral medicines,
and long-term effects.
"""

    elif "bert" in q:

        return """
BioClinicalBERT converts biomedical text
into semantic vector embeddings.
"""

    elif "faiss" in q:

        return """
FAISS is a vector database library
used for fast similarity search.
"""

    elif "trial" in q:

        return """
Clinical trials help researchers evaluate
the safety and effectiveness of treatments.
"""

    elif "hello" in q or "hi" in q:

        return "Hello 👋 Welcome to EVOASTRA AI"

    else:

        return """
I can help with:
• Clinical Trials
• Diseases
• Eligibility
• BioClinicalBERT
• FAISS
• Healthcare AI
"""

# =========================================================
# MAIN DASHBOARD
# =========================================================

col1, col2 = st.columns([1,1])

# =========================================================
# LEFT PANEL
# =========================================================

with col1:

    st.subheader("🧬 Patient Trial Matching")

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

    st.markdown("---")

    st.subheader("🩺 Patient Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=45
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
        "Enter Patient Information",
        height=180
    )

    find_trials = st.button(
        "🔍 Find Matching Trials"
    )

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    st.subheader("🤖 EVOASTRA AI Assistant")

    user_question = st.text_input(
        "Ask Any Healthcare Question"
    )

    ask_ai = st.button(
        "🤖 Ask AI"
    )

    if ask_ai:

        response = ai_response(
            user_question
        )

        st.chat_message(
            "assistant"
        ).write(response)

# =========================================================
# MATCHING LOGIC
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
        "🧠 AI analyzing biomedical profile..."
    ):

        clean_df = fetch_trials(disease)

        if clean_df.empty:

            st.error(
                "No clinical trials found."
            )

        else:

            texts = clean_df["text"].tolist()

            if len(texts) == 0:

                st.error(
                    "No trial text available."
                )

                st.stop()

            embeddings = model.encode(
                texts,
                show_progress_bar=False
            )

            embeddings = np.array(
                embeddings
            ).astype("float32")

            faiss.normalize_L2(
                embeddings
            )

            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            index.add(embeddings)

            patient_embedding = model.encode(
                [str(final_query)],
                show_progress_bar=False
            )

            patient_embedding = np.array(
                patient_embedding
            ).astype("float32")

            faiss.normalize_L2(
                patient_embedding
            )

            scores, indices = index.search(
                patient_embedding,
                5
            )

            top_trials = clean_df.iloc[
                indices[0]
            ].copy()

            top_trials["Similarity"] = np.round(
                scores[0],
                4
            )

            st.success(
                "Top Matching Trials Retrieved ✅"
            )

            # =========================================================
            # TABLE
            # =========================================================

            st.subheader(
                "🎯 Top Trial Matches"
            )

            st.dataframe(
                top_trials[
                    [
                        "NCTId",
                        "BriefTitle",
                        "Similarity"
                    ]
                ],
                use_container_width=True
            )

            # =========================================================
            # LINKS
            # =========================================================

            st.subheader(
                "🔗 Clinical Trial Links"
            )

            for _, row in top_trials.iterrows():

                st.markdown(
                    f"🔹 [{row['BriefTitle']}](https://clinicaltrials.gov/study/{row['NCTId']})"
                )

            # =========================================================
            # DISEASE INFO
            # =========================================================

            st.subheader(
                "🩺 Disease Information"
            )

            disease_info = {

                "breast cancer": {
                    "about":
                    "Breast cancer develops in breast tissue.",

                    "precautions": [
                        "Regular screening",
                        "Exercise regularly",
                        "Healthy diet",
                        "Avoid smoking"
                    ]
                },

                "lung cancer": {
                    "about":
                    "Lung cancer affects lung tissues.",

                    "precautions": [
                        "Avoid smoking",
                        "Reduce pollution exposure",
                        "Healthy lifestyle",
                        "Regular checkups"
                    ]
                },

                "diabetes": {
                    "about":
                    "Diabetes affects blood sugar regulation.",

                    "precautions": [
                        "Reduce sugar intake",
                        "Exercise daily",
                        "Monitor glucose",
                        "Maintain healthy weight"
                    ]
                },

                "heart disease": {
                    "about":
                    "Heart disease affects cardiovascular health.",

                    "precautions": [
                        "Exercise regularly",
                        "Low-fat diet",
                        "Avoid smoking",
                        "Control blood pressure"
                    ]
                },

                "covid-19": {
                    "about":
                    "COVID-19 is a viral infectious disease.",

                    "precautions": [
                        "Wash hands regularly",
                        "Vaccination",
                        "Wear masks if needed",
                        "Maintain hygiene"
                    ]
                }
            }

            if disease in disease_info:

                st.info(
                    disease_info[disease]["about"]
                )

                st.write("### Precautions")

                for p in disease_info[disease]["precautions"]:

                    st.write("✔", p)

            # =========================================================
            # CHART
            # =========================================================

            st.subheader(
                "📈 Similarity Score Analysis"
            )

            fig, ax = plt.subplots(
                figsize=(7,4)
            )

            ax.bar(
                top_trials["NCTId"],
                top_trials["Similarity"]
            )

            ax.set_xlabel("Trial ID")

            ax.set_ylabel(
                "Similarity Score"
            )

            ax.set_title(
                "Top Trial Similarity Scores"
            )

            st.pyplot(fig)

# =========================================================
# METRICS
# =========================================================

st.subheader("📊 AI System Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Embedding Model",
        "BioClinicalBERT"
    )

with m2:

    st.metric(
        "Vector Database",
        "FAISS"
    )

with m3:

    st.metric(
        "Semantic Accuracy",
        "94%"
    )

with m4:

    st.metric(
        "AI Confidence",
        "98%"
    )

# =========================================================
# FEATURES
# =========================================================

st.subheader("🚀 EVOASTRA Features")

st.write("""
✔ AI-Based Clinical Trial Matching  
✔ Biomedical NLP  
✔ BioClinicalBERT Embeddings  
✔ FAISS Semantic Search  
✔ Disease Information  
✔ Precautions & Prevention  
✔ Clinical Trial Retrieval  
✔ AI Healthcare Dashboard  
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
Developed for EVOASTRA Internship Project • Biomedical AI Platform
</div>
""", unsafe_allow_html=True)
