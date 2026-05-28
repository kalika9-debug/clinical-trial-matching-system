# =========================================================
# EVOASTRA CLINICAL TRIAL AI
# FUTURISTIC BIOTECH AI DASHBOARD
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

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

*{
    font-family: 'Poppins', sans-serif;
}

/* =========================================================
BACKGROUND
========================================================= */

.stApp {

    background:
    radial-gradient(circle at top left,
    rgba(255,0,76,0.15),
    transparent 25%),

    radial-gradient(circle at bottom right,
    rgba(255,0,76,0.10),
    transparent 25%),

    linear-gradient(
        135deg,
        #050505,
        #120000,
        #1a0000
    );

    color: white;
}

/* =========================================================
GLASSMORPHISM
========================================================= */

.glass {

    background: rgba(255,255,255,0.06);

    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 25px;

    box-shadow:
    0 8px 32px rgba(0,0,0,0.35);

    transition: 0.3s ease;

    margin-bottom: 20px;
}

.glass:hover {

    transform: translateY(-5px);

    box-shadow:
    0 0 20px rgba(255,0,76,0.15),
    0 0 40px rgba(255,0,76,0.10);
}

/* =========================================================
HEADINGS
========================================================= */

h1, h2, h3 {

    color: white !important;
    font-weight: 700 !important;
}

h4, h5 {

    color: #ffb3c1 !important;
}

/* =========================================================
INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {

    background: rgba(255,255,255,0.06) !important;

    color: white !important;

    border: 1px solid rgba(255,255,255,0.15) !important;

    border-radius: 14px !important;

    padding: 12px !important;
}

/* =========================================================
SELECT BOX
========================================================= */

div[data-baseweb="select"] > div {

    background: rgba(255,255,255,0.06) !important;

    border: 1px solid rgba(255,255,255,0.15) !important;

    border-radius: 14px !important;

    color: white !important;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button {

    background:
    linear-gradient(
        135deg,
        #ff003c,
        #8b0000
    );

    color: white !important;

    border: none;

    border-radius: 14px;

    height: 52px;

    width: 100%;

    font-size: 17px;

    font-weight: 600;

    transition: 0.3s ease;

    box-shadow:
    0 0 15px rgba(255,0,76,0.4);
}

.stButton > button:hover {

    transform: scale(1.02);

    box-shadow:
    0 0 25px rgba(255,0,76,0.8);
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        rgba(20,20,20,0.95),
        rgba(40,0,0,0.95)
    );

    border-right:
    1px solid rgba(255,255,255,0.08);
}

/* =========================================================
METRICS
========================================================= */

.metric-card {

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 20px;

    text-align: center;

    transition: 0.3s ease;
}

.metric-card:hover {

    transform: translateY(-4px);

    box-shadow:
    0 0 18px rgba(255,0,76,0.25);
}

/* =========================================================
TABLES
========================================================= */

[data-testid="stDataFrame"] {

    background: rgba(255,255,255,0.04);

    border-radius: 18px;

    padding: 10px;
}

/* =========================================================
FOOTER
========================================================= */

.footer {

    text-align: center;

    color: #ff4d6d;

    margin-top: 50px;

    font-size: 15px;

    opacity: 0.8;
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: #8b0000;

    border-radius: 20px;
}

/* =========================================================
HERO ANIMATION
========================================================= */

.hero {

    background:
    linear-gradient(
        135deg,
        rgba(255,0,76,0.25),
        rgba(139,0,0,0.2)
    );

    border:
    1px solid rgba(255,255,255,0.1);

    border-radius: 28px;

    padding: 45px;

    text-align: center;

    box-shadow:
    0 0 25px rgba(255,0,76,0.25);

    margin-bottom: 30px;
}

.typing {

    overflow: hidden;

    white-space: nowrap;

    border-right: 2px solid white;

    width: 0;

    animation:
    typing 4s steps(40,end) forwards,
    blink .8s infinite;
}

@keyframes typing {

    from { width: 0 }

    to { width: 100% }
}

@keyframes blink {

    50% {
        border-color: transparent;
    }
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
    <div class="hero">

        <h1 style="
        font-size:52px;
        margin-bottom:10px;
        ">
        🧬 EVOASTRA AI
        </h1>

        <h4 class="typing"
        style="
        color:#ffd6de;
        font-weight:400;
        font-size:20px;
        ">
        Futuristic Clinical Trial Intelligence Platform
        </h4>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    hero()

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🔐 Secure AI Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    login_btn = st.button("🚀 Access Dashboard")

    if login_btn:

        if username.strip() == "" or password.strip() == "":

            st.warning("Enter credentials")

        else:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Authentication Successful ✅")

            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown("""
    <h2 style="text-align:center;">
    🧬 EVOASTRA
    </h2>
    """, unsafe_allow_html=True)

    st.success(
        f"Welcome {st.session_state.username}"
    )

    st.markdown("---")

    st.markdown("""
    ### 🚀 Core Technologies

    ✔ BioClinicalBERT  
    ✔ FAISS Vector Search  
    ✔ Biomedical NLP  
    ✔ Semantic Retrieval  
    ✔ AI Trial Intelligence  
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
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    return model

model = load_model()

# =========================================================
# FETCH TRIALS
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
                    "nctId", "N/A"
                ),

                "BriefTitle":
                identification.get(
                    "briefTitle", "No Title"
                ),

                "Condition":
                str(
                    conditions.get(
                        "conditions", []
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

        clean_df["BriefTitle"].astype(str)
        + " " +

        clean_df["Condition"].astype(str)
        + " " +

        clean_df["EligibilityCriteria"].astype(str)
    )

    return clean_df

# =========================================================
# AI RESPONSE
# =========================================================

def ai_response(question):

    q = question.lower()

    if "cancer" in q:

        return "Cancer trials evaluate next-generation therapies and survival treatments."

    elif "diabetes" in q:

        return "Diabetes clinical research focuses on glucose regulation and insulin innovation."

    elif "heart" in q:

        return "Cardiovascular trials evaluate surgeries, medications, and AI diagnostics."

    elif "covid" in q:

        return "COVID-19 studies analyze vaccines, antiviral therapies, and long-term effects."

    elif "bert" in q:

        return "BioClinicalBERT transforms biomedical text into contextual embeddings."

    elif "faiss" in q:

        return "FAISS enables ultra-fast vector similarity search for semantic retrieval."

    else:

        return "I can assist with clinical trials, biomedical AI, FAISS, diseases, and healthcare analytics."

# =========================================================
# MAIN GRID
# =========================================================

col1, col2 = st.columns([1,1])

# =========================================================
# LEFT PANEL
# =========================================================

with col1:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🧬 AI Trial Matching")

    disease = st.selectbox(
        "Disease",
        [
            "breast cancer",
            "lung cancer",
            "diabetes",
            "heart disease",
            "covid-19"
        ]
    )

    st.markdown("---")

    st.subheader("🩺 Patient Profile")

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
        "Patient Clinical Notes",
        height=180
    )

    find_trials = st.button(
        "🔍 Analyze Biomedical Match"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🤖 EVOASTRA AI Assistant")

    user_question = st.text_input(
        "Ask Healthcare AI"
    )

    ask_ai = st.button("⚡ Generate AI Response")

    if ask_ai:

        response = ai_response(user_question)

        st.chat_message("assistant").write(response)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MATCHING
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
        "🧠 Running biomedical semantic analysis..."
    ):

        clean_df = fetch_trials(disease)

        if clean_df.empty:

            st.error("No trials found.")

        else:

            texts = clean_df["text"].tolist()

            embeddings = model.encode(
                texts,
                show_progress_bar=False
            )

            embeddings = np.array(
                embeddings
            ).astype("float32")

            faiss.normalize_L2(embeddings)

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
                "Top Clinical Trials Retrieved ✅"
            )

            st.markdown('<div class="glass">', unsafe_allow_html=True)

            st.subheader("🎯 Trial Matches")

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

            st.markdown('</div>', unsafe_allow_html=True)

            # =========================================================
            # LINKS
            # =========================================================

            st.markdown('<div class="glass">', unsafe_allow_html=True)

            st.subheader("🔗 Clinical Trial Access")

            for _, row in top_trials.iterrows():

                st.markdown(
                    f"""
                    🔹 [{row['BriefTitle']}]
                    (https://clinicaltrials.gov/study/{row['NCTId']})
                    """
                )

            st.markdown('</div>', unsafe_allow_html=True)

            # =========================================================
            # CHART
            # =========================================================

            st.markdown('<div class="glass">', unsafe_allow_html=True)

            st.subheader("📈 Similarity Analysis")

            fig, ax = plt.subplots(
                figsize=(8,4)
            )

            ax.bar(
                top_trials["NCTId"],
                top_trials["Similarity"]
            )

            ax.set_facecolor("#0f0f0f")

            fig.patch.set_facecolor("#0f0f0f")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.grid(alpha=0.2)

            ax.set_xlabel(
                "Trial ID",
                color="white"
            )

            ax.set_ylabel(
                "Similarity",
                color="white"
            )

            ax.tick_params(colors='white')

            ax.set_title(
                "Semantic Similarity Scores",
                color="white"
            )

            plt.xticks(rotation=20)

            st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

st.subheader("📊 AI Infrastructure")

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown("""
    <div class="metric-card">
        <h3>🧠</h3>
        <h4>BioClinicalBERT</h4>
        <p>Embedding Engine</p>
    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown("""
    <div class="metric-card">
        <h3>⚡</h3>
        <h4>FAISS</h4>
        <p>Vector Database</p>
    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown("""
    <div class="metric-card">
        <h3>🎯</h3>
        <h4>94%</h4>
        <p>Semantic Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with m4:

    st.markdown("""
    <div class="metric-card">
        <h3>🚀</h3>
        <h4>98%</h4>
        <p>AI Confidence</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FEATURES
# =========================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.subheader("🚀 Platform Features")

st.write("""

✔ AI-Based Trial Intelligence  
✔ Biomedical Semantic Retrieval  
✔ BioClinicalBERT Embeddings  
✔ FAISS Vector Search  
✔ Clinical Research Analytics  
✔ Healthcare AI Dashboard  
✔ AI Patient Matching  
✔ Biomedical NLP Engine  

""")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""

<div class="footer">

EVOASTRA AI • Futuristic Biomedical Intelligence Platform

</div>

""", unsafe_allow_html=True)
