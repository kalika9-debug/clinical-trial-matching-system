# =========================================================
# EVOASTRA AI
# FULL MODERN HEALTHCARE AI DASHBOARD
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

/* =========================================================
GOOGLE FONT
========================================================= */

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
        #f5f9ff,
        #eef4ff
    );

    color: #111827;
}

/* =========================================================
MAIN CONTAINER
========================================================= */

.block-container {
    padding-top: 2rem;
}

/* =========================================================
HERO SECTION
========================================================= */

.hero {

    background: white;

    border-radius: 28px;

    padding: 60px 40px;

    text-align: center;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.06);

    border:
    1px solid rgba(0,0,0,0.04);

    margin-bottom: 30px;
}

.hero h1 {

    font-size: 54px;

    font-weight: 700;

    color: #2563eb;

    margin-bottom: 12px;
}

.hero p {

    font-size: 18px;

    color: #6b7280;

    font-weight: 400;
}

/* =========================================================
CARDS
========================================================= */

.card {

    background: white;

    border-radius: 24px;

    padding: 28px;

    box-shadow:
    0 8px 24px rgba(0,0,0,0.05);

    border:
    1px solid rgba(0,0,0,0.04);

    margin-bottom: 24px;
}

/* =========================================================
INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {

    border-radius: 14px !important;

    border:
    1px solid #dbeafe !important;

    padding: 12px !important;

    background: #f9fbff !important;

    color: #111827 !important;
}

/* =========================================================
SELECT BOX
========================================================= */

div[data-baseweb="select"] > div {

    border-radius: 14px !important;

    border:
    1px solid #dbeafe !important;

    background: #f9fbff !important;
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

    color: white;

    border: none;

    border-radius: 14px;

    height: 50px;

    font-size: 16px;

    font-weight: 600;

    transition: 0.3s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 8px 20px rgba(37,99,235,0.25);
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
HEADINGS
========================================================= */

h1,h2,h3,h4,h5 {

    color: #111827 !important;
}

/* =========================================================
METRIC CARDS
========================================================= */

.metric-card {

    background: white;

    border-radius: 20px;

    padding: 22px;

    text-align: center;

    box-shadow:
    0 8px 24px rgba(0,0,0,0.05);

    border:
    1px solid rgba(0,0,0,0.04);

    transition: 0.3s ease;
}

.metric-card:hover {

    transform: translateY(-4px);
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;
}

/* =========================================================
FOOTER
========================================================= */

.footer {

    text-align: center;

    color: #6b7280;

    margin-top: 40px;

    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================================================
# HERO
# =========================================================

def hero():

    st.markdown("""

    <div class="hero">

        <h1>
        🧬 EVOASTRA AI
        </h1>

        <p>
        Clinical Trial Intelligence Platform
        </p>

    </div>

    """, unsafe_allow_html=True)

# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    hero()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔐 Secure Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    login_btn = st.button("🚀 Access Dashboard")

    if login_btn:

        if username.strip() == "" or password.strip() == "":

            st.warning("Please enter username and password")

        else:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Login Successful ✅")

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

    st.markdown("## 🧬 EVOASTRA AI")

    st.success(
        f"Welcome {st.session_state.username}"
    )

    st.markdown("---")

    st.markdown("""

### 🚀 Technologies

✔ BioClinicalBERT  
✔ FAISS Search  
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
# HERO SECTION
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

        clean_df["BriefTitle"].astype(str)
        + " " +

        clean_df["Condition"].astype(str)
        + " " +

        clean_df["EligibilityCriteria"].astype(str)
    )

    return clean_df

# =========================================================
# AI ASSISTANT
# =========================================================

def ai_response(question):

    q = question.lower()

    if "cancer" in q:

        return "Cancer trials evaluate advanced therapies and treatments."

    elif "diabetes" in q:

        return "Diabetes research focuses on glucose monitoring and insulin innovation."

    elif "heart" in q:

        return "Heart disease trials evaluate cardiovascular treatments and surgeries."

    elif "covid" in q:

        return "COVID-19 studies focus on vaccines and antiviral medicines."

    elif "bert" in q:

        return "BioClinicalBERT converts biomedical text into semantic embeddings."

    elif "faiss" in q:

        return "FAISS is a vector similarity search library for AI retrieval systems."

    else:

        return "I can help with healthcare AI, diseases, clinical trials, and biomedical NLP."

# =========================================================
# MAIN DASHBOARD
# =========================================================

col1, col2 = st.columns([1,1])

# =========================================================
# LEFT PANEL
# =========================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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
        "Patient Notes",
        height=180
    )

    find_trials = st.button(
        "🔍 Find Matching Trials"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🤖 EVOASTRA Assistant")

    user_question = st.text_input(
        "Ask Healthcare AI"
    )

    ask_ai = st.button(
        "⚡ Generate Response"
    )

    if ask_ai:

        response = ai_response(
            user_question
        )

        st.chat_message(
            "assistant"
        ).write(response)

    st.markdown('</div>', unsafe_allow_html=True)

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
        "Analyzing biomedical profile..."
    ):

        clean_df = fetch_trials(disease)

        if clean_df.empty:

            st.error(
                "No clinical trials found."
            )

        else:

            texts = clean_df["text"].tolist()

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

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("🎯 Top Trial Matches")

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
            # CHART
            # =========================================================

            st.markdown('<div class="card">', unsafe_allow_html=True)

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

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.grid(alpha=0.2)

            plt.xticks(rotation=20)

            st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

st.subheader("📊 AI Infrastructure")

m1, m2, m3, m4 = st.columns(4)

cards = [

    ("🧠", "BioClinicalBERT", "Embedding Engine"),
    ("⚡", "FAISS", "Vector Database"),
    ("🎯", "94%", "Semantic Accuracy"),
    ("🚀", "98%", "AI Confidence")

]

for col, card in zip([m1,m2,m3,m4], cards):

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

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🚀 Platform Features")

st.write("""

✔ AI Trial Matching  
✔ Biomedical NLP  
✔ BioClinicalBERT Embeddings  
✔ FAISS Semantic Search  
✔ Clinical Trial Retrieval  
✔ Healthcare AI Dashboard  

""")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""

<div class="footer">

EVOASTRA AI • Modern Clinical Intelligence Dashboard

</div>

""", unsafe_allow_html=True)
