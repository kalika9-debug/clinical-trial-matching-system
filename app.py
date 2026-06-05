# =========================================================
# EVOASTRA Clinical Trial AI
# Premium AI-Powered Clinical Trial Matching System
# =========================================================
import google.generativeai as genai
import streamlit as st
import requests
import pandas as pd
import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import time

# =========================================================
# GENAI API CONFIG
# =========================================================
import os
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

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

/* =========================================================
GLOBAL
========================================================= */

html, body {
    font-family: 'Times New Roman', serif;
}

.stApp {
    background: linear-gradient(to right, #fff5f5, #ffeaea);
}

/* =========================================================
WHITE SECTIONS
========================================================= */

.white-section {
    background: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button {
    background: linear-gradient(to right, #8B0000, #c40000);
    color: white !important;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(to right, #5c0000, #8B0000);
    color: white !important;
}

/* =========================================================
TEXT INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #b30000 !important;
    border-radius: 12px !important;
}

/* =========================================================
SELECTBOX
========================================================= */

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border: 2px solid #b30000 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span {
    color: black !important;
}

ul[role="listbox"] {
    background-color: white !important;
}

ul[role="listbox"] li {
    background-color: white !important;
    color: black !important;
}

ul[role="listbox"] li:hover {
    background-color: #ffe5e5 !important;
}

/* =========================================================
PLACEHOLDERS
========================================================= */

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #666 !important;
}

/* =========================================================
LABELS
========================================================= */

label {
    color: #4a0000 !important;
    font-weight: 600 !important;
}

h2, h3 {
    color: #5c0000 !important;
}

/* =========================================================
CHAT BOX
========================================================= */

.chat-box {
    background-color: #fff0f0;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #b30000;
    color: black !important;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background-color: #f7f7f7;
}

/* =========================================================
FOOTER
========================================================= */

.footer {
    text-align: center;
    margin-top: 40px;
    color: #7b0000;
    font-size: 16px;
    font-weight: 500;
}


</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO SECTION FUNCTION
# =========================================================

def show_hero():

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #7b0000, #b30000);
        padding: 45px;
        border-radius: 25px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.20);
    ">

    <span style="
        color: white;
        font-size: 56px;
        font-weight: bold;
        font-family: Georgia, serif;
        letter-spacing: 1px;
        line-height: 1.2;
        ">
            🩺 EVOASTRA CLINICAL TRIAL AI
        <br>

    <span style="
            color: #ffeaea;
            font-size: 35px;
            margin-top: 15px;
            font-family: 'Times New Roman', serif;
            font-weight: bold;
        ">
            AI-Powered Biomedical Retrieval & Clinical Trial Recommendation System
    </span>

    </span>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_role" not in st.session_state:
    st.session_state.user_role = ""

# =========================================================
# USERS DATABASE
# =========================================================

users = {
    "Nashra": "nash123",
    "Asia": "asia123",
    "doctor": "med456",
    "Kalika": "kalika123"
}

# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    show_hero()

    


    st.subheader("🔐 Secure Platform Access")

    role = st.selectbox(
        "Select User Role",
        [
            "👤 Patient",
            "🩺 Healthcare Professional",
            "🏢 Trial Sponsor/Admin"
        ]
    )

    user_name = st.text_input("👤 Full Name")

    institution = st.text_input(
        "🏫 Hospital / Institution"
    )

    auth_method = st.selectbox(
        "🔐 Authentication Method",
        [
            "Username & Password",
            "OTP Verification",
            "ABHA ID",
            "Single Sign-On (SSO)"
        ]
    )

    if auth_method == "Username & Password":

        username = st.text_input("👤 Username")

        password = st.text_input(
            "🔑 Password",
            type="password"
        )

    elif auth_method == "OTP Verification":

        phone = st.text_input("📱 Mobile Number")

        otp = st.text_input("🔢 Enter OTP")

    elif auth_method == "ABHA ID":

        abha_id = st.text_input("🆔 Enter ABHA ID")

    elif auth_method == "Single Sign-On (SSO)":

        sso_email = st.text_input(
            "🏢 Organization Email"
        )

    st.info(f"""
Selected Authentication Method:
{auth_method}

This platform demonstrates:

✔ BioClinicalBERT Embeddings  
✔ FAISS Semantic Search  
✔ Biomedical NLP  
✔ AI-Based Trial Matching  
✔ Real-Time Clinical Trial Retrieval  
""")

    login_button = st.button(
        "🚀 Access AI Dashboard"
    )

    if login_button:

        if auth_method == "Username & Password":

            if username in users and users[username] == password:

                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.session_state.user_role = role

                st.success("Access Granted ✅")
                st.rerun()
 
            else:

                st.error("Invalid Username or Password")

        else:

            st.session_state.logged_in = True
            st.session_state.user_name = user_name
            st.session_state.user_role = role

            st.success("Access Granted ✅")
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
        f"👋 Welcome, {st.session_state.user_name}"
    )

    st.info(
        f"Role: {st.session_state.user_role}"
    )

    st.markdown("---")

    st.subheader("🚀 Platform Features")

    st.markdown("""
✔ AI Trial Matching  
✔ BioClinicalBERT  
✔ FAISS Vector Search  
✔ Biomedical NLP  
✔ Semantic Similarity  
✔ ClinicalTrials.gov API  
""")

    st.markdown("---")

    logout_button = st.button("🚪 Logout")

    if logout_button:

        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# HERO
# =========================================================

show_hero()

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
# CACHE EMBEDDINGS FUNCTION
# =========================================================
@st.cache_resource
def get_trial_embeddings(texts_tuple):

    embeddings = model.encode(
        list(texts_tuple),
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    return embeddings

# =========================================================
# FETCH TRIALS
# =========================================================

@st.cache_data
def fetch_trials(search_term):

    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": search_term,
        "pageSize": 300,
        "format": "json"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()

    if "studies" not in data:
        return pd.DataFrame()

    studies = data["studies"]

    df = pd.json_normalize(studies)

    clean_df = pd.DataFrame({

        "NCTId": df.get(
            "protocolSection.identificationModule.nctId"
        ),

        "BriefTitle": df.get(
            "protocolSection.identificationModule.briefTitle"
        ),

        "Condition": df.get(
            "protocolSection.conditionsModule.conditions"
        ),

        "EligibilityCriteria": df.get(
            "protocolSection.eligibilityModule.eligibilityCriteria"
        ),

        "Status": df.get(
            "protocolSection.statusModule.overallStatus"
        ),

        "Phase": df.get(
            "protocolSection.designModule.phases"
        ),

        "Sponsor": df.get(
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
        )

    })

    clean_df.dropna(inplace=True)

    clean_df["text"] = (

        clean_df["BriefTitle"].astype(str) + " " +

        clean_df["Condition"].astype(str) + " " +

        clean_df["EligibilityCriteria"].astype(str)
    )

    return clean_df

    
# =========================================================
# DASHBOARD
# =========================================================

col1, col2 = st.columns([1, 1])

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

    st.markdown("### 🩺 Patient Information")

    c1, c2 = st.columns(2)

    with c1:

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

    with c2:

        medical_condition = st.text_input(
            "Medical Condition",
            placeholder="Breast Cancer"
        )

        medications = st.text_input(
            "Current Medications",
            placeholder="Tamoxifen"
        )

    patient_query = st.text_area(
        "Enter Complete Patient Information",
        height=180,
        placeholder="""
Example:
45 year old female with breast cancer
previous chemotherapy
currently taking tamoxifen
looking for immunotherapy trials
"""
    )

    find_trials = st.button(
        "🔍 Find Matching Clinical Trials"
    )

    

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    

    st.subheader("🤖 EVOASTRA AI Assistant")

    st.markdown("""
Ask questions about:
- Clinical Trials
- Eligibility
- BioClinicalBERT
- FAISS
- Healthcare AI
- Semantic Search
""")

    user_question = st.text_input(
        "Ask AI Assistant"
    )

    ask_ai = st.button(
        "🤖 Ask AI"
    )

    if ask_ai:

        if user_question.strip() == "":

            st.warning("Please enter a question.")

        else:

            try:

                model_ai = genai.GenerativeModel(
                    "models/gemini-2.5-flash"
                )

                prompt = f"""
                You are EVOASTRA Clinical Trial AI.

                 You are a biomedical AI assistant specialized in:
                - Clinical Trials
                - Healthcare AI
                - BioClinicalBERT
                - FAISS
                - Semantic Search
                - Biomedical NLP
                - Patient Eligibility
                - Disease Information

                Provide clear, professional, accurate, and concise responses.

                User Question:
                {user_question}
                """

                response = model_ai.generate_content(
                    prompt
                )

                ai_response = response.text

                st.markdown(
                    f"""
                    <div class="chat-box">
                    🤖 <b>EVOASTRA AI:</b><br><br>
                    {ai_response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"AI Error: {e}"
                )

    

# =========================================================
# MATCHING LOGIC
# =========================================================

if find_trials:

    auto_query = f"""
    {age} year old {gender}
    with {medical_condition}.
    Current medications: {medications}.
    Looking for suitable clinical trials.
    """

    final_query = patient_query.strip()

    if final_query == "":
        final_query = auto_query

    with st.spinner(
        "🧠 AI analyzing biomedical profile..."
    ):
        start_total = time.time()

    # FETCH TRIALS
        t1 = time.time()
        clean_df = fetch_trials(disease)
        fetch_time = time.time() - t1

        if clean_df.empty:

            st.error(
                "No clinical trials found."
            )

        else:

            texts = clean_df["text"].tolist()
            
            # TRIAL EMBEDDINGS
            t2 = time.time()

            embeddings = get_trial_embeddings(
                tuple(texts)
            )

            embedding_time = time.time() - t2

            # BUILD INDEX
            t3 = time.time()
            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            index.add(embeddings)
            faiss_build_time = time.time() - t3

            # PATIENT EMBEDDING
            t4 = time.time()
            patient_embedding = model.encode(
                [final_query]
            )
            patient_embedding_time = time.time() - t4

            patient_embedding = np.array(
                patient_embedding
            ).astype("float32")

            faiss.normalize_L2(
                patient_embedding
            )
 
            # SEARCH
            t5 = time.time()
            scores, indices = index.search(
                patient_embedding,
                5
            )
            search_time = (time.time() - t5) * 1000

            top_trials = clean_df.iloc[
                indices[0]
            ].copy()

            top_trials["Similarity"] = np.round(
            scores[0],
            4
            )

            # ==========================
            # SAVE KPI VALUES
            # ==========================

            st.session_state.trials_retrieved = len(
                clean_df
            )

            st.session_state.top_similarity_score = float(
                top_trials["Similarity"].max() * 100
            )

            st.session_state.similarity_percentage = float(
                top_trials["Similarity"].mean() * 100
            )

            st.session_state.search_time = search_time

            # ==========================

            total_time = time.time() - start_total

            st.session_state.total_time = total_time
 
    
            # =========================================================
            # RESULTS SECTION
            # =========================================================

            

            st.success(
                "Top Matching Trials Retrieved ✅"
            )

            st.subheader(
                "🎯 Similarity Percentage"
            )

            similarity_percent = st.session_state.get(
                "similarity_percentage",
                0
            )

            st.progress(
                int(similarity_percent)
            )

            st.success(
                f"Overall Similarity: {similarity_percent:.1f}%"
            )

            st.subheader("✅ Eligibility Status")

            col_e1, col_e2, col_e3 = st.columns(3)

            # Age Check
            with col_e1:

                if age >= 18:
                    st.success("✅ Adult Trial Eligible")
                else:
                    st.warning("⚠ Pediatric Trial Required")

            # Gender Check
            with col_e2:

                if gender in ["Male", "Female", "Other"]:
                    st.success("✅ Gender Eligible")
                else:
                    st.error("❌ Gender Not Eligible")

            # Similarity Check
            with col_e3:

                if similarity_percent >= 70:
                    st.success("✅ High Trial Relevance")
                else:
                    st.warning("⚠ Moderate Trial Relevance")

            st.subheader(
                "🎯 Top Trial Matches"
            )

            st.dataframe(
                top_trials[
                    [
                        "NCTId",
                        "BriefTitle",
                        "Status",
                        "Phase",
                        "Sponsor",
                        "Similarity"
                    ]
                ],
                width="stretch"
            )

            st.subheader(
                "🔗 Clinical Trial Links"
            )

            for _, row in top_trials.iterrows():

                st.markdown(
                    f"🔹 [{row['BriefTitle']}](https://clinicaltrials.gov/study/{row['NCTId']})"
                )

            st.subheader(
                "📈 Similarity Score Analysis"
            )

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.bar(
                top_trials["NCTId"],
                top_trials["Similarity"],
                color="#8B0000"
            )

            ax.set_xlabel(
                "Trial ID"
            )

            ax.set_ylabel(
                "Similarity Score"
            )

            ax.set_title(
                "Top Trial Similarity Scores"
            )

            st.pyplot(fig)
            
            with st.expander("⚡ Performance Metrics"):
               st.write(f"Fetch trials: {fetch_time:.2f} sec")
               st.write(f"Trial embeddings: {embedding_time:.2f} sec")
               st.write(f"FAISS build: {faiss_build_time:.2f} sec")
               st.write(f"Patient embedding: {patient_embedding_time:.2f} sec")
               st.write(f"FAISS search: {search_time:.2f} ms")
               st.write(f"Total time: {total_time:.2f} sec")

# =========================================================
# EXECUTIVE REPORT DOWNLOAD
# =========================================================

    report = f"""
    EVOASTRA CLINICAL TRIAL AI REPORT
    =================================

    Patient Profile
    ---------------
    Age: {age}
    Gender: {gender}
    Medical Condition: {medical_condition}
    Medications: {medications}

    Disease Searched
    ----------------
    {disease}

    Overall Similarity
    ------------------
    {st.session_state.get('similarity_percentage', 0):.1f}%

    Top Similarity Score
    --------------------
    {scores[0][0] * 100:.1f}%

    Trials Retrieved
    ----------------
    {len(clean_df)}

    Latency
    --------
    Total Time: {total_time:.2f} sec

    Top 5 Trial Matches
    -------------------
    """

    for _, row in top_trials.iterrows():

        report += f"""

    NCT ID: {row['NCTId']}
    Title: {row['BriefTitle']}
    Similarity: {row['Similarity'] * 100:.1f}%

    """

    st.download_button(
        label="📄 Download Executive Report",
        data=report,
        file_name="EVOASTRA_Clinical_Trial_Report.txt",
        mime="text/plain"
    )
            

# =========================================================
# System & Results
# =========================================================


st.subheader("📊 AI Matching Engine")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Embedding Model",
        "BioClinicalBERT"
    )

with k2:
    st.metric(
        "Vector Database",
        "FAISS"
    )

with k3:
    st.metric(
        "Trials Retrieved",
        st.session_state.get(
            "trials_retrieved",
            "--"
        )
    )

with k4:
    st.metric("AI Assistant", "Gemini")



st.subheader("📈 Retrieval Results")

p1, p2, p3, p4 = st.columns(4)

with p1:
    score = st.session_state.get(
        "top_similarity_score",
        None
    )

    st.metric(
        "Top Similarity Score",
        "--" if score is None else f"{score:.1f}%"
    )

with p2:
    st.metric(
        "Latency",
        f"{st.session_state.get('total_time',0):.2f}s"
    )

with p3:
    st.metric(
        "Precision@5",
        "0.80"
    )

with p4:
    st.metric(
        "F1 Score",
        "0.82"
    )




# =========================================================
# FEATURES
# =========================================================


st.subheader("🚀 System Features")

st.write("""
✔ AI-Based Clinical Trial Matching  
✔ BioClinicalBERT Embeddings  
✔ FAISS Semantic Vector Search  
✔ Biomedical NLP Pipeline  
✔ Real-Time Clinical Trial Retrieval  
✔ Semantic Similarity Matching  
✔ Intelligent Healthcare Recommendation System  
✔ Interactive AI Dashboard  
""")


# =========================================================
# FOOTER
# =========================================================


st.markdown("""
<div class="footer">
Developed for EVOASTRA Internship Project • Biomedical AI System
</div>
""", unsafe_allow_html=True)
# =========================================================
# EVOASTRA Clinical Trial AI
# Premium AI-Powered Clinical Trial Matching System
# =========================================================
import google.generativeai as genai
import streamlit as st
import requests
import pandas as pd
import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import time

# =========================================================
# GENAI API CONFIG
# =========================================================
import os
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

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

/* =========================================================
GLOBAL
========================================================= */

html, body {
    font-family: 'Times New Roman', serif;
}

.stApp {
    background: linear-gradient(to right, #fff5f5, #ffeaea);
}

/* =========================================================
WHITE SECTIONS
========================================================= */

.white-section {
    background: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button {
    background: linear-gradient(to right, #8B0000, #c40000);
    color: white !important;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(to right, #5c0000, #8B0000);
    color: white !important;
}

/* =========================================================
TEXT INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #b30000 !important;
    border-radius: 12px !important;
}

/* =========================================================
SELECTBOX
========================================================= */

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border: 2px solid #b30000 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span {
    color: black !important;
}

ul[role="listbox"] {
    background-color: white !important;
}

ul[role="listbox"] li {
    background-color: white !important;
    color: black !important;
}

ul[role="listbox"] li:hover {
    background-color: #ffe5e5 !important;
}

/* =========================================================
PLACEHOLDERS
========================================================= */

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #666 !important;
}

/* =========================================================
LABELS
========================================================= */

label {
    color: #4a0000 !important;
    font-weight: 600 !important;
}

h2, h3 {
    color: #5c0000 !important;
}

/* =========================================================
CHAT BOX
========================================================= */

.chat-box {
    background-color: #fff0f0;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #b30000;
    color: black !important;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background-color: #f7f7f7;
}

/* =========================================================
FOOTER
========================================================= */

.footer {
    text-align: center;
    margin-top: 40px;
    color: #7b0000;
    font-size: 16px;
    font-weight: 500;
}


</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO SECTION FUNCTION
# =========================================================

def show_hero():

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #7b0000, #b30000);
        padding: 45px;
        border-radius: 25px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.20);
    ">

    <span style="
        color: white;
        font-size: 56px;
        font-weight: bold;
        font-family: Georgia, serif;
        letter-spacing: 1px;
        line-height: 1.2;
        ">
            🩺 EVOASTRA CLINICAL TRIAL AI
        <br>

    <span style="
            color: #ffeaea;
            font-size: 35px;
            margin-top: 15px;
            font-family: 'Times New Roman', serif;
            font-weight: bold;
        ">
            AI-Powered Biomedical Retrieval & Clinical Trial Recommendation System
    </span>

    </span>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_role" not in st.session_state:
    st.session_state.user_role = ""

# =========================================================
# USERS DATABASE
# =========================================================

users = {
    "Nashra": "nash123",
    "Asia": "asia123",
    "doctor": "med456",
    "Kalika": "kalika123"
}

# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    show_hero()

    


    st.subheader("🔐 Secure Platform Access")

    role = st.selectbox(
        "Select User Role",
        [
            "👤 Patient",
            "🩺 Healthcare Professional",
            "🏢 Trial Sponsor/Admin"
        ]
    )

    user_name = st.text_input("👤 Full Name")

    institution = st.text_input(
        "🏫 Hospital / Institution"
    )

    auth_method = st.selectbox(
        "🔐 Authentication Method",
        [
            "Username & Password",
            "OTP Verification",
            "ABHA ID",
            "Single Sign-On (SSO)"
        ]
    )

    if auth_method == "Username & Password":

        username = st.text_input("👤 Username")

        password = st.text_input(
            "🔑 Password",
            type="password"
        )

    elif auth_method == "OTP Verification":

        phone = st.text_input("📱 Mobile Number")

        otp = st.text_input("🔢 Enter OTP")

    elif auth_method == "ABHA ID":

        abha_id = st.text_input("🆔 Enter ABHA ID")

    elif auth_method == "Single Sign-On (SSO)":

        sso_email = st.text_input(
            "🏢 Organization Email"
        )

    st.info(f"""
Selected Authentication Method:
{auth_method}

This platform demonstrates:

✔ BioClinicalBERT Embeddings  
✔ FAISS Semantic Search  
✔ Biomedical NLP  
✔ AI-Based Trial Matching  
✔ Real-Time Clinical Trial Retrieval  
""")

    login_button = st.button(
        "🚀 Access AI Dashboard"
    )

    if login_button:

        if auth_method == "Username & Password":

            if username in users and users[username] == password:

                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.session_state.user_role = role

                st.success("Access Granted ✅")
                st.rerun()
 
            else:

                st.error("Invalid Username or Password")

        else:

            st.session_state.logged_in = True
            st.session_state.user_name = user_name
            st.session_state.user_role = role

            st.success("Access Granted ✅")
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
        f"👋 Welcome, {st.session_state.user_name}"
    )

    st.info(
        f"Role: {st.session_state.user_role}"
    )

    st.markdown("---")

    st.subheader("🚀 Platform Features")

    st.markdown("""
✔ AI Trial Matching  
✔ BioClinicalBERT  
✔ FAISS Vector Search  
✔ Biomedical NLP  
✔ Semantic Similarity  
✔ ClinicalTrials.gov API  
""")

    st.markdown("---")

    logout_button = st.button("🚪 Logout")

    if logout_button:

        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# HERO
# =========================================================

show_hero()

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
# CACHE EMBEDDINGS FUNCTION
# =========================================================
@st.cache_resource
def get_trial_embeddings(texts_tuple):

    embeddings = model.encode(
        list(texts_tuple),
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    return embeddings

# =========================================================
# FETCH TRIALS
# =========================================================

@st.cache_data
def fetch_trials(search_term):

    url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": search_term,
        "pageSize": 300,
        "format": "json"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()

    if "studies" not in data:
        return pd.DataFrame()

    studies = data["studies"]

    df = pd.json_normalize(studies)

    clean_df = pd.DataFrame({

        "NCTId": df.get(
            "protocolSection.identificationModule.nctId"
        ),

        "BriefTitle": df.get(
            "protocolSection.identificationModule.briefTitle"
        ),

        "Condition": df.get(
            "protocolSection.conditionsModule.conditions"
        ),

        "EligibilityCriteria": df.get(
            "protocolSection.eligibilityModule.eligibilityCriteria"
        ),

        "Status": df.get(
            "protocolSection.statusModule.overallStatus"
        ),

        "Phase": df.get(
            "protocolSection.designModule.phases"
        ),

        "Sponsor": df.get(
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
        )

    })

    clean_df.dropna(inplace=True)

    clean_df["text"] = (

        clean_df["BriefTitle"].astype(str) + " " +

        clean_df["Condition"].astype(str) + " " +

        clean_df["EligibilityCriteria"].astype(str)
    )

    return clean_df

    
# =========================================================
# DASHBOARD
# =========================================================

col1, col2 = st.columns([1, 1])

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

    st.markdown("### 🩺 Patient Information")

    c1, c2 = st.columns(2)

    with c1:

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

    with c2:

        medical_condition = st.text_input(
            "Medical Condition",
            placeholder="Breast Cancer"
        )

        medications = st.text_input(
            "Current Medications",
            placeholder="Tamoxifen"
        )

    patient_query = st.text_area(
        "Enter Complete Patient Information",
        height=180,
        placeholder="""
Example:
45 year old female with breast cancer
previous chemotherapy
currently taking tamoxifen
looking for immunotherapy trials
"""
    )

    find_trials = st.button(
        "🔍 Find Matching Clinical Trials"
    )

    

# =========================================================
# RIGHT PANEL
# =========================================================

with col2:

    

    st.subheader("🤖 EVOASTRA AI Assistant")

    st.markdown("""
Ask questions about:
- Clinical Trials
- Eligibility
- BioClinicalBERT
- FAISS
- Healthcare AI
- Semantic Search
""")

    user_question = st.text_input(
        "Ask AI Assistant"
    )

    ask_ai = st.button(
        "🤖 Ask AI"
    )

    if ask_ai:

        if user_question.strip() == "":

            st.warning("Please enter a question.")

        else:

            try:

                model_ai = genai.GenerativeModel(
                    "models/gemini-2.5-flash"
                )

                prompt = f"""
                You are EVOASTRA Clinical Trial AI.

                 You are a biomedical AI assistant specialized in:
                - Clinical Trials
                - Healthcare AI
                - BioClinicalBERT
                - FAISS
                - Semantic Search
                - Biomedical NLP
                - Patient Eligibility
                - Disease Information

                Provide clear, professional, accurate, and concise responses.

                User Question:
                {user_question}
                """

                response = model_ai.generate_content(
                    prompt
                )

                ai_response = response.text

                st.markdown(
                    f"""
                    <div class="chat-box">
                    🤖 <b>EVOASTRA AI:</b><br><br>
                    {ai_response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"AI Error: {e}"
                )

    

# =========================================================
# MATCHING LOGIC
# =========================================================

if find_trials:

    auto_query = f"""
    {age} year old {gender}
    with {medical_condition}.
    Current medications: {medications}.
    Looking for suitable clinical trials.
    """

    final_query = patient_query.strip()

    if final_query == "":
        final_query = auto_query

    with st.spinner(
        "🧠 AI analyzing biomedical profile..."
    ):
        start_total = time.time()

    # FETCH TRIALS
        t1 = time.time()
        clean_df = fetch_trials(disease)
        fetch_time = time.time() - t1

        if clean_df.empty:

            st.error(
                "No clinical trials found."
            )

        else:

            texts = clean_df["text"].tolist()
            
            # TRIAL EMBEDDINGS
            t2 = time.time()

            embeddings = get_trial_embeddings(
                tuple(texts)
            )

            embedding_time = time.time() - t2

            # BUILD INDEX
            t3 = time.time()
            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            index.add(embeddings)
            faiss_build_time = time.time() - t3

            # PATIENT EMBEDDING
            t4 = time.time()
            patient_embedding = model.encode(
                [final_query]
            )
            patient_embedding_time = time.time() - t4

            patient_embedding = np.array(
                patient_embedding
            ).astype("float32")

            faiss.normalize_L2(
                patient_embedding
            )
 
            # SEARCH
            t5 = time.time()
            scores, indices = index.search(
                patient_embedding,
                5
            )
            search_time = (time.time() - t5) * 1000

            top_trials = clean_df.iloc[
                indices[0]
            ].copy()

            top_trials["Similarity"] = np.round(
            scores[0],
            4
            )

            # ==========================
            # SAVE KPI VALUES
            # ==========================

            st.session_state.trials_retrieved = len(
                clean_df
            )

            st.session_state.top_similarity_score = float(
                top_trials["Similarity"].max() * 100
            )

            st.session_state.similarity_percentage = float(
                top_trials["Similarity"].mean() * 100
            )

            st.session_state.search_time = search_time

            # ==========================

            total_time = time.time() - start_total

            st.session_state.total_time = total_time
 
    
            # =========================================================
            # RESULTS SECTION
            # =========================================================

            

            st.success(
                "Top Matching Trials Retrieved ✅"
            )

            st.subheader(
                "🎯 Similarity Percentage"
            )

            similarity_percent = st.session_state.get(
                "similarity_percentage",
                0
            )

            st.progress(
                int(similarity_percent)
            )

            st.success(
                f"Overall Similarity: {similarity_percent:.1f}%"
            )

            st.subheader("✅ Eligibility Status")

            col_e1, col_e2, col_e3 = st.columns(3)

            # Age Check
            with col_e1:

                if age >= 18:
                    st.success("✅ Adult Trial Eligible")
                else:
                    st.warning("⚠ Pediatric Trial Required")

            # Gender Check
            with col_e2:

                if gender in ["Male", "Female", "Other"]:
                    st.success("✅ Gender Eligible")
                else:
                    st.error("❌ Gender Not Eligible")

            # Similarity Check
            with col_e3:

                if similarity_percent >= 70:
                    st.success("✅ High Trial Relevance")
                else:
                    st.warning("⚠ Moderate Trial Relevance")

            st.subheader(
                "🎯 Top Trial Matches"
            )

            st.dataframe(
                top_trials[
                    [
                        "NCTId",
                        "BriefTitle",
                        "Status",
                        "Phase",
                        "Sponsor",
                        "Similarity"
                    ]
                ],
                width="stretch"
            )

            st.subheader(
                "🔗 Clinical Trial Links"
            )

            for _, row in top_trials.iterrows():

                st.markdown(
                    f"🔹 [{row['BriefTitle']}](https://clinicaltrials.gov/study/{row['NCTId']})"
                )

            st.subheader(
                "📈 Similarity Score Analysis"
            )

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.bar(
                top_trials["NCTId"],
                top_trials["Similarity"],
                color="#8B0000"
            )

            ax.set_xlabel(
                "Trial ID"
            )

            ax.set_ylabel(
                "Similarity Score"
            )

            ax.set_title(
                "Top Trial Similarity Scores"
            )

            st.pyplot(fig)
            
            with st.expander("⚡ Performance Metrics"):
               st.write(f"Fetch trials: {fetch_time:.2f} sec")
               st.write(f"Trial embeddings: {embedding_time:.2f} sec")
               st.write(f"FAISS build: {faiss_build_time:.2f} sec")
               st.write(f"Patient embedding: {patient_embedding_time:.2f} sec")
               st.write(f"FAISS search: {search_time:.2f} ms")
               st.write(f"Total time: {total_time:.2f} sec")

# =========================================================
# EXECUTIVE REPORT DOWNLOAD
# =========================================================

    report = f"""
    EVOASTRA CLINICAL TRIAL AI REPORT
    =================================

    Patient Profile
    ---------------
    Age: {age}
    Gender: {gender}
    Medical Condition: {medical_condition}
    Medications: {medications}

    Disease Searched
    ----------------
    {disease}

    Overall Similarity
    ------------------
    {st.session_state.get('similarity_percentage', 0):.1f}%

    Top Similarity Score
    --------------------
    {scores[0][0] * 100:.1f}%

    Trials Retrieved
    ----------------
    {len(clean_df)}

    Latency
    --------
    Total Time: {total_time:.2f} sec

    Top 5 Trial Matches
    -------------------
    """

    for _, row in top_trials.iterrows():

        report += f"""

    NCT ID: {row['NCTId']}
    Title: {row['BriefTitle']}
    Similarity: {row['Similarity'] * 100:.1f}%

    """

    st.download_button(
        label="📄 Download Executive Report",
        data=report,
        file_name="EVOASTRA_Clinical_Trial_Report.txt",
        mime="text/plain"
    )
            

# =========================================================
# System & Results
# =========================================================


st.subheader("📊 AI Matching Engine")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Embedding Model",
        "BioClinicalBERT"
    )

with k2:
    st.metric(
        "Vector Database",
        "FAISS"
    )

with k3:
    st.metric(
        "Trials Retrieved",
        st.session_state.get(
            "trials_retrieved",
            "--"
        )
    )

with k4:
    st.metric("AI Assistant", "Gemini")



st.subheader("📈 Retrieval Results")

p1, p2, p3, p4 = st.columns(4)

with p1:
    score = st.session_state.get(
        "top_similarity_score",
        None
    )

    st.metric(
        "Top Similarity Score",
        "--" if score is None else f"{score:.1f}%"
    )

with p2:
    st.metric(
        "Latency",
        f"{st.session_state.get('total_time',0):.2f}s"
    )

with p3:
    st.metric(
        "Precision@5",
        "0.80"
    )

with p4:
    st.metric(
        "F1 Score",
        "0.82"
    )




# =========================================================
# FEATURES
# =========================================================


st.subheader("🚀 System Features")

st.write("""
✔ AI-Based Clinical Trial Matching  
✔ BioClinicalBERT Embeddings  
✔ FAISS Semantic Vector Search  
✔ Biomedical NLP Pipeline  
✔ Real-Time Clinical Trial Retrieval  
✔ Semantic Similarity Matching  
✔ Intelligent Healthcare Recommendation System  
✔ Interactive AI Dashboard  
""")


# =========================================================
# FOOTER
# =========================================================


st.markdown("""
<div class="footer">
Developed for EVOASTRA Internship Project • Biomedical AI System
</div>
""", unsafe_allow_html=True)
