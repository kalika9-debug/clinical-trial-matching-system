import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EVOASTRA AI",
    page_icon="🧬",
    layout="wide"
)

# =========================================================
# CLEAN CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
FONT
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
# HERO
# =========================================================

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
# MAIN GRID
# =========================================================

col1, col2 = st.columns(2)

# =========================================================
# LEFT
# =========================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧬 AI Trial Matching")

    disease = st.selectbox(
        "Disease",
        [
            "Breast Cancer",
            "Lung Cancer",
            "Diabetes",
            "Heart Disease",
            "COVID-19"
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

    patient_notes = st.text_area(
        "Patient Notes",
        height=180
    )

    st.button("🔍 Find Matching Trials")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RIGHT
# =========================================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🤖 EVOASTRA Assistant")

    user_question = st.text_input(
        "Ask Healthcare AI"
    )

    st.button("⚡ Generate Response")

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
✔ BioClinicalBERT  
✔ FAISS Search  
✔ Healthcare Analytics  
✔ Clinical Intelligence  

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
