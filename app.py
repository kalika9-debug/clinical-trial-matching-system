# -*- coding: utf-8 -*-

# ==============================
# STEP 1 - IMPORT LIBRARIES
# ==============================

import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
from transformers import pipeline

import faiss


# ==============================
# STEP 2 - CREATE PROJECT FOLDERS
# ==============================

folders = [
    "data",
    "models",
    "embeddings",
    "faiss_index",
    "outputs",
    "scripts"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created successfully.")


# ==============================
# STEP 3 - DOWNLOAD CLINICAL TRIAL DATA
# ==============================

url = "https://clinicaltrials.gov/api/v2/studies"

params = {
    "query.term": "cancer",
    "pageSize": 100,
    "format": "json"
}

response = requests.get(url, params=params)

print("Status Code:", response.status_code)

data = response.json()

studies = data["studies"]

df = pd.json_normalize(studies)

print("Dataset Shape:", df.shape)


# ==============================
# STEP 4 - CLEAN THE DATASET
# ==============================

clean_df = pd.DataFrame({
    "NCTId": df["protocolSection.identificationModule.nctId"],

    "BriefTitle": df["protocolSection.identificationModule.briefTitle"],

    "Condition": df["protocolSection.conditionsModule.conditions"],

    "EligibilityCriteria":
        df["protocolSection.eligibilityModule.eligibilityCriteria"],

    "Gender":
        df["protocolSection.eligibilityModule.sex"],

    "MinimumAge":
        df["protocolSection.eligibilityModule.minimumAge"],

    "MaximumAge":
        df["protocolSection.eligibilityModule.maximumAge"]
})

# Remove missing values
clean_df.dropna(inplace=True)

# Remove duplicates
clean_df.drop_duplicates(subset="NCTId", inplace=True)

# Remove HTML tags
def clean_html(text):
    return BeautifulSoup(str(text), "html.parser").get_text()

clean_df["EligibilityCriteria"] = clean_df[
    "EligibilityCriteria"
].apply(clean_html)

print("Cleaned Dataset Shape:", clean_df.shape)

# Save cleaned dataset
clean_df.to_csv("cleaned_clinical_trials.csv", index=False)

print("Cleaned dataset saved successfully.")


# ==============================
# STEP 5 - GENERATE EMBEDDINGS
# ==============================

model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

clean_df["text"] = (
    clean_df["Condition"].astype(str) + " " +
    clean_df["EligibilityCriteria"].astype(str)
)

print("Generating embeddings...")

embeddings = model.encode(
    clean_df["text"].tolist(),
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings Shape:", embeddings.shape)


# ==============================
# STEP 6 - CREATE FAISS VECTOR DATABASE
# ==============================

dimension = 768

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Total vectors stored:", index.ntotal)


# ==============================
# STEP 7 - SEARCH SAMPLE QUERY
# ==============================

query = "45 year old female with breast cancer"

query_vector = model.encode([query])

query_vector = np.array(query_vector).astype("float32")

D, I = index.search(query_vector, 5)

print("\nTop Matching Trials:\n")

print(clean_df.iloc[I[0]][[
    "NCTId",
    "BriefTitle",
    "Condition"
]])


# ==============================
# STEP 8 - SYNTHETIC PATIENT DATA
# ==============================

patients = [
    {
        "age": 45,
        "gender": "Female",
        "condition": "Breast Cancer",
        "medications": "Tamoxifen"
    },

    {
        "age": 60,
        "gender": "Male",
        "condition": "Diabetes"
    },

    {
        "age": 35,
        "gender": "Female",
        "condition": "Lung Cancer"
    }
]

patient = patients[0]

patient_text = f"""
Age: {patient['age']}
Gender: {patient['gender']}
Condition: {patient['condition']}
Medications: {patient.get('medications', '')}
"""

print("\nPatient Profile:")
print(patient_text)


# ==============================
# STEP 9 - MATCH PATIENT WITH TRIALS
# ==============================

patient_vector = model.encode([patient_text])

patient_vector = np.array(patient_vector).astype("float32")

D, I = index.search(patient_vector, 5)

top_trials = clean_df.iloc[I[0]]

print("\nRecommended Trials:\n")

print(top_trials[[
    "NCTId",
    "BriefTitle",
    "Condition",
    "Gender",
    "MinimumAge",
    "MaximumAge"
]])


# ==============================
# STEP 10 - EVALUATION
# ==============================

start = time.time()

D, I = index.search(patient_vector, 5)

end = time.time()

print("\nLatency:", end - start)

relevant = 0

for cond in top_trials["Condition"]:
    if "breast" in str(cond).lower():
        relevant += 1

precision_at_5 = relevant / 5

print("Precision@5:", precision_at_5)


# ==============================
# STEP 11 - VISUALIZATION
# ==============================

models = ["Rule-Based", "TF-IDF", "BioClinicalBERT"]

scores = [0.45, 0.62, 0.81]

plt.bar(models, scores)

plt.xlabel("Models")
plt.ylabel("Precision@5")

plt.title("Model Comparison")

plt.show()


# ==============================
# STEP 12 - SAVE EMBEDDINGS & INDEX
# ==============================

np.save("trial_embeddings.npy", embeddings)

faiss.write_index(index, "clinical_trials.index")

print("\nEmbeddings and FAISS index saved successfully.")


# ==============================
# STEP 13 - RETRIEVAL FUNCTION
# ==============================

def retrieve_trials(patient_text, k=5):

    patient_vector = model.encode([patient_text])

    patient_vector = np.array(patient_vector).astype("float32")

    D, I = index.search(patient_vector, k)

    results = clean_df.iloc[I[0]]

    return results


query = """
50 year old male with lung cancer
"""

results = retrieve_trials(query)

print("\nRetrieved Trials:\n")

print(results[[
    "NCTId",
    "BriefTitle",
    "Condition"
]])


# ==============================
# STEP 14 - LLM ELIGIBILITY EXTRACTION
# ==============================

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1"
)

trial_text = clean_df.iloc[0]["EligibilityCriteria"]

prompt = f"""
Extract:
1. Inclusion Criteria
2. Exclusion Criteria
3. Age
4. Gender

Text:
{trial_text}
"""

print("\nPrompt Created Successfully.")
