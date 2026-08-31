import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import re
from collections import Counter

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Sleep Coach MVP", page_icon="🌙", layout="wide")

st.title("🌙 AI Sleep Coach MVP")
st.caption("A Hybrid System Combining Machine Learning Predictive Analytics & RAG-Powered Conversational AI")

# Retrieve API Key securely from Streamlit Secrets
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", None)

if not openrouter_api_key:
    st.sidebar.warning("⚠️ OpenRouter API Key missing in Streamlit Secrets.")
else:
    st.sidebar.success("🔒 API Key loaded securely from Secrets!")

# ==============================================================================
# LOAD ARTIFACTS (.pkl files)
# ==============================================================================
@st.cache_resource
def load_pickle_artifacts():
    ml_payload = joblib.load('sleep_model.pkl')
    rag_payload = joblib.load('lightweight_rag_components.pkl')
    return ml_payload, rag_payload

try:
    ml_payload, rag_payload = load_pickle_artifacts()
    
    # Resilient ML Payload Unpacking
    if isinstance(ml_payload, dict):
        ml_model = ml_payload.get('model')
        scaler = ml_payload.get('scaler', None)
    elif isinstance(ml_payload, (list, tuple)):
        ml_model = ml_payload[0]
        scaler = ml_payload[1] if len(ml_payload) > 1 else None
    else:
        ml_model = ml_payload
        scaler = None
        
    # Resilient RAG Payload Unpacking
    if isinstance(rag_payload, dict):
        rag_chunks = rag_payload.get('chunks', rag_payload.get('documents', []))
    else:
        rag_chunks = rag_payload

    st.sidebar.success("✅ Models & Text Chunks Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading .pkl files: {e}")
    st.error("Please ensure `sleep_model.pkl` and `lightweight_rag_components.pkl` are in your GitHub root folder.")
    st.stop()

# ==============================================================================
# LIGHTWEIGHT KEYWORD MATCHING ENGINE
# ==============================================================================
def search_raw_text_chunks(query, chunks, top_k=3):
    stopwords = {"i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it", 
                 "what", "which", "who", "whom", "this", "that", "am", "is", "are", "was", 
                 "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", 
                 "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                 "while", "of", "at", "by", "for", "with", "about", "against", "to", "then"}
    
    query_tokens = [
        word for word in re.findall(r'\b\w+\b', query.lower()) 
        if word not in stopwords and len(word) > 2
    ]
    
    if not query_tokens:
        query_tokens = [w for w in query.lower().split() if len(w) > 2]

    scored_chunks = []
    
    for item in chunks:
        text_content = item['text'] if isinstance(item, dict) else str(item)
        source_doc = item.get('source', 'Sleep Guideline') if isinstance(item, dict) else 'Knowledge Base'
        
        chunk_tokens = re.findall(r'\b\w+\b', text_content.lower())
        chunk_token_counts = Counter(chunk_tokens)
        
        overlap_score = sum(chunk_token_counts[token] for token in query_tokens if token in chunk_token_counts)
        scored_chunks.append((overlap_score, text_content, source_doc))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

# ==============================================================================
# INTERFACE TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["📊 ML Sleepiness Predictor", "💬 RAG AI Sleep Coach", "📈 System Architecture & Evaluation"])

# ------------------------------------------------------------------------------
# TAB 1: ML Model Interface
# ------------------------------------------------------------------------------
with tab1:
    st.header("ML Next-Day Sleepiness Score Predictor (KSS)")
    st.write("This module estimates your Karolinska Sleepiness Scale (KSS) score (1 = Extremely Alert, 9 = Extremely Sleepy).")
    
    col1, col2 = st.columns(2)
    with col1:
        sleep_dur = st.slider("Sleep Duration (Hours)", 3.0, 12.0, 7.0, 0.5)
        bedtime = st.slider("Bedtime Hour (24h format)", 20.0, 28.0, 23.0, 0.5, help="23 = 11 PM, 24 = Midnight, 26 = 2 AM")
        caffeine = st.selectbox("Caffeine Cups (After 2 PM)", [0, 1, 2, 3, 4, 5])
        
        if st.button("Run ML Prediction"):
            input_df = pd.DataFrame([[sleep_dur, bedtime, caffeine]], 
                                    columns=['sleep_duration', 'bedtime_hour', 'caffeine_intake'])
            
            if scaler is not None:
                inputs_scaled = scaler.transform(input_df)
                raw_pred = ml_model.predict(inputs_scaled)[0]
            else:
                raw_pred = ml_model.predict(input_df)[0]
            
            # Bound prediction score between 1.0 and 9.0 (Standard KSS Scale)
            predicted_score = round(float(np.clip(raw_pred, 1.0, 9.0)), 1)
            
            st.session_state['predicted_score'] = predicted_score
            st.session_state['user_dur'] = sleep_dur
            st.session_state['user_bed'] = bedtime
            
            st.metric(label="Predicted KSS Alertness-Sleepiness Score", value=f"{predicted_score} / 9")
            st.warning("⚠️ **Model Performance Note:** $R^2 \\approx 0.13$. High variance expected due to limited linear predictive power in the underlying dataset.")

# ------------------------------------------------------------------------------
# TAB 2: RAG AI Coach Interface
# ------------------------------------------------------------------------------
with tab2:
    st.header("RAG-Grounded AI Sleep Coach")
    st.write("Grounded in medical knowledge extracted from CDC, NSF, Harvard, and NIH guidelines.")
    
    mode = st.radio(
        "Select Coaching Strategy Mode:", 
        ["Mode 1: Morning Check-in & Habit Reflection", 
         "Mode 2: Bedtime Procrastination & Negotiation Coach"]
    )
    
    # Retrieve current KSS Score or assign default fallback
    current_kss = st.session_state.get('predicted_score', 5.0)
    
    st.info(f"📊 **Current Model Prediction:** User KSS Score is **{current_kss} / 9** (1 = Alert, 9 = Extremely Sleepy)")

    default_text = f"My predicted KSS sleepiness score is {current_kss}/9 based on {st.session_state.get('user_dur', 7.0)} hours of sleep. What recommendations do you have for me?"
    user_query = st.text_area("Your Sleep Question or Check-in:", value=default_text)

    if st.button("Generate RAG Response"):
        if not openrouter_api_key:
            st.error("API Key not found. Please set `OPENROUTER_API_KEY` in Streamlit secrets.")
        else:
            with st.spinner("Executing keyword retrieval & querying OpenRouter..."):
                top_matches = search_raw_text_chunks(user_query, rag_chunks, top_k=3)
                context_str = "\n\n".join([f"Source ({m[2]}): {m[1]}" for m in top_matches])
                
                # Dynamic System Prompts Injecting predicted KSS Score
                if "Mode 1" in mode:
                    system_prompt = f"""You are a helpful sleep coach assistant.
The user's predicted Karolinska Sleepiness Scale (KSS) score is {current_kss}/9 (where 1=Extremely Alert and 9=Extremely Sleepy).
Acknowledge their predicted KSS score directly in your advice.
Using the scientific context below, write a concise answer (2-3 sentences max).
Do NOT provide medical advice or diagnose conditions. Keep your response supportive and non-medical.

CONTEXT:
{context_str}

USER CHECK-IN:
{user_query}"""
                else:
                    system_prompt = f"""You are an accountability Sleep Coach dealing with bedtime procrastination.
The user's predicted Karolinska Sleepiness Scale (KSS) score is {current_kss}/9 (where 1=Extremely Alert and 9=Extremely Sleepy).
Explicitly reference their predicted KSS score to highlight the trade-off between immediate activity gain vs. sacrificed cognitive alertness tomorrow.
Do NOT provide medical advice. Keep your response supportive and non-medical (2-3 sentences max).

CONTEXT:
{context_str}

USER NEGOTIATION:
{user_query}"""

                try:
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {openrouter_api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "nvidia/nemotron-3.5-lightning:free",
                        "messages": [{"role": "user", "content": system_prompt}],
                        "max_tokens": 300,
                        "reasoning": {"enabled": False}
                    }
                    
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    res_json = response.json()
                    
                    ai_response = res_json['choices'][0]['message']['content']
                    
                    st.success("### AI Coach Guidance")
                    st.write(ai_response)
                    
                    with st.expander("🔍 View Retrieved Knowledge Context"):
                        for match in top_matches:
                            st.caption(f"**From {match[2]} (Match Score: {match[0]}):** {match[1]}")
                            
                except Exception as e:
                    st.error(f"OpenRouter API Error: {e}")

# ------------------------------------------------------------------------------
# TAB 3: System Architecture & Evaluation
# ------------------------------------------------------------------------------
with tab3:
    st.header("Module Integration & Evaluation Analysis")
    
    st.subheader("1. System Architecture")
    st.markdown("""
    * **Predictive ML Module:** Linear Regression ($R^2 \\approx 0.13$) estimating KSS Score (`sleep_model.pkl`).
    * **Lightweight RAG Engine:** Fast keyword overlap algorithm operating on raw text chunks (`lightweight_rag_components.pkl`).
    * **Inference Engine:** OpenRouter Free API running `nvidia/nemotron-3.5-lightning:free`.
    """)
    
    st.subheader("2. Module Evaluation & Findings")
    st.info("""
    **Core Question:** *What additional value does the ML module provide given its limited predictive performance?*
    
    **Analysis:** 
    The ML model predicts the user's Karolinska Sleepiness Scale (KSS) score. Passing this KSS metric directly into both coaching modes allows the agent to calibrate its urgency and tone based on predicted fatigue.
    """)
