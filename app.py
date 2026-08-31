import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import re
from collections import Counter
from datetime import datetime, timedelta

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

# Dynamic KSS Predictor Utility
def predict_kss(sleep_dur, bedtime_hour, caffeine_intake=0):
    input_features = np.array([[sleep_dur, bedtime_hour, caffeine_intake]])
    try:
        if scaler is not None:
            if hasattr(scaler, "feature_names_in_"):
                input_df = pd.DataFrame(input_features, columns=scaler.feature_names_in_)
                inputs_scaled = scaler.transform(input_df)
            else:
                inputs_scaled = scaler.transform(input_features)
            raw_pred = ml_model.predict(inputs_scaled)[0]
        else:
            if hasattr(ml_model, "feature_names_in_"):
                input_df = pd.DataFrame(input_features, columns=ml_model.feature_names_in_)
                raw_pred = ml_model.predict(input_df)[0]
            else:
                raw_pred = ml_model.predict(input_features)[0]
        return round(float(np.clip(raw_pred, 1.0, 9.0)), 1)
    except Exception:
        return 5.0

# Generate 24h Time Options for Dropdowns
time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

# ==============================================================================
# INTERFACE TABS
# ==============================================================================
tab1, tab2 = st.tabs(["💬 RAG AI Sleep Coach", "📈 System Architecture & Evaluation"])

# ------------------------------------------------------------------------------
# TAB 1: RAG AI Sleep Coach Interface
# ------------------------------------------------------------------------------
with tab1:
    st.header("AI Sleep Coach")
    st.write("Grounded in medical knowledge extracted from CDC, NSF, Harvard, and NIH guidelines.")
    
    mode = st.radio(
        "Select Coaching Strategy Mode:", 
        ["Mode 1: Morning Check-in & Habit Reflection", 
         "Mode 2: Bedtime Procrastination & Negotiation Coach"]
    )
    
    st.markdown("---")
    
    if "Mode 1" in mode:
        col1, col2, col3 = st.columns(3)
        with col1:
            bedtime_str = st.selectbox("Previous Night Bedtime", time_options, index=44)  # 22:00
        with col2:
            wake_str = st.selectbox("Morning Wake Up Time", time_options, index=14)       # 07:00
        with col3:
            user_self_kss = st.selectbox(
                "Rate your current alertness-sleepiness levels (1 = extremely alert; 10 = extremely sleepy)",
                options=list(range(1, 11)),
                index=4
            )
            
        user_query = st.text_area("Type in your Sleep Question or Check-in Reflection", height=120)
        
        if st.button("Generate Personalized Feedback"):
            # Calculate sleep duration automatically
            t_bed = datetime.strptime(bedtime_str, "%H:%M")
            t_wake = datetime.strptime(wake_str, "%H:%M")
            if t_wake <= t_bed:
                t_wake += timedelta(days=1)
            sleep_duration = (t_wake - t_bed).total_seconds() / 3600.0
            
            bedtime_hour = t_bed.hour + (t_bed.minute / 60.0)
            if bedtime_hour < 12:
                bedtime_hour += 24  # Normalize late hours
                
            predicted_kss = predict_kss(sleep_duration, bedtime_hour)
            st.session_state['latest_kss'] = predicted_kss
            
            # KSS Score Display Box
            st.info(
                f"📊 **ML Next-Day Sleepiness Score Predictor (KSS):** **{predicted_kss} / 9**\n\n"
                f"*Note: Karolinska Sleepiness Scale (KSS) score (1 = Extremely Alert, 9 = Extremely Sleepy).* | "
                f"Calculated Sleep Duration: **{sleep_duration:.1f} hrs**"
            )
            
            if not openrouter_api_key:
                st.error("API Key not found. Please set `OPENROUTER_API_KEY` in Streamlit secrets.")
            else:
                with st.spinner("Executing keyword retrieval & querying OpenRouter..."):
                    top_matches = search_raw_text_chunks(user_query, rag_chunks, top_k=3)
                    context_str = "\n\n".join([f"Source ({m[2]}): {m[1]}" for m in top_matches])
                    
                    system_prompt = f"""You are a helpful sleep coach assistant.
The user's predicted Karolinska Sleepiness Scale (KSS) score is {predicted_kss}/9 (1=Extremely Alert, 9=Extremely Sleepy), based on {sleep_duration:.1f} hours of sleep (Bedtime: {bedtime_str}, Wake time: {wake_str}).
The user self-reported their current alertness-sleepiness as {user_self_kss}/10.
Acknowledge their predicted KSS score and sleep stats directly in your advice.
Using the scientific context below, write a concise answer (2-3 sentences max).
Do NOT provide medical advice or diagnose conditions. Keep your response supportive and non-medical.

CONTEXT:
{context_str}

USER REFLECTION:
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

    else:
        # MODE 2: Bedtime Procrastination & Negotiation Coach
        col1, col2 = st.columns(2)
        with col1:
            current_time_str = st.selectbox("What time is it now?", time_options, index=46) # 23:00
            target_wake_str = st.selectbox("What time are you aiming to get up tomorrow?", time_options, index=14) # 07:00
        with col2:
            aim_sleep = st.slider(
                "How much sleep are you aiming for? (7-9 hours of sleep is recommended; below 7 hours mean sleep deprivation)",
                min_value=0.0, max_value=12.0, value=8.0, step=0.5
            )

        user_query = st.text_area("Type in your rationale to delay sleep tonight (i.e. Why are you putting off sleep?)", height=120)

        if st.button("Generate Personalized Feedback"):
            # Calculate potential available sleep if going to bed right now
            t_now = datetime.strptime(current_time_str, "%H:%M")
            t_wake = datetime.strptime(target_wake_str, "%H:%M")
            if t_wake <= t_now:
                t_wake += timedelta(days=1)
            available_sleep = (t_wake - t_now).total_seconds() / 3600.0
            
            bedtime_hour = t_now.hour + (t_now.minute / 60.0)
            if bedtime_hour < 12:
                bedtime_hour += 24
                
            predicted_kss = predict_kss(available_sleep, bedtime_hour)
            st.session_state['latest_kss'] = predicted_kss
            
            # KSS Score Display Box
            st.info(
                f"📊 **ML Next-Day Sleepiness Score Predictor (KSS):** **{predicted_kss} / 9**\n\n"
                f"*Note: Karolinska Sleepiness Scale (KSS) score (1 = Extremely Alert, 9 = Extremely Sleepy).* | "
                f"Max Available Sleep: **{available_sleep:.1f} hrs** (Target: **{aim_sleep} hrs**)"
            )

            if not openrouter_api_key:
                st.error("API Key not found. Please set `OPENROUTER_API_KEY` in Streamlit secrets.")
            else:
                with st.spinner("Executing keyword retrieval & querying OpenRouter..."):
                    top_matches = search_raw_text_chunks(user_query, rag_chunks, top_k=3)
                    context_str = "\n\n".join([f"Source ({m[2]}): {m[1]}" for m in top_matches])
                    
                    system_prompt = f"""You are an accountability Sleep Coach dealing with bedtime procrastination.
The current time is {current_time_str}, and the user aims to wake up at {target_wake_str} (available sleep: {available_sleep:.1f} hrs vs target sleep: {aim_sleep} hrs).
Their predicted Karolinska Sleepiness Scale (KSS) score tomorrow will be {predicted_kss}/9 (where 1=Extremely Alert and 9=Extremely Sleepy).
Explicitly reference their predicted KSS score to highlight the trade-off between immediate activity gain vs. sacrificed cognitive alertness tomorrow.
Do NOT provide medical advice. Keep your response supportive and non-medical (2-3 sentences max).

CONTEXT:
{context_str}

USER NEGOTIATION RATIONALE:
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
# TAB 2: System Architecture & Evaluation
# ------------------------------------------------------------------------------
with tab2:
    st.header("Module Integration & Evaluation Analysis")
    
    st.subheader("1. System Architecture")
    st.markdown("""
    * **Predictive ML Module:** Linear Regression ($R^2 \\approx 0.13$) estimating KSS Score (`sleep_model.pkl`) calculated from user schedule inputs.
    * **Lightweight RAG Engine:** Fast keyword overlap algorithm operating on raw text chunks (`lightweight_rag_components.pkl`).
    * **Inference Engine:** OpenRouter Free API running `nvidia/nemotron-3.5-lightning:free`.
    """)
    
    st.subheader("2. Module Evaluation & Findings")
    st.info("""
    **Core Question:** *What additional value does the ML module provide given its limited predictive performance?*
    
    **Analysis:** 
    The ML model predicts the user's Karolinska Sleepiness Scale (KSS) score based on calculated sleep durations and bedtime timing. Passing this KSS metric directly into both coaching modes allows the agent to calibrate its urgency and tone based on predicted fatigue.
    """)
