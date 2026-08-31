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

# Custom CSS targeting Streamlit's border containers reliably across all UI re-renders
st.markdown("""
<style>
    /* Mode 1 - Card 1: Sky Blue */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#card-bedtime) {
        background-color: #e0f2fe !important;
        border: 2px solid #38bdf8 !important;
        border-radius: 20px !important;
        padding: 10px !important;
    }
    /* Mode 1 - Card 2 / Mode 2 - Card 1: Light Green */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#card-green) {
        background-color: #dcfce7 !important;
        border: 2px solid #4ade80 !important;
        border-radius: 20px !important;
        padding: 10px !important;
    }
    /* Mode 2 - Card 2: Light Yellow */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#card-yellow) {
        background-color: #fef9c3 !important;
        border: 2px solid #facc15 !important;
        border-radius: 20px !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

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
    
    if isinstance(ml_payload, dict):
        ml_model = ml_payload.get('model')
        scaler = ml_payload.get('scaler', None)
    elif isinstance(ml_payload, (list, tuple)):
        ml_model = ml_payload[0]
        scaler = ml_payload[1] if len(ml_payload) > 1 else None
    else:
        ml_model = ml_payload
        scaler = None
        
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
# LIGHTWEIGHT KEYWORD MATCHING ENGINE & KSS PREDICTOR
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

# ==============================================================================
# HELPER FUNCTIONS FOR TIME SELECTION (Hour Range: 0 to 12)
# ==============================================================================
def render_time_picker(label_prefix, default_hour=10, default_minute=0, default_period="PM"):
    st.markdown(f"#### {label_prefix}")
    col_period, col_hr, col_min = st.columns(3)
    
    with col_period:
        period = st.selectbox(
            "Period", 
            ["AM", "PM"], 
            index=0 if default_period == "AM" else 1, 
            key=f"{label_prefix}_period"
        )
    with col_hr:
        hour_12 = st.selectbox(
            "Hour", 
            list(range(0, 13)), 
            index=default_hour, 
            key=f"{label_prefix}_hour"
        )
    with col_min:
        minute = st.selectbox(
            "Minute", 
            [f"{m:02d}" for m in range(60)], 
            index=default_minute, 
            key=f"{label_prefix}_minute"
        )
        
    hr_24 = hour_12 % 12
    if period == "PM":
        hr_24 += 12
        
    return hr_24, int(minute), f"{hour_12:02d}:{minute} {period}"

# Cleans out <think>...</think> tags and caps output to <= 3 sentences
def clean_and_trim_response(text):
    # Remove hidden reasoning/thinking traces
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Enforce max 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    if len(sentences) > 3:
        return " ".join(sentences[:3])
    return cleaned

# ==============================================================================
# INTERFACE TABS DEFINITION
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
        # Sky Blue Card Container
        with st.container(border=True):
            st.markdown('<span id="card-bedtime"></span>', unsafe_allow_html=True)
            bed_hr, bed_min, bedtime_display = render_time_picker(
                "Previous Night Bedtime", default_hour=10, default_minute=0, default_period="PM"
            )
        
        st.write("")
        
        # Light Green Card Container
        with st.container(border=True):
            st.markdown('<span id="card-green"></span>', unsafe_allow_html=True)
            wake_hr, wake_min, wake_display = render_time_picker(
                "Morning Wake Up Time", default_hour=7, default_minute=0, default_period="AM"
            )
        
        st.write("")
        
        user_self_kss = st.slider(
            "Rate your current alertness-sleepiness levels (1 = extremely alert; 10 = extremely sleepy)",
            min_value=0, max_value=12, value=5, step=1
        )
            
        user_query = st.text_area("Type in your Sleep Question or Check-in Reflection", height=120)
        
        if st.button("Generate Personalized Feedback"):
            t_bed = datetime(2026, 1, 1, bed_hr, bed_min)
            t_wake = datetime(2026, 1, 1, wake_hr, wake_min)
            if t_wake <= t_bed:
                t_wake += timedelta(days=1)
                
            sleep_duration = (t_wake - t_bed).total_seconds() / 3600.0
            bedtime_hour_decimal = bed_hr + (bed_min / 60.0)
            if bedtime_hour_decimal < 12:
                bedtime_hour_decimal += 24
                
            predicted_kss = predict_kss(sleep_duration, bedtime_hour_decimal)
            st.session_state['latest_kss'] = predicted_kss
            
            st.info(
                f"📊 **ML Next-Day Sleepiness Score Predictor (KSS):** **{predicted_kss} / 9**\n\n"
                f"*Note: Karolinska Sleepiness Scale (KSS) score (1 = Extremely Alert, 9 = Extremely Sleepy).* | "
                f"Calculated Sleep Duration: **{sleep_duration:.1f} hrs** ({bedtime_display} to {wake_display})"
            )
            
            if not openrouter_api_key:
                st.error("API Key not found. Please set `OPENROUTER_API_KEY` in Streamlit secrets.")
            else:
                with st.spinner("Executing keyword retrieval & querying OpenRouter..."):
                    top_matches = search_raw_text_chunks(user_query, rag_chunks, top_k=3)
                    context_str = "\n\n".join([f"Source ({m[2]}): {m[1]}" for m in top_matches])
                    
                    system_prompt = f"""You are a helpful sleep coach assistant.
STRICT RULE: Do NOT output any internal thinking or reasoning tags. Provide your final response directly in 1 to 3 sentences maximum.

The user's predicted Karolinska Sleepiness Scale (KSS) score is {predicted_kss}/9 (1=Extremely Alert, 9=Extremely Sleepy), based on {sleep_duration:.1f} hours of sleep (Bedtime: {bedtime_display}, Wake time: {wake_display}).
The user self-reported their current alertness-sleepiness as {user_self_kss}/12.
Acknowledge their predicted KSS score and sleep stats directly in your advice.
Using the scientific context below, write a supportive response.

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
                            "max_tokens": 1000
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=12)
                        response.raise_for_status()
                        res_json = response.json()
                        raw_ai_response = res_json['choices'][0]['message']['content']
                        
                        # Strip <think> tags and trim to 3 sentences max
                        final_response = clean_and_trim_response(raw_ai_response)
                        
                        st.success("### AI Coach Guidance")
                        st.write(final_response)
                        
                        with st.expander("🔍 View Retrieved Knowledge Context"):
                            seen_sources = set()
                            for match in top_matches:
                                source_name = match[2]
                                if source_name not in seen_sources:
                                    st.markdown(f"• **{source_name}**")
                                    seen_sources.add(source_name)
                    except Exception as e:
                        st.error(f"OpenRouter API Error: {e}")

    else:
        # Mode 2 - Light Green Card Container
        with st.container(border=True):
            st.markdown('<span id="card-green"></span>', unsafe_allow_html=True)
            now_hr, now_min, now_display = render_time_picker(
                "What time is it now?", default_hour=11, default_minute=0, default_period="PM"
            )
        
        st.write("")
        
        # Mode 2 - Yellow Card Container
        with st.container(border=True):
            st.markdown('<span id="card-yellow"></span>', unsafe_allow_html=True)
            target_hr, target_min, target_display = render_time_picker(
                "What time are you aiming to get up tomorrow?", default_hour=7, default_minute=0, default_period="AM"
            )
        
        st.write("")
        
        aim_sleep = st.slider(
            "How much sleep are you aiming for? (7-9 hours of sleep is recommended; below 7 hours means sleep deprivation)",
            min_value=0.0, max_value=12.0, value=8.0, step=0.5
        )

        user_query = st.text_area("Type in your rationale to delay sleep tonight (i.e. Why are you putting off sleep?)", height=120)

        if st.button("Generate Personalized Feedback"):
            t_now = datetime(2026, 1, 1, now_hr, now_min)
            t_wake = datetime(2026, 1, 1, target_hr, target_min)
            if t_wake <= t_now:
                t_wake += timedelta(days=1)
                
            available_sleep = (t_wake - t_now).total_seconds() / 3600.0
            bedtime_hour_decimal = now_hr + (now_min / 60.0)
            if bedtime_hour_decimal < 12:
                bedtime_hour_decimal += 24
                
            predicted_kss = predict_kss(available_sleep, bedtime_hour_decimal)
            st.session_state['latest_kss'] = predicted_kss
            
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
STRICT RULE: Do NOT output any internal thinking or reasoning tags. Provide your final response directly in 1 to 3 sentences maximum.

The current time is {now_display}, and the user aims to wake up at {target_display} (available sleep: {available_sleep:.1f} hrs vs target sleep: {aim_sleep} hrs).
Their predicted Karolinska Sleepiness Scale (KSS) score tomorrow will be {predicted_kss}/9 (where 1=Extremely Alert and 9=Extremely Sleepy).
Explicitly reference their predicted KSS score to highlight the trade-off between immediate activity gain vs. sacrificed cognitive alertness tomorrow.

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
                            "max_tokens": 1000
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=12)
                        response.raise_for_status()
                        res_json = response.json()
                        raw_ai_response = res_json['choices'][0]['message']['content']
                        
                        # Strip <think> tags and trim to 3 sentences max
                        final_response = clean_and_trim_response(raw_ai_response)
                        
                        st.success("### AI Coach Guidance")
                        st.write(final_response)
                        
                        with st.expander("🔍 View Retrieved Knowledge Context"):
                            seen_sources = set()
                            for match in top_matches:
                                source_name = match[2]
                                if source_name not in seen_sources:
                                    st.markdown(f"• **{source_name}**")
                                    seen_sources.add(source_name)
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
    
    **Analysis:** The ML model predicts the user's Karolinska Sleepiness Scale (KSS) score based on calculated sleep durations and bedtime timing. Passing this KSS metric directly into both coaching modes allows the agent to calibrate its urgency and tone based on predicted fatigue.
    """)
