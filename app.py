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

# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key", 
    type="password", 
    help="Enter your key from openrouter.ai (using nvidia/nemotron-3.5-lightning:free)"
)

# ==============================================================================
# LOAD ARTIFACTS (.pkl files)
# ==============================================================================
@st.cache_resource
def load_pickle_artifacts():
    # Load ML Model Components
    ml_payload = joblib.load('sleep_model.pkl')
    
    # Load Lightweight RAG Components
    rag_payload = joblib.load('lightweight_rag_components.pkl')
    
    return ml_payload, rag_payload

try:
    ml_payload, rag_payload = load_pickle_artifacts()
    
    # --- RESILIENT ML PAYLOAD UNPACKING ---
    if isinstance(ml_payload, dict):
        ml_model = ml_payload.get('model')
        scaler = ml_payload.get('scaler', None)
    elif isinstance(ml_payload, (list, tuple)):
        ml_model = ml_payload[0]
        scaler = ml_payload[1] if len(ml_payload) > 1 else None
    else:
        # ml_payload is the raw LinearRegression object directly
        ml_model = ml_payload
        scaler = None
        
    # --- RESILIENT RAG PAYLOAD UNPACKING ---
    if isinstance(rag_payload, dict):
        rag_chunks = rag_payload.get('chunks', rag_payload.get('documents', []))
    else:
        rag_chunks = rag_payload

    st.sidebar.success("✅ Models & Lightweight Text Chunks Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading .pkl files: {e}")
    st.error("Please ensure `sleep_model.pkl` and `lightweight_rag_components.pkl` are present in your GitHub repository root folder.")
    st.stop()

    st.sidebar.success("✅ Models & Lightweight Text Chunks Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading .pkl files: {e}")
    st.error("Please ensure `sleep_model.pkl` and `lightweight_rag_components.pkl` are present in your GitHub repository root folder.")
    st.stop()

# ==============================================================================
# LIGHTWEIGHT KEYWORD MATCHING ENGINE
# ==============================================================================
def search_raw_text_chunks(query, chunks, top_k=3):
    """
    Performs instant keyword overlap search over raw text chunks 
    using normalized token matching and frequency scoring.
    """
    # Simple stopword list to ignore common English non-informative words
    stopwords = {"i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it", 
                 "what", "which", "who", "whom", "this", "that", "am", "is", "are", "was", 
                 "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", 
                 "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                 "while", "of", "at", "by", "for", "with", "about", "against", "to", "then"}
    
    # Tokenize user query: lowercase and strip non-alphanumeric chars
    query_tokens = [
        word for word in re.findall(r'\b\w+\b', query.lower()) 
        if word not in stopwords and len(word) > 2
    ]
    
    if not query_tokens:
        # Fallback to simple split if all words were stripped
        query_tokens = [w for w in query.lower().split() if len(w) > 2]

    scored_chunks = []
    
    for item in chunks:
        # Extract text content and source title
        text_content = item['text'] if isinstance(item, dict) else str(item)
        source_doc = item.get('source', 'Sleep Guideline') if isinstance(item, dict) else 'Knowledge Base'
        
        # Tokenize target text chunk
        chunk_tokens = re.findall(r'\b\w+\b', text_content.lower())
        chunk_token_counts = Counter(chunk_tokens)
        
        # Calculate Term Overlap Score
        overlap_score = sum(chunk_token_counts[token] for token in query_tokens if token in chunk_token_counts)
        
        scored_chunks.append((overlap_score, text_content, source_doc))
    
    # Sort descending by relevance score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Return top_k matching chunks
    return scored_chunks[:top_k]

# ==============================================================================
# INTERFACE TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["📊 ML Sleepiness Predictor", "💬 RAG AI Sleep Coach", "📈 System Architecture & Evaluation"])

# ------------------------------------------------------------------------------
# TAB 1: ML Model Interface
# ------------------------------------------------------------------------------
with tab1:
    st.header("ML Next-Day Sleepiness Score Predictor")
    st.write("This module utilizes a linear regression model trained on UK sleep data to estimate daytime sleepiness.")
    
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
            
            predicted_score = round(float(np.clip(raw_pred, 1.0, 10.0)), 1)
            
            st.session_state['predicted_score'] = predicted_score
            st.session_state['user_dur'] = sleep_dur
            st.session_state['user_bed'] = bedtime
            
            st.metric(label="Predicted Daytime Sleepiness Score", value=f"{predicted_score} / 10")
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
    
    default_text = ""
    if 'predicted_score' in st.session_state:
        default_text = f"I slept {st.session_state['user_dur']} hours last night and my predicted sleepiness score is {st.session_state['predicted_score']}/10. What should I do tonight?"
    
    user_query = st.text_area("Your Sleep Question or Check-in:", value=default_text)

    if st.button("Generate RAG Response"):
        if not openrouter_api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        else:
            with st.spinner("Executing instant keyword retrieval over text chunks & querying OpenRouter..."):
                
                # 1. Instant Keyword Search over Lightweight .pkl Chunks
                top_matches = search_raw_text_chunks(user_query, rag_chunks, top_k=3)
                
                # Format retrieved chunks for Context Injection
                context_str = "\n\n".join([f"Source ({m[2]}): {m[1]}" for m in top_matches])
                
                # 2. Construct Prompt based on Mode
                if "Mode 1" in mode:
                    system_prompt = f"""You are a helpful sleep coach assistant.
Using the scientific context below, write a concise answer (2-3 sentences max).
Do NOT provide medical advice or diagnose conditions. Keep your response supportive and non-medical.

CONTEXT:
{context_str}

USER CHECK-IN:
{user_query}"""
                else:
                    system_prompt = f"""You are an accountability Sleep Coach dealing with bedtime procrastination.
Highlight the explicit trade-off between immediate activity gain vs. sacrificed cognitive alertness tomorrow.
Do NOT provide medical advice. Keep your response supportive and non-medical (2-3 sentences max).

CONTEXT:
{context_str}

USER NEGOTIATION:
{user_query}"""

                # 3. OpenRouter API Request
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
                    
                    # Expandable Context Viewer
                    with st.expander("🔍 View Retrieved Knowledge Context (Keyword Matched)"):
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
    * **Predictive ML Module:** Linear Regression ($R^2 \\approx 0.13$) trained on UK Sleep Dataset metrics (`sleep_model.pkl`).
    * **Lightweight RAG Engine:** Fast keyword overlap algorithm operating on raw text chunks extracted from 5 medical guidelines (`lightweight_rag_components.pkl`).
    * **Inference Engine:** OpenRouter Free API running `nvidia/nemotron-3.5-lightning:free`.
    """)
    
    st.subheader("2. Module Evaluation & Findings")
    st.info("""
    **Core Question:** *What additional value does the ML module provide given its limited predictive performance?*
    
    **Analysis:** 
    The ML model serves as an experimental personalization component. While its statistical predictive power is low ($R^2 \\approx 0.13$), passing its predicted sleepiness score directly into the RAG module context enables the conversational agent to tailor its tone and urgency. The RAG architecture serves as the primary, factual foundation of the MVP.
    """)
