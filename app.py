import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import re
from collections import Counter
from datetime import datetime, timedelta

# ==============================================================================
# HELPER FUNCTIONS FOR 12-HOUR TIME SELECTION & CONVERSION
# ==============================================================================
def render_time_picker(label_prefix, default_hour=10, default_minute=0, default_period="PM"):
    """
    Renders 3 side-by-side dropdowns for Period (AM/PM), Hour (1-12), and Minute (00-59).
    Returns a datetime.time object representing the 24-hour equivalent.
    """
    st.markdown(f"**{label_prefix}**")
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
            list(range(1, 13)), 
            index=default_hour - 1, 
            key=f"{label_prefix}_hour"
        )
    with col_min:
        minute = st.selectbox(
            "Minute", 
            [f"{m:02d}" for m in range(60)], 
            index=default_minute, 
            key=f"{label_prefix}_minute"
        )
        
    # Convert 12-hour format to 24-hour format
    hr_24 = hour_12 % 12
    if period == "PM":
        hr_24 += 12
        
    return hr_24, int(minute), f"{hour_12:02d}:{minute} {period}"

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
        # MODE 1: 3 Side-by-Side Dropdowns for Bedtime & Wake Time
        col1, col2 = st.columns(2)
        
        with col1:
            bed_hr, bed_min, bedtime_display = render_time_picker(
                "Previous Night Bedtime", default_hour=10, default_minute=0, default_period="PM"
            )
            
        with col2:
            wake_hr, wake_min, wake_display = render_time_picker(
                "Morning Wake Up Time", default_hour=7, default_minute=0, default_period="AM"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        user_self_kss = st.selectbox(
            "Rate your current alertness-sleepiness levels (1 = extremely alert; 10 = extremely sleepy)",
            options=list(range(1, 11)),
            index=4
        )
            
        user_query = st.text_area("Type in your Sleep Question or Check-in Reflection", height=120)
        
        if st.button("Generate Personalized Feedback"):
            # Calculate total sleep duration
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
            
            # KSS Score Display Box
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
The user's predicted Karolinska Sleepiness Scale (KSS) score is {predicted_kss}/9 (1=Extremely Alert, 9=Extremely Sleepy), based on {sleep_duration:.1f} hours of sleep (Bedtime: {bedtime_display}, Wake time: {wake_display}).
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
                            "max_tokens": 300
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=10)
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
            now_hr, now_min, now_display = render_time_picker(
                "What time is it now?", default_hour=11, default_minute=0, default_period="PM"
            )
            
        with col2:
            target_hr, target_min, target_display = render_time_picker(
                "What time are you aiming to get up tomorrow?", default_hour=7, default_minute=0, default_period="AM"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Fixed typo: changed "mean" to "means"
        aim_sleep = st.slider(
            "How much sleep are you aiming for? (7-9 hours of sleep is recommended; below 7 hours means sleep deprivation)",
            min_value=0.0, max_value=12.0, value=8.0, step=0.5
        )

        user_query = st.text_area("Type in your rationale to delay sleep tonight (i.e. Why are you putting off sleep?)", height=120)

        if st.button("Generate Personalized Feedback"):
            # Calculate potential available sleep
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
The current time is {now_display}, and the user aims to wake up at {target_display} (available sleep: {available_sleep:.1f} hrs vs target sleep: {aim_sleep} hrs).
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
                            "max_tokens": 300
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=10)
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
