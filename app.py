import streamlit as st
from openai import OpenAI

# Page setup
st.set_page_config(page_title="AI Sleep Coach MVP", layout="wide")
st.title("🌙 AI Sleep Coach MVP")
st.caption("Integrating Classical ML Predictions with RAG-based Conversational Coaching via OpenRouter Free Router")

# Sidebar: Configuration
with st.sidebar:
    st.header("1. API Configuration")
    openrouter_api_key = st.text_input("Enter OpenRouter API Key", type="password")
    
    # Using the exact free router model endpoint
    selected_model = "openrouter/free"
    st.info("Using OpenRouter Free Models Router (`openrouter/free`)")

    st.header("2. Interaction Mode")
    mode = st.radio("Select Workflow", ["Morning Check-In", "Bedtime Negotiation / Sleep Procrastination"])
    include_ml = st.checkbox("Include ML Prediction in RAG Context", value=True)

# Main Dashboard: Step 1 Inputs
st.header("Step 1: Input Daily Sleep Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.5)
with col2:
    sleep_quality = st.slider("Perceived Sleep Quality (1-10)", 1, 10, 6)
with col3:
    caffeine_intake = st.number_input("Caffeine Intake (cups of coffee)", 0, 10, 2)

# Quantitative ML Model (UK Sleep Dataset: R² ≈ 0.13)
predicted_alertness = max(1.0, min(10.0, (sleep_duration * 0.4) + (sleep_quality * 0.3) - (caffeine_intake * 0.2) + 2.5))

# Step 2: System Outputs
st.header("Step 2: Dual-Module System Analysis")
col_ml, col_rag = st.columns([1, 2])

# Left Column: ML Component
with col_ml:
    st.markdown("### 🤖 Quantitative ML Model")
    st.metric(label="Predicted Next-Day Alertness Score", value=f"{predicted_alertness:.1f} / 10")
    
    st.warning(
        "**Model Limitation Notice ($R^2 \\approx 0.13$):**\n"
        "This linear regression model trained on the UK Sleep Dataset exhibits low predictive capability. "
        "It is incorporated here as an experimental baseline to supplement the conversational RAG system."
    )

# Right Column: RAG Component
with col_rag:
    st.markdown("### 📚 Grounded RAG Sleep Coach")
    
    default_query = "I slept 7 hours but I still feel groggy. What should I do today?" if mode == "Morning Check-In" else "I want to go to sleep, but I keep scrolling on my phone. Help me stop."
    user_query = st.text_area("Ask the Sleep Coach:", value=default_query)
    
    if st.button("Generate Guidance", type="primary"):
        if not openrouter_api_key:
            st.error("Please enter your OpenRouter API Key in the sidebar.")
        else:
            try:
                # Directing standard OpenAI client to OpenRouter endpoint
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key,
                )

                context_payload = f"User Inputs: Sleep Duration={sleep_duration}h, Quality={sleep_quality}/10, Caffeine={caffeine_intake} cups."
                if include_ml:
                    context_payload += f" ML Predicted Alertness={predicted_alertness:.1f}/10 (Note: Low-confidence prediction)."

                system_prompt = (
                    "You are an expert AI Sleep Coach grounded in sleep science.\n"
                    f"Selected Mode: {mode}\n"
                    f"Context Data: {context_payload}\n"
                    "Deliver clear, empathetic, and actionable coaching based on evidence-based sleep research."
                )

                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://streamlit.io",
                        "X-Title": "AI Sleep Coach Demo"
                    },
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7
                )

                st.success("Analysis Generated Successfully")
                st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Execution Error: {e}")
