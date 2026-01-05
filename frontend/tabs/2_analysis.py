import streamlit as st
from api_client import get_prediction, get_explanation

def render():
    st.header("Step 2: AI Prediction & Insights")
    
    # 1. Safety Check
    if not st.session_state.get('trained'):
        st.warning("⚠️ Please train a model in Step 1 first.")
        return

    # 2. Get the "Winning" Features from Session State
    features = st.session_state.get('feature_names', [])
    target_name = st.session_state.get('target_name', 'Target')
    
    st.info(f"✨ Using the **{len(features)} optimized features** selected by the AI.")

    # 3. Dynamic Input Form
    inputs = {}
    
    # Use 3 columns for a cleaner layout
    cols = st.columns(3)
    for i, col_name in enumerate(features):
        with cols[i % 3]:
            # Create a text input for each feature
            # Default value is empty, user types in it
            val = st.text_input(f"{col_name}", key=f"in_{col_name}")
            inputs[col_name] = val

    # Helper function to format inputs safely
    def get_formatted_inputs():
        formatted = {}
        missing = []
        for k, v in inputs.items():
            if v == "":
                missing.append(k)
                continue
            try:
                # Try converting to float/int
                if "." in v:
                    formatted[k] = float(v)
                else:
                    formatted[k] = int(v)
            except:
                # Fallback to string (categorical)
                formatted[k] = v
        return formatted, missing

    st.markdown("---")

    # --- BUTTON 1: PREDICT (Fast) ---
    if st.button("🚀 Run Prediction", type="primary"):
        formatted_inputs, missing_fields = get_formatted_inputs()
        
        if missing_fields:
            st.error(f"⚠️ Please fill in all fields. Missing: {', '.join(missing_fields)}")
        else:
            with st.spinner("Calculating..."):
                pred_res = get_prediction(formatted_inputs)
            
            if "error" in pred_res:
                st.error(f"Prediction Failed: {pred_res['error']}")
            else:
                # Save to session state so we can use it for explanation later
                st.session_state['last_input'] = formatted_inputs
                st.session_state['last_result'] = pred_res
                st.session_state['explanation_result'] = None # Reset old explanation

    # --- DISPLAY RESULTS ---
    if st.session_state.get('last_result'):
        res = st.session_state['last_result']
        
        st.subheader("🎯 Prediction Result")
        
        # Create a nice metric card
        c1, c2 = st.columns(2)
        
        # Check if it's Classification (has probability) or Regression
        if res.get('probability') is not None:
            # Classification
            pred_class = res['prediction']
            prob = res['probability']
            
            c1.metric(f"Predicted {target_name}", f"Class {pred_class}")
            
            # Color code the probability (Risk Meter)
            if prob > 0.7:
                c2.error(f"Probability: {prob*100:.1f}% (High)")
            elif prob < 0.3:
                c2.success(f"Probability: {prob*100:.1f}% (Low)")
            else:
                c2.warning(f"Probability: {prob*100:.1f}% (Medium)")
        else:
            # Regression
            val = res['prediction']
            c1.metric(f"Predicted {target_name}", f"{val:.2f}")
            c2.metric("Confidence", "High (Regression)")

        # --- BUTTON 2: EXPLAIN (Slow/LLM) ---
        st.markdown("---")
        st.info("Want to understand WHY the AI made this decision?")
        
        if st.button("🧠 Generate AI Explanation"):
            current_inputs = st.session_state['last_input']
            current_pred = st.session_state['last_result']['prediction']
            current_prob = st.session_state['last_result'].get('probability') # Might be None
            
            with st.spinner("Analyzing SHAP values & generating narrative..."):
                explain_res = get_explanation(current_inputs, current_pred, current_prob)
                st.session_state['explanation_result'] = explain_res

    # --- DISPLAY EXPLANATION ---
    if st.session_state.get('explanation_result'):
        exp = st.session_state['explanation_result']
        
        st.subheader("📝 AI Narrative")
        st.success(exp.get('narrative', 'No narrative generated.'))
        
        st.subheader("📊 Key Drivers")
        st.caption("These features pushed the model towards this decision:")
        st.bar_chart(exp.get('shap_values', {}))