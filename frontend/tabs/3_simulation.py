import streamlit as st
from api_client import get_prediction, get_counterfactual

def render():
    st.header("Step 3: What-If Simulator")

    # 1. Safety Check
    if not st.session_state.get('trained'):
        st.warning("⚠️ Please train a model in Step 1 first.")
        return

    features = st.session_state.get('feature_names', [])
    
    # 2. Layout: Sidebar Controls vs Main Result
    st.subheader("🎛️ Adjust Features")
    
    # We try to grab values from Tab 2 if they exist
    last_inputs = st.session_state.get('last_input', {})

    current_sim_inputs = {}
    
    # Dynamic Sliders for numeric, Dropdowns for categorical
    # (Since we don't know exact types here easily without complex logic, 
    # we will use text/number inputs for robustness, or assume numeric for sliders)
    
    cols = st.columns(3)
    for i, col_name in enumerate(features):
        with cols[i % 3]:
            # Get default value from previous tab or set to 0
            default_val = last_inputs.get(col_name, 0)
            
            # Create a number input (Acting as a slider/adjuster)
            # We use text_input because it handles both int/float/string gracefully
            val = st.text_input(f"Adjust {col_name}", value=str(default_val), key=f"sim_{col_name}")
            
            # Save to our simulation dict
            current_sim_inputs[col_name] = val

    # Helper to format
    def get_clean_inputs():
        clean = {}
        for k, v in current_sim_inputs.items():
            try:
                if "." in v:
                    clean[k] = float(v)
                else:
                    clean[k] = int(v)
            except:
                clean[k] = v
        return clean

    st.markdown("---")
    
    # 3. Real-Time Simulation
    if st.button("🔄 Re-Simulate Outcome", type="primary"):
        clean_inputs = get_clean_inputs()
        
        with st.spinner("Simulating..."):
            res = get_prediction(clean_inputs)
        
        if "error" in res:
            st.error(f"Simulation Error: {res['error']}")
        else:
            # Display Big Metric
            c1, c2 = st.columns(2)
            if res.get('probability') is not None:
                # Class
                new_prob = res['probability'] * 100
                c1.metric("New Prediction", f"Class {res['prediction']}")
                c2.metric("New Risk Probability", f"{new_prob:.1f}%")
            else:
                # Reg
                c1.metric("New Predicted Value", f"{res['prediction']:.2f}")

    # 4. Strategy Generator (Counterfactuals)
    st.markdown("### 💡 AI Strategy Generator")
    st.info("Ask the AI: *'What is the minimum change needed to get a different outcome?'*")
    
# ... inside frontend/tabs/3_simulation.py

if st.button("✨ Generate Strategy"):
        clean_inputs = get_clean_inputs()
        
        with st.spinner("Finding the easiest path to a better outcome..."):
            cf_res = get_counterfactual(clean_inputs)
            
        if "error" in cf_res:
            st.error(f"Strategy Failed: {cf_res['error']}")
        elif "message" in cf_res:
            st.warning(cf_res['message'])
        else:
            # --- 🛠️ SAFETY FIX: UNWRAP NESTED RESPONSE ---
            # If the result is wrapped like {'0': {...data...}}, unwrap it.
            # We check if there is only 1 key and its value is a dictionary.
            if len(cf_res) == 1 and isinstance(list(cf_res.values())[0], dict):
                first_key = list(cf_res.keys())[0]
                cf_res = cf_res[first_key]
            # ---------------------------------------------

            st.success("Strategy Found! Change these features:")
            
            # Compare Old vs New
            diff_cols = st.columns(3) # Fixed to 3 columns for better layout
            
            changes_found = False
            col_counter = 0
            
            for feat, new_val in cf_res.items():
                old_val = clean_inputs.get(feat, "N/A")
                
                # Only show the specific features that CHANGED
                # We convert to string to ensure fair comparison (e.g. 1 vs 1.0)
                if str(old_val) != str(new_val):
                    changes_found = True
                    with diff_cols[col_counter % 3]:
                        st.metric(
                            label=f"Change {feat}", 
                            value=new_val, 
                            delta=f"From {old_val}"
                        )
                    col_counter += 1
            
            if not changes_found:
                st.info("The AI found that your current inputs already match the desired outcome! No changes needed.")