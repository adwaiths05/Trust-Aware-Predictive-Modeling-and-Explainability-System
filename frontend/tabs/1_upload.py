import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from api_client import upload_file, train_model

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

def render():
    st.header("Step 1: Intelligent Auto-ML Training")

    # --- 1. Data Selection ---
    data_source = st.radio("📂 Source:", ["Existing File", "Upload New"], horizontal=True)
    df = None
    
    if data_source == "Existing File":
        if config.DATA_DIR.exists():
            files = [f.name for f in config.DATA_DIR.glob("*.csv")]
            if files:
                selected_filename = st.selectbox("Select Dataset:", files)
                if selected_filename:
                    df = pd.read_csv(config.DATA_DIR / selected_filename)
                    st.session_state['filename'] = selected_filename
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if st.button("⬆️ Upload"):
                upload_file(uploaded, uploaded.name)
                st.session_state['filename'] = uploaded.name
                st.rerun()

    # --- 2. Configuration ---
    if df is not None:
        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            all_cols = df.columns.tolist()
            target_col = st.selectbox("🎯 Target Variable:", all_cols, index=len(all_cols)-1)
        with c2:
            st.write("⚙️ **Engine Mode**")
            auto_opt = st.checkbox("Auto-Select Best Algorithm", value=True)
            if auto_opt:
                if len(all_cols) <= 12:
                    st.caption("✅ Dataset is small. Using **Brute Force** (100% Optimal).")
                else:
                    st.caption("⚡ Dataset is large. Using **Stepwise Selection** (Smart Pruning).")

        # --- 3. Train & Visualize ---
        if st.button("🚀 Run Auto-ML Engine", type="primary"):
            filename = st.session_state.get('filename')
            
            with st.spinner("🤖 Running Optimization Algorithms..."):
                metrics = train_model(filename, target_col, [], auto_opt)
                
                if "error" in metrics:
                    st.error(metrics['error'])
                else:
                    st.session_state['trained'] = True
                    st.session_state['feature_names'] = metrics['kept_features']
                    st.session_state['target_name'] = target_col
                    
                    st.balloons()
                    
                    # --- DASHBOARD UI ---
                    st.markdown("### 🏆 Training Report")
                    
                    # 1. Top Level Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Problem Type", metrics.get("detected_type", "").title())
                    m2.metric("Algorithm", "Hybrid Auto-ML")
                    
                    if metrics.get("detected_type") == "classification":
                        m3.metric("Accuracy", f"{metrics.get('accuracy',0)*100:.1f}%")
                        m4.metric("F1 Score", metrics.get('f1',0))
                    else:
                        m3.metric("RMSE", metrics.get('rmse',0))
                        m4.metric("R2 Score", metrics.get('r2',0))

                    # 2. Tabs for Deep Dive
                    tab1, tab2, tab3 = st.tabs(["🧬 Feature Selection", "🔥 Correlation Map", "📊 Importance"])
                    
                    with tab1:
                        st.success(f"✅ Selected {len(metrics['kept_features'])} Best Features")
                        st.write("The algorithm determined these features contain the most signal:")
                        st.markdown(" ".join([f"`{f}`" for f in metrics['kept_features']]))
                        
                        dropped = [c for c in all_cols if c not in metrics['kept_features'] and c != target_col]
                        if dropped:
                            st.error(f"❌ Dropped {len(dropped)} Weak Features")
                            st.caption("These were removed because they reduced model accuracy:")
                            st.write(", ".join(dropped))

                    with tab2:
                        st.write("How features relate to each other (Darker = Stronger Link)")
                        # Compute correlation on Frontend for display
                        numeric_df = df[metrics['kept_features']].select_dtypes(include=['number'])
                        if not numeric_df.empty:
                            corr = numeric_df.corr()
                            # Use Pandas styling for a heatmap
                            st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None))
                        else:
                            st.info("Not enough numeric features for correlation map.")

                    with tab3:
                        st.write("Feature Contribution to Target")
                        # Use the correlation values sent from backend or calculate local
                        if "feature_correlations" in metrics and metrics["feature_correlations"]:
                            imp_df = pd.DataFrame(
                                list(metrics["feature_correlations"].items()), 
                                columns=["Feature", "Correlation with Target"]
                            ).set_index("Feature")
                            # Sort by absolute value
                            imp_df["abs"] = imp_df["Correlation with Target"].abs()
                            imp_df = imp_df.sort_values("abs", ascending=False).drop(columns=["abs"])
                            
                            st.bar_chart(imp_df)
                        else:
                            st.info("Importance charts available for numeric targets/features.")