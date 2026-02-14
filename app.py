import streamlit as st
import pandas as pd
import os
from model import train_model, predict_days
from gemini_extractor import extract_delays

# Page configuration
st.set_page_config(page_title="TrailerFlow Dashboard", layout="wide")

#adding background color
st.markdown("""
<style>
.stApp {
  background: linear-gradient(
    135deg,
    #eef2ff 0%,
    #e0c3fc 40%,
    #fbc2eb 100%
  );
}
</style>
""", unsafe_allow_html=True)

#adding background color


# Constants
CSV_PATH = "trailerflow_dataset_100.csv"
HIGH_RISK_THRESHOLD = 7  # Days

# --- Data & Model Loading ---
@st.cache_resource
def get_trained_model(csv_path):
    return train_model(csv_path)

@st.cache_data
def load_initial_data(csv_path):
    df = pd.read_csv(csv_path)
    # Initialize extra columns if they don't exist
    for col in ["parts_days", "accident_days", "training_days"]:
        if col not in df.columns:
            df[col] = 0
    return df

# Initialize session state for data persistence
if 'data' not in st.session_state:
    st.session_state.data = load_initial_data(CSV_PATH)

# Train/Get model
model = get_trained_model(CSV_PATH)

# Calculate baseline averages for autofill
baselines = st.session_state.data.groupby('trailer_type')['estimated_days'].mean().to_dict()

# Helper function to update predictions
def update_predictions():
    df = st.session_state.data
    predictions = []
    for _, row in df.iterrows():
        pred = predict_days(
            model, 
            row['trailer_type'], 
            row['estimated_days'], 
            row.get('parts_days', 0), 
            row.get('accident_days', 0), 
            row.get('training_days', 0)
        )
        predictions.append(pred)
    st.session_state.data['predicted_days'] = predictions

# Initial prediction run
if 'predicted_days' not in st.session_state.data.columns:
    update_predictions()

# --- UI Components ---
st.title("TrailerFlow")

# Top Metric Cards
total_active = len(st.session_state.data)
avg_pred = st.session_state.data['predicted_days'].mean()
high_risk = len(st.session_state.data[st.session_state.data['predicted_days'] > HIGH_RISK_THRESHOLD])

m1, m2, m3 = st.columns(3)
m1.metric("Total Active Trailers", total_active)
m2.metric("Avg Predicted Days", f"{avg_pred:.1f}")
m3.metric("High Risk Count", high_risk)

st.divider()

# --- Kanban Scheduler ---
st.subheader("Kanban Bay Scheduler")

# Rank trailers by predicted days (shortest first)
ranked_df = st.session_state.data.sort_values(by='predicted_days').reset_index(drop=True)

# Define Columns
cols = st.columns(5)
bay_names = ["Bay 1", "Bay 2", "Bay 3", "Bay 4", "Queue"]

for i, col in enumerate(cols):
    with col:
        st.markdown(f"### {bay_names[i]}")
        if i < 4:
            # Bays 1-4 get one trailer each
            if i < len(ranked_df):
                trailer = ranked_df.iloc[i]
                # Card Styling
                risk = "Red" if trailer['predicted_days'] > HIGH_RISK_THRESHOLD else "Amber" if trailer['predicted_days'] > 4 else "Green"
                st.info(f"**{trailer['trailer_id']}**\n\nType: {trailer['trailer_type']}\n\nPred: {trailer['predicted_days']} days\n\nRisk: {risk}")
            else:
                st.write("Empty")
        else:
            # Queue gets the rest
            if len(ranked_df) > 4:
                for _, trailer in ranked_df.iloc[4:7].iterrows():
                    risk = "Red" if trailer['predicted_days'] > HIGH_RISK_THRESHOLD else "Amber" if trailer['predicted_days'] > 4 else "Green"
                    with st.expander(f"{trailer['trailer_id']} - {trailer['predicted_days']} days"):
                        st.write(f"Type: {trailer['trailer_type']}")
                        st.write(f"Risk: {risk}")
            else:
                st.write("Empty")

st.divider()

# --- Actions Section ---
col_add, col_email = st.columns(2)

with col_add:
    st.subheader("Add New Trailer")
    with st.form("add_trailer"):
        t_type = st.selectbox("Trailer Type", options=list(baselines.keys()))
        
        # Auto-generate ID
        existing_ids = st.session_state.data['trailer_id'].tolist()
        numeric_ids = [int(tid[1:]) for tid in existing_ids if tid.startswith('T') and tid[1:].isdigit()]
        next_id = f"T{max(numeric_ids) + 1 if numeric_ids else 101}"
        
        est_days = st.number_input("Estimated Days", value=float(baselines.get(t_type, 5.0)))
        
        if st.form_submit_button("Add & Predict"):
            new_row = {
                "trailer_id": next_id,
                "trailer_type": t_type,
                "estimated_days": est_days,
                "parts_days": 0,
                "accident_days": 0,
                "training_days": 0,
                "actual_days": 0 # Placeholder
            }
            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
            update_predictions()
            st.success(f"Added {next_id}")
            st.rerun()

with col_email:
    st.subheader("Email Update Section")
    email_input = st.text_area("Paste email update here", height=150, placeholder="e.g. Trailer T103 has a 3 day delay for parts...")
    
    if 'extracted_json' not in st.session_state:
        st.session_state.extracted_json = None

    if st.button("Extract Delay Using AI"):
        if email_input:
            with st.spinner("AI is extracting details..."):
                extracted = extract_delays(email_input)
                st.session_state.extracted_json = extracted
        else:
            st.warning("Please paste an email first.")

    if st.session_state.extracted_json:
        st.json(st.session_state.extracted_json)
        
        if st.button("Recalculate Schedule"):
            target_id = st.session_state.extracted_json.get("trailer_id")
            if target_id in st.session_state.data['trailer_id'].values:
                # Update the row
                idx = st.session_state.data.index[st.session_state.data['trailer_id'] == target_id][0]
                st.session_state.data.at[idx, 'parts_days'] = st.session_state.extracted_json.get('parts_days', 0)
                st.session_state.data.at[idx, 'accident_days'] = st.session_state.extracted_json.get('accident_days', 0)
                st.session_state.data.at[idx, 'training_days'] = st.session_state.extracted_json.get('training_days', 0)
                
                update_predictions()
                st.success(f"Updated {target_id} and recalculated schedule.")
                st.session_state.extracted_json = None
                st.rerun()
            else:
                st.error(f"Trailer ID {target_id} not found in active dataset.")

# --- Data Table ---
with st.expander("View Full Dataset"):
    st.dataframe(st.session_state.data)
