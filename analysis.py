import streamlit as st
import pandas as pd
import os
import re
import json

st.title("CFPB Credit Card Complaint Analysis")

# --- Configurable section ---
tool_keywords = ['TSYS', 'OMEGA', 'EPM', 'CTL', 'CTI', 'CTR', 'Case Management Software', 'Document Scanner']
team_keywords = ['CAO', 'Customer Service', 'Operations', 'CTI', 'CTR', 'Research Team']

# --- Helper functions ---
def extract_matches(text, keywords):
    if pd.isna(text): return []
    matches = [kw for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', str(text), re.IGNORECASE)]
    return matches

def extract_driver_info(text):
    if pd.isna(text): return None, None
    driver, subdriver = None, None
    matches = re.findall(r'(driver|subdriver)\s*[:=]\s*(\w+)', str(text), flags=re.IGNORECASE)
    for match in matches:
        if match[0].lower() == 'driver':
            driver = match[1]
        elif match[0].lower() == 'subdriver':
            subdriver = match[1]
    return driver, subdriver

def parse_json_field(field):
    try:
        if isinstance(field, str) and field.strip().startswith('{'):
            return json.loads(field)
    except Exception:
        return None
    return None

def summarize_action(row):
    activity_type = row.get('CASE_ACTIVITY_TYPE_CD', '')
    note = row.get('parsed_note', {}) or {}
    details = row.get('parsed_details', {}) or {}

    summary_parts = [f"Type: {activity_type}"]

    if 'status' in details:
        summary_parts.append(f"Status changed to '{details['status']}'")
    if 'team' in details:
        summary_parts.append(f"Team involved: {details['team']}")
    if 'tool' in details:
        summary_parts.append(f"Tool used: {details['tool']}")
    if 'comment' in note:
        summary_parts.append(f"Comment: {note['comment']}")
    if 'reason' in note:
        summary_parts.append(f"Reason: {note['reason']}")
    
    return "; ".join(summary_parts)

def load_all_csvs(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, filename))
            df['complaint_id'] = filename.replace('.csv', '')
            all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# --- File uploader ---
folder_path = st.text_input("Enter path to folder containing complaint CSVs")
if folder_path and os.path.isdir(folder_path):
    st.success("Loading data from folder...")
    df = load_all_csvs(folder_path)

    st.write("Raw data sample:", df.head())

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['complaint_id', 'timestamp'])

    df['step_number'] = df.groupby('complaint_id').cumcount() + 1
    df['prev_timestamp'] = df.groupby('complaint_id')['timestamp'].shift(1)
    df['step_duration_mins'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 60.0
    df['step_duration_mins'] = df['step_duration_mins'].fillna(0)

    df['parsed_note'] = df['ACTIVITY_NOTE'].apply(parse_json_field)
    df['parsed_details'] = df['ACTIVITY_DETAILS'].apply(parse_json_field)

    df['tools_used'] = df['ACTIVITY_NOTE'].apply(lambda x: extract_matches(x, tool_keywords))
    df['teams_contacted'] = df['ACTIVITY_NOTE'].apply(lambda x: extract_matches(x, team_keywords))

    df[['driver', 'subdriver']] = df['ACTIVITY_NOTE'].apply(lambda x: pd.Series(extract_driver_info(x)))

    df['step_action_summary'] = df.apply(summarize_action, axis=1)

    summary_indicators = ['CASE_CLOSED', 'CASE_CLOSURE_REASON', 'DISPUTE_REASON_UPDATE']
    df['is_summary'] = df['CASE_ACTIVITY_TYPE_CD'].isin(summary_indicators)
    summary_texts = df[df['is_summary']][['complaint_id', 'CASE_ACTIVITY_TYPE_CD', 'ACTIVITY_NOTE', 'ACTIVITY_DETAILS']]

    summary_df = df.groupby('complaint_id').agg({
        'step_number': 'max',
        'step_duration_mins': 'sum',
        'tools_used': lambda x: list(set(sum(x, []))),
        'teams_contacted': lambda x: list(set(sum(x, []))),
        'driver': lambda x: next((i for i in x if pd.notna(i)), None),
        'subdriver': lambda x: next((i for i in x if pd.notna(i)), None)
    }).rename(columns={
        'step_number': 'total_steps',
        'step_duration_mins': 'total_TAT_minutes'
    }).reset_index()

    st.subheader("Complaint Summary")
    st.dataframe(summary_df)

    if st.checkbox("Show full processed data"):
        st.dataframe(df)

    if st.checkbox("Show extracted summaries"):
        st.dataframe(summary_texts)

else:
    st.warning("Please enter a valid folder path to begin analysis.")
