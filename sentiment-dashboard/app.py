import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Title and description
st.title("📊 Sentiment Analysis Dashboard")
st.markdown("Analyze the sentiment of text or batch files (CSV) using a powerful AI model.")

# Initialize Hugging Face API key
api_key = st.sidebar.text_input("Enter your Hugging Face API Token:", type="password")
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {"Authorization": f"Bearer {api_key}"}

# Function to query the Hugging Face API
def query_sentiment(text):
    """
    Sends text to the Hugging Face API for sentiment analysis.
    Returns a label ('positive', 'negative', 'neutral') and a confidence score.
    """
    if not api_key:
        st.error("Please enter your Hugging Face API token in the sidebar.")
        return None, None

    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list):
            top_prediction = result[0][0]
            return top_prediction['label'], top_prediction['score']
        else:
            st.error(f"Unexpected API response: {result}")
            return None, None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling the API: {e}")
        return None, None

# --- TAB 1: Single Text Analysis ---
tab1, tab2 = st.tabs(["Single Text Analysis", "Batch File Analysis"])

with tab1:
    st.header("Analyze Single Text")
    text_input = st.text_area("Paste your text here:", "I absolutely love this product! It's fantastic.", height=150)
    
    if st.button("Analyze Sentiment", key="analyze_single"):
        with st.spinner('🤖 AI is thinking...'):
            label, score = query_sentiment(text_input)
            
        if label and score:
            label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
            readable_label = label_map.get(label, label)
            
            st.success("Analysis Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", readable_label)
            with col2:
                st.metric("Confidence Score", f"{score:.2%}")

with tab2:
    st.header("Analyze a Batch File")
    uploaded_file = st.file_uploader("Choose a CSV file with a 'text' column", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview of your data:**")
        st.dataframe(df.head())
        
        if 'text' not in df.columns:
            st.error("The CSV file must contain a column named 'text'.")
        else:
            if st.button("Analyze File", key="analyze_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, row in enumerate(df['text']):
                    percent_complete = int((i + 1) / len(df) * 100)
                    progress_bar.progress(percent_complete)
                    status_text.text(f"Analyzing {i+1} of {len(df)}...")
                    
                    label, score = query_sentiment(str(row))
                    if label and score:
                        label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
                        readable_label = label_map.get(label, label)
                        results.append({'text': row, 'sentiment': readable_label, 'confidence': score})
                    else:
                        results.append({'text': row, 'sentiment': 'Error', 'confidence': 0})
                
                results_df = pd.DataFrame(results)
                status_text.text("✅ Analysis complete!")
                
                st.subheader("Results")
                st.dataframe(results_df)
                
                fig = px.bar(results_df['sentiment'].value_counts(), 
                             title="Distribution of Sentiments",
                             labels={'value': 'Count', 'index': 'Sentiment'})
                st.plotly_chart(fig, use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )