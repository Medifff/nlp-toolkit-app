import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# Set the page configuration
st.set_page_config(layout="wide", page_title="NLP Toolkit App")

# Use caching decorators to load the models only once
@st.cache_resource
def load_sentiment_pipeline():
    """Loads the pre-trained sentiment analysis pipeline."""
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_summarization_pipeline():
    """Loads the pre-trained text summarization pipeline."""
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_ner_pipeline():
    """Loads the pre-trained Named Entity Recognition pipeline."""
    return pipeline("ner", grouped_entities=True)

@st.cache_resource
def load_text_generation_pipeline():
    """Loads the pre-trained text generation pipeline."""
    return pipeline("text-generation", model="gpt2")

# --- Main App ---
st.title("Multi-Purpose NLP Toolkit")

# Load the models with error handling
try:
    sentiment_pipeline = load_sentiment_pipeline()
    summarization_pipeline = load_summarization_pipeline()
    ner_pipeline = load_ner_pipeline()
    text_generation_pipeline = load_text_generation_pipeline()
except Exception as e:
    st.error(f"Error loading a model: {e}")
    st.stop()

# --- UI Tabs for different modes ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentiment", "Batch Sentiment", "Summarization", "NER", "Text Generation"])

# --- Tab 1: Single Text Analysis ---
with tab1:
    st.header("Analyze a Single Piece of Text")
    st.write("Enter text and the model will classify its sentiment as POSITIVE or NEGATIVE.")

    user_text = st.text_area(
        "Paste your text here:",
        height=150,
        value="I recently saw the new blockbuster movie and it was fantastic! The acting was superb and the plot was thrilling.",
        key="single_text_area"
    )

    if st.button("Analyze Single Text"):
        if user_text:
            with st.spinner("Analyzing..."):
                result = sentiment_pipeline(user_text)
                label = result[0]['label']
                score = result[0]['score']
                
                st.subheader("Analysis Result")
                if label == 'POSITIVE':
                    st.success(f"Sentiment: {label} 👍")
                else:
                    st.error(f"Sentiment: {label} 👎")
                st.info(f"**Confidence Score:** {score:.2%}")
        else:
            st.warning("Please enter some text to analyze.")

# --- Tab 2: Batch CSV Analysis ---
with tab2:
    st.header("Analyze a Batch of Texts from a CSV File")
    st.write("Upload a CSV file with a column of text to analyze the sentiment for each row.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("File Uploaded Successfully. Here's a preview:")
            st.dataframe(df.head())

            text_column = st.selectbox(
                "Which column contains the text you want to analyze?",
                options=df.columns
            )

            if st.button("Analyze CSV File"):
                if text_column:
                    with st.spinner("Analyzing all rows... This may take a while."):
                        def analyze_row(text):
                            if pd.isna(text) or not isinstance(text, str):
                                return {'label': 'N/A', 'score': 0}
                            return sentiment_pipeline(text)[0]

                        progress_bar = st.progress(0, text="Analyzing...")
                        total_rows = len(df)
                        
                        results = [analyze_row(row) for i, row in enumerate(df[text_column])]
                        
                        results_df = pd.DataFrame(results)
                        df_with_sentiment = pd.concat([df, results_df.rename(columns={'label': 'sentiment_label', 'score': 'sentiment_score'})], axis=1)

                    st.subheader("Analysis Complete!")
                    sentiment_counts = df_with_sentiment['sentiment_label'].value_counts()
                    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
                    st.plotly_chart(fig)
                    st.write("Full Data with Sentiment Analysis:")
                    st.dataframe(df_with_sentiment)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Tab 3: Text Summarization ---
with tab3:
    st.header("Summarize a Long Article")
    st.write("Paste a long piece of text and the model will generate a concise summary.")

    article_text = st.text_area(
        "Paste the article text here:",
        height=300,
        value="""(Paste a long article here to test)"""
    )
    
    min_len = st.slider("Minimum summary length", 50, 500, 100)
    max_len = st.slider("Maximum summary length", 100, 1000, 200)

    if st.button("Summarize Text"):
        if article_text:
            if min_len >= max_len:
                st.error("Error: Minimum length must be less than maximum length.")
            else:
                with st.spinner("Generating summary..."):
                    summary_result = summarization_pipeline(
                        article_text, 
                        max_length=max_len, 
                        min_length=min_len, 
                        do_sample=False
                    )
                    
                    st.subheader("Generated Summary")
                    st.success(summary_result[0]['summary_text'])
        else:
            st.warning("Please paste some text to summarize.")

# --- Tab 4: Named Entity Recognition (NER) ---
with tab4:
    st.header("Find Named Entities in Text")
    st.write("Paste text and the model will find and categorize entities like people (PER), organizations (ORG), and locations (LOC).")

    ner_text = st.text_area(
        "Paste your text for NER here:",
        height=200,
        value="Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California. Tim Cook is the current CEO."
    )

    if st.button("Find Entities"):
        if ner_text:
            with st.spinner("Finding entities..."):
                entities = ner_pipeline(ner_text)
                
                st.subheader("Found Entities")
                if entities:
                    df_entities = pd.DataFrame(entities)
                    df_entities.rename(columns={'entity_group': 'Category', 'word': 'Entity', 'score': 'Confidence'}, inplace=True)
                    st.dataframe(df_entities[['Category', 'Entity', 'Confidence']])
                else:
                    st.info("No named entities were found in the text.")
        else:
            st.warning("Please enter some text to analyze.")

# --- Tab 5: Text Generation ---
with tab5:
    st.header("Generate Text from a Prompt")
    st.write("Enter a starting phrase and the model will continue writing from it.")

    prompt_text = st.text_input(
        "Enter your prompt here:",
        value="In a world where technology and magic co-exist,"
    )

    max_length_gen = st.slider("Maximum generation length", 50, 300, 100)

    if st.button("Generate Text"):
        if prompt_text:
            with st.spinner("Generating text..."):
                generated_result = text_generation_pipeline(
                    prompt_text,
                    max_length=max_length_gen,
                    num_return_sequences=1
                )
                
                st.subheader("Generated Text")
                st.info(generated_result[0]['generated_text'])
        else:
            st.warning("Please enter a prompt to generate text.")
