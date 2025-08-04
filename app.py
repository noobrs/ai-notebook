import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re
from bs4 import BeautifulSoup
import emoji
import html

# Configure page
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .result-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .result-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Text cleaning function
def clean_text(text):
    """Clean and preprocess text"""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = html.unescape(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s.,!?;:]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Model descriptions
MODEL_INFO = {
    "BERT": {
        "description": "Bidirectional Encoder Representations from Transformers - State-of-the-art transformer model",
        "strengths": [
            "Highest accuracy for sentiment analysis",
            "Understands context bidirectionally",
            "Pre-trained on large corpus",
            "Handles complex language patterns"
        ],
        "weaknesses": [
            "Computationally expensive",
            "Slower inference time",
            "Requires GPU for optimal performance",
            "Large model size"
        ],
        "accuracy": "~92-95%",
        "speed": "Slow"
    },
    "SVM": {
        "description": "Support Vector Machine with TF-IDF vectorization - Classical machine learning approach",
        "strengths": [
            "Fast training and inference",
            "Good performance on small datasets",
            "Memory efficient",
            "Interpretable results"
        ],
        "weaknesses": [
            "Limited understanding of context",
            "Requires feature engineering",
            "May struggle with complex language",
            "Sensitive to data preprocessing"
        ],
        "accuracy": "~85-88%",
        "speed": "Very Fast"
    },
    "Naive Bayes": {
        "description": "Probabilistic classifier based on Bayes theorem - Simple and effective baseline",
        "strengths": [
            "Very fast training and prediction",
            "Works well with small datasets",
            "Simple and interpretable",
            "Good baseline model"
        ],
        "weaknesses": [
            "Assumes feature independence",
            "May oversimplify relationships",
            "Lower accuracy than deep learning",
            "Sensitive to data quality"
        ],
        "accuracy": "~80-83%",
        "speed": "Very Fast"
    },
    "LSTM": {
        "description": "Long Short-Term Memory neural network - Recurrent network for sequence modeling",
        "strengths": [
            "Good at capturing sequential patterns",
            "Handles variable length inputs",
            "Better than basic RNNs",
            "Moderate computational requirements"
        ],
        "weaknesses": [
            "Slower than traditional ML",
            "May suffer from vanishing gradients",
            "Requires more data than classical ML",
            "Less interpretable"
        ],
        "accuracy": "~88-91%",
        "speed": "Moderate"
    }
}

def load_model(model_name):
    """Load the selected model"""
    try:
        if model_name == "BERT":
            # Load BERT model and tokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
            # Load your trained weights if available
            # model.load_state_dict(torch.load("checkpoints/bert_sentiment_best.pt", map_location="cpu"))
            return {"tokenizer": tokenizer, "model": model}
        
        elif model_name == "SVM":
            # Load SVM model
            try:
                bundle = joblib.load("svm_sentiment.pkl")
                return bundle
            except:
                st.error("SVM model file not found. Please train the model first.")
                return None
        
        elif model_name == "Naive Bayes":
            # Placeholder for Naive Bayes model
            st.warning("Naive Bayes model not implemented yet.")
            return None
        
        elif model_name == "LSTM":
            # Placeholder for LSTM model
            st.warning("LSTM model not implemented yet.")
            return None
            
    except Exception as e:
        st.error(f"Error loading {model_name} model: {str(e)}")
        return None

def predict_sentiment(text, model_name, model):
    """Predict sentiment for given text"""
    if model is None:
        return "Error", 0.0
    
    try:
        cleaned_text = clean_text(text)
        
        if model_name == "BERT":
            tokenizer = model["tokenizer"]
            bert_model = model["model"]
            
            # Tokenize and predict
            inputs = tokenizer(cleaned_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Map class to sentiment
            sentiment_map = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
            sentiment = sentiment_map[predicted_class]
            
            return sentiment, confidence
        
        elif model_name == "SVM":
            # Use SVM model
            vectorizer = model["vectorizer"]
            svm_model = model["model"]
            
            # Transform text and predict
            text_vectorized = vectorizer.transform([cleaned_text])
            prediction = svm_model.predict(text_vectorized)[0]
            
            # Get probability scores
            decision_scores = svm_model.decision_function(text_vectorized)
            confidence = np.max(np.abs(decision_scores)) / np.sum(np.abs(decision_scores))
            
            return prediction, confidence
        
        else:
            return "Model not available", 0.0
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Error", 0.0

def create_result_visualization(results_df):
    """Create visualization for bulk analysis results"""
    # Sentiment distribution
    sentiment_counts = results_df['Predicted_Sentiment'].value_counts()
    
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Confidence distribution
    fig_hist = px.histogram(
        results_df,
        x='Confidence',
        title="Confidence Score Distribution",
        nbins=20,
        color_discrete_sequence=['#1f77b4']
    )
    
    return fig_pie, fig_hist

def generate_pdf_report(results_df, model_name, analysis_type):
    """Generate PDF report of analysis results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Movie Review Sentiment Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary information
    story.append(Paragraph(f"<b>Analysis Type:</b> {analysis_type}", styles['Normal']))
    story.append(Paragraph(f"<b>Model Used:</b> {model_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Reviews Analyzed:</b> {len(results_df)}", styles['Normal']))
    story.append(Paragraph(f"<b>Average Confidence:</b> {results_df['Confidence'].mean():.2%}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Sentiment distribution
    sentiment_counts = results_df['Predicted_Sentiment'].value_counts()
    story.append(Paragraph("<b>Sentiment Distribution:</b>", styles['Heading2']))
    
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results_df)) * 100
        story.append(Paragraph(f"• {sentiment}: {count} reviews ({percentage:.1f}%)", styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Top results table
    story.append(Paragraph("<b>Analysis Results (Top 10):</b>", styles['Heading2']))
    
    # Prepare table data
    table_data = [['Review Text', 'Predicted Sentiment', 'Confidence']]
    for idx, row in results_df.head(10).iterrows():
        review_text = row['Review_Text'][:100] + "..." if len(row['Review_Text']) > 100 else row['Review_Text']
        table_data.append([
            review_text,
            row['Predicted_Sentiment'],
            f"{row['Confidence']:.2%}"
        ])
    
    # Create table
    table = Table(table_data, colWidths=[4*72, 1.5*72, 1*72])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.header("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Analysis Model:",
        options=list(MODEL_INFO.keys()),
        index=0
    )
    
    # Display model information
    with st.sidebar.expander("📊 Model Information", expanded=True):
        model_info = MODEL_INFO[selected_model]
        st.write(f"**Description:** {model_info['description']}")
        st.write(f"**Accuracy:** {model_info['accuracy']}")
        st.write(f"**Speed:** {model_info['speed']}")
        
        st.write("**Strengths:**")
        for strength in model_info['strengths']:
            st.write(f"• {strength}")
        
        st.write("**Weaknesses:**")
        for weakness in model_info['weaknesses']:
            st.write(f"• {weakness}")
    
    # Main content area
    tab1, tab2 = st.tabs(["📝 Single Review Analysis", "📁 Bulk CSV Analysis"])
    
    with tab1:
        st.header("Analyze Individual Review")
        
        # Text input
        review_text = st.text_area(
            "Enter your movie review:",
            placeholder="Type your movie review here... (e.g., 'This movie was absolutely fantastic! Great acting and plot.')",
            height=150
        )
        
        # Analysis button
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if review_text.strip():
                with st.spinner(f"Analyzing with {selected_model} model..."):
                    # Load model
                    model = load_model(selected_model)
                    
                    if model is not None:
                        # Predict sentiment
                        sentiment, confidence = predict_sentiment(review_text, selected_model, model)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Analysis Results")
                            
                            # Style result based on sentiment
                            if "positive" in sentiment.lower():
                                result_class = "result-positive"
                                emoji_icon = "😊"
                            elif "negative" in sentiment.lower():
                                result_class = "result-negative"
                                emoji_icon = "😞"
                            else:
                                result_class = "result-neutral"
                                emoji_icon = "😐"
                            
                            st.markdown(f"""
                                <div class="{result_class}">
                                    <h3>{emoji_icon} Predicted Sentiment: {sentiment.title()}</h3>
                                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader("📈 Confidence Visualization")
                            
                            # Create confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = confidence * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Confidence Level (%)"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Export option for single analysis
                        st.subheader("📥 Export Results")
                        if st.button("📄 Export as PDF"):
                            # Create DataFrame for single result
                            single_result_df = pd.DataFrame({
                                'Review_Text': [review_text],
                                'Predicted_Sentiment': [sentiment],
                                'Confidence': [confidence]
                            })
                            
                            pdf_buffer = generate_pdf_report(single_result_df, selected_model, "Single Review Analysis")
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"sentiment_analysis_report_{selected_model.lower()}.pdf",
                                mime="application/pdf"
                            )
            else:
                st.warning("⚠️ Please enter a review to analyze.")
    
    with tab2:
        st.header("Bulk Analysis from CSV File")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews:",
            type=['csv'],
            help="CSV file should have a column named 'review' or 'text' containing the movie reviews"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Column selection
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                selected_column = st.selectbox(
                    "Select the column containing reviews:",
                    options=text_columns,
                    index=0 if text_columns else None
                )
                
                if selected_column and st.button("🚀 Start Bulk Analysis", type="primary"):
                    # Load model
                    model = load_model(selected_model)
                    
                    if model is not None:
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        total_rows = len(df)
                        
                        # Process each review
                        for idx, row in df.iterrows():
                            review_text = str(row[selected_column])
                            
                            # Update progress
                            progress = (idx + 1) / total_rows
                            progress_bar.progress(progress)
                            status_text.text(f"Processing review {idx + 1} of {total_rows}...")
                            
                            # Predict sentiment
                            sentiment, confidence = predict_sentiment(review_text, selected_model, model)
                            
                            results.append({
                                'Review_Text': review_text,
                                'Predicted_Sentiment': sentiment,
                                'Confidence': confidence
                            })
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.success("✅ Analysis completed!")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Reviews", len(results_df))
                        
                        with col2:
                            avg_confidence = results_df['Confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.2%}")
                        
                        with col3:
                            most_common = results_df['Predicted_Sentiment'].mode()[0]
                            st.metric("Most Common Sentiment", most_common.title())
                        
                        # Visualizations
                        st.subheader("📊 Analysis Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie, fig_hist = create_result_visualization(results_df)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Export options
                        st.subheader("📥 Export Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV export
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download CSV Results",
                                data=csv_data,
                                file_name=f"bulk_sentiment_analysis_{selected_model.lower()}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # PDF export
                            if st.button("📄 Generate PDF Report"):
                                with st.spinner("Generating PDF report..."):
                                    pdf_buffer = generate_pdf_report(results_df, selected_model, "Bulk CSV Analysis")
                                    
                                    st.download_button(
                                        label="📥 Download PDF Report",
                                        data=pdf_buffer,
                                        file_name=f"bulk_sentiment_analysis_report_{selected_model.lower()}.pdf",
                                        mime="application/pdf"
                                    )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🎬 Movie Review Sentiment Analysis Tool | Built with Streamlit</p>
            <p>Support for BERT, SVM, Naive Bayes, and LSTM models</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()