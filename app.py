import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Load trained model
model = joblib.load("news_category_prediction.pkl")

# Page config
st.set_page_config(page_title="News Category Prediction", page_icon="📰", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: #222;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dataset Insights"])

# ---------------- Prediction Page ----------------
if page == "Prediction":
    st.title("📰 News Category Prediction")
    st.write("Predict the category of a news article based on its content.")

    input_text = st.text_area("Enter your News Content:", "")

    if st.button("Classify"):
        if input_text.strip():
            prediction = model.predict([input_text])[0]
            probabilities = model.predict_proba([input_text])[0]

            st.subheader(f"✅ Predicted Category: **{prediction}**")

            # Probabilities DataFrame
            prob_df = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probabilities
            })
            st.write("### Category Probabilities")
            st.bar_chart(prob_df.set_index("Category"))

            # --- Downloadable Results ---
            st.write("### 📥 Download Prediction Report")

            # CSV Export
            result_df = pd.DataFrame({"Input Text": [input_text], "Predicted Category": [prediction]})
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download as CSV", data=csv, file_name="prediction.csv", mime="text/csv")

            # PDF Export
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph("📰 News Category Prediction Report", styles["Title"]))
            story.append(Spacer(1, 12))

            # Input text
            story.append(Paragraph(f"<b>Input Text:</b> {input_text}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Prediction
            story.append(Paragraph(f"<b>Predicted Category:</b> {prediction}", styles["Heading2"]))
            story.append(Spacer(1, 12))

            # Probabilities Table
            prob_data = [["Category", "Probability"]] + prob_df.values.tolist()
            table = Table(prob_data, colWidths=[200, 150])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(Paragraph("<b>Category Probabilities:</b>", styles["Heading3"]))
            story.append(table)
            story.append(Spacer(1, 20))

            # Build PDF
            doc.build(story)
            pdf_buffer.seek(0)

            st.download_button(
                label="Download as PDF",
                data=pdf_buffer,
                file_name="prediction_report.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("⚠️ Please enter some text to classify.")

# ---------------- Dataset Insights Page ----------------
elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")

    try:
        df = pd.read_csv("dataset/BBC_News_Train.csv")
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write("### Category Distribution")
        category_counts = df["Category"].value_counts()
        st.bar_chart(category_counts)
    except Exception as e:
        st.error(f"Dataset not found or error loading dataset: {e}")
