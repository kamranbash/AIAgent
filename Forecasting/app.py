import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from prophet import Prophet
from dotenv import load_dotenv
from groq import Groq  # if you plan to integrate further AI functionality

# Load API key securely (for additional AI features)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📊", layout="wide")

st.title("Revenue Forecasting with Prophet Algorithm")
st.markdown("Upload an Excel file that contains the columns **Date** and **Revenue**.")

# File uploader widget for Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read the Excel file
    data = pd.read_excel(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(data.head())
    
    # Validate the required columns
    if "Date" not in data.columns or "Revenue" not in data.columns:
        st.error("The Excel file must contain 'Date' and 'Revenue' columns.")
    else:
        # Prepare the data for Prophet:
        # Rename columns to ds (date) and y (target variable) as required by Prophet.
        df = data.rename(columns={"Date": "ds", "Revenue": "y"})
        # Convert ds to datetime
        df["ds"] = pd.to_datetime(df["ds"])
        
        st.subheader("🔄 Data for Forecasting")
        st.dataframe(df.head())
        
        # Initialize and train the Prophet model
        model = Prophet()
        model.fit(df)
        
        # Let the user define the forecasting horizon (number of days)
        forecast_periods = st.number_input("Enter the number of days to forecast", min_value=1, value=30, step=1)
        
        # Create future dataframe and generate forecast
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        st.subheader("📈 Forecast Output")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        # Plot the forecast
        st.markdown("#### Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        
        # Plot forecast components (trend, weekly, yearly)
        st.markdown("#### Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
        
        # Example of integrating additional AI features (using Groq) if needed:
        st.subheader("🤖 AI-Generated Commentary")
        client = Groq(api_key=GROQ_API_KEY)
        # Here, you could generate a prompt that includes key insights from the forecast.
        # This is just a template prompt.
        prompt = f"""
        You are a financial expert. Analyze the following forecast data and provide:
        - Key trends in the revenue forecast.
        - Any potential concerns or risks.
        - Recommendations based on the forecast.
        Forecast Data (in JSON): {forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(orient='records')}
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in financial forecasting."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        ai_commentary = response.choices[0].message.content
        st.markdown(ai_commentary)
