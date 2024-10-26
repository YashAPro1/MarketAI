import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px

# Configure Gemini AI
GEMINI_API_KEY = 'AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8' 
genai.configure(api_key=GEMINI_API_KEY)

class GeminiConsumerAnalyzer:
    def _init_(self):
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_customer_insights(self, customer_data):
        prompt = f"""
        Analyze this Indian customer's shopping behavior:
        - Language: {customer_data['preferred_language']}
        - Purchase Amount: ₹{customer_data['purchase_amount']}
        - Visit Frequency: {customer_data['visit_frequency']} times
        - Payment Mode: {customer_data['preferred_payment_mode']}
        - Product Category: {customer_data['product_category']}
        - Age Group: {customer_data['age_group']}
        
        Provide possible predictions of customer:
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                }
            )
            return response.text
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def generate_regional_strategy(self, regional_data):
        prompt = f"""
        Create a detailed marketing strategy for these Indian regions:
        Region Summary:
        {regional_data.to_string()}
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                }
            )
            return response.text
        except Exception as e:
            return f"Error generating strategy: {str(e)}"

def create_visualizations(df):
    figs = []
    figs.append(px.bar(df, x='state', y='purchase_amount', title='Purchase Amount by State'))
    figs.append(px.histogram(df, x='visit_frequency', title='Visit Frequency Distribution'))
    payment_dist = df['preferred_payment_mode'].value_counts()
    figs.append(px.pie(values=payment_dist.values, names=payment_dist.index, title='Payment Mode Distribution'))
    age_dist = df['age_group'].value_counts()
    figs.append(px.pie(values=age_dist.values, names=age_dist.index, title='Age Group Distribution'))
    return figs

def create_sample_data():
    data = {
        'customer_id': list(range(1, 21)),
        'state': np.random.choice(['Maharashtra', 'Tamil Nadu', 'West Bengal', 'Punjab', 'Gujarat', 'Karnataka', 'Delhi', 'Rajasthan'], 20),
        'preferred_language': np.random.choice(['Marathi', 'Tamil', 'Bengali', 'Hindi', 'Gujarati', 'Kannada', 'Hindi', 'Hindi'], 20),
        'purchase_amount': np.random.randint(500, 5000, 20),
        'visit_frequency': np.random.randint(1, 10, 20),
        'preferred_payment_mode': np.random.choice(['UPI', 'Cash', 'Card', 'Net Banking'], 20),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home & Living', 'Beauty', 'Books'], 20),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], 20),
        'date': pd.date_range(start='2024-01-01', periods=20)
    }
    return pd.DataFrame(data)

def main():
    st.title("🛍 Indian Consumer Analysis Dashboard")
    st.sidebar.header("Select Analysis Type")
    
    df = create_sample_data()
    analyzer = GeminiConsumerAnalyzer()

    analysis_type = st.sidebar.radio("Choose analysis type:", ["Behaviour Analysis", "Regional Analysis"])

    if analysis_type == "Behaviour Analysis":
        st.subheader("Individual Customer Analysis")
        customer_id = st.selectbox("Select Customer ID", df['customer_id'].tolist())
        customer_data = df[df['customer_id'] == customer_id].iloc[0]
        insights = analyzer.generate_customer_insights(customer_data)
        st.markdown(insights)
        
        st.subheader("Customer Visualizations")
        for fig in create_visualizations(df):
            st.plotly_chart(fig)

    elif analysis_type == "Regional Analysis":
        st.subheader("Regional Strategy Analysis")
        regional_summary = df.groupby('state').agg({
            'purchase_amount': 'mean',
            'visit_frequency': 'mean',
            'preferred_payment_mode': lambda x: x.mode()[0]
        }).round(2)
        
        strategy = analyzer.generate_regional_strategy(regional_summary)
        st.markdown(strategy)
        
        st.subheader("Regional Visualizations")
        for fig in create_visualizations(df):
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()