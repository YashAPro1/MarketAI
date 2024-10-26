import os
from typing import List, Dict
import pandas as pd
import chromadb
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

def create_sample_dataset():
    """Create a sample product dataset"""
    data = {
        'product_id': range(1, 11),
        'product_details': [
            "Premium Smart Watch with fitness tracking, perfect for tech-savvy young professionals who enjoy fitness. Great for Diwali gifting.",
            "Traditional Silk Saree with modern designs, ideal for women aged 25-50 who appreciate cultural wear. Popular during festivals.",
            "Organic Skincare Gift Set, suitable for beauty enthusiasts and perfect for Eid celebrations.",
            "Gaming Laptop with RGB keyboard, targeted at young gamers and students. Top choice for Dussehra sales.",
            "Handcrafted Silver Jewelry Set, perfect for women who love traditional accessories. Ideal for wedding season.",
            "Smart Home Security System, great for security-conscious family households. Popular during holiday season.",
            "Premium Coffee Maker, ideal for coffee enthusiasts and working professionals. Great for house warming gifts.",
            "Fitness Equipment Set, perfect for health-conscious individuals starting their fitness journey.",
            "Designer Handbag Collection, targeted at fashion-forward women aged 20-40. Popular during shopping festivals.",
            "Educational Tablet with learning apps, ideal for students aged 8-15. Great for academic season."
        ],
        'target_audience': [
            "Tech-savvy young professionals, fitness enthusiasts, age 25-40",
            "Traditional women, cultural event attendees, age 25-50",
            "Beauty enthusiasts, organic product lovers, age 20-45",
            "Students, young gamers, tech enthusiasts, age 15-30",
            "Traditional jewelry lovers, wedding shoppers, age 25-55",
            "Family households, security-conscious homeowners",
            "Coffee lovers, working professionals, home enthusiasts",
            "Fitness beginners, health-conscious individuals, age 20-50",
            "Fashion-forward women, luxury shoppers, age 20-40",
            "Students, parents of school children, age 8-15"
        ],
        'price_range': [  # Added price range for better campaign planning
            "₹15,000 - ₹25,000",
            "₹8,000 - ₹50,000",
            "₹2,000 - ₹5,000",
            "₹60,000 - ₹120,000",
            "₹20,000 - ₹100,000",
            "₹25,000 - ₹45,000",
            "₹15,000 - ₹30,000",
            "₹10,000 - ₹30,000",
            "₹25,000 - ₹75,000",
            "₹15,000 - ₹30,000"
        ],
        'category': [  # Added category for better organization
            "Electronics",
            "Fashion",
            "Beauty & Personal Care",
            "Electronics",
            "Jewelry",
            "Home & Living",
            "Home Appliances",
            "Sports & Fitness",
            "Fashion Accessories",
            "Electronics"
        ]
    }
    return pd.DataFrame(data)

class EventFinder:
    def __init__(self, llm):
        self.llm = llm
    
    def get_events(self, start_date: str, end_date: str, location: str) -> List[Dict]:
        """
        Use Gemini to identify festivals and events between given dates for a specific location
        """
        events_prompt = f"""
        Identify the major festivals, cultural events, shopping seasons, and holidays between {start_date} and {end_date} in {location}.
        
        Consider:
        1. Religious festivals
        2. Cultural celebrations
        3. Shopping seasons
        4. School/College schedules
        5. Local events specific to {location}
        6. National holidays
        7. Corporate/Business events
        8. Seasonal changes and related events
        
        For each event, provide:
        - Name of the event
        - Date (must be between {start_date} and {end_date})
        - Type (festival/holiday/shopping/cultural/seasonal)
        - Description
        - Significance for marketing and sales
        - Shopping patterns associated with the event
        
        Format the response as a structured list of events with clear headers and categories.
        """
        
        try:
            response = self.llm.invoke(events_prompt)
            return str(response)
        except Exception as e:
            st.error(f"Error generating event list: {str(e)}")
            return []

class ProductRecommender:
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        self.chroma_client = chromadb.Client()
        self.event_finder = EventFinder(self.llm)
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare and index the product data"""
        documents = [
            f"Product: {row['product_details']}\nTarget Audience: {row['target_audience']}"
            for _, row in df.iterrows()
        ]
        
        self.vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def get_recommendations(self, user_profile: str, events_list: str, k: int = 4) -> Dict:
        """Get product recommendations based on user profile and events"""
        query = f"""
        Based on the following information, suggest the most relevant products:
        
        User Profile: {user_profile}
        Upcoming Events: {events_list}
        
        Analyze the product details and target audience carefully to make personalized recommendations.
        Consider the specific events and cultural context when recommending products.
        Explain why each recommended product would be suitable for this user and the upcoming events.
        """
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        try:
            result = qa_chain({"query": query})
            return result
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return {"result": "Error generating recommendations", "source_documents": []}
    
    def generate_campaign_recommendations(self, 
                                       start_date: str, 
                                       end_date: str, 
                                       location: str,
                                       events_data: str,
                                       products: List[Dict],
                                       user_profile: str) -> str:
        """
        Generate campaign recommendations based on events and products
        """
        campaign_prompt = f"""
        Based on the following information:

        Date Range: {start_date} to {end_date}
        Location: {location}

        Upcoming Events and Festivals:
        {events_data}

        Available Products:
        {[doc.page_content for doc in products]}

        User Profile:
        {user_profile}

        Generate 2-3 strategic marketing campaigns that:
        1. Align with the location-specific events and festivals
        2. Consider local cultural sensitivities and preferences
        3. Target the right audience at the right time
        4. Maximize seasonal and cultural opportunities
        
        For each campaign provide:
        1. Campaign Title (creative and culturally relevant)
        2. Event/Festival Connection
        3. Target Audience
        4. Campaign Duration (with specific dates)
        5. Key Marketing Channels (consider local preferences)
        6. Promotional Strategy
        7. Cultural Integration Aspects
        8. Expected Outcomes
        9. Budget Allocation Suggestions
        10. Local Partnership Opportunities
        
        Ensure the campaigns are:
        - Culturally appropriate for {location}
        - Well-spaced across the given date range
        - Taking advantage of peak shopping periods
        - Integrated with local customs and practices
        """
        
        try:
            campaign_response = self.llm.invoke(campaign_prompt)
            return str(campaign_response)
        except Exception as e:
            return f"Error generating marketing campaigns: {str(e)}"

def main():
    st.title("🎯 Location-Based Campaign Recommender")
    
    try:
        recommender = ProductRecommender()
        df = create_sample_dataset()  # Using the same sample dataset as before
        
        with st.spinner("Preparing product data..."):
            recommender.prepare_data(df)
        
        # Location Selection
        st.subheader("📍 Campaign Location")
        location = st.text_input(
            "Enter location (City/State/Country):",
            "Mumbai, India"
        )
        
        # Date Range Selection
        st.subheader("📅 Campaign Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                min_value=datetime.now().date(),
                value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                min_value=start_date,
                value=start_date + timedelta(days=90)
            )
        
        # User Profile
        st.subheader("👤 Target Audience Profile")
        user_profile = st.text_area(
            "Enter target audience characteristics:",
            "Tech-savvy young professionals, age 25-40, urban lifestyle, interested in latest gadgets and fashion"
        )
        
        if st.button("🎯 Generate Campaign Recommendations"):
            with st.spinner("Analyzing local events and generating campaign recommendations..."):
                # First get events for the location and date range
                events_data = recommender.event_finder.get_events(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    location=location
                )
                
                # Get product recommendations based on events
                recommendations = recommender.get_recommendations(
                    user_profile=user_profile,
                    events_list=events_data
                )
                
                # Generate campaigns based on events and recommended products
                campaigns = recommender.generate_campaign_recommendations(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    location=location,
                    events_data=events_data,
                    products=recommendations["source_documents"],
                    user_profile=user_profile
                )
                
                # Display Results
                st.subheader("📅 Events & Festivals")
                st.markdown(events_data)
                
                st.subheader("📊 Campaign Recommendations")
                st.markdown(campaigns)
                
                # Display supporting information
                with st.expander("🎁 Recommended Products"):
                    st.write(recommendations["result"])
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your configuration and try again.")

if __name__ == "__main__":
    main()