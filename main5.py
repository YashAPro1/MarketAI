from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st
from datetime import datetime, timedelta
import os
# from recommender_utils import ProductRecommender

class EventFinder:
    def __init__(self, llm):
        self.llm = llm
    
    def get_events(self, start_date: str, end_date: str, location: str) -> str:
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
        - Add which products they can Market - Name the products and add their description
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
            return f"Error generating event list: {str(e)}"

class ProductRecommender:
    def __init__(self):
        """Initialize the recommender with LLM and embeddings"""
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        # Load the existing vector store
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        self.event_finder = EventFinder(self.llm)
    
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
            search_type="mmr",  # Using MMR for more diverse results
            search_kwargs={
                "k": k,
                "fetch_k": k * 2,
                "lambda_mult": 0.7
            }
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
            return {"result": f"Error generating recommendations: {str(e)}", "source_documents": []}
    
    def generate_campaign_recommendations(self, 
                                       start_date: str, 
                                       end_date: str, 
                                       location: str,
                                       events_data: str,
                                       products: List[Dict],
                                       user_profile: str) -> str:
        """Generate campaign recommendations based on events and products"""
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
        """
        
        try:
            campaign_response = self.llm.invoke(campaign_prompt)
            return str(campaign_response)
        except Exception as e:
            return f"Error generating marketing campaigns: {str(e)}"
        


def main():
    st.title("🎯 Location-Based Campaign Recommender")
    
    try:
        # Check if vector store exists
        if not os.path.exists("./chroma_db"):
            st.error("Vector store not found. Please run data_indexer.py first!")
            return
        
        recommender = ProductRecommender()
        
        # Location Selection
        st.subheader("📍 Campaign Location")
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox(
                "State:",
                ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Gujarat", "Other"]
            )
        with col2:
            city = st.text_input("City:", "Mumbai")
        location = f"{city}, {state}"
        
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
        
        # Target Audience Profile
        st.subheader("👤 Target Audience Profile")
        
        # More structured audience profile input
        age_range = st.select_slider(
            "Age Range",
            options=list(range(15, 76, 5)),
            value=(25, 40)
        )
        
        interests = st.multiselect(
            "Interests",
            ["Technology", "Fashion", "Beauty", "Sports", "Culture", "Food", 
             "Travel", "Education", "Entertainment", "Home & Living"],
            default=["Technology", "Fashion"]
        )
        
        occupation = st.selectbox(
            "Primary Occupation",
            ["Working Professional", "Student", "Business Owner", "Homemaker", 
             "Retired", "Other"]
        )
        
        # Construct user profile
        user_profile = f"{occupation} aged {age_range[0]}-{age_range[1]}, "
        user_profile += f"interested in {', '.join(interests)}"
        
        if st.button("🎯 Generate Campaign Recommendations"):
            with st.spinner("Analyzing local events and generating campaign recommendations..."):
                # Get events for the location and date range
                events_data = recommender.event_finder.get_events(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    location=location
                )
                
                # Get product recommendations
                recommendations = recommender.get_recommendations(
                    user_profile=user_profile,
                    events_list=events_data
                )
                
                # Generate campaigns
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
                
                # Show product recommendations in expandable section
                with st.expander("🎁 Recommended Products"):
                    st.write(recommendations["result"])
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your configuration and try again.")


if __name__ == "__main__":
    main()