# app.py
import os
from typing import Dict
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductRecommender:
    def __init__(self):
        """Initialize the recommender with pre-existing vector store"""
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
        
    def get_recommendations(self, user_profile: str, location: str, upcoming_events: str, k: int = 4) -> Dict:
        """Get culturally relevant product recommendations"""
        # Create a more specific query combining all user inputs
        combined_query = f"""
        Find products specifically matching these exact criteria:
        
        User Details:
        - Profile: {user_profile}
        - Location: {location}
        - Events: {upcoming_events}
        
        Focus on:
        1. Products matching the user's specific age group and interests in {user_profile}
        2. Local availability and cultural significance in {location}
        3. Immediate relevance to {upcoming_events}
        4. Price range appropriate for mentioned occasions
        
        Consider regional factors:
        - Local festivals and customs of {location}
        - Weather and seasonal considerations
        - Regional shopping patterns and preferences
        """
        
        # Get relevant documents using semantic search
        docs = self.vectorstore.similarity_search_with_relevance_scores(
            combined_query,
            k=k*2  # Get more candidates for diversity
        )
        
        # Filter out low relevance scores
        relevant_docs = [doc for doc, score in docs if score > 0.7]
        
        # If we have too few results, get more with a broader search
        if len(relevant_docs) < k:
            additional_docs = self.vectorstore.similarity_search(
                combined_query,
                k=k-len(relevant_docs)
            )
            relevant_docs.extend(additional_docs)
        
        # Limit to top k results
        relevant_docs = relevant_docs[:k]
        
        # Create a detailed prompt for the LLM to analyze the retrieved products
        analysis_prompt = f"""
        Analyze these products for the following user:
        
        User Context:
        {user_profile}
        Currently in: {location}
        Upcoming events: {upcoming_events}
        
        For each product, explain:
        1. Specific reasons why it matches this user's profile
        2. How it relates to their location and upcoming events
        3. Cultural significance and appropriateness
        4. Practical considerations (timing, weather, occasion)
        
        Products to analyze:
        {[doc.page_content for doc in relevant_docs]}
        """
        
        try:
            # Get LLM analysis
            analysis = self.llm.predict(analysis_prompt)
            
            return {
                "result": analysis,
                "source_documents": relevant_docs
            }
            
        except Exception as e:
            return {
                "result": f"Error generating recommendations: {str(e)}",
                "source_documents": []
            }

def main():
    st.title("🎁 Indian Cultural Smart Product Recommender")
    
    try:
        if not os.path.exists("./chroma_db"):
            st.error("Vector store not found. Please run data_indexer.py first!")
            return
            
        recommender = ProductRecommender()
        
        # Add more specific input fields
        st.subheader("👤 User Profile")
        age = st.number_input("Age:", min_value=1, max_value=100, value=25)
        occupation = st.selectbox("Occupation:", 
            ["Student", "Working Professional", "Business Owner", "Homemaker", "Retired", "Other"])
        interests = st.multiselect("Interests:",
            ["Technology", "Fashion", "Sports", "Arts", "Culture", "Food", "Travel", "Music"])
        
        user_profile = f"{age}-year-old {occupation}, interested in {', '.join(interests)}"
        
        st.subheader("📍 Location")
        state = st.selectbox("State:", 
            ["Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Gujarat"])  # Add more states
        city = st.text_input("City:", "Bangalore")
        location = f"{city}, {state}"
        
        st.subheader("📅 Upcoming Events")
        events = st.multiselect("Select upcoming events:",
            ["Diwali", "Wedding", "House Warming", "Birthday", "Anniversary"])
        timeframe = st.selectbox("When?", 
            ["Within a week", "Within a month", "2-3 months away"])
        upcoming_events = f"{', '.join(events)} {timeframe}"
        
        if st.button("🔍 Get Personalized Recommendations"):
            with st.spinner("Generating culturally relevant recommendations..."):
                recommendations = recommender.get_recommendations(
                    user_profile=user_profile,
                    location=location,
                    upcoming_events=upcoming_events,
                    k=4
                )
                
                st.subheader("🌟 Recommended Products")
                st.write(recommendations["result"])
                
                st.subheader("📋 Product Details")
                for doc in recommendations["source_documents"]:
                    st.info(doc.page_content)
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your API key and try again.")
if __name__ == "__main__":
    main()