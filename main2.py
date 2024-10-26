import os
from typing import List, Dict
import pandas as pd
import chromadb
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Updated import
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Sample product dataset
def create_sample_dataset():
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
        ]
    }
    return pd.DataFrame(data)

class ProductRecommender:
    def __init__(self):
        # if not os.getenv("GOOGLE_API_KEY"):
        #     raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        # Initialize Google Generative AI
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare and index the product data"""
        # Combine product details and target audience for better context
        documents = [
            f"Product: {row['product_details']}\nTarget Audience: {row['target_audience']}"
            for _, row in df.iterrows()
        ]
        
        # Create vector store
        self.vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
    def get_recommendations(self, user_profile: str, upcoming_events: str, k: int = 4) -> Dict:
        """Get product recommendations based on user profile and upcoming events"""
        # Create query template
        query = f"""
        Based on the following user profile and upcoming events, suggest the most relevant products:
        
        User Profile: {user_profile}
        Upcoming Events: {upcoming_events}
        
        Analyze the product details and target audience carefully to make personalized recommendations.
        Explain why each recommended product would be suitable for this user.
        """
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get recommendations
        try:
            result = qa_chain({"query": query})
            return result
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return {"result": "Error generating recommendations", "source_documents": []}

def main():
    st.title("🎁 Smart Product Recommender for Marketing")
    
    # Check for API key
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     st.error("Please set your GOOGLE_API_KEY in the .env file")
    #     return
        
    try:
        # Initialize recommender
        recommender = ProductRecommender()
        
        # Create and index sample dataset
        df = create_sample_dataset()
        
        with st.spinner("Preparing product data..."):
            recommender.prepare_data(df)
        
        # User input
        st.subheader("👤 User Profile")
        user_profile = st.text_area(
            "Enter user preferences and interests:",
            "25-year-old working professional, interested in technology and fitness, enjoys outdoor activities"
        )
        
        st.subheader("📅 Upcoming Events")
        upcoming_events = st.text_area(
            "Enter upcoming events/festivals in the next month:",
            "Diwali festival in 3 weeks, office party next week"
        )
        
        if st.button("🔍 Get Recommendations"):
            with st.spinner("Generating personalized recommendations..."):
                # Get recommendations
                recommendations = recommender.get_recommendations(
                    user_profile=user_profile,
                    upcoming_events=upcoming_events
                )
                
                # Display recommendations
                st.subheader("🌟 Recommended Products")
                st.write(recommendations["result"])
                
                # Display source documents
                st.subheader("📋 Matching Products Details")
                for doc in recommendations["source_documents"]:
                    st.info(doc.page_content)
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your API key and try again.")

if __name__ == "__main__":
    main()