import os
from typing import List, Dict
from datetime import datetime, timedelta
import random
import pandas as pd
import chromadb
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

def generate_upcoming_events(location: str, culture: str) -> List[Dict]:
    """Generate relevant upcoming events based on location and culture"""
    # Define cultural and regional events
    events_database = {
        'Indian': {
            'Major Festivals': [
                ('Diwali', 'October-November'),
                ('Holi', 'March'),
                ('Navratri', 'September-October'),
                ('Durga Puja', 'September-October'),
                ('Ganesh Chaturthi', 'August-September')
            ],
            'Regional': {
                'North': ['Lohri', 'Baisakhi', 'Karva Chauth'],
                'South': ['Pongal', 'Onam', 'Ugadi'],
                'West': ['Gudi Padwa', 'Janmashtami'],
                'East': ['Bihu', 'Chhath Puja']
            }
        },
        'Western': {
            'Major Festivals': [
                ('Christmas', 'December'),
                ('New Year', 'December-January'),
                ('Halloween', 'October'),
                ('Thanksgiving', 'November'),
                ('Easter', 'March-April')
            ]
        },
        'Common': [
            'Birthday',
            'Anniversary',
            'House Warming',
            'Graduation',
            'Wedding Season',
            'Back to School',
            'Summer Vacation',
            'Winter Break',
            'Shopping Festival',
            'Corporate Events'
        ]
    }
    
    current_date = datetime.now()
    upcoming_events = []
    
    # Add culture-specific events
    if culture in events_database:
        for festival, month in events_database[culture]['Major Festivals']:
            upcoming_events.append({
                'event': festival,
                'type': 'Cultural Festival',
                'importance': 'High'
            })
            
        # Add regional events if applicable
        if 'Regional' in events_database[culture]:
            region = location.split('-')[0] if '-' in location else 'North'
            if region in events_database[culture]['Regional']:
                for event in events_database[culture]['Regional'][region]:
                    upcoming_events.append({
                        'event': event,
                        'type': 'Regional Festival',
                        'importance': 'Medium'
                    })
    
    # Add common events
    for event in random.sample(events_database['Common'], 3):
        upcoming_events.append({
            'event': event,
            'type': 'Common',
            'importance': 'Medium'
        })
    
    return upcoming_events

def create_expanded_dataset():
    """Create an expanded product dataset with more variety and details"""
    # Product categories
    categories = {
        'Electronics': {
            'brands': ['TechPro', 'SmartLife', 'DigiMax', 'EliteGear', 'InnovateX'],
            'price_ranges': ['Budget', 'Mid-range', 'Premium', 'Luxury']
        },
        'Fashion': {
            'brands': ['StyleVogue', 'EthnicElegance', 'ModernWear', 'LuxeFashion', 'TrendSetters'],
            'price_ranges': ['Affordable', 'Premium', 'Designer', 'Luxury']
        },
        'Home & Living': {
            'brands': ['HomeCraft', 'LivingLux', 'ElegantSpaces', 'HomeHarmony', 'SpaceStyle'],
            'price_ranges': ['Value', 'Premium', 'Luxury', 'Ultra-Luxury']
        },
        'Beauty & Personal Care': {
            'brands': ['GlowUp', 'NatureCare', 'BeautyBliss', 'OrganicLife', 'LuxeBeauty'],
            'price_ranges': ['Drugstore', 'Premium', 'Luxury', 'Professional']
        }
    }

    products = []
    product_id = 1

    # Electronics
    electronics_products = [
        {
            'name': 'Smart Watch',
            'variants': ['Fitness Tracker', 'Premium Watch', 'Sport Edition', 'Limited Edition'],
            'features': ['Heart Rate Monitor', 'Sleep Tracking', 'GPS', 'Water Resistant'],
            'occasions': ['Fitness Goals', 'Tech Gifts', 'Professional Use', 'Active Lifestyle']
        },
        {
            'name': 'Wireless Earbuds',
            'variants': ['Sport Edition', 'Premium Audio', 'Active Noise Cancelling', 'Basic'],
            'features': ['Bluetooth 5.0', 'Water Resistant', 'Touch Controls', 'Long Battery Life'],
            'occasions': ['Music Lovers', 'Workout', 'Travel', 'Daily Commute']
        },
        # Add 20 more electronics products...
    ]

    # Fashion
    fashion_products = [
        {
            'name': 'Traditional Wear',
            'variants': ['Saree', 'Lehenga', 'Kurta Set', 'Ethnic Fusion'],
            'features': ['Handcrafted', 'Designer', 'Festival Special', 'Wedding Collection'],
            'occasions': ['Festivals', 'Weddings', 'Cultural Events', 'Special Occasions']
        },
        {
            'name': 'Western Wear',
            'variants': ['Casual', 'Formal', 'Party Wear', 'Business'],
            'features': ['Trendy', 'Comfortable', 'Premium Fabric', 'Designer'],
            'occasions': ['Office Wear', 'Party', 'Casual Outings', 'Special Events']
        },
        # Add 20 more fashion products...
    ]

    # Generate 100 products
    for category, details in categories.items():
        base_products = electronics_products if category == 'Electronics' else fashion_products
        
        for _ in range(25):  # 25 products per category
            base_product = random.choice(base_products)
            brand = random.choice(details['brands'])
            price_range = random.choice(details['price_ranges'])
            variant = random.choice(base_product['variants'])
            features = random.sample(base_product['features'], random.randint(2, len(base_product['features'])))
            occasions = random.sample(base_product['occasions'], random.randint(2, len(base_product['occasions'])))
            
            product_details = f"{brand} {base_product['name']} - {variant} ({price_range}). Features: {', '.join(features)}."
            target_audience = f"Ideal for {', '.join(occasions)}. Suitable for {price_range} segment customers who value {random.choice(features)}."
            
            products.append({
                'product_id': product_id,
                'category': category,
                'brand': brand,
                'product_details': product_details,
                'target_audience': target_audience,
                'price_range': price_range,
                'cultural_relevance': random.choice(['Universal', 'Indian', 'Western', 'Asian', 'Middle Eastern'])
            })
            
            product_id += 1

    return pd.DataFrame(products)

class EnhancedProductRecommender:
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
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare and index the product data"""
        documents = [
            f"Category: {row['category']}\nBrand: {row['brand']}\nProduct: {row['product_details']}\n"
            f"Target Audience: {row['target_audience']}\nPrice Range: {row['price_range']}\n"
            f"Cultural Relevance: {row['cultural_relevance']}"
            for _, row in df.iterrows()
        ]
        
        self.vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
    def get_recommendations(self, user_profile: Dict, upcoming_events: List[Dict], k: int = 2) -> Dict:
        """Get product recommendations based on enhanced user profile and upcoming events"""
        # Create detailed query
        events_str = "\n".join([f"- {event['event']} ({event['type']}, Importance: {event['importance']})" 
                              for event in upcoming_events])
        
        query = f"""
        Please analyze the following user profile and upcoming events to suggest the most relevant products:
        
        User Profile:
        - Age: {user_profile['age']}
        - Location: {user_profile['location']}
        - Culture: {user_profile['culture']}
        - Interests: {user_profile['interests']}
        - Budget Preference: {user_profile['budget']}
        
        Upcoming Events:
        {events_str}
        
        Please recommend {k} specific products that:
        1. Match the user's cultural background and location
        2. Fit their budget preference
        3. Are relevant to their interests and upcoming events
        4. Consider seasonal and festival-specific needs
        
        For each recommendation, explain:
        - Why it's suitable for this specific user
        - How it relates to their upcoming events
        - Why it's culturally appropriate
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

def main():
    st.title("🎁 Enhanced Smart Product Recommender")
    
    try:
        recommender = EnhancedProductRecommender()
        
        # Create and index expanded dataset
        df = create_expanded_dataset()
        
        with st.spinner("Preparing product data..."):
            recommender.prepare_data(df)
        
        # Enhanced user profile inputs
        st.subheader("👤 User Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age:", min_value=15, max_value=100, value=25)
            location = st.selectbox("Location:", [
                "North-India", "South-India", "East-India", "West-India",
                "USA", "Europe", "Middle East", "Southeast Asia"
            ])
            
        with col2:
            culture = st.selectbox("Cultural Background:", [
                "Indian", "Western", "Asian", "Middle Eastern", "Mixed"
            ])
            budget = st.selectbox("Budget Preference:", [
                "Budget", "Mid-range", "Premium", "Luxury"
            ])
        
        interests = st.text_area(
            "Interests and Preferences:",
            "technology, fitness, fashion, reading"
        )
        
        # Generate upcoming events automatically
        upcoming_events = generate_upcoming_events(location, culture)
        
        st.subheader("📅 Upcoming Events")
        st.write("Automatically detected events based on your profile:")
        for event in upcoming_events:
            st.write(f"• {event['event']} - {event['type']} (Importance: {event['importance']})")
        
        user_profile = {
            "age": age,
            "location": location,
            "culture": culture,
            "interests": interests,
            "budget": budget
        }
        
        if st.button("🔍 Get Personalized Recommendations"):
            with st.spinner("Generating personalized recommendations..."):
                recommendations = recommender.get_recommendations(
                    user_profile=user_profile,
                    upcoming_events=upcoming_events
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