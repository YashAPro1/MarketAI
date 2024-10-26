# data_indexer.py
import os
from typing import List, Dict
import pandas as pd
import chromadb
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_enhanced_dataset():
    # Your existing create_enhanced_dataset function remains the same
    data = {
        'product_id': range(1, 16),
        'product_name': [
            "Premium Smart Watch",
            "Banarasi Silk Saree",
            "Ayurvedic Beauty Kit",
            "Gaming Laptop",
            "Temple Design Jewelry",
            "Smart Home System",
            "South Indian Coffee Set",
            "Yoga Equipment Kit",
            "Designer Kurti Collection",
            "Educational Tablet",
            "Traditional Dhoti Set",
            "Gujarati Thali Set",
            "Kerala Massage Oil Kit",
            "Bengali Festival Wear",
            "Rajasthani Home Decor"
        ],
        'product_details': [
            "Premium Smart Watch with fitness tracking, heart monitoring, and local language support. Features festival calendar and prayer time notifications.",
            "Authentic Banarasi Silk Saree with traditional motifs and modern designs. Pure silk with real zari work, perfect for ceremonies.",
            "Natural Ayurvedic Beauty Kit with ingredients sourced from Kerala. Contains kumkumadi oil, ubtan, and herbal face masks.",
            "High-performance Gaming Laptop with multilingual keyboard support and pre-installed Indian gaming content.",
            "22K Gold Temple Design Jewelry Set inspired by South Indian architecture. Includes necklace, earrings, and bangles.",
            "Smart Home System with Indian voltage compatibility, local language support, and festival mode lighting.",
            "Premium South Indian Coffee Making Set with traditional brass filter, tumbler, and aromatic coffee powder.",
            "Complete Yoga Equipment Kit with mat, blocks, straps, and meditation cushion. Includes guided yoga videos in regional languages.",
            "Designer Kurti Collection featuring regional embroidery styles - Lucknowi, Gujarati, and Rajasthani work.",
            "Educational Tablet with content in 12 Indian languages, CBSE/ICSE curriculum support.",
            "Handwoven Cotton Dhoti Set with Angavastram, perfect for traditional ceremonies and temple visits.",
            "Authentic Gujarati Thali Set in copper, with compartments for all traditional dishes and accompaniments.",
            "Traditional Kerala Ayurvedic Massage Oil Kit with wooden massagers and instruction manual.",
            "Bengali Festival Wear Collection including kurtas, dhotis, and sarees suitable for Durga Puja.",
            "Hand-painted Rajasthani Home Decor Set including wall plates, puppets, and mirror work cushions."
        ],
        'target_audience': [
            "Tech-savvy professionals in metro cities, age 25-40, health-conscious",
            "Traditional women in North India, cultural events, age 25-60, luxury segment",
            "Natural beauty enthusiasts, Kerala/South Indian audience, age 20-45",
            "Student gamers and tech enthusiasts, tier 1-2 cities, age 15-30",
            "South Indian families, wedding shoppers, traditional jewelry lovers",
            "Modern Indian families in apartments, tech-adopters, metro cities",
            "South Indian coffee lovers, traditional households, premium segment",
            "Yoga practitioners, health-conscious individuals, spiritual audience",
            "Working women, fashion-conscious, tier 1-2 cities, age 20-45",
            "K-12 students, urban and semi-urban families, educational focus",
            "Traditional men, temple visitors, South Indian demographic",
            "Gujarati families, traditional cooking enthusiasts, gift segment",
            "Ayurveda followers, Kerala demographic, wellness focused",
            "Bengali community, cultural event participants, all age groups",
            "Home decor enthusiasts, art lovers, premium segment"
        ],
        'regional_relevance': [
            "Pan-India with multi-language support",
            "North India, especially UP, Bihar, MP",
            "Kerala and South Indian markets",
            "Pan-India urban markets",
            "South India, especially TN, Karnataka",
            "Metro cities across India",
            "South India, coffee-drinking regions",
            "Pan-India with spiritual significance",
            "Pan-India with regional design elements",
            "Pan-India educational market",
            "South India traditional market",
            "Gujarat and Western India",
            "Kerala and Ayurveda followers",
            "Bengal and Eastern India",
            "Rajasthan and North India"
        ],
        'cultural_occasions': [
            "Daily use, Diwali, New Year",
            "Weddings, Diwali, Karva Chauth",
            "Year-round, Wedding season",
            "Diwali, New Year, Exam season",
            "Weddings, Temple festivals, Akshaya Tritiya",
            "Griha Pravesh, Diwali",
            "Daily use, House warming",
            "Daily use, International Yoga Day",
            "Festival season, Daily wear",
            "Academic year start, Children's Day",
            "Temple festivals, Traditional events",
            "House warming, Gujarati New Year",
            "Monsoon season, Wedding gifts",
            "Durga Puja, Bengali New Year",
            "House warming, Diwali decoration"
        ]
    }
    return pd.DataFrame(data)

def create_vector_store():
    """Create and persist the vector store"""
    print("Creating vector store...")
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
    )
    
    # Get dataset
    df = create_enhanced_dataset()
    
    # Prepare documents
    documents = [
        f"""Product: {row['product_name']}
        Details: {row['product_details']}
        Target Audience: {row['target_audience']}
        Regional Relevance: {row['regional_relevance']}
        Cultural Occasions: {row['cultural_occasions']}"""
        for _, row in df.iterrows()
    ]
    
    # Create and persist vector store
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Persist the vector store
    vectorstore.persist()
    print("Vector store created and persisted successfully!")
    
if __name__ == "__main__":
    create_vector_store()