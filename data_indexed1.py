import os
import pandas as pd
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_sample_dataset() -> pd.DataFrame:
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
        'price_range': [
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
        'category': [
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

def index_products():
    """Index the product data into the vector store"""
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key="AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"
        )
        
        # Get product data
        df = create_sample_dataset()
        
        # Prepare documents for indexing
        documents = [
            f"Product: {row['product_details']}\n"
            f"Target Audience: {row['target_audience']}\n"
            f"Price Range: {row['price_range']}\n"
            f"Category: {row['category']}"
            for _, row in df.iterrows()
        ]
        
        # Create and persist vector store
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"Successfully indexed {len(documents)} products!")
        return True
        
    except Exception as e:
        print(f"Error indexing products: {str(e)}")
        return False

if __name__ == "__main__":
    success = index_products()
    if success:
        print("Product indexing completed successfully!")
    else:
        print("Failed to index products. Please check the error messages above.")