import streamlit as st
import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np
import json
import asyncio
from io import StringIO

class MarketMatcher:
    def __init__(self, api_key: str):  # Fixed: Changed _init_ to __init__
        """Initialize the MarketMatcher with Google AI API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.scaler = StandardScaler()
    
    def load_and_analyze_data(self, 
                            user_data_path: str, 
                            product_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Load user and product data from CSV files and analyze their columns"""
        # Load data
        user_df = pd.read_csv(user_data_path)
        product_df = pd.read_csv(product_data_path)
        
        # Analyze columns
        user_columns = self._analyze_columns(user_df)
        product_columns = self._analyze_columns(product_df)
        
        columns_analysis = {
            'user_columns': user_columns,
            'product_columns': product_columns,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return user_df, product_df, columns_analysis
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze dataframe columns for their characteristics and matching potential"""
        analysis = {}
        
        for column in df.columns:
            column_type = str(df[column].dtype)
            unique_values = df[column].nunique()
            null_percentage = (df[column].isnull().sum() / len(df)) * 100
            
            # Determine column characteristics
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            is_categorical = pd.api.types.is_categorical_dtype(df[column]) or (
                not is_numeric and unique_values / len(df) < 0.05
            )
            
            # Assign relevance score (1-10)
            relevance_score = self._calculate_relevance_score(
                is_numeric,
                is_categorical,
                unique_values,
                null_percentage
            )
            
            # Assign matching weight (0-1)
            matching_weight = relevance_score / 10.0
            
            analysis[column] = {
                'data_type': column_type,
                'unique_values': int(unique_values),
                'null_percentage': float(null_percentage),
                'is_numeric': is_numeric,
                'is_categorical': is_categorical,
                'relevance_score': relevance_score,
                'matching_weight': matching_weight
            }
        
        return analysis
    
    def _calculate_relevance_score(self,
                                 is_numeric: bool,
                                 is_categorical: bool,
                                 unique_values: int,
                                 null_percentage: float) -> int:
        """Calculate relevance score for a column based on its characteristics"""
        score = 7  # Base score
        
        # Adjust score based on characteristics
        if is_numeric:
            score += 1
        if is_categorical:
            score += 1
        if null_percentage > 50:
            score -= 3
        elif null_percentage > 20:
            score -= 1
        if unique_values < 2:
            score -= 2
        
        # Ensure score is between 1 and 10
        return max(1, min(10, score))
    
    def preprocess_data(self,
                       df: pd.DataFrame,
                       columns_analysis: Dict[str, Any],
                       data_type: str) -> pd.DataFrame:
        """Preprocess dataframe for matching analysis"""
        analysis_key = f'{data_type}_columns'
        relevant_columns = [
            col for col, info in columns_analysis[analysis_key].items()
            if info['relevance_score'] > 5
        ]
        
        processed_df = df[relevant_columns].copy()
        
        # Handle each column based on its characteristics
        for column in relevant_columns:
            col_info = columns_analysis[analysis_key][column]
            
            if col_info['is_numeric']:
                # Fill nulls with median and scale
                processed_df[column] = processed_df[column].fillna(
                    processed_df[column].median()
                )
                processed_df[column] = self.scaler.fit_transform(
                    processed_df[column].values.reshape(-1, 1)
                )
            else:
                # For categorical, fill nulls with mode and one-hot encode
                processed_df[column] = processed_df[column].fillna(
                    processed_df[column].mode()[0]
                )
                if col_info['is_categorical']:
                    dummies = pd.get_dummies(
                        processed_df[column],
                        prefix=column,
                        dummy_na=True
                    )
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    processed_df.drop(column, axis=1, inplace=True)
        
        return processed_df
    
    def _calculate_similarity(self,
                            user_vector: pd.Series,
                            product_vector: pd.Series,
                            user_weights: Dict[str, float],
                            product_weights: Dict[str, float]) -> float:
        """Calculate weighted similarity between user and product vectors"""
        # Normalize weights
        total_weight = sum(user_weights.values()) + sum(product_weights.values())
        normalized_weights = {
            **{k: v/total_weight for k, v in user_weights.items()},
            **{k: v/total_weight for k, v in product_weights.items()}
        }
        
        # Calculate weighted Euclidean distance
        weighted_diff_squared = 0
        for column in normalized_weights:
            if column in user_vector and column in product_vector:
                diff = user_vector[column] - product_vector[column]
                weighted_diff_squared += normalized_weights[column] * (diff ** 2)
        
        # Convert distance to similarity score (0-1)
        similarity = 1 / (1 + np.sqrt(weighted_diff_squared))
        return float(similarity)

    def find_user_matches(self,
                         user_id: int,
                         user_df: pd.DataFrame,
                         product_df: pd.DataFrame,
                         columns_analysis: Dict[str, Any],
                         top_n: int = 3) -> Optional[Dict[str, Any]]:
        """Find the best product matches for a specific user ID"""
        # Check if user exists
        user_row = user_df[user_df['user_id'] == user_id]
        if user_row.empty:
            return {
                "error": f"User ID {user_id} not found",
                "details": "Please provide a valid user ID"
            }
            
        # Preprocess both datasets
        processed_user_df = self.preprocess_data(user_df, columns_analysis, 'user')
        processed_product_df = self.preprocess_data(product_df, columns_analysis, 'product')
        
        # Get matching weights
        user_weights = {
            col: info['matching_weight']
            for col, info in columns_analysis['user_columns'].items()
            if info['relevance_score'] > 5
        }
        
        product_weights = {
            col: info['matching_weight']
            for col, info in columns_analysis['product_columns'].items()
            if info['relevance_score'] > 5
        }
        
        # Get user vector
        user_idx = user_df[user_df['user_id'] == user_id].index[0]
        user_vector = processed_user_df.iloc[user_idx]
        
        # Calculate similarity scores
        similarities = []
        for prod_idx in range(len(processed_product_df)):
            product_vector = processed_product_df.iloc[prod_idx]
            
            # Calculate weighted similarity
            similarity = self._calculate_similarity(
                user_vector,
                product_vector,
                user_weights,
                product_weights
            )
            
            similarities.append({
                'product_id': product_df.iloc[prod_idx]['product_id'],
                'similarity_score': similarity,
                'product_data': product_df.iloc[prod_idx].to_dict()
            })
        
        # Sort by similarity score and get top N
        top_matches = sorted(
            similarities,
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:top_n]
        
        return {
            'user_id': user_id,
            'user_data': user_row.iloc[0].to_dict(),
            'matches': top_matches
        }
    
    async def analyze_user_matches(self, 
                                 user_matches: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis for a specific user's matches"""
        analysis_prompt = f"""
        Analyze product matches for this specific user:
        User Profile: {user_matches}
        
        Provide a detailed analysis including:
        1. User profile summary
        2. Why these products match this user
        3. Key matching factors for each product
        4. Personalized marketing recommendations
        5. Potential cross-sell/upsell opportunities
        
        Return a valid JSON with this structure:
        {{
            "user_profile_summary": {{
                "key_characteristics": [],
                "buying_potential": "description",
                "market_segment": "segment_name"
            }},
            "product_matches": [
                {{
                    "product_id": "id",
                    "match_reasons": [],
                    "marketing_angles": [],
                    "timing_recommendation": "description"
                }}
            ],
            "marketing_recommendations": {{
                "primary_channel": "channel_name",
                "message_tone": "description",
                "key_features_to_highlight": [],
                "promotional_suggestions": []
            }},
            "cross_sell_opportunities": []
        }}
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            return json.loads(response.text.replace('```','').replace('json',''))
        except Exception as e:
            return {
                "error": "User match analysis failed",
                "details": str(e)
            }
    
    async def get_user_recommendations(self,
                                     user_id: int,
                                     user_data_path: str,
                                     product_data_path: str,
                                     top_n: int = 3) -> Dict[str, Any]:
        """Get personalized product recommendations and analysis for a specific user"""
        try:
            # Load and analyze data
            user_df, product_df, columns_analysis = self.load_and_analyze_data(
                user_data_path,
                product_data_path
            )

            # Find matches for specific user
            user_matches = self.find_user_matches(
                user_id,
                user_df,
                product_df,
                columns_analysis,
                top_n
            )
            
            if "error" in user_matches:
                return user_matches
            
            # Generate detailed analysis for user
            analysis = await self.analyze_user_matches(user_matches)
            
            return {
                "user_matches": user_matches,
                "detailed_analysis": analysis
            }
            
        except Exception as e:
            return {
                "error": "User recommendation pipeline failed",
                "details": str(e)
            }

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Streamlit UI
st.set_page_config(page_title="Market Matcher", page_icon="🎯", layout="wide")

st.title("🎯 Market Matcher - Product Marketing System")

# API Key Input
api_key = "AIzaSyC0PEcIDIkGpM0y34W4jPIKXvrJZQbPdA8"

# File Upload Section # Use margin for better vertical alignment
st.sidebar.title("Mind Market")
st.sidebar.header("Data Upload")
user_data_file = st.sidebar.file_uploader("Upload User Data CSV", type=['csv'])
product_data_file = st.sidebar.file_uploader("Upload Product Data CSV", type=['csv'])

if api_key and user_data_file and product_data_file:
    try:
        # Initialize matcher
        matcher = MarketMatcher(api_key=api_key)
        
        # Save uploaded files temporarily
        user_df = pd.read_csv(StringIO(user_data_file.getvalue().decode()))
        product_df = pd.read_csv(StringIO(product_data_file.getvalue().decode()))
        
        # Save files temporarily
        user_df.to_csv("temp_user_data.csv", index=False)
        product_df.to_csv("temp_product_data.csv", index=False)
        
        # Display available user IDs
        user_ids = user_df['user_id'].unique()
        selected_user = st.sidebar.selectbox("Select User ID", user_ids)
        
        # Number of recommendations
        top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
        
        if st.sidebar.button("Get Recommendations"):
            with st.spinner("Analyzing data and generating recommendations..."):
                # Create async function wrapper
                async def get_recommendations():
                    return await matcher.get_user_recommendations(
                        user_id=selected_user,
                        user_data_path="temp_user_data.csv",
                        product_data_path="temp_product_data.csv",
                        top_n=top_n
                    )
                
                # Run async function
                recommendations = asyncio.run(get_recommendations())
                formatted_data = convert_to_native(recommendations)
                
                # Display Results in Tabs
                tab1, tab2, tab3 = st.tabs(["User Profile", "Product Matches", "Marketing Analysis"])
                
                with tab1:
                    st.header("User Profile")
                    if "user_matches" in formatted_data:
                        user_data = formatted_data["user_matches"]["user_data"]
                        st.json(user_data)
                        
                        if "detailed_analysis" in formatted_data:
                            profile_summary = formatted_data["detailed_analysis"]["user_profile_summary"]
                            st.subheader("Profile Analysis")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Key Characteristics:")
                                for char in profile_summary["key_characteristics"]:
                                    st.write(f"• {char}")
                            
                            with col2:
                                st.write(f"Market Segment: {profile_summary['market_segment']}")
                                st.write(f"Buying Potential: {profile_summary['buying_potential']}")
                
                with tab2:
                    st.header("Product Matches")
                    if "user_matches" in formatted_data:
                        for idx, match in enumerate(formatted_data["user_matches"]["matches"]):
                            with st.expander(f"Match #{idx + 1} - Score: {match['similarity_score']:.2f}"):
                                st.json(match["product_data"])
                                
                                if "detailed_analysis" in formatted_data:
                                    product_analysis = next(
                                        (p for p in formatted_data["detailed_analysis"]["product_matches"] 
                                         if str(p["product_id"]) == str(match["product_id"])),
                                        None
                                    )
                                    
                                    if product_analysis:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("Match Reasons:")
                                            for reason in product_analysis["match_reasons"]:
                                                st.write(f"• {reason}")
                                        
                                        with col2:
                                            st.write("Marketing Angles:")
                                            for angle in product_analysis["marketing_angles"]:
                                                st.write(f"• {angle}")
                                            
                                        st.write(f"Timing: {product_analysis['timing_recommendation']}")
                
                with tab3:
                    st.header("Marketing Analysis")
                    if "detailed_analysis" in formatted_data:
                        marketing = formatted_data["detailed_analysis"]["marketing_recommendations"]
                        cross_sell = formatted_data["detailed_analysis"]["cross_sell_opportunities"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Marketing Recommendations")
                            st.write(f"Primary Channel: {marketing['primary_channel']}")
                            st.write(f"Message Tone: {marketing['message_tone']}")
                            
                            st.write("Key Features to Highlight:")
                            for feature in marketing['key_features_to_highlight']:
                                st.write(f"• {feature}")
                        
                        with col2:
                            st.subheader("Promotional Suggestions")
                            for promo in marketing['promotional_suggestions']:
                                st.write(f"• {promo}")
                            
                            st.subheader("Cross-Sell Opportunities")
                            for opportunity in cross_sell:
                                st.write(f"• {opportunity}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Welcome!, Enjoy this Service!!.")
    
    # Sample Data Format
    st.header("Expected Data Format")
    
    st.subheader("User Data CSV Format:")
    st.code("""
    user_id,age,income,location,preferences
    1,28,75000,New York,["tech","fashion"]
    2,35,95000,San Francisco,["sports","outdoor"]
    """)
    
    st.subheader("Product Data CSV Format:")
    st.code("""
    product_id,name,category,price,features
    101,Premium Laptop,Electronics,1200,["high-performance","lightweight"]
    102,Running Shoes,Sports,120,["cushioning","durability"]
    """)