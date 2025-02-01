import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
from typing import Dict, List, Tuple

class RetailIntelligenceModel:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        """
        Initialize the Retail Intelligence Model with MongoDB connection
        """
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        
        # Preprocessing components
        self.preprocessor = None
        self.sales_predictor = None
        self.recommendation_model = None
        self.store_placement_model = None

    def prepare_customer_features(self) -> pd.DataFrame:
        """
        Extract and prepare customer features for modeling
        """
        # Fetch customer and transaction data
        customers = list(self.db.customers.find())
        transactions = list(self.db.transactions.find())
        
        # Transform data into a DataFrame
        customer_features = []
        for customer in customers:
            # Aggregate customer transaction data
            customer_txns = [
                txn for txn in transactions 
                if txn['customer_id'] == customer['customer_id']
            ]
            
            features = {
                'customer_id': customer['customer_id'],
                'age': customer['personal_info']['age'],
                'gender': customer['personal_info']['gender'],
                'loyalty_tier': customer['loyalty_info']['membership_tier'],
                'total_transactions': len(customer_txns),
                'total_spend': sum(
                    txn['payment']['final_amount'] 
                    for txn in customer_txns
                ),
                'avg_transaction_value': sum(
                    txn['payment']['final_amount'] 
                    for txn in customer_txns
                ) / (len(customer_txns) or 1),
                'preferred_categories': ','.join(
                    customer['shopping_preferences']['preferred_categories']
                ),
                'preferred_payment_methods': ','.join(
                    customer['shopping_preferences']['preferred_payment_methods']
                )
            }
            customer_features.append(features)
        
        return pd.DataFrame(customer_features)

    def prepare_store_features(self) -> pd.DataFrame:
        """
        Extract and prepare store features for placement model
        """
        stores = list(self.db.stores.find())
        transactions = list(self.db.transactions.find())
        
        store_features = []
        for store in stores:
            # Find transactions for this store
            store_txns = [
                txn for txn in transactions 
                if txn['store_id'] == store['store_id']
            ]
            
            features = {
                'store_id': store['store_id'],
                'state': store['location']['state'],
                'city': store['location']['city'],
                'category': "Hypermarket",
                'total_store_revenue': sum(
                    txn['payment']['final_amount'] 
                    for txn in store_txns
                ),
                'transaction_count': len(store_txns),
                'avg_transaction_value': sum(
                    txn['payment']['final_amount'] 
                    for txn in store_txns
                ) / (len(store_txns) or 1),
                'latitude': store['location']['coordinates']['latitude'],
                'longitude': store['location']['coordinates']['longitude']
            }
            store_features.append(features)
        
        return pd.DataFrame(store_features)

    def build_customer_segmentation_model(self):
        """
        Build a model for customer segmentation and behavior prediction
        """
        # Prepare data
        df = self.prepare_customer_features()
        
        # Prepare features and target
        features = ['age', 'total_transactions', 'total_spend', 'avg_transaction_value']
        categorical_features = ['gender', 'loyalty_tier']
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Clustering model for customer segmentation
        from sklearn.cluster import KMeans
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clusterer', KMeans(n_clusters=4, random_state=42))
        ])
        
        # Fit the model
        pipeline.fit(df[features + categorical_features])
        
        # Add cluster labels to DataFrame
        df['customer_segment'] = pipeline.named_steps['clusterer'].labels_
        
        return pipeline, df

    def build_sales_prediction_model(self):
        """
        Build a model to predict sales based on store and customer features
        """
        # Prepare store features
        store_df = self.prepare_store_features()
        
        # Prepare features
        features = [
            'latitude', 'longitude', 
            'transaction_count', 'avg_transaction_value'
        ]
        categorical_features = ['state', 'city', 'category']
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Regression model for sales prediction
        sales_predictor = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Prepare target (total store revenue)
        X = store_df[features + categorical_features]
        y = store_df['total_store_revenue']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        sales_predictor.fit(X_train, y_train)
        
        return sales_predictor, X_test, y_test

    def build_recommendation_system(self):
        """
        Build a recommendation system based on customer and product data
        """
        # Fetch transaction data
        transactions = list(self.db.transactions.find())
        products = list(self.db.products.find())
        
        # Create product-customer interaction matrix
        interaction_data = []
        for txn in transactions:
            for item in txn['items']:
                interaction_data.append({
                    'customer_id': txn['customer_id'],
                    'product_id': item['product_id'],
                    'quantity': item['quantity']
                })
        
        interaction_df = pd.DataFrame(interaction_data)
        
        # Simple collaborative filtering approach
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create customer-product interaction matrix
        customer_product_matrix = interaction_df.pivot_table(
            index='customer_id', 
            columns='product_id', 
            values='quantity', 
            fill_value=0
        )
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(customer_product_matrix)
        
        return {
            'interaction_matrix': customer_product_matrix,
            'similarity_matrix': similarity_matrix
        }

    def recommend_products(self, customer_id: str, top_n: int = 5):
        """
        Generate product recommendations for a specific customer
        """
        # Fetch transaction data
        transactions = list(self.db.transactions.find())
        
        # Create interaction DataFrame
        interaction_data = []
        for txn in transactions:
            for item in txn['items']:
                interaction_data.append({
                    'customer_id': txn['customer_id'],
                    'product_id': item['product_id'],
                    'quantity': item['quantity']
                })
        
        interaction_df = pd.DataFrame(interaction_data)
        
        # Load recommendation system
        rec_system = self.build_recommendation_system()
        
        # Check if customer exists in interaction matrix
        if customer_id not in rec_system['interaction_matrix'].index:
            return []
        
        # Find customer index
        customer_index = rec_system['interaction_matrix'].index.get_loc(customer_id)
        
        # Get similar customers
        similar_customers = rec_system['similarity_matrix'][customer_index]
        
        # Find top similar customers
        top_similar_customers = similar_customers.argsort()[::-1][1:6]
        
        # Aggregate recommended products
        recommended_product_ids = set()
        for idx in top_similar_customers:
            similar_customer_id = rec_system['interaction_matrix'].index[idx]
            customer_products = interaction_df[
                interaction_df['customer_id'] == similar_customer_id
            ]['product_id'].unique()
            
            recommended_product_ids.update(customer_products)
        
        return list(recommended_product_ids)[:top_n]

    def predict_store_performance(self, store_features: Dict) -> float:
        """
        Predict store sales performance based on features
        """
        # Use the sales prediction model to forecast store performance
        sales_predictor, X_test, y_test = self.build_sales_prediction_model()
        
        # Prepare input features
        input_features = pd.DataFrame([store_features])
        
        # Predict sales
        predicted_sales = sales_predictor.predict(input_features)
        
        return predicted_sales[0]

    def generate_store_placement_recommendations(self, num_recommendations: int = 5):
        """
        Generate recommendations for new store locations
        """
        # Analyze existing stores and their performance
        store_df = self.prepare_store_features()
        
        # Clustering to identify high-potential areas
        from sklearn.cluster import KMeans
        
        # Use geographical and performance features
        placement_features = ['latitude', 'longitude', 'total_store_revenue']
        X = store_df[placement_features]
        
        # Cluster stores
        kmeans = KMeans(n_clusters=num_recommendations, random_state=42)
        store_df['cluster'] = kmeans.fit_predict(X)
        
        # Find cluster centroids with highest potential
        cluster_performance = store_df.groupby('cluster')['total_store_revenue'].mean()
        top_clusters = cluster_performance.nlargest(num_recommendations)
        
        # Generate recommendations
        recommendations = []
        for cluster in top_clusters.index:
            cluster_stores = store_df[store_df['cluster'] == cluster]
            centroid = kmeans.cluster_centers_[cluster]
            
            recommendations.append({
                'latitude': centroid[0],
                'longitude': centroid[1],
                'estimated_revenue': cluster_performance[cluster],
                'nearby_cities': cluster_stores['city'].unique().tolist()
            })
        
        return recommendations

    def save_models(self, path_prefix: str = 'retail_intelligence_'):
        """
        Save trained models for future use
        """
        # Save customer segmentation model
        joblib.dump(
            self.build_customer_segmentation_model()[0], 
            f'{path_prefix}customer_segmentation.joblib'
        )
        
        # Save sales prediction model
        sales_predictor, _, _ = self.build_sales_prediction_model()
        joblib.dump(sales_predictor, f'{path_prefix}sales_predictor.joblib')

def main():
    # Initialize and train the model
    retail_model = RetailIntelligenceModel()
    
    # Demonstrate key functionalities
    print("1. Customer Segmentation:")
    _, customer_segments = retail_model.build_customer_segmentation_model()
    print(customer_segments['customer_segment'].value_counts())
    
    print("\n2. Sales Prediction Model:")
    sales_predictor, X_test, y_test = retail_model.build_sales_prediction_model()
    print(f"Model Score: {sales_predictor.score(X_test, y_test)}")
    
    print("\n3. Store Placement Recommendations:")
    store_recommendations = retail_model.generate_store_placement_recommendations()
    for i, rec in enumerate(store_recommendations, 1):
        print(f"Recommendation {i}:")
        print(f"  Location: {rec['latitude']}, {rec['longitude']}")
        print(f"  Estimated Revenue: ₹{rec['estimated_revenue']:,.2f}")
        print(f"  Nearby Cities: {', '.join(rec['nearby_cities'])}")
    
    # Save models for future use
    retail_model.save_models()

if __name__ == "__main__":
    main()