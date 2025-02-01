import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import pickle

class RetailPredictionModel:
    def __init__(self):
        self.customer_features = ['age', 'income', 'family_size']
        self.outlet_features = ['commercial_viability', 'competition_score', 'luxury_index']
        self.location_features = ['commercial_score', 'public_transport_accessibility', 'population_density']
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Models
        self.outlet_preference_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.sales_prediction_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        self.customer_segment_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

    def load_and_preprocess_data(self):
        """Load and preprocess all datasets"""
        
        # Load datasets
        print("Loading datasets...")
        customers = pd.read_csv('customers.csv')
        outlets = pd.read_csv('integrated_outlets.csv')
        transactions = pd.read_csv('integrated_transactions.csv')
        locations = pd.read_csv('mumbai_locations.csv')
        
        # Encode categorical variables
        categorical_columns = {
            'customers': ['gender', 'occupation', 'marital_status', 'residential_area'],
            'outlets': ['category', 'brand', 'target_demographic', 'outlet_size', 'area_type'],
            'locations': ['primary_demographic', 'residential_density'],
            'transactions': ['payment_method', 'time_of_day', 'season']
        }
        
        for dataset_name, columns in categorical_columns.items():
            for col in columns:
                if col in locals()[dataset_name].columns:
                    self.label_encoders[f"{dataset_name}_{col}"] = LabelEncoder()
                    locals()[dataset_name][col] = self.label_encoders[f"{dataset_name}_{col}"].fit_transform(
                        locals()[dataset_name][col]
                    )
        
        # Convert dates to datetime and extract features
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        transactions['hour'] = transactions['transaction_date'].dt.hour
        transactions['day_of_week'] = transactions['transaction_date'].dt.dayofweek
        transactions['month'] = transactions['transaction_date'].dt.month
        
        return customers, outlets, transactions, locations

    def prepare_features(self, customers, outlets, transactions, locations):
        """Prepare feature matrices for training"""
        
        print("Preparing features...")
        
        # Merge datasets
        transaction_features = transactions.merge(
            customers[['customer_id'] + self.customer_features],
            on='customer_id',
            how='left'
        ).merge(
            outlets[['outlet_id'] + self.outlet_features],
            on='outlet_id',
            how='left'
        ).merge(
            locations[['location_id'] + self.location_features],
            on='location_id',
            how='left'
        )
        
        # Create feature matrices
        X_features = transaction_features[
            self.customer_features +
            self.outlet_features +
            self.location_features +
            ['hour', 'day_of_week', 'month', 'weekend_flag']
        ]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        return X_scaled, transaction_features

    def train_models(self, X, transaction_features):
        """Train all prediction models"""
        
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            transaction_features['outlet_id'],
            test_size=0.2,
            random_state=42
        )
        
        # Train outlet preference model
        print("\nTraining outlet preference model...")
        self.outlet_preference_model.fit(X_train, y_train)
        y_pred = self.outlet_preference_model.predict(X_test)
        print("\nOutlet Preference Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Train sales prediction model
        print("\nTraining sales prediction model...")
        X_sales_train, X_sales_test, y_sales_train, y_sales_test = train_test_split(
            X,
            transaction_features['purchase_amount'],
            test_size=0.2,
            random_state=42
        )
        
        self.sales_prediction_model.fit(X_sales_train, y_sales_train)
        y_sales_pred = self.sales_prediction_model.predict(X_sales_test)
        sales_mse = mean_squared_error(y_sales_test, y_sales_pred)
        sales_r2 = r2_score(y_sales_test, y_sales_pred)
        
        print("\nSales Prediction Model Performance:")
        print(f"Mean Squared Error: {sales_mse:.2f}")
        print(f"R² Score: {sales_r2:.2f}")
        
        # Train customer segmentation model
        print("\nTraining customer segmentation model...")
        customer_features = transaction_features[self.customer_features].drop_duplicates()
        n_clusters = 4  # Number of customer segments
        self.customer_segment_model.fit(
            customer_features,
            pd.qcut(customer_features['income'], n_clusters, labels=range(n_clusters))
        )
        
        return X_test, y_test, X_sales_test, y_sales_test

    def analyze_feature_importance(self, feature_names):
        """Analyze and print feature importance"""
        
        print("\nFeature Importance Analysis:")
        
        # Outlet preference model feature importance
        outlet_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.outlet_preference_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features for outlet preference:")
        print(outlet_importance.head())
        
        # Sales prediction model feature importance
        sales_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.sales_prediction_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features for sales prediction:")
        print(sales_importance.head())

    def save_models(self):
        """Save trained models and preprocessors"""
        
        print("\nSaving models...")
        
        # Save models
        with open('outlet_preference_model.pkl', 'wb') as f:
            pickle.dump(self.outlet_preference_model, f)
        
        with open('sales_prediction_model.pkl', 'wb') as f:
            pickle.dump(self.sales_prediction_model, f)
        
        with open('customer_segment_model.pkl', 'wb') as f:
            pickle.dump(self.customer_segment_model, f)
        
        # Save preprocessors
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Models and preprocessors saved successfully!")

    def make_predictions(self, X_test):
        """Make predictions using trained models"""
        
        predictions = {
            'outlet_preferences': self.outlet_preference_model.predict(X_test),
            'predicted_sales': self.sales_prediction_model.predict(X_test),
            'customer_segments': self.customer_segment_model.predict(
                X_test[:, :len(self.customer_features)]
            )
        }
        
        return predictions

    def generate_recommendations(self, location_data, customer_data):
        """Generate store location recommendations"""
        
        print("\nGenerating Location Recommendations...")
        
        recommendations = []
        for _, location in location_data.iterrows():
            # Calculate location score based on multiple factors
            # Calculate normalized population density score
            pop_density_score = location['population_density'] / location_data['population_density'].max()
            
            demographic_score = location['commercial_score'] * 0.4 + \
                              location['public_transport_accessibility'] * 0.3 + \
                              pop_density_score * 0.3
            
            # Calculate potential customer base
            potential_customers = len(customer_data[
                (customer_data['residential_area'] == location['primary_demographic']) &
                (customer_data['income'] > 50000)  # Adjust threshold as needed
            ])
            
            recommendations.append({
                'location_id': location['location_id'],
                'area_name': location['area_name'],
                'demographic_score': demographic_score,
                'potential_customers': potential_customers,
                'recommendation_score': demographic_score * np.log1p(potential_customers)
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df.sort_values('recommendation_score', ascending=False)

# Main execution
if __name__ == "__main__":
    # Initialize model
    model = RetailPredictionModel()
    
    # Load and preprocess data
    customers, outlets, transactions, locations = model.load_and_preprocess_data()
    
    # Prepare features
    X, transaction_features = model.prepare_features(customers, outlets, transactions, locations)
    
    # Train models
    X_test, y_test, X_sales_test, y_sales_test = model.train_models(X, transaction_features)
    
    # Analyze feature importance
    feature_names = (
        model.customer_features +
        model.outlet_features +
        model.location_features +
        ['hour', 'day_of_week', 'month', 'weekend_flag']
    )
    model.analyze_feature_importance(feature_names)
    
    # Generate location recommendations
    recommendations = model.generate_recommendations(locations, customers)
    print("\nTop 5 Recommended Locations:")
    print(recommendations.head())
    
    # Save models
    model.save_models()
    
    print("\nTraining complete! Models are ready for predictions.")