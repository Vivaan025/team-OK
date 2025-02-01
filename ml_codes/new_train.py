import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from collections import defaultdict
import joblib
import warnings
warnings.filterwarnings('ignore')

class RetailPredictor:
    def __init__(self, mongodb_uri='mongodb://localhost:27017/'):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['indian_retail']
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None

    def prepare_features(self):
        """Prepare features with domain-specific knowledge"""
        print("Fetching data...")
        
        # Fetch data
        customers = list(self.db.customers.find())
        transactions = list(self.db.transactions.find())
        stores = list(self.db.stores.find())
        products = list(self.db.products.find())
        
        # Create mappings
        store_categories = {s['store_id']: s['category'] for s in stores}
        store_details = {s['store_id']: s for s in stores}
        product_categories = {p['product_id']: p['category'] for p in products}
        
        print("Processing features...")
        features_list = []
        
        for customer in customers:
            customer_id = customer['customer_id']
            customer_txns = [t for t in transactions if t['customer_id'] == customer_id]
            
            if not customer_txns:
                continue
            
            # Sort transactions by date
            customer_txns.sort(key=lambda x: pd.to_datetime(x['date']))
            
            # Get the last N transactions for recent behavior
            recent_txns = customer_txns[-10:] if len(customer_txns) > 10 else customer_txns
            
            # Basic customer info
            features = {
                'customer_id': customer_id,
                'age': customer['personal_info']['age'],
                'gender': customer['personal_info']['gender'],
                'city': customer['personal_info']['address']['city'],
                'state': customer['personal_info']['address']['state'],
                'membership_tier': customer['loyalty_info']['membership_tier'],
            }
            
            # Recent transaction patterns
            recent_amounts = [t['payment']['final_amount'] for t in recent_txns]
            features.update({
                'recent_avg_amount': np.mean(recent_amounts),
                'recent_max_amount': max(recent_amounts),
                'recent_min_amount': min(recent_amounts),
                'recent_std_amount': np.std(recent_amounts) if len(recent_amounts) > 1 else 0
            })
            
            # Store category frequencies (recent)
            recent_store_visits = [store_categories[t['store_id']] for t in recent_txns]
            store_freq = pd.Series(recent_store_visits).value_counts()
            total_visits = len(recent_store_visits)
            
            for store_type in set(store_categories.values()):
                features[f'recent_{store_type.lower()}_ratio'] = (
                    store_freq.get(store_type, 0) / total_visits
                )
            
            # Product category preferences (recent)
            category_amounts = defaultdict(float)
            category_quantities = defaultdict(int)
            total_amount = 0
            total_items = 0
            
            for txn in recent_txns:
                for item in txn['items']:
                    category = product_categories[item['product_id']]
                    amount = item['quantity'] * item['final_price']
                    category_amounts[category] += amount
                    category_quantities[category] += item['quantity']
                    total_amount += amount
                    total_items += item['quantity']
            
            for category in set(product_categories.values()):
                if total_amount > 0:
                    features[f'recent_{category.lower()}_spend_ratio'] = (
                        category_amounts[category] / total_amount
                    )
                    features[f'recent_{category.lower()}_quantity_ratio'] = (
                        category_quantities[category] / total_items if total_items > 0 else 0
                    )
                else:
                    features[f'recent_{category.lower()}_spend_ratio'] = 0
                    features[f'recent_{category.lower()}_quantity_ratio'] = 0
            
            # Shopping time patterns (recent)
            times = pd.Series([pd.to_datetime(t['date']).hour for t in recent_txns])
            features.update({
                'morning_ratio': len(times[times.between(6, 11)]) / len(times),
                'afternoon_ratio': len(times[times.between(12, 16)]) / len(times),
                'evening_ratio': len(times[times.between(17, 21)]) / len(times),
                'night_ratio': len(times[(times >= 22) | (times <= 5)]) / len(times)
            })
            
            # Payment patterns (recent)
            payment_methods = [t['payment']['method'] for t in recent_txns]
            features['primary_payment'] = max(set(payment_methods), key=payment_methods.count)
            payment_counts = pd.Series(payment_methods).value_counts()
            
            for method in ['UPI', 'Credit Card', 'Debit Card', 'Cash', 'Net Banking']:
                features[f'payment_{method.lower().replace(" ", "_")}_ratio'] = (
                    payment_counts.get(method, 0) / len(payment_methods)
                )
            
            # Festival shopping behavior (recent)
            festival_txns = [t for t in recent_txns if t['festivals'] and len(t['festivals']) > 0]
            features['festival_shopping_ratio'] = len(festival_txns) / len(recent_txns)
            
            # Get preferred store (target) - based on most recent visits
            features['preferred_store'] = max(set(recent_store_visits), 
                                           key=recent_store_visits.count)
            
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'city', 'state', 'membership_tier', 
                          'primary_payment', 'preferred_store']
        
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['float64']).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Prepare final features
        X = df.drop(['customer_id', 'preferred_store'], axis=1)
        y = df['preferred_store']
        
        print(f"Final feature shape: {X.shape}")
        return X, y

    def train_model(self):
        """Train an optimized LightGBM model"""
        print("Training model...")
        
        # Prepare data
        X, y = self.prepare_features()
        
        # Implement k-fold cross validation
        from sklearn.model_selection import StratifiedKFold
        
        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        
        print("\nStarting cross-validation training...")
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Set optimal parameters based on feature importance insights
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_error',
            'boosting_type': 'gbdt',
            'num_leaves': 96,          # Increased for better store pattern capture
            'max_depth': 12,           # Increased depth
            'learning_rate': 0.01,     # Slightly increased for faster convergence
            'feature_fraction': 0.7,   # Focus on most important features
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 15,
            'lambda_l1': 0.05,         # Reduced regularization
            'lambda_l2': 0.05,
            'min_gain_to_split': 0.05,
            'verbose': -1,
            'random_state': 42
        }
        
        fold_models = []
        feature_importance_list = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nTraining Fold {fold}/{n_folds}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,  # Increased number of rounds
                valid_sets=[train_data, valid_data],
                callbacks=callbacks
            )
            
            # Evaluate fold
            y_pred = np.argmax(model.predict(X_test), axis=1)
            fold_accuracy = np.mean(y_pred == y_test)
            scores.append(fold_accuracy)
            
            print(f"Fold {fold} Accuracy: {fold_accuracy:.4f}")
            
            # Store model and importance
            fold_models.append(model)
            
            # Get feature importance for this fold
            importance = pd.DataFrame({
                'feature': model.feature_name(),
                'importance': model.feature_importance(importance_type='gain')
            })
            feature_importance_list.append(importance)
        
        # Calculate average accuracy
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        print("\nCross-validation accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
        
        # Get average feature importance across folds
        all_importance = pd.concat(feature_importance_list)
        avg_importance = all_importance.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features (Average across folds):")
        print(avg_importance.head(15))
        
        # Use the best model
        best_model_idx = np.argmax(scores)
        self.model = fold_models[best_model_idx]
        best_accuracy = max(scores)
        
        print(f"\nBest fold accuracy: {best_accuracy:.4f}")
        
        # Final importance from best model
        importance = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(importance.head(15))
        
        # Save model
        self.save_model()
        
        return best_accuracy

    def save_model(self, filename='retail_optimized_model'):
        """Save model and preprocessing objects"""
        # Save LightGBM model
        self.model.save_model(f'{filename}.txt')
        
        # Save preprocessing objects
        preprocessing = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        joblib.dump(preprocessing, f'{filename}_preprocessing.joblib')
        print(f"\nModel saved as {filename}.txt")
        print(f"Preprocessing objects saved as {filename}_preprocessing.joblib")

def main():
    predictor = RetailPredictor()
    accuracy = predictor.train_model()
    print(f"\nTraining completed with accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    try:
        print("Starting Optimized Retail Preference Model Training...")
        main()
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise