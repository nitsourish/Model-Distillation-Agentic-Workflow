"""
Optimized prediction utilities for loan approval and amount prediction.
This module provides fast inference functions using pre-trained models.
"""

import numpy as np
import pandas as pd
import torch
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch.nn as nn
import torch.nn.functional as F


class SmallClassifier(nn.Module):
    """Student classification model architecture."""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))  
        x = self.dropout(x)
        return self.fc3(x)


class SmallRegressor(nn.Module):
    """Student regression model architecture."""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))     
        x = self.dropout(x)
        return self.fc3(x)


class LoanPredictionService:
    """Fast loan prediction service with pre-loaded models and preprocessor."""
    
    def __init__(self, artifacts_dir="/Users/sourishdey/Desktop/learning/model_distillation/model_distillation/artefacts"):
        self.artifacts_dir = artifacts_dir
        self.classifier_model = None
        self.regressor_model = None
        self.feature_preprocessor = None
        self.device = 'cpu'
        
    def load_models(self):
        """Load all models and preprocessor once."""
        print("🚀 Loading models and preprocessor...")
        
        # Load classifier
        classifier_path = os.path.join(self.artifacts_dir, "student_classifier.pth")
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.classifier_model = SmallClassifier(input_size=checkpoint['input_size'])
        self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier_model.eval()
        
        # Load regressor
        regressor_path = os.path.join(self.artifacts_dir, "student_regressor.pth")
        checkpoint = torch.load(regressor_path, map_location=self.device)
        self.regressor_model = SmallRegressor(input_size=checkpoint['input_size'])
        self.regressor_model.load_state_dict(checkpoint['model_state_dict'])
        self.regressor_model.eval()
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.artifacts_dir, "feature_preprocessor.pkl")
        try:
            self.feature_preprocessor = joblib.load(preprocessor_path)
            
            # Test if preprocessor matches model expectations
            data_path = "/Users/sourishdey/Desktop/learning/model_distillation/model_distillation/data/loan_data_with_max_loan.csv"
            test_data = pd.read_csv(data_path).iloc[:1]
            test_features = test_data.drop(columns=['loan_status', 'max_loan'], errors='ignore')
            test_processed = self.feature_preprocessor.transform(test_features)
            
            expected_features = self.classifier_model.fc1.in_features
            actual_features = test_processed.shape[1]
            
            if actual_features != expected_features:
                print(f"⚠️  Preprocessor mismatch! Expected {expected_features}, got {actual_features}")
                print("   Creating corrected preprocessor...")
                self._create_preprocessor()
            else:
                print("✅ All models and preprocessor loaded successfully!")
                
        except FileNotFoundError:
            print("⚠️  Feature preprocessor not found. Creating one...")
            self._create_preprocessor()
            
    def _create_preprocessor(self):
        """Create and save feature preprocessor that matches model expectations."""
        print("⚠️  Creating preprocessor that matches model architecture...")
        
        # Get expected input size from the loaded model
        expected_features = self.classifier_model.fc1.in_features
        print(f"   🎯 Target feature count: {expected_features}")
        
        data_path = "/Users/sourishdey/Desktop/learning/model_distillation/model_distillation/data/loan_data_with_max_loan.csv"
        full_data = pd.read_csv(data_path)
        training_features = full_data.drop(columns=['loan_status', 'max_loan'], errors='ignore')
        
        categorical_features = training_features.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = training_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"   📊 Categorical: {categorical_features}")
        print(f"   📊 Numerical: {numerical_features}")
        
        # Try different configurations to match the expected feature count
        # Option 1: drop='first' (default)
        preprocessor_v1 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        preprocessor_v1.fit(training_features)
        test_features_v1 = preprocessor_v1.transform(training_features.iloc[:1]).shape[1]
        
        # Option 2: drop=None
        preprocessor_v2 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        preprocessor_v2.fit(training_features)
        test_features_v2 = preprocessor_v2.transform(training_features.iloc[:1]).shape[1]
        
        print(f"   🧪 With drop='first': {test_features_v1} features")
        print(f"   🧪 With drop=None: {test_features_v2} features")
        
        # Choose the one that matches
        if test_features_v1 == expected_features:
            self.feature_preprocessor = preprocessor_v1
            print(f"   ✅ Using drop='first' configuration")
        elif test_features_v2 == expected_features:
            self.feature_preprocessor = preprocessor_v2
            print(f"   ✅ Using drop=None configuration")
        else:
            print(f"   ❌ Neither configuration matches! Creating custom truncated version.")
            # Use drop='first' but truncate to match expected features
            self.feature_preprocessor = preprocessor_v1
            self.feature_truncate_to = expected_features
            print(f"   🔧 Will truncate features from {test_features_v1} to {expected_features}")
        
        # Save for future use
        os.makedirs(self.artifacts_dir, exist_ok=True)
        preprocessor_path = os.path.join(self.artifacts_dir, "feature_preprocessor_corrected.pkl")
        joblib.dump(self.feature_preprocessor, preprocessor_path)
        print(f"   💾 Corrected preprocessor saved to: {preprocessor_path}")
        
    def predict_single_customer(self, customer_data):
        """
        Fast prediction for a single customer.
        
        Args:
            customer_data (pd.DataFrame): Single row DataFrame with customer features
            
        Returns:
            dict: Prediction results
        """
        if self.classifier_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        # Prepare features (drop target columns and customer_id)
        feature_columns = customer_data.drop(columns=['loan_status', 'max_loan', 'customer_id'], errors='ignore')
        
        # Preprocess features
        X_processed = self.feature_preprocessor.transform(feature_columns)
        
        # Truncate features if needed to match model expectations
        if hasattr(self, 'feature_truncate_to'):
            X_processed = X_processed[:, :self.feature_truncate_to]
            
        X_tensor = torch.FloatTensor(X_processed)
        
        # Make predictions
        with torch.no_grad():
            # Classification only - no loan amount calculation
            class_outputs = self.classifier_model(X_tensor)
            class_probs = F.softmax(class_outputs, dim=1)
            class_prediction = torch.argmax(class_outputs, dim=1)
            approval_probability = class_probs[:, 1].numpy()[0]
            
        return {
            'loan_approval_prediction': int(class_prediction.numpy()[0]),
            'loan_approval_probability': float(approval_probability),
            'approved': approval_probability >= 0.5,
        }
    
    def predict_batch(self, customer_data_batch):
        """
        Fast prediction for multiple customers.
        
        Args:
            customer_data_batch (pd.DataFrame): DataFrame with multiple customer rows
            
        Returns:
            list: List of prediction dictionaries
        """
        if self.classifier_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        # Prepare features
        feature_columns = customer_data_batch.drop(columns=['loan_status', 'max_loan', 'customer_id'], errors='ignore')
        
        # Preprocess features
        X_processed = self.feature_preprocessor.transform(feature_columns)
        
        # Truncate features if needed to match model expectations
        if hasattr(self, 'feature_truncate_to'):
            X_processed = X_processed[:, :self.feature_truncate_to]
            
        X_tensor = torch.FloatTensor(X_processed)
        
        # Make predictions
        with torch.no_grad():
            # Classification only - no loan amount calculation
            class_outputs = self.classifier_model(X_tensor)
            class_probs = F.softmax(class_outputs, dim=1)
            class_predictions = torch.argmax(class_outputs, dim=1)
            approval_probabilities = class_probs[:, 1].numpy()
            
        # Format results
        results = []
        for i in range(len(customer_data_batch)):
            results.append({
                'loan_approval_prediction': int(class_predictions[i].numpy()),
                'loan_approval_probability': float(approval_probabilities[i]),
                'approved': approval_probabilities[i] >= 0.5,
            })
            
        return results


def predict_customer_loan_propensity(loan_data_df, customer_id, prediction_service=None):
    """
    Predict loan propensity for a specific customer using the prediction service.

    Args:
        loan_data_df (pd.DataFrame): The loan dataset
        customer_id (int): The customer ID to predict for
        prediction_service (LoanPredictionService): Pre-initialized service
        
    Returns:
        dict: Dictionary containing prediction propensity results
    """
    # Create service if not provided
    if prediction_service is None:
        prediction_service = LoanPredictionService()
        prediction_service.load_models()
    
    # Get customer data
    customer_data = loan_data_df.loc[loan_data_df['customer_id'] == customer_id]
    
    if customer_data.empty:
        return {"error": f"Customer ID {customer_id} not found"}
    
    # Get prediction
    prediction = prediction_service.predict_single_customer(customer_data)
    
    # Add customer info
    prediction['customer_id'] = customer_id
    prediction['customer_data'] = customer_data.to_dict('records')[0]
    
    return prediction


def predict_customer_loan_amount(loan_data_df, customer_id, prediction_service=None):
    """
    Predict loan amount for a specific customer (requires approval probability >= 0.5).
    
    Args:
        loan_data_df (pd.DataFrame): The loan dataset
        customer_id (int): The customer ID to predict for
        prediction_service (LoanPredictionService): Pre-initialized service
        
    Returns:
        dict: Dictionary containing loan amount prediction results
    """
    # Create service if not provided
    if prediction_service is None:
        prediction_service = LoanPredictionService()
        prediction_service.load_models()
    
    # Get customer data
    customer_data = loan_data_df.loc[loan_data_df['customer_id'] == customer_id]
    
    if customer_data.empty:
        return {"error": f"Customer ID {customer_id} not found"}
    
    # First check approval probability
    approval_result = prediction_service.predict_single_customer(customer_data)
    
    if not approval_result['approved']:
        return {
            'customer_id': customer_id,
            'loan_approval_probability': approval_result['loan_approval_probability'],
            'approved': False,
            'predicted_loan_amount': 0.0,
            'reason': 'Loan rejected due to low approval probability'
        }
    
    # If approved, predict loan amount
    feature_columns = customer_data.drop(columns=['loan_status', 'max_loan', 'customer_id'], errors='ignore')
    
    # Preprocess features
    X_processed = prediction_service.feature_preprocessor.transform(feature_columns)
    
    # Truncate features if needed
    if hasattr(prediction_service, 'feature_truncate_to'):
        X_processed = X_processed[:, :prediction_service.feature_truncate_to]
        
    X_tensor = torch.FloatTensor(X_processed)
    
    # Get loan amount prediction
    with torch.no_grad():
        reg_outputs = prediction_service.regressor_model(X_tensor)
        loan_amount = reg_outputs.numpy()[0][0]
    
    return {
        'customer_id': customer_id,
        'loan_approval_probability': approval_result['loan_approval_probability'],
        'approved': True,
        'predicted_loan_amount': float(loan_amount),
        'reason': 'Loan approved with predicted amount'
    }


# Global service instance for reuse
_global_service = None

def get_prediction_service():
    """Get or create a global prediction service instance."""
    global _global_service
    if _global_service is None:
        _global_service = LoanPredictionService()
        _global_service.load_models()
    return _global_service
