from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class SVMModel:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.scaler = StandardScaler()  # For feature scaling     
        self.model = None   

    def prepare_data(self, selected_features):
        # Extract the selected features and the target variable
        X = self.data[selected_features]
        y = self.data[self.target_column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self):
        # Initialize the SVM model
        self.model = SVC(kernel='rbf', random_state=42)
    

    def train_with_selected_features(self, selected_features):
        # Ensure the model is initialized
        if not self.model:
            self.build_model()

        # Prepare the data with the selected features
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(selected_features)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model on the test set
        y_pred = self.model.predict(X_test_scaled)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        
        return test_f1
