from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

class NNModel:
    def __init__(self, data, target_column, n_features=10):
        """
        Initializes the NN model with the dataset and target variable.
        """
        self.data = data
        self.target_column = target_column
        self.model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        # Automatically exclude the target column from feature names
        self.feature_names = self.data.columns.drop([target_column]).tolist()
        self.n_features = n_features
    
    def select_features(self):
        """
        Selects the top k features based on the ANOVA F-test.
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        selector = SelectKBest(f_classif, k=self.n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features

    def prepare_data(self, selected_features=None):
        """
        Prepares data for training and testing. Scales features and splits the dataset.
        Uses all features if 'selected_features' is None or empty.
        """
        if not selected_features:
            # If no specific features are selected, use all features
            selected_features = self.feature_names
        
        X = self.data[selected_features]
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_evaluate(self, selected_features=None):
        """
        Trains the NN model with the selected features (or all if none are specified) and evaluates its performance.
        """
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(selected_features)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='macro')
        test_recall = recall_score(y_test, y_pred, average='macro')
        test_f1 = f1_score(y_test, y_pred, average='macro')
        return {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'selected_features': selected_features
        }

    def train_with_selected_features(self, selected_features=None):
        """
        Trains the model with the selected features (or all if none are specified) and evaluates its performance using F1 score.
        Acts as an alias to 'train_and_evaluate' for clarity and consistency.
        """
        return self.train_and_evaluate(selected_features)
    
    def train_with_SK_feature_selection(self):
        selected_features = self.select_features()
        return self.train_and_evaluate(selected_features)
