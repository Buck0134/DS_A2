import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class RFModelWithFeatureSelection:
    def __init__(self, df, target_column, n_features=10):
        self.df = df  # DataFrame containing the data
        self.target_column = target_column  # Name of the target variable column
        self.n_features = n_features  # Number of features to select
        self.pipeline = None  # Will hold the pipeline including feature selection and model

    # def prepare_data(self):
    #     X = self.df.drop(self.target_column, axis=1)
    #     y = self.df[self.target_column]
    #     return train_test_split(X, y, test_size=0.2, random_state=42)

    def prepare_data(self):
        # Drop rows with NaN values first
        df_cleaned = self.df.dropna()
        
        # Drop constant features
        selector = VarianceThreshold()
        df_reduced = df_cleaned.loc[:, df_cleaned.columns != self.target_column]
        df_reduced = pd.DataFrame(selector.fit_transform(df_reduced), columns=df_reduced.columns[selector.get_support()])
        
        # Re-attach the target column
        df_reduced[self.target_column] = df_cleaned[self.target_column]
        
        # After cleaning and reduction, check if the DataFrame is empty
        if df_reduced.empty:
            raise ValueError("After preprocessing, the DataFrame is empty.")
        
        # Convert categorical features to dummy variables and extract the target variable
        X_transformed = pd.get_dummies(df_reduced.drop(self.target_column, axis=1))
        y = df_reduced[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
        # Return the transformed DataFrame columns for feature name mapping
        return X_train, X_test, y_train, y_test, X_transformed.columns

    def build_model(self):
        # Define a pipeline that includes feature selection and a classifier
        self.pipeline = Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif, k=self.n_features)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    
    def train(self):
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data()
        if not self.pipeline:
            self.build_model()
        self.pipeline.fit(X_train, y_train)
        
        # Prediction
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        # Compute scores
        training_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        training_precision = precision_score(y_train, y_pred_train, average='macro')
        test_precision = precision_score(y_test, y_pred_test, average='macro')
        training_recall = recall_score(y_train, y_pred_train, average='macro')
        test_recall = recall_score(y_test, y_pred_test, average='macro')
        training_f1 = f1_score(y_train, y_pred_train, average='macro')
        test_f1 = f1_score(y_test, y_pred_test, average='macro')
        
        # Return scores including precision and recall
        return {
            'training_accuracy': training_accuracy,
            'training_precision': training_precision,
            'training_recall': training_recall,
            'training_f1': training_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }


    def train_with_selected_features(self, selected_features):
        # Adapt method to ensure pipeline is properly initialized and used
        # First, prepare the data using the selected features
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data()

        # Check if the pipeline has been initialized; if not, build it
        if not self.pipeline:
            self.build_model()

        # Filter the dataset to include only the selected features
        X_train_selected = X_train[feature_names[selected_features]]
        X_test_selected = X_test[feature_names[selected_features]]

        # Now, fit the model using only the selected features
        self.pipeline.fit(X_train_selected, y_train)

        # Perform predictions and evaluate the model using the test set
        y_pred_test = self.pipeline.predict(X_test_selected)
        test_f1 = f1_score(y_test, y_pred_test, average='macro')

        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='macro')
        test_recall = recall_score(y_test, y_pred_test, average='macro')
        test_f1 = f1_score(y_test, y_pred_test, average='macro')

        # Return the trained pipeline and the test F1 score
        # return self.pipeline, test_f1
        return {
            'Model':self.pipeline,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'selected_features': selected_features
        }
    
    

class RFModel:
    def __init__(self, df, target_column):
        self.df = df  # DataFrame containing the data
        self.target_column = target_column  # Name of the target variable column
        self.pipeline = None  # Will hold the model

    def prepare_data(self):
        # Drop rows with NaN values
        df_cleaned = self.df.dropna()
        
        # Drop constant features
        selector = VarianceThreshold()
        df_reduced = df_cleaned.loc[:, df_cleaned.columns != self.target_column]
        df_reduced = pd.DataFrame(selector.fit_transform(df_reduced), columns=df_reduced.columns[selector.get_support()])
        
        # Re-attach the target column
        df_reduced[self.target_column] = df_cleaned[self.target_column]
        
        # Check if the DataFrame is empty after preprocessing
        if df_reduced.empty:
            raise ValueError("After preprocessing, the DataFrame is empty.")
        
        # Convert categorical features to dummy variables and extract the target variable
        X = pd.get_dummies(df_reduced.drop(self.target_column, axis=1))
        y = df_reduced[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self):
        # Define a model without feature selection
        self.pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    
    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        if not self.pipeline:
            self.build_model()
        self.pipeline.fit(X_train, y_train)
        
        # Prediction
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        # Compute scores
        training_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        training_precision = precision_score(y_train, y_pred_train, average='macro')
        test_precision = precision_score(y_test, y_pred_test, average='macro')
        training_recall = recall_score(y_train, y_pred_train, average='macro')
        test_recall = recall_score(y_test, y_pred_test, average='macro')
        training_f1 = f1_score(y_train, y_pred_train, average='macro')
        test_f1 = f1_score(y_test, y_pred_test, average='macro')
        
        # Return scores including precision and recall
        return {
            'training_accuracy': training_accuracy,
            'training_precision': training_precision,
            'training_recall': training_recall,
            'training_f1': training_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
