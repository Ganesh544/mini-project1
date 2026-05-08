import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, VotingRegressor, RandomForestClassifier, RandomForestRegressor
import matplotlib
matplotlib.use('Agg')

from sklearn.base import BaseEstimator, RegressorMixin, clone


import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

class Tao_Tree_classifier:
    def __init__(self, n_estimators=1, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """Fit Random Forest Classifier logic"""
        n_samples, n_features = X.shape
        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = resample(X, y, replace=True, random_state=self.random_state + i)
            
            # Random subset of features
            if self.max_features == "sqrt":
                n_sub_features = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_sub_features = int(np.log2(n_features))
            else:
                n_sub_features = n_features
            
            feature_indices = np.random.choice(n_features, n_sub_features, replace=False)

            tree = RandomForestClassifier()
            tree.fit(X, y)
            
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        """Majority voting for classification"""
        preds = []
        for tree, feature_indices in self.trees:
            preds.append(tree.predict(X))
        
        preds = np.array(preds).T  # shape: (n_samples, n_trees)
        
        # Majority vote
        final_preds = [np.bincount(row).argmax() for row in preds]
        return np.array(final_preds)


class Tao_Tree_regressor:
    def __init__(self, n_estimators=1, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        np.random.seed(self.random_state)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = resample(X, y, replace=True, random_state=self.random_state + i)
            
            if self.max_features == "sqrt":
                n_sub_features = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_sub_features = int(np.log2(n_features))
            else:
                n_sub_features = n_features
            
            feature_indices = np.random.choice(n_features, n_sub_features, replace=False)

            tree = RandomForestRegressor()
            tree.fit(X, y)
            
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        """Average predictions for regression"""
        preds = []
        for tree, feature_indices in self.trees:
            preds.append(tree.predict(X))
        
        preds = np.array(preds).T  # shape: (n_samples, n_trees)
        
        # Average predictions across trees
        final_preds = np.mean(preds, axis=1)
        return final_preds


class MLModelManager:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.y1 = None
        self.X_train = None
        self.X_test = None
        self.y_class_train = None
        self.y_class_test = None
        self.y_reg_train = None
        self.y_reg_test = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.classification_models = {}
        self.regression_models = {}
        self.classification_results = {}
        self.regression_results = {}
        
        # Model directories
        self.model_dir = 'models'
        self.results_dir = 'results'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self, filepath):
        """Load and preprocess the dataset."""
        self.df = pd.read_csv(filepath)
        self.X, self.y, self.y1, self.label_encoders = self.preprocess_data(self.df)
        self.split_data()
    
    def preprocess_data(self, df, is_train=True, label_encoders=None):
        """Preprocess the dataset."""
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if is_train:
            label_encoders = {}
            # Encode categorical variables
            for col in df.select_dtypes(include='object').columns:
                if col not in ['Attack Type']:  # Don't encode target variable yet
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
            
            # Encode target variable separately
            if 'Attack Type' in df.columns:
                le_target = LabelEncoder()
                df['Attack Type'] = le_target.fit_transform(df['Attack Type'].astype(str))
                label_encoders['Attack Type'] = le_target
        else:
            for col in df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
        
        # Fill missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        if is_train:
            # Define features and targets
            feature_cols = [col for col in df.columns if col not in ['Attack Type', 'Throughput (Mbps)']]
            X = df[feature_cols]
            y = df['Attack Type'] if 'Attack Type' in df.columns else None
            y1 = df['Throughput (Mbps)'] if 'Throughput (Mbps)' in df.columns else None
            return X, y, y1, label_encoders
        else:
            return df
    
    def split_data(self):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_class_train, self.y_class_test, self.y_reg_train, self.y_reg_test = train_test_split(
            self.X, self.y, self.y1, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Apply SMOTE for data balancing on classification data
        self.X_train_balanced, self.y_class_train_balanced = self.smote.fit_resample(self.X_train_scaled, self.y_class_train)
    
    def train_models(self, selected_models):
        """Train selected models."""
        results = {}
        
        for model_name in selected_models:
            # Check if models already exist
            clf_path = os.path.join(self.model_dir, f'{model_name}_classifier.pkl')
            reg_path = os.path.join(self.model_dir, f'{model_name}_regressor.pkl')
            
            if os.path.exists(clf_path) and os.path.exists(reg_path):
                # Load existing models
                self.classification_models[model_name] = joblib.load(clf_path)
                self.regression_models[model_name] = joblib.load(reg_path)
                results[model_name] = 'Loaded existing models'
            else:
                # Train new models
                clf, reg = self._train_model_pair(model_name)
                self.classification_models[model_name] = clf
                self.regression_models[model_name] = reg
                
                # Save models
                joblib.dump(clf, clf_path)
                joblib.dump(reg, reg_path)
                results[model_name] = 'Trained and saved new models'
        
        # Evaluate all models
        self._evaluate_models()
        return results
    
    def _train_model_pair(self, model_name):
        """Train a classifier and regressor pair for the given model."""
        if model_name == 'SVM':
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            reg = SVR(kernel='rbf')
        elif model_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5)
            reg = KNeighborsRegressor(n_neighbors=5)
        elif model_name == 'TaoTree':
            clf =  Tao_Tree_classifier()
            reg = Tao_Tree_regressor()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train models - use balanced data for classification, original for regression
        clf.fit(self.X_train_balanced, self.y_class_train_balanced)
        reg.fit(self.X_train_scaled, self.y_reg_train)
        
        return clf, reg
    
    def _evaluate_models(self):
        """Evaluate all trained models."""
        self.classification_results = {}
        self.regression_results = {}
        
        for model_name in self.classification_models.keys():
            # Classification evaluation
            clf = self.classification_models[model_name]
            y_pred = clf.predict(self.X_test_scaled)
            y_prob = clf.predict_proba(self.X_test_scaled) if hasattr(clf, 'predict_proba') else None
            
            self.classification_results[model_name] = {
                'accuracy': accuracy_score(self.y_class_test, y_pred) * 100,
                'precision': precision_score(self.y_class_test, y_pred, average='macro') * 100,
                'recall': recall_score(self.y_class_test, y_pred, average='macro') * 100,
                'f1_score': f1_score(self.y_class_test, y_pred, average='macro') * 100,
                'confusion_matrix': confusion_matrix(self.y_class_test, y_pred),
                'classification_report': classification_report(self.y_class_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            # Regression evaluation
            reg = self.regression_models[model_name]
            y_reg_pred = reg.predict(self.X_test_scaled)
            
            self.regression_results[model_name] = {
                'mae': mean_absolute_error(self.y_reg_test, y_reg_pred),
                'mse': mean_squared_error(self.y_reg_test, y_reg_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_reg_test, y_reg_pred)),
                'r2_score': r2_score(self.y_reg_test, y_reg_pred),
                'predictions': y_reg_pred,
                'actual': self.y_reg_test
            }
    
    def generate_eda_plots(self):
        """Generate EDA plots and return as base64 encoded images."""
        plots = {}
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Attack Type Distribution
        axes[0, 0].hist(self.y, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Attack Types')
        axes[0, 0].set_xlabel('Attack Type (Encoded)')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Throughput Distribution
        axes[0, 1].hist(self.y1, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Throughput')
        axes[0, 1].set_xlabel('Throughput (Mbps)')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Latency vs Attack Type
        if 'Latency (ms)' in self.X.columns:
            attack_types = sorted(self.y.unique())
            latency_by_attack = [self.X.loc[self.y == at, 'Latency (ms)'].values for at in attack_types]
            axes[0, 2].boxplot(latency_by_attack, labels=attack_types)
            axes[0, 2].set_title('Latency by Attack Type')
            axes[0, 2].set_xlabel('Attack Type')
            axes[0, 2].set_ylabel('Latency (ms)')
        
        # 4. Signal Strength vs Throughput
        if 'Signal Strength' in self.X.columns:
            signal_types = sorted(self.X['Signal Strength'].unique())
            throughput_by_signal = [self.y1.loc[self.X['Signal Strength'] == st].values for st in signal_types]
            axes[1, 0].boxplot(throughput_by_signal, labels=signal_types)
            axes[1, 0].set_title('Throughput by Signal Strength')
            axes[1, 0].set_xlabel('Signal Strength')
            axes[1, 0].set_ylabel('Throughput (Mbps)')
        
        # 5. Latency vs Throughput scatter
        if 'Latency (ms)' in self.X.columns:
            axes[1, 1].scatter(self.X['Latency (ms)'], self.y1, alpha=0.6, color='coral')
            axes[1, 1].set_title('Throughput vs Latency')
            axes[1, 1].set_xlabel('Latency (ms)')
            axes[1, 1].set_ylabel('Throughput (Mbps)')
        
        # 6. Correlation heatmap
        df_corr = self.X.copy()
        df_corr['Throughput'] = self.y1
        df_corr['Attack_Type'] = self.y
        
        # Select numeric columns and limit to top correlations for readability
        numeric_cols = df_corr.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = df_corr[numeric_cols].corr()
        
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[1, 2].set_yticklabels(corr_matrix.columns)
        axes[1, 2].set_title('Correlation Heatmap')
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plots['eda_combined'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plots
    
    def get_classification_results(self):
        """Get classification results with plots."""
        results = {}
        
        for model_name, metrics in self.classification_results.items():
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            confusion_plot = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            results[model_name] = {
                'metrics': metrics,
                'confusion_plot': confusion_plot
            }
        
        return results
    
    def get_regression_results(self):
        """Get regression results with plots."""
        results = {}
        
        for model_name, metrics in self.regression_results.items():
            # Create actual vs predicted plot
            plt.figure(figsize=(8, 6))
            plt.scatter(metrics['actual'], metrics['predictions'], alpha=0.6)
            plt.plot([metrics['actual'].min(), metrics['actual'].max()], 
                    [metrics['actual'].min(), metrics['actual'].max()], 'r--', lw=2)
            plt.xlabel('Actual Throughput')
            plt.ylabel('Predicted Throughput')
            plt.title(f'{model_name} - Actual vs Predicted Throughput')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            scatter_plot = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            results[model_name] = {
                'metrics': metrics,
                'scatter_plot': scatter_plot
            }
        
        return results
    
    def get_feature_names(self):
        """Get feature names for the prediction form."""
        if self.X is not None:
            return list(self.X.columns)
        return []
    
    def predict(self, input_data, model_name):
        """Make predictions for single input."""
        if model_name not in self.classification_models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Convert input to dataframe
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for col in self.X.columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        input_df = input_df[self.X.columns]
        
        # Scale input
        input_scaled = self.scaler.transform(input_df)
        
        # Make predictions
        clf = self.classification_models[model_name]
        reg = self.regression_models[model_name]
        
        attack_prediction = clf.predict(input_scaled)[0]
        attack_probability = clf.predict_proba(input_scaled)[0] if hasattr(clf, 'predict_proba') else None
        throughput_prediction = reg.predict(input_scaled)[0]
        
        # Decode attack type if label encoder exists
        if 'Attack Type' in self.label_encoders:
            attack_type_name = self.label_encoders['Attack Type'].inverse_transform([attack_prediction])[0]
        else:
            attack_type_name = str(attack_prediction)
        
        return {
            'attack_type': attack_type_name,
            'attack_probability': attack_probability,
            'throughput': throughput_prediction
        }
    
    def predict_batch(self, test_df, model_name):
        """Make predictions for batch data."""
        if model_name not in self.classification_models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Preprocess test data
        test_processed = self.preprocess_data(test_df, is_train=False, label_encoders=self.label_encoders)
        
        # Ensure all features are present
        for col in self.X.columns:
            if col not in test_processed.columns:
                test_processed[col] = 0
        
        # Reorder columns
        test_processed = test_processed[self.X.columns]
        
        # Scale data
        test_scaled = self.scaler.transform(test_processed)
        
        # Make predictions
        clf = self.classification_models[model_name]
        reg = self.regression_models[model_name]
        
        attack_predictions = clf.predict(test_scaled)
        throughput_predictions = reg.predict(test_scaled)
        
        # Decode attack types
        if 'Attack Type' in self.label_encoders:
            attack_types = self.label_encoders['Attack Type'].inverse_transform(attack_predictions)
        else:
            attack_types = attack_predictions
        
        results = pd.DataFrame({
            'Attack_Type_Prediction': attack_types,
            'Throughput_Prediction': throughput_predictions
        })
        
        return results
