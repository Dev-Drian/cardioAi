import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
    
    def prepare_features(self, df):
        """
        Prepara las características y la variable objetivo
        """
        # Seleccionar características principales
        features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
        
        X = df[features].copy()
        y = df['cardio'].copy()
        
        return X, y
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Entrena todos los modelos de machine learning
        """
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definir modelos con sus parámetros optimizados
        models_config = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='liblinear'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10
            ),
            'SVM': SVC(
                random_state=42,
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Naive Bayes': GaussianNB()
        }
        
        trained_models = {}
        metrics = {}
        
        for name, model in models_config.items():
            # Entrenar modelo
            if name in ['SVM', 'K-Nearest Neighbors', 'Naive Bayes', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Validación cruzada
            if name in ['SVM', 'K-Nearest Neighbors', 'Naive Bayes', 'Logistic Regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Almacenar resultados
            trained_models[name] = model
            metrics[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std()
            }
        
        self.models = trained_models
        self.is_trained = True
        
        return {
            'models': trained_models,
            'metrics': metrics
        }
    
    def predict_single(self, model_name, features):
        """
        Realiza predicción para una sola muestra
        """
        if not self.is_trained or model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no está entrenado")
        
        model = self.models[model_name]
        
        # Escalar si es necesario
        if model_name in ['SVM', 'K-Nearest Neighbors', 'Naive Bayes', 'Logistic Regression']:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features_scaled)[0][1]
            else:
                probability = None
        else:
            prediction = model.predict(features.reshape(1, -1))[0]
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features.reshape(1, -1))[0][1]
            else:
                probability = None
        
        return prediction, probability
    
    def get_feature_importance(self, model_name):
        """
        Obtiene la importancia de las características para modelos que la soporten
        """
        if not self.is_trained or model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no está entrenado")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
    
    def get_model_parameters(self, model_name):
        """
        Obtiene los parámetros del modelo entrenado
        """
        if not self.is_trained or model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no está entrenado")
        
        return self.models[model_name].get_params()
