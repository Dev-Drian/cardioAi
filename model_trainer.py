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
from sklearn.feature_selection import SelectKBest, f_classif

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        self.feature_selector = SelectKBest(f_classif, k=8)  # Seleccionar las 8 mejores características
    
    def prepare_features(self, df):
        """
        Prepara las características y la variable objetivo
        """
        # Seleccionar características principales en un orden específico
        features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
        
        # Asegurar que todas las características existan
        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Característica '{feature}' no encontrada en el dataset")
        
        X = df[features].copy()
        y = df['cardio'].copy()
        
        return X, y
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Entrena todos los modelos de machine learning de forma optimizada
        """
        # Para datasets grandes, usar una muestra para entrenamiento más rápido
        if len(X_train) > 10000:
            from sklearn.model_selection import train_test_split
            X_train_sample, X_val, y_train_sample, y_val = train_test_split(
                X_train, y_train, train_size=10000, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            X_train_sample, X_val, y_train_sample, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train_sample)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Seleccionar mejores características
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_sample)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Configuración de modelos con parámetros optimizados
        models_config = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear',
                C=0.1,  # Regularización más fuerte
                penalty='l2',
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        trained_models = {}
        metrics = {}
        
        for name, model in models_config.items():
            # Entrenar modelo
            if name == 'Logistic Regression':
                model.fit(X_train_selected, y_train_sample)
                y_pred = model.predict(X_test_selected)
                # Validación cruzada más robusta
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train_selected, y_train_sample, cv=cv, scoring='accuracy')
            else:
                model.fit(X_train_sample, y_train_sample)
                y_pred = model.predict(X_test)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train_sample, y_train_sample, cv=cv, scoring='accuracy')
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
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
        
        # Escalar y seleccionar características
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Realizar predicción
        prediction = model.predict(features_selected)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_selected)[0][1]
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
