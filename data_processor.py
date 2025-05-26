import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def process_data(self, df):
        """
        Procesa el dataset cardiovascular desde el formato original
        """
        # Hacer una copia del dataframe
        df_processed = df.copy()
        
        # Si la primera fila contiene solo 'A', la eliminamos (header issue)
        if len(df_processed.columns) == 1 and str(df_processed.iloc[0, 0]).strip() == 'A':
            df_processed = df_processed.drop(df_processed.index[0])
        
        # Nombrar las columnas correctamente
        expected_columns = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                          'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
        
        if len(df_processed.columns) == 1:
            # Si los datos están en una sola columna, separarlos
            # Asegurar que es string y no está vacío
            first_col = df_processed.columns[0]
            df_processed = df_processed.dropna()  # Eliminar filas vacías
            df_processed[first_col] = df_processed[first_col].astype(str)
            df_processed = df_processed[first_col].str.split(';', expand=True)
        
        # Asignar nombres de columnas
        df_processed.columns = expected_columns[:len(df_processed.columns)]
        
        # Convertir tipos de datos
        for col in df_processed.columns:
            if col != 'id':
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Eliminar filas con muchos valores nulos
        df_processed = df_processed.dropna(thresh=len(df_processed.columns) * 0.7)
        
        # Limpiar datos anómalos
        df_processed = self._clean_anomalies(df_processed)
        
        # Crear características adicionales
        df_processed = self._create_features(df_processed)
        
        return df_processed
    
    def _clean_anomalies(self, df):
        """
        Limpia valores anómalos en el dataset
        """
        # Eliminar filas con valores nulos
        df = df.dropna()
        
        # Filtrar valores de presión arterial anómalos
        df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
        df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
        df = df[df['ap_hi'] > df['ap_lo']]  # Sistólica debe ser mayor que diastólica
        
        # Filtrar altura y peso razonables
        df = df[(df['height'] >= 120) & (df['height'] <= 220)]
        df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
        
        # Filtrar edad razonable (convertir días a años)
        df['age_years'] = df['age'] / 365.25
        df = df[(df['age_years'] >= 18) & (df['age_years'] <= 100)]
        
        return df
    
    def _create_features(self, df):
        """
        Crea características adicionales útiles para el modelo
        """
        # BMI (Índice de Masa Corporal)
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
        # Presión arterial media
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        
        # Diferencia de presión arterial (pulse pressure)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        
        # Categorías de BMI
        df['bmi_category'] = pd.cut(df['bmi'], 
                                   bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=[1, 2, 3, 4])  # 1=underweight, 2=normal, 3=overweight, 4=obese
        df['bmi_category'] = df['bmi_category'].astype(int)
        
        # Categorías de edad
        df['age_category'] = pd.cut(df['age_years'], 
                                   bins=[0, 35, 50, 65, float('inf')], 
                                   labels=[1, 2, 3, 4])  # 1=young, 2=middle, 3=senior, 4=elderly
        df['age_category'] = df['age_category'].astype(int)
        
        # Hipertensión (presión alta)
        df['hypertension'] = ((df['ap_hi'] > 140) | (df['ap_lo'] > 90)).astype(int)
        
        # Factores de riesgo combinados
        df['risk_factors'] = df['smoke'] + df['alco'] + (1 - df['active'])
        
        return df
    
    def get_feature_names(self):
        """
        Retorna los nombres de las características para el modelo
        """
        return ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    
    def get_target_name(self):
        """
        Retorna el nombre de la variable objetivo
        """
        return 'cardio'
