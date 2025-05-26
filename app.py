import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai


import os

# Cargar variables de entorno


# Configurar Gemini
GOOGLE_API_KEY = "AIzaSyCS1tJIfKS8XBVclWxc-UnkVVGTowq8iGU"
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("⚠️ No se encontró la API key de Google. Por favor, configura tu API key en el archivo .env")
    model = None

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizations import Visualizations

def main():
    st.set_page_config(
        page_title="CardioAI - Predictor Cardiovascular",
        page_icon="💝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado para diseño innovador
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .floating-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .floating-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0,0,0,0.15);
    }
    
    .neon-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .neon-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
    }
    
    .pulse-animation {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .7; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
        animation: pulse 2s infinite;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66, #40c057) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .form-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Título con animación
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="gradient-text" style="font-size: 4rem; margin-bottom: 0.5rem;">
            💝 CardioAI
        </h1>
        <h3 style="color: #666; font-family: 'Poppins', sans-serif; font-weight: 300;">
            Predictor Cardiovascular Inteligente
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox(
        "Selecciona una página:",
        ["🏠 Inicio", "📊 Exploración de Datos", "🔮 Predicciones", "📈 Comparación de Modelos", "🤖 Chat IA - Gemini"]
    )
    
    # Cargar datos
    @st.cache_data
    def load_data():
        try:
            # Leer el archivo CSV con formato especial
            df = pd.read_csv('attached_assets/cardio_train.csv', header=None)
            
            # Procesar el dataset
            processor = DataProcessor()
            df_processed = processor.process_data(df)
            return df_processed, processor
        except FileNotFoundError:
            st.error("❌ No se pudo encontrar el archivo 'cardio_train.csv'. Por favor, asegúrate de que el archivo esté en la carpeta 'attached_assets'.")
            return None, None
        except Exception as e:
            st.error(f"❌ Error al cargar los datos: {str(e)}")
            return None, None
    
    df, processor = load_data()
    
    if df is None:
        st.stop()
    
    # Mostrar información básica del dataset
    st.sidebar.markdown("### 📋 Información del Dataset")
    st.sidebar.info(f"**Registros:** {len(df):,}")
    st.sidebar.info(f"**Características:** {len(df.columns)-1}")
    st.sidebar.info(f"**Casos positivos:** {df['cardio'].sum():,} ({df['cardio'].mean()*100:.1f}%)")
    
    # Inicializar modelos solo cuando sea necesario
    if 'models_initialized' not in st.session_state:
        with st.spinner("🤖 Inicializando modelos de IA... Por favor espera un momento"):
            trainer = ModelTrainer()
            X, y = trainer.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            results = trainer.train_all_models(X_train, X_test, y_train, y_test)
            
            # Guardar en session state
            st.session_state['trainer'] = trainer  # Guardar el trainer completo
            st.session_state['models'] = results['models']
            st.session_state['metrics'] = results['metrics']
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns.tolist()
            st.session_state['models_initialized'] = True
            
        st.success("✅ ¡Modelos entrenados y listos para usar!")
        st.rerun()
    
    # Navegación por páginas
    if page == "🏠 Inicio":
        landing_page(df)
    elif page == "📊 Exploración de Datos":
        exploration_page(df, processor)
    elif page == "🔮 Predicciones":
        prediction_page(df)
    elif page == "📈 Comparación de Modelos":
        comparison_page(df)
    elif page == "🤖 Chat IA - Gemini":
        chat_gemini_page(df)

def exploration_page(df, processor):
    st.header("📊 Exploración de Datos Cardiovasculares")
    
    # Crear instancia de visualizaciones
    viz = Visualizations()
    
    # Estadísticas descriptivas
    st.subheader("📈 Estadísticas Descriptivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Variables Numéricas")
        numeric_stats = df.describe()
        st.dataframe(numeric_stats.round(2))
    
    with col2:
        st.markdown("#### Distribución de la Variable Objetivo")
        cardio_counts = df['cardio'].value_counts()
        fig_pie = px.pie(
            values=cardio_counts.values,
            names=['Sin Enfermedad Cardiovascular', 'Con Enfermedad Cardiovascular'],
            title="Distribución de Enfermedades Cardiovasculares",
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Visualizaciones interactivas
    st.subheader("📊 Visualizaciones Interactivas")
    
    # Selección de tipo de gráfico
    chart_type = st.selectbox(
        "Selecciona el tipo de análisis:",
        ["Distribución por Edad", "Análisis de Presión Arterial", "Factores de Riesgo", "Correlación de Variables", "Distribución por Género"]
    )
    
    if chart_type == "Distribución por Edad":
        viz.plot_age_distribution(df, st)
    elif chart_type == "Análisis de Presión Arterial":
        viz.plot_blood_pressure_analysis(df, st)
    elif chart_type == "Factores de Riesgo":
        viz.plot_risk_factors(df, st)
    elif chart_type == "Correlación de Variables":
        viz.plot_correlation_matrix(df, st)
    elif chart_type == "Distribución por Género":
        viz.plot_gender_analysis(df, st)
    
    # Datos originales
    st.markdown("---")
    st.subheader("🔍 Vista del Dataset")
    
    if st.checkbox("Mostrar datos originales"):
        st.dataframe(df.head(100))
        
        # Opción de descarga
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar datos procesados (CSV)",
            data=csv,
            file_name="cardio_data_processed.csv",
            mime="text/csv"
        )

def training_page(df):
    st.header("🤖 Entrenamiento de Modelos de Machine Learning")
    
    # Preparar datos
    trainer = ModelTrainer()
    X, y = trainer.prepare_features(df)
    
    # División de datos
    st.subheader("⚙️ Configuración del Entrenamiento")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.4, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Semilla aleatoria", 1, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    st.info(f"📊 Datos de entrenamiento: {len(X_train):,} | Datos de prueba: {len(X_test):,}")
    
    # Entrenar modelos
    if st.button("🚀 Entrenar Todos los Modelos", type="primary"):
        with st.spinner("Entrenando modelos..."):
            results = trainer.train_all_models(X_train, X_test, y_train, y_test)
            
            # Guardar en session state
            st.session_state['models'] = results['models']
            st.session_state['metrics'] = results['metrics']
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns.tolist()
        
        st.success("✅ ¡Modelos entrenados exitosamente!")
    
    # Mostrar resultados si existen
    if 'models' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Resultados del Entrenamiento")
        
        # Tabla de métricas
        metrics_df = pd.DataFrame(st.session_state['metrics']).T
        metrics_df = metrics_df.round(4)
        metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
        
        # Colorear la tabla según el rendimiento
        def color_metrics(val):
            if val >= 0.9:
                return 'background-color: #d4edda; color: #155724;'
            elif val >= 0.8:
                return 'background-color: #fff3cd; color: #856404;'
            else:
                return 'background-color: #f8d7da; color: #721c24;'
        
        styled_df = metrics_df.style.applymap(color_metrics).set_properties(**{
            'padding': '10px',
            'border': '1px solid #dee2e6',
            'text-align': 'center',
            'font-weight': 'bold'
        }).set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#f8f9fa'),
                       ('color', '#212529'),
                       ('font-weight', 'bold'),
                       ('text-align', 'center'),
                       ('padding', '12px'),
                       ('border', '1px solid #dee2e6')]},
            {'selector': 'td',
             'props': [('padding', '10px'),
                       ('border', '1px solid #dee2e6')]},
            {'selector': 'tr:hover',
             'props': [('background-color', '#f5f5f5')]}
        ])
        st.dataframe(styled_df, use_container_width=True)
        
        # Modelo destacado
        best_model = metrics_df.index[0]
        best_accuracy = metrics_df.loc[best_model, 'Accuracy']
        
        st.success(f"🏆 **Mejor modelo:** {best_model} con {best_accuracy:.1%} de precisión")
        
        # Análisis detallado del mejor modelo
        st.subheader(f"🔍 Análisis Detallado: {best_model}")
        
        if best_model == 'Logistic Regression':
            # Mostrar coeficientes del modelo
            model = st.session_state['models'][best_model]
            feature_importance = pd.DataFrame({
                'Feature': st.session_state['feature_names'],
                'Coefficient': np.abs(model.coef_[0])
            }).sort_values('Coefficient', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title=f"Importancia de Características - {best_model}",
                color='Coefficient',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Mostrar métricas de validación cruzada
            st.markdown("### 📊 Métricas de Validación Cruzada")
            cv_mean = metrics_df.loc[best_model, 'CV_Mean']
            cv_std = metrics_df.loc[best_model, 'CV_Std']
            st.info(f"""
            - **Media de Validación Cruzada:** {cv_mean:.3f}
            - **Desviación Estándar:** {cv_std:.3f}
            - **Rango de Confianza:** [{cv_mean - 2*cv_std:.3f}, {cv_mean + 2*cv_std:.3f}]
            """)
            
            # Mostrar matriz de confusión
            st.markdown("### 🎯 Matriz de Confusión")
            y_pred = model.predict(st.session_state['X_test'])
            cm = confusion_matrix(st.session_state['y_test'], y_pred)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Matriz de Confusión - {best_model}",
                labels=dict(x="Predicho", y="Real"),
                x=['Sin Riesgo', 'Con Riesgo'],
                y=['Sin Riesgo', 'Con Riesgo'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Mostrar reporte de clasificación
            st.markdown("### 📋 Reporte de Clasificación")
            report = classification_report(st.session_state['y_test'], y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        elif best_model == 'Decision Tree':
            # Información del árbol de decisión
            model = st.session_state['models'][best_model]
            st.write(f"**Profundidad del árbol:** {model.get_depth()}")
            st.write(f"**Número de hojas:** {model.get_n_leaves()}")
            
            # Feature importance para Decision Tree
            feature_importance = pd.DataFrame({
                'Feature': st.session_state['feature_names'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 10 Características Más Importantes - {best_model}",
                color='Importance',
                color_continuous_scale='Plasma'
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)

def prediction_page(df):
    st.markdown("""
    <div class="floating-card">
        <h2 style="text-align: center; color: #667eea; font-family: 'Poppins', sans-serif;">
            🔮 Predictor Cardiovascular Inteligente
        </h2>
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
            Análisis instantáneo con Inteligencia Artificial - Modelo de Regresión Logística
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estado del modelo principal (solo Regresión Logística)
    if 'metrics' in st.session_state:
        lr_accuracy = st.session_state['metrics']['Logistic Regression']['Accuracy']
        
        st.markdown(f"""
        <div class="prediction-result">
            <h3 style="margin-bottom: 1rem;">✅ Modelo CardioAI Activo</h3>
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div>
                    <h4>🧠 Regresión Logística</h4>
                    <p>Algoritmo Principal</p>
                </div>
                <div>
                    <h4>{lr_accuracy:.1%}</h4>
                    <p>Precisión Verificada</p>
                </div>
                <div>
                    <h4>⚡ Instantáneo</h4>
                    <p>Resultado en Tiempo Real</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="form-container">
        <h3 style="text-align: center; color: #667eea; margin-bottom: 2rem;">
            📋 Datos del Paciente
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulario mejorado con diseño más entendible
    st.markdown("### 👤 Información Personal")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("🎂 Edad del paciente", 18, 100, 50, help="Edad en años completos")
        gender = st.selectbox("👫 Género", ["Mujer", "Hombre"], help="Selecciona el género del paciente")
    
    with col2:
        height = st.slider("📏 Altura (cm)", 120, 220, 170, help="Altura en centímetros")
        weight = st.slider("⚖️ Peso (kg)", 30, 200, 70, help="Peso en kilogramos")
    
    st.markdown("### 🩺 Mediciones Médicas")
    col1, col2 = st.columns(2)
    
    with col1:
        ap_hi = st.slider("💓 Presión Sistólica (mmHg)", 80, 250, 120, help="Presión arterial sistólica - normal: 90-120")
        ap_lo = st.slider("💗 Presión Diastólica (mmHg)", 40, 150, 80, help="Presión arterial diastólica - normal: 60-80")
    
    with col2:
        cholesterol = st.selectbox("🧪 Nivel de Colesterol", ["Normal", "Sobre el normal", "Muy alto"], 
                                  help="Nivel de colesterol en sangre")
        gluc = st.selectbox("🍯 Nivel de Glucosa", ["Normal", "Sobre el normal", "Muy alto"],
                           help="Nivel de glucosa en sangre")
    
    st.markdown("### 🎯 Estilo de Vida")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        smoke = st.selectbox("🚭 ¿Fuma?", ["No", "Sí"], help="¿El paciente fuma tabaco?")
    
    with col2:
        alco = st.selectbox("🍷 ¿Consume alcohol?", ["No", "Sí"], help="¿El paciente consume alcohol regularmente?")
    
    with col3:
        active = st.selectbox("🏃‍♂️ ¿Hace ejercicio?", ["No", "Sí"], help="¿El paciente hace actividad física regular?")
    
    # Convertir inputs a formato del modelo
    def convert_inputs():
        # Calcular edad en días (aproximadamente)
        age_days = age * 365
        
        # Calcular BMI
        bmi = weight / ((height/100) ** 2)
        
        # Convertir variables categóricas
        gender_num = 2 if gender == "Hombre" else 1
        chol_map = {"Normal": 1, "Sobre el normal": 2, "Muy alto": 3}
        gluc_map = {"Normal": 1, "Sobre el normal": 2, "Muy alto": 3}
        
        # Crear array en el orden correcto de características
        return np.array([[
            age_days,      # age
            gender_num,    # gender
            height,        # height
            weight,        # weight
            ap_hi,         # ap_hi
            ap_lo,         # ap_lo
            chol_map[cholesterol],  # cholesterol
            gluc_map[gluc],         # gluc
            1 if smoke == "Sí" else 0,    # smoke
            1 if alco == "Sí" else 0,     # alco
            1 if active == "Sí" else 0,   # active
            bmi            # bmi
        ]]), bmi  # Retornar también el bmi calculado
    
    # Realizar predicción con diseño mejorado
    if st.button("🔬 Analizar Riesgo Cardiovascular", type="primary"):
        input_data, bmi = convert_inputs()  # Capturar el bmi retornado
        
        # Usar el trainer guardado en session state
        trainer = st.session_state['trainer']
        prediction, probability = trainer.predict_single('Logistic Regression', input_data)
        
        # Guardar datos del usuario en session_state
        st.session_state['user_data'] = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'Alto' if prediction == 1 else 'Bajo'
        }
        
        # Mostrar resultado con diseño innovador
        risk_class = "risk-high" if prediction == 1 else "risk-low"
        risk_text = "ALTO RIESGO" if prediction == 1 else "BAJO RIESGO"
        risk_icon = "🚨" if prediction == 1 else "✅"
        
        st.markdown(f"""
        <div class="prediction-result {risk_class}">
            <h2 style="margin-bottom: 1rem;">{risk_icon} Resultado del Análisis</h2>
            <div style="display: flex; justify-content: space-around; align-items: center; margin: 2rem 0;">
                <div style="text-align: center;">
                    <h3 style="font-size: 2.5rem; margin-bottom: 0.5rem;">{probability:.1%}</h3>
                    <p style="font-size: 1.2rem;">Probabilidad de Riesgo</p>
                </div>
                <div style="text-align: center;">
                    <h3 style="font-size: 2rem; margin-bottom: 0.5rem;">{risk_text}</h3>
                    <p style="font-size: 1.2rem;">Clasificación IA</p>
                </div>
                <div style="text-align: center;">
                    <h3 style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</h3>
                    <p style="font-size: 1.2rem;">Regresión Logística</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretación personalizada
        st.markdown("### 📋 Interpretación Médica")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="floating-card" style="border-left: 5px solid #ff6b6b;">
                <h4 style="color: #e74c3c;">⚠️ Atención Requerida</h4>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    El análisis indica una <strong>probabilidad elevada</strong> de riesgo cardiovascular ({probability:.1%}). 
                    Se recomienda <strong>consulta médica inmediata</strong> para evaluación profesional.
                </p>
                
                <h5 style="color: #c0392b; margin-top: 1.5rem;">🎯 Factores de Riesgo Identificados:</h5>
                <ul style="font-size: 1rem; line-height: 1.5;">
                    {chr(10).join([f"<li>🔴 <strong>{factor}:</strong> {value}</li>" for factor, value in [
                        ("Presión Arterial", f"{ap_hi}/{ap_lo} mmHg"),
                        ("Colesterol", cholesterol),
                        ("Glucosa", gluc),
                        ("IMC", f"{bmi:.1f}"),
                        ("Tabaquismo", "Sí" if smoke == "Sí" else "No"),
                        ("Alcohol", "Sí" if alco == "Sí" else "No"),
                        ("Actividad Física", "No" if active == "No" else "Sí")
                    ]])}
                </ul>
                
                <h5 style="color: #c0392b; margin-top: 1.5rem;">🎯 Acciones Recomendadas:</h5>
                <ul style="font-size: 1rem; line-height: 1.5;">
                    <li>👨‍⚕️ <strong>Prioritario:</strong> Agendar cita con cardiólogo</li>
                    <li>🏃‍♂️ <strong>Ejercicio:</strong> Actividad física supervisada</li>
                    <li>🥗 <strong>Nutrición:</strong> Dieta cardio-saludable</li>
                    <li>🚭 <strong>Estilo de vida:</strong> Eliminar factores de riesgo</li>
                    <li>📊 <strong>Monitoreo:</strong> Control regular de presión arterial</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="floating-card" style="border-left: 5px solid #51cf66;">
                <h4 style="color: #27ae60;">✅ Resultado Favorable</h4>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    El análisis muestra una <strong>baja probabilidad</strong> de riesgo cardiovascular ({probability:.1%}). 
                    Los parámetros analizados se encuentran dentro de <strong>rangos favorables</strong>.
                </p>
                
                <h5 style="color: #2e8b57; margin-top: 1.5rem;">💚 Estado Actual:</h5>
                <ul style="font-size: 1rem; line-height: 1.5;">
                    {chr(10).join([f"<li>✅ <strong>{factor}:</strong> {value}</li>" for factor, value in [
                        ("Presión Arterial", f"{ap_hi}/{ap_lo} mmHg"),
                        ("Colesterol", cholesterol),
                        ("Glucosa", gluc),
                        ("IMC", f"{bmi:.1f}"),
                        ("Tabaquismo", "No" if smoke == "No" else "Sí"),
                        ("Alcohol", "No" if alco == "No" else "Sí"),
                        ("Actividad Física", "Sí" if active == "Sí" else "No")
                    ]])}
                </ul>
                
                <h5 style="color: #2e8b57; margin-top: 1.5rem;">💚 Recomendaciones de Mantenimiento:</h5>
                <ul style="font-size: 1rem; line-height: 1.5;">
                    <li>🎯 <strong>Continuar:</strong> Mantener hábitos saludables actuales</li>
                    <li>🏃‍♂️ <strong>Actividad:</strong> Ejercicio regular (150 min/semana)</li>
                    <li>🍎 <strong>Alimentación:</strong> Dieta balanceada y rica en nutrientes</li>
                    <li>📅 <strong>Prevención:</strong> Chequeos médicos anuales</li>
                    <li>😊 <strong>Bienestar:</strong> Manejo del estrés y descanso adecuado</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Información sobre el modelo
        st.markdown("### 🤖 Acerca del Análisis")
        st.info(f"""
        **Modelo Utilizado:** Regresión Logística - Algoritmo de IA especializado en clasificación médica
        
        **Precisión del Modelo:** {st.session_state['metrics']['Logistic Regression']['Accuracy']:.1%}
        
        **Datos Analizados:** 12 parámetros cardiovasculares (edad, presión arterial, colesterol, estilo de vida, etc.)
        
        **Entrenamiento:** Basado en 68,603 casos reales de pacientes cardiovasculares
        
        ⚠️ **Importante:** Este análisis es una herramienta de apoyo. Siempre consulte con un profesional médico para un diagnóstico definitivo.
        """)
        
        # Marcar que se realizó una predicción para habilitar el chat
        st.session_state['prediction_made'] = True
        
        # Mensaje de desbloqueo del chat
        st.success("🔓 **¡Chat IA desbloqueado!** Ahora puede acceder al asistente inteligente de Gemini en la sección 'Chat IA - Gemini'")

def comparison_page(df):
    st.header("📈 Comparación Exhaustiva de Modelos")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Primero debes entrenar los modelos en la página de 'Entrenamiento de Modelos'.")
        return
    
    # Métricas generales
    st.subheader("📊 Tabla Comparativa de Métricas")
    
    metrics_df = pd.DataFrame(st.session_state['metrics']).T
    metrics_df = metrics_df.round(4)
    metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
    
    # Colorear la tabla
    def color_metrics(val):
        if val >= 0.9:
            return 'background-color: #d4edda; color: #155724;'
        elif val >= 0.8:
            return 'background-color: #fff3cd; color: #856404;'
        else:
            return 'background-color: #f8d7da; color: #721c24;'
    
    styled_df = metrics_df.style.applymap(color_metrics).set_properties(**{
        'padding': '10px',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-weight': 'bold'
    }).set_table_styles([
        {'selector': 'th',
         'props': [('background-color', '#f8f9fa'),
                   ('color', '#212529'),
                   ('font-weight', 'bold'),
                   ('text-align', 'center'),
                   ('padding', '12px'),
                   ('border', '1px solid #dee2e6')]},
        {'selector': 'td',
         'props': [('padding', '10px'),
                   ('border', '1px solid #dee2e6')]},
        {'selector': 'tr:hover',
         'props': [('background-color', '#f5f5f5')]}
    ])
    st.dataframe(styled_df, use_container_width=True)
    
    # Gráfico de barras comparativo
    st.subheader("📊 Comparación Visual de Métricas")
    
    metric_to_plot = st.selectbox(
        "Selecciona la métrica a visualizar:",
        ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    )
    
    fig_comparison = px.bar(
        x=metrics_df.index,
        y=metrics_df[metric_to_plot],
        title=f"Comparación de {metric_to_plot} entre Modelos",
        color=metrics_df[metric_to_plot],
        color_continuous_scale='Viridis',
        text=metrics_df[metric_to_plot].round(3)
    )
    fig_comparison.update_traces(textposition='outside')
    fig_comparison.update_layout(
        xaxis_title="Modelos",
        yaxis_title=metric_to_plot,
        showlegend=False
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Matriz de confusión para cada modelo
    st.subheader("🔍 Matrices de Confusión")
    
    model_to_analyze = st.selectbox("Selecciona un modelo para análisis detallado:", list(st.session_state['models'].keys()))
    
    if model_to_analyze:
        trainer = st.session_state['trainer']
        X_test_scaled = trainer.scaler.transform(st.session_state['X_test'])
        X_test_selected = trainer.feature_selector.transform(X_test_scaled)
        
        if model_to_analyze == 'Logistic Regression':
            y_pred = trainer.models[model_to_analyze].predict(X_test_selected)
        else:
            y_pred = trainer.models[model_to_analyze].predict(st.session_state['X_test'])
        
        # Matriz de confusión
        cm = confusion_matrix(st.session_state['y_test'], y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Matriz de Confusión - {model_to_analyze}",
                labels=dict(x="Predicho", y="Real"),
                x=['Sin Riesgo', 'Con Riesgo'],
                y=['Sin Riesgo', 'Con Riesgo'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Reporte de clasificación
            report = classification_report(st.session_state['y_test'], y_pred, output_dict=True)
            
            st.markdown(f"#### 📋 Reporte Detallado - {model_to_analyze}")
            st.markdown(f"**Precisión Global:** {report['accuracy']:.3f}")
            st.markdown("**Métricas Detalladas del Modelo:**")
            for key, value in report.items():
                if isinstance(value, dict) and key not in ['accuracy', 'macro avg', 'weighted avg']:
                    st.markdown(f"- **Clase {key}:** Precisión {value.get('precision', 0):.3f}, Recall {value.get('recall', 0):.3f}")
    
    # Curvas ROC
    st.subheader("📈 Curvas ROC/AUC")
    
    fig_roc = go.Figure()
    
    for model_name, model in st.session_state['models'].items():
        if hasattr(model, 'predict_proba'):
            if model_name == 'Logistic Regression':
                y_prob = model.predict_proba(X_test_selected)[:, 1]
            else:
                y_prob = model.predict_proba(st.session_state['X_test'])[:, 1]
        else:
            # Para SVM
            if model_name == 'Logistic Regression':
                y_prob = model.decision_function(X_test_selected)
            else:
                y_prob = model.decision_function(st.session_state['X_test'])
        
        fpr, tpr, _ = roc_curve(st.session_state['y_test'], y_prob)
        auc_score = auc(fpr, tpr)
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'{model_name} (AUC = {auc_score:.3f})',
            mode='lines'
        ))
    
    # Línea diagonal
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Línea Base (AUC = 0.5)',
        showlegend=False
    ))
    
    fig_roc.update_layout(
        title='Curvas ROC - Comparación de Todos los Modelos',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Ranking final
    st.subheader("🏆 Ranking Final de Modelos")
    
    ranking_df = metrics_df.copy()
    ranking_df['Ranking'] = range(1, len(ranking_df) + 1)
    ranking_df = ranking_df[['Ranking', 'Accuracy', 'Precision', 'Recall', 'F1_Score']]
    
    st.dataframe(ranking_df, use_container_width=True)
    
    # Recomendación final
    best_model = ranking_df.index[0]
    st.success(f"🎯 **Modelo Recomendado:** {best_model}")
    st.info(f"📊 Este modelo obtuvo la mejor precisión general: {ranking_df.loc[best_model, 'Accuracy']:.1%}")

def landing_page(df):
    """
    Página de inicio con diseño atractivo y resumen del sistema
    """
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; font-size: 3rem; margin-bottom: 1rem;">
            🫀 Sistema Predictivo Cardiovascular
        </h1>
        <p style="color: white; font-size: 1.5rem; margin-bottom: 1.5rem;">
            Tecnología de Inteligencia Artificial para la Detección Temprana de Riesgos Cardiovasculares
        </p>
        <p style="color: white; font-size: 1.1rem;">
            Análisis avanzado con 6 algoritmos de Machine Learning para predicciones precisas
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estadísticas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total de Registros",
            value=f"{len(df):,}",
            delta="Dataset completo"
        )
    
    with col2:
        cardio_rate = df['cardio'].mean() * 100
        st.metric(
            label="❤️ Tasa de Enfermedad",
            value=f"{cardio_rate:.1f}%",
            delta=f"{df['cardio'].sum():,} casos"
        )
    
    with col3:
        avg_age = df['age_years'].mean()
        st.metric(
            label="👥 Edad Promedio",
            value=f"{avg_age:.1f} años",
            delta="Población adulta"
        )
    
    with col4:
        if 'metrics' in st.session_state:
            metrics_df = pd.DataFrame(st.session_state['metrics']).T
            best_accuracy = metrics_df['Accuracy'].max()
            st.metric(
                label="🎯 Precisión Real",
                value=f"{best_accuracy:.1%}",
                delta="Modelos entrenados"
            )
        else:
            st.metric(
                label="🎯 Precisión Estimada",
                value="73.0%",
                delta="Modelos entrenados"
            )
    
    st.markdown("---")
    
    # Características principales
    st.subheader("🚀 Características Principales del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 Inteligencia Artificial Avanzada
        - **3 Algoritmos Optimizados**: Logistic Regression, Decision Tree, Random Forest
        - **Validación Cruzada**: Métricas confiables y precisas
        - **Feature Engineering**: Creación automática de características relevantes
        - **Escalado Automático**: Normalización inteligente de datos
        
        ### 📊 Análisis Completo
        - **Exploración Interactiva**: Visualizaciones dinámicas con Plotly
        - **Correlaciones**: Análisis de factores de riesgo
        - **Segmentación**: Análisis por edad, género y factores de riesgo
        """)
    
    with col2:
        st.markdown("""
        ### 🔮 Predicciones Precisas
        - **Consenso de Modelos**: Múltiples algoritmos para mayor precisión
        - **Probabilidades**: Cálculo de riesgo cardiovascular
        - **Recomendaciones**: Consejos personalizados basados en IA
        - **Interfaz Intuitiva**: Fácil de usar para profesionales de la salud
        
        ### 📈 Comparación de Modelos
        - **Métricas Detalladas**: Accuracy, Precision, Recall, F1-Score
        - **Visualizaciones**: Gráficos comparativos interactivos
        - **Matriz de Confusión**: Análisis detallado de rendimiento
        """)
    
    # Call to Action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
        <h3 style="color: white; margin-bottom: 1rem;">¿Listo para comenzar?</h3>
        <p style="color: white; font-size: 1.1rem;">
            Explora los datos, entrena modelos de IA y realiza predicciones cardiovasculares precisas
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualización de muestra
    st.subheader("📊 Vista Previa de los Datos")
    
    # Distribución de riesgo cardiovascular
    cardio_dist = df['cardio'].value_counts()
    fig_preview = px.pie(
        values=cardio_dist.values,
        names=['Sin Riesgo Cardiovascular', 'Con Riesgo Cardiovascular'],
        title="Distribución de Riesgo Cardiovascular en el Dataset",
        color_discrete_sequence=['#00CC96', '#EF553B'],
        hole=0.4
    )
    fig_preview.update_traces(textposition='inside', textinfo='percent+label')
    fig_preview.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_preview, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Datos del Dataset")
        st.info(f"**Total de pacientes:** {len(df):,}")
        st.info(f"**Pacientes con riesgo:** {df['cardio'].sum():,}")
        st.info(f"**Pacientes sin riesgo:** {(df['cardio'] == 0).sum():,}")
        st.success("✅ Datos procesados y listos para análisis")


def chat_gemini_page(df):
    """
    Página de chat con Gemini para análisis y recomendaciones
    """
    st.header("🤖 Chat IA - Asistente Cardiovascular con Gemini")
    
    # Verificar si Gemini está configurado
    if model is None:
        st.error("""
        ⚠️ El modelo Gemini no está configurado correctamente.
        
        Para usar el chat, necesitas:
        1. Crear un archivo `.env` en la raíz del proyecto
        2. Agregar tu API key de Google: `GOOGLE_API_KEY=tu_api_key_aquí`
        3. Reiniciar la aplicación
        """)
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h3 style="color: white; margin-bottom: 1rem;">🧠 Asistente Inteligente Cardiovascular</h3>
        <p style="color: white; font-size: 1.1rem;">
            Chatea con Gemini IA para obtener análisis y recomendaciones personalizadas sobre tu salud cardiovascular
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Explicación del funcionamiento del sistema
    st.markdown("""
    <div class="floating-card" style="margin-bottom: 2rem;">
        <h4 style="color: #667eea;">📋 Cómo funciona el sistema:</h4>
        <ol style="font-size: 1rem; line-height: 1.6;">
            <li>📊 <strong>Recolección de datos:</strong> Ingresa tus datos personales y médicos en la página de predicciones</li>
            <li>🤖 <strong>Análisis IA:</strong> El sistema procesa tus datos usando algoritmos de machine learning</li>
            <li>🎯 <strong>Predicción:</strong> Se calcula tu riesgo cardiovascular y probabilidad</li>
            <li>💡 <strong>Recomendaciones:</strong> Recibe consejos personalizados basados en tus resultados</li>
            <li>💬 <strong>Chat IA:</strong> Consulta dudas específicas sobre tu salud cardiovascular</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay predicciones realizadas
    if 'prediction_made' not in st.session_state:
        st.markdown("""
        <div class="floating-card" style="text-align: center; border-left: 5px solid #f39c12;">
            <h3 style="color: #e67e22;">🔒 Chat IA Bloqueado</h3>
            <p style="font-size: 1.1rem; color: #666;">
                Para acceder al asistente inteligente de Gemini, primero debe realizar una predicción cardiovascular.
            </p>
            <p style="font-size: 1rem; color: #888;">
                Vaya a la página de <strong>Predicciones</strong> y analice un paciente para desbloquear esta función.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Obtener datos del usuario de session_state
    user_data = st.session_state.get('user_data', {})
    if not user_data:
        st.warning("⚠️ No hay datos de usuario disponibles. Por favor, realice una predicción primero.")
        return
    
    # Preparar contexto para Gemini
    context = f"""
    Eres un asistente médico especializado en salud cardiovascular. 
    Analiza los siguientes datos del paciente y proporciona recomendaciones personalizadas.
    
    IMPORTANTE: Formatea tu respuesta usando HTML con el siguiente estilo profesional:
    
    Estructura tu respuesta así:
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">Análisis de Factores de Riesgo</h4>
        <ul style="color: #34495e; line-height: 1.6;">
            <li><strong style="color: #2980b9;">Factor 1:</strong> Explicación</li>
            <li><strong style="color: #2980b9;">Factor 2:</strong> Explicación</li>
        </ul>
    </div>
    
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;">
        <h4 style="color: #2c3e50; border-bottom: 2px solid #27ae60; padding-bottom: 10px;">Recomendaciones Específicas</h4>
        <ul style="color: #34495e; line-height: 1.6;">
            <li><strong style="color: #27ae60;">Recomendación 1:</strong> Explicación</li>
            <li><strong style="color: #27ae60;">Recomendación 2:</strong> Explicación</li>
        </ul>
    </div>
    
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;">
        <h4 style="color: #2c3e50; border-bottom: 2px solid #e67e22; padding-bottom: 10px;">Consejos Prácticos</h4>
        <ul style="color: #34495e; line-height: 1.6;">
            <li><strong style="color: #e67e22;">Consejo 1:</strong> Explicación</li>
            <li><strong style="color: #e67e22;">Consejo 2:</strong> Explicación</li>
        </ul>
    </div>
    
    Usa estos colores para diferentes elementos:
    - Títulos principales: #2c3e50 (azul oscuro)
    - Factores de riesgo: #2980b9 (azul)
    - Recomendaciones: #27ae60 (verde)
    - Consejos: #e67e22 (naranja)
    - Texto normal: #34495e (gris oscuro)
    - Advertencias: #c0392b (rojo)
    
    Datos del paciente:
    - Edad: {user_data.get('age', 'N/A')} años
    - Género: {user_data.get('gender', 'N/A')}
    - Altura: {user_data.get('height', 'N/A')} cm
    - Peso: {user_data.get('weight', 'N/A')} kg
    - IMC: {user_data.get('bmi', 'N/A'):.1f}
    - Presión Arterial: {user_data.get('ap_hi', 'N/A')}/{user_data.get('ap_lo', 'N/A')} mmHg
    - Colesterol: {user_data.get('cholesterol', 'N/A')}
    - Glucosa: {user_data.get('gluc', 'N/A')}
    - Tabaquismo: {user_data.get('smoke', 'N/A')}
    - Alcohol: {user_data.get('alco', 'N/A')}
    - Actividad Física: {user_data.get('active', 'N/A')}
    - Riesgo Cardiovascular: {user_data.get('risk_level', 'N/A')} ({user_data.get('probability', 'N/A'):.1%})
    """
    
    # Sugerencias de preguntas personalizadas
    st.subheader("💡 Preguntas Sugeridas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Análisis de mi salud"):
            user_question = "¿Cómo está mi salud cardiovascular según mis datos?"
            st.session_state.user_input = user_question
    
    with col2:
        if st.button("🎯 Factores de riesgo"):
            user_question = "¿Cuáles son mis principales factores de riesgo cardiovascular?"
            st.session_state.user_input = user_question
    
    with col3:
        if st.button("💊 Recomendaciones"):
            user_question = "¿Qué recomendaciones específicas me das para mejorar mi salud cardiovascular?"
            st.session_state.user_input = user_question
    
    # Input del usuario
    user_input = st.text_input(
        "Escribe tu pregunta sobre tu salud cardiovascular:",
        key="user_input_field",
        value=st.session_state.get('user_input', '')
    )
    
    if st.button("Enviar 🚀") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        try:
            # Generar respuesta con Gemini
            prompt = f"{context}\n\nPregunta del usuario: {user_input}"
            response = model.generate_content(prompt)
            ai_response = response.text
            
            # Asegurar que la respuesta tenga el formato HTML correcto
            if not ai_response.startswith('<'):
                ai_response = f"""
                <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {ai_response.replace(chr(10), '<br>')}
                </div>
                """
            
        except Exception as e:
            ai_response = f"""
            <div style="background-color: #fdf2f2; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #c0392b; border-bottom: 2px solid #c0392b; padding-bottom: 10px;">⚠️ Error</h4>
                <p style="color: #34495e;">Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo.</p>
                <p style="color: #c0392b; font-size: 0.9em;">Error: {str(e)}</p>
            </div>
            """
        
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.session_state.user_input = ""
    
    # Mostrar historial de chat con mejor formato
    if st.session_state.chat_history:
        st.subheader("💬 Conversación")
        
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #f0f7ff; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong style="color: #2980b9;">👤 Tú:</strong><br>
                    <p style="margin: 0.5rem 0 0 0; color: #2c3e50;">{message['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong style="color: #2c3e50;">🤖 Gemini IA:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Botón para limpiar chat
    if st.button("🗑️ Limpiar Conversación"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()
