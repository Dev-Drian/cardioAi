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

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizations import Visualizations

def main():
    st.set_page_config(
        page_title="Sistema Predictivo Cardiovascular",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("❤️ Sistema Predictivo Cardiovascular")
    st.markdown("---")
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox(
        "Selecciona una página:",
        ["📊 Exploración de Datos", "🤖 Entrenamiento de Modelos", "🔮 Predicciones", "📈 Comparación de Modelos"]
    )
    
    # Cargar datos
    @st.cache_data
    def load_data():
        try:
            # Intentar cargar el archivo desde la ubicación esperada
            df = pd.read_csv('attached_assets/cardio_train.csv', sep=';')
            
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
    
    # Navegación por páginas
    if page == "📊 Exploración de Datos":
        exploration_page(df, processor)
    elif page == "🤖 Entrenamiento de Modelos":
        training_page(df)
    elif page == "🔮 Predicciones":
        prediction_page(df)
    elif page == "📈 Comparación de Modelos":
        comparison_page(df)

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
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Modelo destacado
        best_model = metrics_df.index[0]
        best_accuracy = metrics_df.loc[best_model, 'Accuracy']
        
        st.success(f"🏆 **Mejor modelo:** {best_model} con {best_accuracy:.1%} de precisión")
        
        # Análisis detallado del mejor modelo
        st.subheader(f"🔍 Análisis Detallado: {best_model}")
        
        if best_model == 'Random Forest':
            # Feature importance para Random Forest
            model = st.session_state['models'][best_model]
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
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
        
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
    st.header("🔮 Predicciones Individuales")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Primero debes entrenar los modelos en la página de 'Entrenamiento de Modelos'.")
        return
    
    st.subheader("📝 Ingresa los Datos del Paciente")
    
    # Formulario de entrada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Edad (años)", 18, 100, 50)
        gender = st.selectbox("Género", ["Mujer", "Hombre"])
        height = st.slider("Altura (cm)", 120, 220, 170)
        weight = st.slider("Peso (kg)", 30, 200, 70)
    
    with col2:
        ap_hi = st.slider("Presión Sistólica", 80, 250, 120)
        ap_lo = st.slider("Presión Diastólica", 40, 150, 80)
        cholesterol = st.selectbox("Colesterol", ["Normal", "Sobre el normal", "Muy alto"])
        gluc = st.selectbox("Glucosa", ["Normal", "Sobre el normal", "Muy alto"])
    
    with col3:
        smoke = st.selectbox("¿Fuma?", ["No", "Sí"])
        alco = st.selectbox("¿Consume alcohol?", ["No", "Sí"])
        active = st.selectbox("¿Hace actividad física?", ["No", "Sí"])
    
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
        
        return np.array([[
            age_days, gender_num, height, weight, ap_hi, ap_lo,
            chol_map[cholesterol], gluc_map[gluc],
            1 if smoke == "Sí" else 0,
            1 if alco == "Sí" else 0,
            1 if active == "Sí" else 0,
            bmi
        ]])
    
    # Realizar predicción
    if st.button("🔬 Realizar Predicción", type="primary"):
        input_data = convert_inputs()
        
        st.markdown("---")
        st.subheader("📊 Resultados de la Predicción")
        
        # Predicciones de todos los modelos
        predictions = {}
        probabilities = {}
        
        for model_name, model in st.session_state['models'].items():
            pred = model.predict(input_data)[0]
            predictions[model_name] = pred
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_data)[0][1]
                probabilities[model_name] = prob
            else:
                # Para SVM sin probabilidad
                decision = model.decision_function(input_data)[0]
                prob = 1 / (1 + np.exp(-decision))  # Aproximación sigmoide
                probabilities[model_name] = prob
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Predicciones por Modelo")
            for model_name, pred in predictions.items():
                risk_level = "🔴 ALTO RIESGO" if pred == 1 else "🟢 BAJO RIESGO"
                st.write(f"**{model_name}:** {risk_level}")
        
        with col2:
            st.markdown("#### 📊 Probabilidades de Riesgo")
            prob_df = pd.DataFrame({
                'Modelo': list(probabilities.keys()),
                'Probabilidad': [f"{p:.1%}" for p in probabilities.values()],
                'Valor': list(probabilities.values())
            })
            
            fig_prob = px.bar(
                prob_df,
                x='Modelo',
                y='Valor',
                title="Probabilidad de Enfermedad Cardiovascular",
                color='Valor',
                color_continuous_scale='RdYlGn_r',
                text='Probabilidad'
            )
            fig_prob.update_traces(textposition='outside')
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # Consenso de modelos
        avg_prob = np.mean(list(probabilities.values()))
        consensus_pred = 1 if avg_prob > 0.5 else 0
        
        st.markdown("---")
        st.subheader("🏆 Consenso de Modelos")
        
        if consensus_pred == 1:
            st.error(f"⚠️ **RIESGO CARDIOVASCULAR ELEVADO** - Probabilidad promedio: {avg_prob:.1%}")
            st.markdown("**Recomendaciones:**")
            st.markdown("- 👨‍⚕️ Consultar con un cardiólogo")
            st.markdown("- 🏃‍♂️ Aumentar la actividad física")
            st.markdown("- 🥗 Mejorar la dieta")
            st.markdown("- 🚭 Evitar el tabaco y alcohol")
        else:
            st.success(f"✅ **RIESGO CARDIOVASCULAR BAJO** - Probabilidad promedio: {avg_prob:.1%}")
            st.markdown("**Recomendaciones:**")
            st.markdown("- 💚 Mantener hábitos saludables")
            st.markdown("- 🏃‍♂️ Continuar con actividad física regular")
            st.markdown("- 🍎 Mantener una dieta balanceada")
            st.markdown("- 📅 Revisiones médicas periódicas")

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
            return 'background-color: #d4edda'
        elif val >= 0.8:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = metrics_df.style.applymap(color_metrics)
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
        model = st.session_state['models'][model_to_analyze]
        y_pred = model.predict(st.session_state['X_test'])
        
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
            st.markdown(f"**Precisión Clase 0:** {report['0']['precision']:.3f}")
            st.markdown(f"**Precisión Clase 1:** {report['1']['precision']:.3f}")
            st.markdown(f"**Recall Clase 0:** {report['0']['recall']:.3f}")
            st.markdown(f"**Recall Clase 1:** {report['1']['recall']:.3f}")
            st.markdown(f"**F1-Score Macro:** {report['macro avg']['f1-score']:.3f}")
    
    # Curvas ROC
    st.subheader("📈 Curvas ROC/AUC")
    
    fig_roc = go.Figure()
    
    for model_name, model in st.session_state['models'].items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(st.session_state['X_test'])[:, 1]
        else:
            # Para SVM
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

if __name__ == "__main__":
    main()
