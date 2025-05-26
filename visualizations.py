import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizations:
    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_age_distribution(self, df, st):
        """
        Gráfico de distribución de edad por presencia de enfermedad cardiovascular
        """
        # Crear grupos de edad
        age_groups = pd.cut(df['age_years'], bins=[18, 30, 40, 50, 60, 70, 100], 
                           labels=['18-30', '30-40', '40-50', '50-60', '60-70', '70+'])
        
        df_temp = df.copy()
        df_temp['age_group'] = age_groups
        df_temp['cardio_status'] = df_temp['cardio'].map({0: 'Sin Enfermedad', 1: 'Con Enfermedad'})
        
        # Gráfico de barras agrupadas
        fig = px.histogram(
            df_temp,
            x='age_group',
            color='cardio_status',
            barmode='group',
            title='Distribución de Enfermedades Cardiovasculares por Grupo de Edad',
            labels={'age_group': 'Grupo de Edad', 'count': 'Número de Pacientes'},
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        
        fig.update_layout(
            xaxis_title="Grupo de Edad",
            yaxis_title="Número de Pacientes",
            legend_title="Estado Cardiovascular"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas adicionales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Edad Promedio (Sin Enfermedad)", f"{df[df['cardio']==0]['age_years'].mean():.1f} años")
        with col2:
            st.metric("Edad Promedio (Con Enfermedad)", f"{df[df['cardio']==1]['age_years'].mean():.1f} años")
    
    def plot_blood_pressure_analysis(self, df, st):
        """
        Análisis de presión arterial
        """
        # Scatter plot de presión sistólica vs diastólica
        df_temp = df.copy()
        df_temp['cardio_status'] = df_temp['cardio'].map({0: 'Sin Enfermedad', 1: 'Con Enfermedad'})
        
        fig = px.scatter(
            df_temp.sample(n=min(5000, len(df_temp))),  # Muestra para mejor rendimiento
            x='ap_lo',
            y='ap_hi',
            color='cardio_status',
            title='Relación entre Presión Sistólica y Diastólica',
            labels={'ap_lo': 'Presión Diastólica (mmHg)', 'ap_hi': 'Presión Sistólica (mmHg)'},
            color_discrete_sequence=['#2E8B57', '#DC143C'],
            opacity=0.6
        )
        
        # Agregar líneas de referencia para hipertensión
        fig.add_hline(y=140, line_dash="dash", line_color="red", 
                     annotation_text="Hipertensión Sistólica (≥140)")
        fig.add_vline(x=90, line_dash="dash", line_color="red", 
                     annotation_text="Hipertensión Diastólica (≥90)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas de presión arterial
        st.subheader("📊 Estadísticas de Presión Arterial")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hypertension_rate = df['hypertension'].mean() * 100
            st.metric("Tasa de Hipertensión", f"{hypertension_rate:.1f}%")
        
        with col2:
            avg_systolic = df['ap_hi'].mean()
            st.metric("Presión Sistólica Promedio", f"{avg_systolic:.0f} mmHg")
        
        with col3:
            avg_diastolic = df['ap_lo'].mean()
            st.metric("Presión Diastólica Promedio", f"{avg_diastolic:.0f} mmHg")
        
        with col4:
            avg_pulse_pressure = df['pulse_pressure'].mean()
            st.metric("Presión de Pulso Promedio", f"{avg_pulse_pressure:.0f} mmHg")
    
    def plot_risk_factors(self, df, st):
        """
        Análisis de factores de riesgo
        """
        # Calcular prevalencia de factores de riesgo
        risk_factors = {
            'Tabaquismo': df['smoke'].mean() * 100,
            'Consumo de Alcohol': df['alco'].mean() * 100,
            'Sedentarismo': (1 - df['active']).mean() * 100,
            'Colesterol Alto': (df['cholesterol'] > 1).mean() * 100,
            'Glucosa Alta': (df['gluc'] > 1).mean() * 100,
            'Sobrepeso/Obesidad': (df['bmi'] > 25).mean() * 100
        }
        
        # Gráfico de barras horizontal
        fig = px.bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            title='Prevalencia de Factores de Riesgo Cardiovascular',
            labels={'x': 'Prevalencia (%)', 'y': 'Factor de Riesgo'},
            color=list(risk_factors.values()),
            color_continuous_scale='Reds'
        )
        
        fig.update_traces(text=[f"{v:.1f}%" for v in risk_factors.values()], textposition='outside')
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis por enfermedad cardiovascular
        st.subheader("🔍 Factores de Riesgo por Presencia de Enfermedad")
        
        risk_by_cardio = []
        factors = ['smoke', 'alco', 'cholesterol', 'gluc']
        factor_names = ['Tabaquismo', 'Alcohol', 'Colesterol Alto', 'Glucosa Alta']
        
        for factor, name in zip(factors, factor_names):
            if factor in ['cholesterol', 'gluc']:
                no_cardio = (df[df['cardio']==0][factor] > 1).mean() * 100
                with_cardio = (df[df['cardio']==1][factor] > 1).mean() * 100
            else:
                no_cardio = df[df['cardio']==0][factor].mean() * 100
                with_cardio = df[df['cardio']==1][factor].mean() * 100
            
            risk_by_cardio.extend([
                {'Factor': name, 'Grupo': 'Sin Enfermedad', 'Prevalencia': no_cardio},
                {'Factor': name, 'Grupo': 'Con Enfermedad', 'Prevalencia': with_cardio}
            ])
        
        risk_df = pd.DataFrame(risk_by_cardio)
        
        fig2 = px.bar(
            risk_df,
            x='Factor',
            y='Prevalencia',
            color='Grupo',
            barmode='group',
            title='Comparación de Factores de Riesgo por Presencia de Enfermedad Cardiovascular',
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def plot_correlation_matrix(self, df, st):
        """
        Matriz de correlación de variables numéricas
        """
        # Seleccionar variables numéricas relevantes
        numeric_vars = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
        
        corr_matrix = df[numeric_vars].corr()
        
        # Crear heatmap con plotly
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="Matriz de Correlación de Variables",
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlaciones más fuertes con la variable objetivo
        cardio_corr = corr_matrix['cardio'].abs().sort_values(ascending=False)[1:]  # Excluir autocorrelación
        
        st.subheader("🎯 Variables Más Correlacionadas con Enfermedad Cardiovascular")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 5 Correlaciones Positivas")
            for var, corr in cardio_corr.head().items():
                st.write(f"**{var}:** {corr:.3f}")
        
        with col2:
            # Gráfico de barras de correlaciones
            fig_corr = px.bar(
                x=cardio_corr.head().values,
                y=cardio_corr.head().index,
                orientation='h',
                title="Correlaciones con Enfermedad Cardiovascular",
                color=cardio_corr.head().values,
                color_continuous_scale='Viridis'
            )
            fig_corr.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_corr, use_container_width=True)
    
    def plot_gender_analysis(self, df, st):
        """
        Análisis por género
        """
        df_temp = df.copy()
        df_temp['gender_label'] = df_temp['gender'].map({1: 'Mujer', 2: 'Hombre'})
        df_temp['cardio_status'] = df_temp['cardio'].map({0: 'Sin Enfermedad', 1: 'Con Enfermedad'})
        
        # Distribución por género
        fig1 = px.histogram(
            df_temp,
            x='gender_label',
            color='cardio_status',
            barmode='group',
            title='Distribución de Enfermedades Cardiovasculares por Género',
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Métricas por género
        st.subheader("📊 Estadísticas por Género")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👩 Mujeres")
            women_data = df[df['gender'] == 1]
            cardio_rate_women = women_data['cardio'].mean() * 100
            avg_age_women = women_data['age_years'].mean()
            avg_bmi_women = women_data['bmi'].mean()
            
            st.metric("Tasa de Enfermedad Cardiovascular", f"{cardio_rate_women:.1f}%")
            st.metric("Edad Promedio", f"{avg_age_women:.1f} años")
            st.metric("BMI Promedio", f"{avg_bmi_women:.1f}")
        
        with col2:
            st.markdown("#### 👨 Hombres")
            men_data = df[df['gender'] == 2]
            cardio_rate_men = men_data['cardio'].mean() * 100
            avg_age_men = men_data['age_years'].mean()
            avg_bmi_men = men_data['bmi'].mean()
            
            st.metric("Tasa de Enfermedad Cardiovascular", f"{cardio_rate_men:.1f}%")
            st.metric("Edad Promedio", f"{avg_age_men:.1f} años")
            st.metric("BMI Promedio", f"{avg_bmi_men:.1f}")
        
        # Análisis de BMI por género
        fig2 = px.box(
            df_temp,
            x='gender_label',
            y='bmi',
            color='cardio_status',
            title='Distribución de BMI por Género y Estado Cardiovascular',
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        
        st.plotly_chart(fig2, use_container_width=True)
