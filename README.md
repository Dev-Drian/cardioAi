# CardioAI - Predictor Cardiovascular

Sistema de predicción cardiovascular basado en Inteligencia Artificial que utiliza múltiples algoritmos de machine learning para predecir el riesgo cardiovascular.

## 🚀 Despliegue en Streamlit Cloud

Para desplegar esta aplicación en Streamlit Cloud, sigue estos pasos:

1. Crea una cuenta en [Streamlit Cloud](https://streamlit.io/cloud) si aún no tienes una.

2. Sube tu código a un repositorio de GitHub.

3. En Streamlit Cloud:
   - Haz clic en "New app"
   - Selecciona tu repositorio
   - Selecciona el archivo principal: `app.py`
   - En "Advanced settings", agrega la variable de entorno:
     - `GOOGLE_API_KEY`: Tu API key de Google para Gemini

4. Haz clic en "Deploy"

## 📋 Requisitos Locales

Para ejecutar el proyecto localmente:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## 🔑 Configuración de API Key

Para usar la funcionalidad de chat con Gemini:

1. Obtén una API key de Google AI Studio
2. Crea un archivo `.env` en la raíz del proyecto
3. Agrega tu API key: `GOOGLE_API_KEY=tu_api_key_aquí`

## 📁 Estructura del Proyecto

- `app.py`: Aplicación principal
- `data_processor.py`: Procesamiento de datos
- `model_trainer.py`: Entrenamiento de modelos
- `visualizations.py`: Visualizaciones y gráficos
- `requirements.txt`: Dependencias del proyecto
- `.streamlit/config.toml`: Configuración de Streamlit

## 🛠️ Tecnologías Utilizadas

- Streamlit
- Python
- Scikit-learn
- Plotly
- Google Gemini AI
- Pandas
- NumPy
