import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="🏠 Predictor de Precios de Casas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .sidebar-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Cargar el dataset original para análisis"""
    try:
        df = pd.read_csv('data/train.csv')
        return df
    except:
        st.error("❌ No se pudo cargar el dataset. Asegúrate de que el archivo train.csv esté en la carpeta data/")
        return None

@st.cache_resource
def load_models():
    """Cargar modelos y artefactos guardados"""
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return rf_model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {e}")
        return None, None, None

def preprocess_input_data(input_data, feature_columns, scaler):
    """Preprocesar datos de entrada para predicción"""
    # 1. Crear DataFrame con los datos de entrada
    df = pd.DataFrame([input_data])
    
    # 2. Manejar datos faltantes
    # Debes replicar exactamente la lógica de tu API
    df['Alley'] = df['Alley'].fillna('NA')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
    df['PoolQC'] = df['PoolQC'].fillna('NA')
    df['Fence'] = df['Fence'].fillna('NA')
    df['MiscFeature'] = df['MiscFeature'].fillna('NA')
    df['LotFrontage'] = df['LotFrontage'].fillna(69.0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df.apply(
        lambda row: row['YearBuilt'] if pd.isna(row['GarageYrBlt']) else row['GarageYrBlt'], axis=1
    )

    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False) # type: ignore
    
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    numeric_columns_to_scale = df_final.select_dtypes(include=[np.number]).columns

    return df_final

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🏠 Dashboard de Predicción de Precios de Casas</h1>', 
                unsafe_allow_html=True)
    
    # Cargar datos y modelos
    df = load_data()
    rf_model, scaler, feature_columns = load_models()
    
    if df is None or rf_model is None:
        st.stop()
    
    # Sidebar para navegación
    st.sidebar.markdown('<div class="sidebar-header">📊 Navegación</div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Selecciona una página:",
        ["🏠 Predictor", "📈 Análisis Exploratorio", "🤖 Comparación de Modelos", "📊 Estadísticas del Dataset"]
    )
    
    if page == "🏠 Predictor":
        show_predictor_page(df, rf_model, scaler, feature_columns)
    elif page == "📈 Análisis Exploratorio":
        show_analysis_page(df)
    elif page == "🤖 Comparación de Modelos":
        show_models_comparison(df)
    else:
        show_statistics_page(df)

def show_predictor_page(df, rf_model, scaler, feature_columns):
    """Página principal del predictor"""
    st.markdown("## 🎯 Predictor de Precios")
    st.markdown("Ingresa las características de la casa para obtener una predicción de precio:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Características Principales")
        overall_qual = st.selectbox("Calidad General (1-10)", range(1, 11), index=6)
        gr_liv_area = st.number_input("Área Habitable (sq ft)", min_value=334, max_value=5642, value=1500)
        total_bsmt_sf = st.number_input("Área Total Sótano (sq ft)", min_value=0, max_value=6110, value=1000)
        garage_area = st.number_input("Área del Garaje (sq ft)", min_value=0, max_value=1418, value=500)
        year_built = st.number_input("Año de Construcción", min_value=1872, max_value=2010, value=2000)
        
        st.markdown("### 📐 Dimensiones")
        first_flr_sf = st.number_input("1er Piso (sq ft)", min_value=334, max_value=4692, value=800)
        second_flr_sf = st.number_input("2do Piso (sq ft)", min_value=0, max_value=2065, value=0)
        lot_area = st.number_input("Área del Lote (sq ft)", min_value=1300, max_value=215245, value=10000)
    
    with col2:
        st.markdown("### 🏠 Detalles de la Casa")
        neighborhood = st.selectbox("Vecindario", df['Neighborhood'].unique())
        house_style = st.selectbox("Estilo de Casa", df['HouseStyle'].unique())
        exterior_qual = st.selectbox("Calidad Exterior", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2)
        kitchen_qual = st.selectbox("Calidad Cocina", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2)
        
        st.markdown("### 🚿 Baños y Habitaciones")
        full_bath = st.number_input("Baños Completos", min_value=0, max_value=3, value=2)
        bedroom = st.number_input("Habitaciones", min_value=0, max_value=8, value=3)
        garage_cars = st.number_input("Capacidad Garaje (autos)", min_value=0, max_value=4, value=2)
    
    # Botón de predicción
    if st.button("🔮 Predecir Precio", type="primary"):
        # Crear diccionario con datos de entrada
        input_data = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            'GarageArea': garage_area,
            'YearBuilt': year_built,
            '1stFlrSF': first_flr_sf,
            '2ndFlrSF': second_flr_sf,
            'LotArea': lot_area,
            'Neighborhood': neighborhood,
            'HouseStyle': house_style,
            'ExterQual': exterior_qual,
            'KitchenQual': kitchen_qual,
            'FullBath': full_bath,
            'Bedroom': bedroom,
            'GarageCars': garage_cars,
            # Valores por defecto para otras características
            'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 60,
            'Street': 'Pave', 'Alley': 'NA', 'LotShape': 'Reg',
            'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LotConfig': 'Inside',
            'LandSlope': 'Gtl', 'Condition1': 'Norm', 'Condition2': 'Norm',
            'BldgType': '1Fam', 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg',
            'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None', 'MasVnrArea': 0, 'ExterCond': 'TA',
            'Foundation': 'PConc', 'BsmtQual': 'TA', 'BsmtCond': 'TA',
            'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ', 'BsmtFinSF1': total_bsmt_sf//2,
            'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': total_bsmt_sf//2,
            'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y',
            'Electrical': 'SBrkr', 'LowQualFinSF': 0, 'BsmtFullBath': 0,
            'BsmtHalfBath': 0, 'HalfBath': 0, 'Kitchen': 1,
            'TotRmsAbvGrd': bedroom + 2, 'Functional': 'Typ', 'Fireplaces': 0,
            'FireplaceQu': 'NA', 'GarageType': 'Attchd', 'GarageYrBlt': year_built,
            'GarageFinish': 'RFn', 'GarageQual': 'TA', 'GarageCond': 'TA',
            'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 0,
            'EnclosedPorch': 0, '3SsnPorch': 0, 'ScreenPorch': 0,
            'PoolArea': 0, 'PoolQC': 'NA', 'Fence': 'NA',
            'MiscFeature': 'NA', 'MiscVal': 0, 'MoSold': 6,
            'YrSold': 2008, 'SaleType': 'WD', 'SaleCondition': 'Normal',
            'YearRemodAdd': year_built
        }
        
        try:
            # Preprocesar datos
            X_pred = preprocess_input_data(input_data, feature_columns, scaler)
            
            # Hacer predicción
            prediction = rf_model.predict(X_pred)[0]
            
            # Mostrar resultado
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: white; margin-bottom: 1rem;">💰 Precio Predicho</h2>
                <h1 style="color: #FFD700; font-size: 3rem; margin: 0;">${prediction:,.0f}</h1>
                <p style="color: #E0E0E0; margin-top: 1rem;">Basado en las características ingresadas</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Análisis del precio
            avg_price = df['SalePrice'].mean()
            if prediction > avg_price * 1.5:
                st.success("🏆 ¡Casa de lujo! El precio está muy por encima del promedio.")
            elif prediction > avg_price:
                st.info("📈 Casa por encima del precio promedio del mercado.")
            elif prediction > avg_price * 0.7:
                st.warning("📊 Casa con precio cerca del promedio del mercado.")
            else:
                st.error("📉 Casa con precio por debajo del promedio del mercado.")
                
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

def show_analysis_page(df):
    """Página de análisis exploratorio"""
    st.markdown("## 📈 Análisis Exploratorio de Datos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Total Casas", f"{len(df):,}")
    with col2:
        st.metric("💰 Precio Promedio", f"${df['SalePrice'].mean():,.0f}")
    with col3:
        st.metric("📊 Precio Mediano", f"${df['SalePrice'].median():,.0f}")
    with col4:
        st.metric("🔄 Desv. Estándar", f"${df['SalePrice'].std():,.0f}")
    
    # Gráficos interactivos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribución de Precios")
        fig = px.histogram(df, x='SalePrice', nbins=50, 
                          title='Distribución de Precios de Casas')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏘️ Precios por Vecindario")
        neighborhood_prices = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=True)
        fig = px.bar(x=neighborhood_prices.values, y=neighborhood_prices.index,
                    orientation='h', title='Precio Promedio por Vecindario')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏠 Área vs Precio")
        fig = px.scatter(df, x='GrLivArea', y='SalePrice',
                        title='Área Habitable vs Precio de Venta',
                        trendline="ols")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏗️ Calidad vs Precio")
        quality_prices = df.groupby('OverallQual')['SalePrice'].mean()
        fig = px.line(x=quality_prices.index, y=quality_prices.values,
                     title='Calidad General vs Precio Promedio',
                     markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de correlación
    st.markdown("### 🔗 Correlación entre Variables Numéricas")
    numeric_cols = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 
                   'YearBuilt', 'GarageArea', 'FullBath', 'TotRmsAbvGrd']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                   title="Matriz de Correlación")
    st.plotly_chart(fig, use_container_width=True)

def show_models_comparison(df):
    """Página de comparación de modelos"""
    st.markdown("## 🤖 Comparación de Modelos")
    
    # Métricas de rendimiento (simuladas basadas en tu análisis)
    models_performance = {
        'Modelo': ['Regresión Lineal', 'Regresión Polinómica', 'Random Forest'],
        'R²': [0.4414, 0.8631, 0.8905],
        'MAE': [21200, 21627, 17634],
        'RMSE': [65460, 32400, 28986]
    }
    
    perf_df = pd.DataFrame(models_performance)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Métricas de Rendimiento")
        st.dataframe(perf_df, use_container_width=True)
        
        # Gráfico de R²
        fig = px.bar(perf_df, x='Modelo', y='R²', 
                    title='Comparación R² por Modelo',
                    color='R²', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Error Absoluto Medio (MAE)")
        fig = px.bar(perf_df, x='Modelo', y='MAE',
                    title='MAE por Modelo (menor es mejor)',
                    color='MAE', color_continuous_scale='reds_r')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 RMSE por Modelo")
        fig = px.bar(perf_df, x='Modelo', y='RMSE',
                    title='RMSE por Modelo (menor es mejor)',
                    color='RMSE', color_continuous_scale='oranges_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # Importancia de características
    st.markdown("### 🏆 Top 15 Características Más Importantes (Random Forest)")
    
    # Datos de importancia (basados en tu análisis)
    feature_importance = {
        'Característica': ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '2ndFlrSF', 
                          'BsmtFinSF1', '1stFlrSF', 'LotArea', 'GarageArea',
                          'YearBuilt', 'GarageCars', 'LotFrontage', 'TotRmsAbvGrd',
                          'YearRemodAdd', 'OpenPorchSF', 'FullBath'],
        'Importancia': [0.557672, 0.121496, 0.034023, 0.033340, 0.028375,
                       0.026401, 0.016683, 0.015096, 0.012254, 0.011868,
                       0.008371, 0.006708, 0.006405, 0.006239, 0.006200]
    }
    
    imp_df = pd.DataFrame(feature_importance)
    fig = px.bar(imp_df, x='Importancia', y='Característica',
                orientation='h', title='Importancia de Características')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_statistics_page(df):
    """Página de estadísticas del dataset"""
    st.markdown("## 📊 Estadísticas del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Estadísticas Descriptivas - Precio")
        price_stats = df['SalePrice'].describe()
        for stat, value in price_stats.items():
            if stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                st.metric(stat.title(), f"${value:,.0f}")
    
    with col2:
        st.markdown("### 🏠 Distribución por Tipo de Casa")
        house_style_counts = df['HouseStyle'].value_counts()
        fig = px.pie(values=house_style_counts.values, 
                    names=house_style_counts.index,
                    title='Distribución por Estilo de Casa')
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots
    st.markdown("### 📦 Análisis de Precios por Categorías")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x='OverallQual', y='SalePrice',
                    title='Precio por Calidad General')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seleccionar algunos vecindarios principales
        top_neighborhoods = df['Neighborhood'].value_counts().head(10).index
        df_filtered = df[df['Neighborhood'].isin(top_neighborhoods)]
        fig = px.box(df_filtered, x='Neighborhood', y='SalePrice',
                    title='Precio por Vecindario (Top 10)')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de datos faltantes
    st.markdown("### ❌ Análisis de Datos Faltantes")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores Faltantes': missing_data.values,
            'Porcentaje': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ ¡No hay datos faltantes en el dataset!")

if __name__ == "__main__":
    main()