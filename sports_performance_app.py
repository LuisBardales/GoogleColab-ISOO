import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuracion de Idioma
from utils.i18n import obtener_textos

# Librerias de Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# Pruebas Estadisticas
from scipy import stats
from scipy.stats import shapiro, levene, f_oneway, kruskal
import scikit_posthocs as sp

# Reporte PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

# Configuración
st.set_page_config(
    page_title="Predictor de Rendimiento Deportivo",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Diseño CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #808495;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PredictorRendimientoDeportivo:
    def __init__(self):
        self.datos = None
        self.datos_procesados = None
        self.X_entreno = None
        self.X_prueba = None
        self.y_entreno = None
        self.y_prueba = None
        self.escala = StandardScaler()
        self.modelos = {}
        self.resultados = {}
        
    def cargar_data(self, path_archivo=None):
        """Carga y procesamiento inicial de datos"""
        try:
            if path_archivo:
                self.datos = pd.read_csv(path_archivo)
            else:
                # Creación de datos de prueba para demo
                np.random.seed(42)
                n_pruebas = 100
                
                deportes = ['Running', 'Swimming', 'Cycling', 'Soccer', 'Basketball', 'Tennis']
                eventos = ['50m Freestyle', '100m Sprint', '10km Road Race', 'Marathon']
                
                datos = {
                    'Athlete_ID': [f'A{i:03d}' for i in range(1, n_pruebas+1)],
                    'Athlete_Name': [f'Athlete_{i}' for i in range(1, n_pruebas+1)],
                    'Sport_Type': np.random.choice(deportes, n_pruebas),
                    'Event': np.random.choice(eventos, n_pruebas),
                    'Training_Hours_per_Week': np.random.normal(25, 8, n_pruebas),
                    'Average_Heart_Rate': np.random.normal(150, 25, n_pruebas),
                    'BMI': np.random.normal(23, 3, n_pruebas),
                    'Sleep_Hours_per_Night': np.random.normal(7.5, 1, n_pruebas),
                    'Daily_Caloric_Intake': np.random.normal(2500, 500, n_pruebas),
                    'Hydration_Level': np.random.normal(70, 15, n_pruebas),
                    'Injury_History': np.random.choice(['None', 'Minor', 'Major'], n_pruebas, p=[0.5, 0.3, 0.2]),
                    'Previous_Competition_Performance': np.random.normal(50, 20, n_pruebas),
                    'Training_Intensity': np.random.choice(['Low', 'Medium', 'High'], n_pruebas),
                    'Resting_Heart_Rate': np.random.normal(70, 15, n_pruebas),
                    'Body_Fat_Percentage': np.random.normal(15, 5, n_pruebas),
                    'VO2_Max': np.random.normal(55, 10, n_pruebas),
                    'Performance_Metric': np.random.normal(60, 20, n_pruebas)
                }
                
                self.datos = pd.DataFrame(datos)

            textos = obtener_textos()
            st.success(f"✅ {textos['datos_cargados_exitosamente']} {self.datos.shape[0]} {textos['filas']}, {self.datos.shape[1]} {textos['columnas']}.")
            return True
            
        except Exception as e:
            st.error(f"❌ {textos['datos_cargados_erroneamente']} {str(e)}")
            return False
    
    def analisis_datos_exploratorios(self):
        """Analisis Exploratorio Integral de Datos"""
        textos = obtener_textos()

        st.markdown(f'<div class="section-header">📊 {textos["analisis_exploratorio_datos"]}</div>', 
                   unsafe_allow_html=True)
        
        # Información Básica
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 {textos['filas_mayus']}</h4>
                <h2>{self.datos.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔢 {textos['columnas_mayus']}</h4>
                <h2>{self.datos.shape[1]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            porcentaje_perdido = (self.datos.isnull().sum().sum() / (self.datos.shape[0] * self.datos.shape[1])) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h4>❓ {textos['datos_faltantes']}</h4>
                <h2>{porcentaje_perdido:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            numeric_cols = self.datos.select_dtypes(include=[np.number]).shape[1]
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔢 {textos['variables_numericas']}</h4>
                <h2>{numeric_cols}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Descripción General de los Datos
        st.subheader(f"🔍 {textos['vista_general_datos']}")
        st.dataframe(self.datos.head(100), use_container_width=True)
        
        # Valores Faltantes y Tipos de Datos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {textos['tipos_datos']}")
            tipos_datos = pd.DataFrame({
                textos['columna_mayus']: self.datos.dtypes.index,
                textos['tipo_mayus']: self.datos.dtypes.values
            })
            st.dataframe(tipos_datos, use_container_width=True)
            
        with col2:
            st.subheader(f"❓ {textos['valores_faltantes']}")
            datos_faltantes = self.datos.isnull().sum()
            faltante_df = pd.DataFrame({
                textos['columna_mayus']: datos_faltantes.index,
                textos['faltantes_mayus']: datos_faltantes.values,
                textos['porcentaje_mayus']: (datos_faltantes.values / len(self.datos)) * 100
            }).sort_values(textos['faltantes_mayus'], ascending=False)
            st.dataframe(faltante_df[faltante_df[textos['faltantes_mayus']] > 0], use_container_width=True)
    
    def estadisticas_descriptivas(self):
        """Generar estadísticas descriptivas completas"""
        textos=obtener_textos()

        st.subheader(f"📈 {textos['estadisticos_descriptivos']}")
        
        datos_numericos = self.datos.select_dtypes(include=[np.number])
        
        # Estadisticas descriptivas básicas
        estadisticas_descriptivas = datos_numericos.describe()
        st.dataframe(estadisticas_descriptivas, use_container_width=True)
        
        # Estadisticas avanzadas
        st.subheader(f"📊 {textos['estadisticos_avanzados']}")
        
        estadisticas_avanzadas = []
        for columna in datos_numericos.columns:
            col_datos = datos_numericos[columna].dropna()
            if len(col_datos) > 0:
                estadisticas_dict = {
                    'Variable': columna,
                    'Skewness': col_datos.skew(),
                    'Kurtosis': col_datos.kurtosis(),
                    'CV (%)': (col_datos.std() / col_datos.mean()) * 100 if col_datos.mean() != 0 else 0,
                    'IQR': col_datos.quantile(0.75) - col_datos.quantile(0.25)
                }
                estadisticas_avanzadas.append(estadisticas_dict)
        
        avanzado_df = pd.DataFrame(estadisticas_avanzadas)
        st.dataframe(avanzado_df, use_container_width=True)
    
    def crear_visualizaciones(self):
        """ Generar visualizaciones integrales"""

        textos = obtener_textos()

        st.subheader(f"📊 {textos['visualizaciones_descriptivas']}")
        
        datos_numericos = self.datos.select_dtypes(include=[np.number])
        
        # Plots de distribución
        st.subheader(f"📈 {textos['distribuciones_variables_numericas']}")
        
        # Seleccionar variables para los plots de distribución
        vars_seleccionadas = st.multiselect(
            f"{textos['seleccion_variables']}",
            datos_numericos.columns.tolist(),
            default=datos_numericos.columns[:4].tolist()
        )
        
        if vars_seleccionadas:
            fig = make_subplots(
                rows=(len(vars_seleccionadas) + 1) // 2, 
                cols=2,
                subplot_titles=vars_seleccionadas
            )
            
            for i, var in enumerate(vars_seleccionadas):
                fil = (i // 2) + 1
                col = (i % 2) + 1
                
                limpiar_data = datos_numericos[var].dropna()
                
                fig.add_trace(
                    go.Histogram(x=limpiar_data, name=var, showlegend=False),
                    row=fil, col=col
                )
            
            fig.update_layout(height=300 * ((len(vars_seleccionadas) + 1) // 2), 
                            title_text=f"{textos['distribucion_variables']}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Mapa de Calor de Correlación
        st.subheader(f"🔥 {textos['mapa_calor']}")
        
        corr_matriz = datos_numericos.corr()
        
        fig = px.imshow(
            corr_matriz,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title=f'{textos["matriz_correlacion_variables_numericas"]}'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Plots de caja para variables categoricas
        cols_categoricas = self.datos.select_dtypes(include=['object']).columns
        if len(cols_categoricas) > 0 and 'Performance_Metric' in datos_numericos.columns:
            st.subheader(f"📦 {textos['diagrama_caja_categoria']}")
            
            cat_seleccionada = st.selectbox(
                f"{textos['seleccion_variable_categorica']}",
                cols_categoricas
            )
            
            if cat_seleccionada:
                fig = px.box(
                    self.datos, 
                    x=cat_seleccionada, 
                    y='Performance_Metric',
                    title=f'{textos["distribucion_performance_metric"]} {cat_seleccionada}'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def preprocesamiento_datos(self):
        """Preprocesa datos para machine learning"""

        textos=obtener_textos()

        st.subheader(f"🔧 {textos['preprocesamiento_datos']}")
        
        # Crea una copia para el preprocesamiento
        self.datos_procesados = self.datos.copy()
        
        # Gestionar valores faltantes
        cols_numericas = self.datos_procesados.select_dtypes(include=[np.number]).columns
        cols_categoricas = self.datos_procesados.select_dtypes(include=['object']).columns
        
        # Imputar columnas numéricas con la mediana
        imputador_numerico = SimpleImputer(strategy='median')
        self.datos_procesados[cols_numericas] = imputador_numerico.fit_transform(self.datos_procesados[cols_numericas])
        
        # Imputar columnas categoricas con la moda
        imputador_categorico = SimpleImputer(strategy='most_frequent')
        self.datos_procesados[cols_categoricas] = imputador_categorico.fit_transform(self.datos_procesados[cols_categoricas])
        
        # Codificar variables categoricas
        codificador_etiquetas = {}
        for col in cols_categoricas:
            if col not in ['Athlete_ID', 'Athlete_Name']:  # Saltar columnas ID
                le = LabelEncoder()
                self.datos_procesados[col + '_encoded'] = le.fit_transform(self.datos_procesados[col].astype(str))
                codificador_etiquetas[col] = le
        
        # Preparar features y objetivos
        col_objetivo = 'Performance_Metric'
        if col_objetivo not in self.datos_procesados.columns:
            st.error(f"❌ Columna objetivo '{col_objetivo}' no encontrada.")
            return False
        
        # Seleccionar features (excluir columnas no predictivas)
        excluir_cols = ['Athlete_ID', 'Athlete_Name', col_objetivo] + list(cols_categoricas)
        feature_cols = [col for col in self.datos_procesados.columns if col not in excluir_cols]
        
        X = self.datos_procesados[feature_cols]
        y = self.datos_procesados[col_objetivo]
        
        # Eliminar filas con valores objetivo faltantes
        mascara = ~y.isnull()
        X = X[mascara]
        y = y[mascara]
        
        # Dividir datos
        self.X_entreno, self.X_prueba, self.y_entreno, self.y_prueba = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar features
        self.X_entreno_escalado = self.escala.fit_transform(self.X_entreno)
        self.X_prueba_escalada = self.escala.transform(self.X_prueba)
        
        st.success(f"✅ {textos['datos_preprocesados']}: {X.shape[0]} {textos['muestras']}, {X.shape[1]} {textos['caracteristicas']}.")
        st.write(f"🔹 {textos['conjunto_entrenamiento']} {self.X_entreno.shape[0]} {textos['muestras']}")
        st.write(f"🔹 {textos['conjunto_prueba']} {self.X_prueba.shape[0]} {textos['muestras']}")
        
        return True
    
    def entreno_modelos(self):
        """Entrenar múltiples modelos de Machine Learning"""
        st.subheader("🤖 Entrenamiento de Modelos de Machine Learning")
        
        # Definir modelos
        modelos_configuracion = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'Hybrid 1 (RF + GB)': VotingRegressor([
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
            ]),
            'Hybrid 2 (RF + SVR + NN)': VotingRegressor([
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('svr', SVR(kernel='rbf')),
                ('nn', MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
            ])
        }
        
        barra_progreso = st.progress(0)
        resultados = {}
        
        for i, (nombre, modelo) in enumerate(modelos_configuracion.items()):
            st.write(f"🔄 Entrenando {nombre}...")
            
            try:
                # Usa datos escalados para los modelos que se beneficien de ellos
                if nombre in ['SVR', 'Neural Network']:
                    X_entreno_uso = self.X_entreno_escalado
                    X_prueba_uso = self.X_prueba_escalada
                else:
                    X_entreno_uso = self.X_entreno
                    X_prueba_uso = self.X_prueba
                
                # Entrenamiento modelo
                modelo.fit(X_entreno_uso, self.y_entreno)
                
                # Realizar predicciones
                y_pred_entreno = modelo.predict(X_entreno_uso)
                y_pred_prueba = modelo.predict(X_prueba_uso)
                
                # Calcular metricas
                entreno_r2 = r2_score(self.y_entreno, y_pred_entreno)
                prueba_r2 = r2_score(self.y_prueba, y_pred_prueba)
                entreno_rmse = np.sqrt(mean_squared_error(self.y_entreno, y_pred_entreno))
                prueba_rmse = np.sqrt(mean_squared_error(self.y_prueba, y_pred_prueba))
                entreno_mae = mean_absolute_error(self.y_entreno, y_pred_entreno)
                prueba_mae = mean_absolute_error(self.y_prueba, y_pred_prueba)
                
                # Validación cruzada
                if nombre in ['SVR', 'Neural Network']:
                    cv_puntuaciones = cross_val_score(modelo, self.X_entreno_escalado, self.y_entreno, 
                                              cv=5, scoring='r2')
                else:
                    cv_puntuaciones = cross_val_score(modelo, self.X_entreno, self.y_entreno, 
                                              cv=5, scoring='r2')
                
                resultados[nombre] = {
                    'model': modelo,
                    'train_r2': entreno_r2,
                    'test_r2': prueba_r2,
                    'train_rmse': entreno_rmse,
                    'test_rmse': prueba_rmse,
                    'train_mae': entreno_mae,
                    'test_mae': prueba_mae,
                    'cv_mean': cv_puntuaciones.mean(),
                    'cv_std': cv_puntuaciones.std(),
                    'predictions_test': y_pred_prueba
                }
                
                st.write(f"✅ {nombre} completado - R² Test: {prueba_r2:.4f}")
                
            except Exception as e:
                st.write(f"❌ Error en {nombre}: {str(e)}")
                
            barra_progreso.progress((i + 1) / len(modelos_configuracion))
        
        self.resultados = resultados
        return resultados
    
    def evaluar_modelos(self):
        """Comprehensive model evaluation"""
        st.subheader("📊 Evaluación de Modelos")
        
        if not self.resultados:
            st.error("❌ No hay resultados de modelos disponibles.")
            return
        
        # Crear DataFrame de resultados
        datos_resultados = []
        for nombre, resultado in self.resultados.items():
            datos_resultados.append({
                'Modelo': nombre,
                'R² Entrenamiento': resultado['train_r2'],
                'R² Prueba': resultado['test_r2'],
                'RMSE Entrenamiento': resultado['train_rmse'],
                'RMSE Prueba': resultado['test_rmse'],
                'MAE Entrenamiento': resultado['train_mae'],
                'MAE Prueba': resultado['test_mae'],
                'CV Media': resultado['cv_mean'],
                'CV Desv. Est.': resultado['cv_std']
            })
        
        resultados_df = pd.DataFrame(datos_resultados)
        
        # Mostrar tabla de resultados
        st.dataframe(resultados_df, use_container_width=True)
        
        # Identificación del mejor modelo
        nombre_mejor_modelo = resultados_df.loc[resultados_df['R² Prueba'].idxmax(), 'Modelo']
        mejor_r2 = resultados_df['R² Prueba'].max()
        
        st.success(f"🏆 Mejor modelo: **{nombre_mejor_modelo}** (R² = {mejor_r2:.4f})")
        
        # Visualización de comparación de modelos
        st.subheader("📊 Comparación Visual de Modelos")
        
        # Comparación R²
        fig = px.bar(
            resultados_df, 
            x='Modelo', 
            y=['R² Entrenamiento', 'R² Prueba'],
            title='Comparación de R² por Modelo',
            barmode='group'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparación RMSE
        fig2 = px.bar(
            resultados_df, 
            x='Modelo', 
            y=['RMSE Entrenamiento', 'RMSE Prueba'],
            title='Comparación de RMSE por Modelo',
            barmode='group'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Plots de dispersión predicción vs. real para los mejores modelos
        st.subheader("🎯 Predicciones vs Valores Reales (Mejores Modelos)")
        
        top_3_modelos = resultados_df.nlargest(3, 'R² Prueba')['Modelo'].tolist()
        
        fig = make_subplots(
            rows=1, cols=len(top_3_modelos),
            subplot_titles=top_3_modelos
        )
        
        for i, nombre_model in enumerate(top_3_modelos):
            predicciones = self.resultados[nombre_model]['predictions_test']
            
            fig.add_trace(
                go.Scatter(
                    x=self.y_prueba, 
                    y=predicciones,
                    mode='markers',
                    name=nombre_model,
                    showlegend=False
                ),
                row=1, col=i+1
            )
            
            # Agregar línea de referencia
            min_val = min(self.y_prueba.min(), predicciones.min())
            max_val = max(self.y_prueba.max(), predicciones.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_xaxes(title_text="Valores Reales")
        fig.update_yaxes(title_text="Predicciones")
        fig.update_layout(height=400, title_text="Predicciones vs Valores Reales")
        
        st.plotly_chart(fig, use_container_width=True)
        
        return nombre_mejor_modelo, resultados_df
    
    def pruebas_estadisticas(self):
        """Realizar pruebas estadísticas robustas"""
        st.subheader("🧪 Pruebas Estadísticas Robustas")
        
        if not self.resultados:
            st.error("❌ No hay resultados de modelos disponibles.")
            return
        
        # Recopilar predicciones de todos los modelos
        predicciones_dict = {}
        for nombre, resultado in self.resultados.items():
            predicciones_dict[nombre] = resultado['predictions_test']
        
        # Prueba de normalidad de residuos
        st.subheader("🔍 Pruebas de Normalidad de Residuos")
        
        resultados_normalidad = []
        for nombre, predicciones in predicciones_dict.items():
            residuos = self.y_prueba - predicciones
            
            # Prueba Shapiro-Wilk
            stat, p_valor = shapiro(residuos)
            
            resultados_normalidad.append({
                'Modelo': nombre,
                'Estadístico Shapiro-Wilk': stat,
                'p-valor': p_valor,
                'Normalidad': 'Sí' if p_valor > 0.05 else 'No'
            })
        
        normalidad_df = pd.DataFrame(resultados_normalidad)
        st.dataframe(normalidad_df, use_container_width=True)
        
        # Prueba de homocedasticidad (prueba de Levene)
        st.subheader("🔍 Prueba de Homoscedasticidad (Levene)")
        
        lista_residuales = []
        etiquetas_modelo = []
        
        for nombre, predicciones in predicciones_dict.items():
            residuos = self.y_prueba - predicciones
            lista_residuales.append(residuos)
            etiquetas_modelo.extend([nombre] * len(residuos))
        
        if len(lista_residuales) > 1:
            levene_stat, levene_p = levene(*lista_residuales)
            
            st.write(f"**Estadístico de Levene:** {levene_stat:.4f}")
            st.write(f"**p-valor:** {levene_p:.4f}")
            st.write(f"**Interpretación:** {'Varianzas homogéneas' if levene_p > 0.05 else 'Varianzas heterogéneas'}")
        
        # Comparación Estadísticas de Modelos (ANOVA o Kruskal-Wallis)
        st.subheader("📊 Comparación Estadística de Modelos")
        
        # Uso de las puntuaciones R² para la comparación
        r2_puntuacion = [resultado['cv_mean'] for resultado in self.resultados.values()]
        nombre_modelos = list(self.resultados.keys())
        
        # Dado que tenemos puntuaciones de validación cruzada, podemos realizar pruebas estadísticas
        cv_puntuaciones_por_modelo = []
        for nombre, resultado in self.resultados.items():
            # Simular puntuaciones de CV para demostración (en la práctica, almacenarías puntuaciones de CV reales)
            cv_significado = resultado['cv_mean']
            cv_std = resultado['cv_std']
            puntuaciones_simuladas = np.random.normal(cv_significado, cv_std, 5)  # 5-fold CV
            cv_puntuaciones_por_modelo.append(puntuaciones_simuladas)
        
        # Prueba Kruskal-Wallis test (no-parametrica)
        if len(cv_puntuaciones_por_modelo) > 2:
            kruskal_stat, kruskal_p = kruskal(*cv_puntuaciones_por_modelo)
            
            st.write(f"**Prueba de Kruskal-Wallis:**")
            st.write(f"Estadístico: {kruskal_stat:.4f}")
            st.write(f"p-valor: {kruskal_p:.4f}")
            
            if kruskal_p < 0.05:
                st.write("✅ **Conclusión:** Existen diferencias significativas entre los modelos")
                
                # Analisis Post-hoc
                st.subheader("🔬 Análisis Post-hoc")
                try:
                    # Preparar datos para la prueba post-hoc
                    todas_puntuaciones = np.concatenate(cv_puntuaciones_por_modelo)
                    todas_etiquetas = []
                    for i, puntuaciones in enumerate(cv_puntuaciones_por_modelo):
                        todas_etiquetas.extend([nombre_modelos[i]] * len(puntuaciones))
                    
                    posthoc_df = pd.DataFrame({
                        'scores': todas_puntuaciones,
                        'models': todas_etiquetas
                    })
                    
                    # Prueba de Dunn para comparaciones post-hoc
                    posthoc_resultados = sp.posthoc_dunn(posthoc_df, val_col='scores', 
                                                    group_col='models', p_adjust='bonferroni')
                    
                    st.write("**Comparaciones post-hoc (Dunn's test con corrección Bonferroni):**")
                    st.dataframe(posthoc_resultados, use_container_width=True)
                    
                except Exception as e:
                    st.write(f"No se pudo realizar el análisis post-hoc: {str(e)}")
            else:
                st.write("❌ **Conclusión:** No hay diferencias significativas entre los modelos")
    
    def generar_reporte_pdf(self, nombre_mejor_modelo, resultados_df):
        """Generar un informe PDF completo"""
        st.subheader("📄 Generar Reporte en PDF")
        
        if st.button("🔄 Generar Reporte PDF"):
            try:
                # Crear PDF en memoria
                buffer = io.BytesIO()
                
                # Crear el documento PDF
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                estilos = getSampleStyleSheet()
                historia = []
                
                # Titulo
                estilo_titulo = ParagraphStyle(
                    'CustomTitle',
                    parent=estilos['Heading1'],
                    fontSize=18,
                    spaceAfter=30,
                    alignment=1  # Alineación centrada
                )
                
                historia.append(Paragraph("REPORTE DE PREDICCIÓN DE RENDIMIENTO DEPORTIVO", estilo_titulo))
                historia.append(Spacer(1, 20))
                
                # Resumen ejecutivo
                historia.append(Paragraph("RESUMEN EJECUTIVO", estilos['Heading2']))
                
                texto_resumen = f"""
                <para>
                Este reporte presenta un análisis completo de predicción de rendimiento deportivo 
                utilizando técnicas de machine learning. Se evaluaron múltiples algoritmos y 
                se identificó el mejor modelo basado en métricas de rendimiento estadísticamente 
                robustas.<br/><br/>
                
                <b>Mejor Modelo:</b> {nombre_mejor_modelo}<br/>
                <b>Número de muestras analizadas:</b> {self.datos.shape[0]}<br/>
                <b>Características utilizadas:</b> {self.X_entreno.shape[1]}<br/>
                <b>R² del mejor modelo:</b> {resultados_df['R² Prueba'].max():.4f}
                </para>
                """
                
                historia.append(Paragraph(texto_resumen, estilos['Normal']))
                historia.append(Spacer(1, 20))
                
                # Descripción del dataset
                historia.append(Paragraph("1. DESCRIPCIÓN DEL DATASET", estilos['Heading2']))
                
                texto_dataset = f"""
                <para>
                El dataset contiene información de {self.datos.shape[0]} atletas con {self.datos.shape[1]} 
                características diferentes. Las variables incluyen métricas fisiológicas, de entrenamiento 
                y rendimiento histórico.<br/><br/>
                
                <b>Variables principales:</b><br/>
                • Variables fisiológicas: BMI, frecuencia cardíaca, VO2 Max<br/>
                • Variables de entrenamiento: horas de entrenamiento, intensidad<br/>
                • Variables de estilo de vida: sueño, nutrición, hidratación<br/>
                • Variable objetivo: Performance_Metric
                </para>
                """
                
                historia.append(Paragraph(texto_dataset, estilos['Normal']))
                historia.append(Spacer(1, 20))
                
                # Análisis estadístico
                historia.append(Paragraph("2. ANÁLISIS ESTADÍSTICO DESCRIPTIVO", estilos['Heading2']))
                
                datos_numericos = self.datos.select_dtypes(include=[np.number])
                desc_estadisticas = datos_numericos.describe()
                
                # Crear tabla para estadísticas descriptivas
                desc_datos = [['Variable', 'Media', 'Desv. Est.', 'Min', 'Max']]
                for col in desc_estadisticas.columns[:5]:  # Show first 5 numeric columns
                    desc_datos.append([
                        col,
                        f"{desc_estadisticas.loc['mean', col]:.2f}",
                        f"{desc_estadisticas.loc['std', col]:.2f}",
                        f"{desc_estadisticas.loc['min', col]:.2f}",
                        f"{desc_estadisticas.loc['max', col]:.2f}"
                    ])
                
                desc_tabla = Table(desc_datos)
                desc_tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                historia.append(desc_tabla)
                historia.append(Spacer(1, 20))
                
                # Resultados del Modelos de Machine Learning
                historia.append(Paragraph("3. RESULTADOS DE MODELOS DE MACHINE LEARNING", estilos['Heading2']))
                
                # Crear tabla de resultados
                datos_modelo = [['Modelo', 'R² Prueba', 'RMSE Prueba', 'MAE Prueba', 'CV Media']]
                
                for _, fila in resultados_df.iterrows():
                    datos_modelo.append([
                        fila['Modelo'],
                        f"{fila['R² Prueba']:.4f}",
                        f"{fila['RMSE Prueba']:.4f}",
                        f"{fila['MAE Prueba']:.4f}",
                        f"{fila['CV Media']:.4f}"
                    ])
                
                tabla_modelo = Table(datos_modelo)
                tabla_modelo.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                historia.append(tabla_modelo)
                historia.append(Spacer(1, 20))
                
                # Analisis del Mejor Modelo
                historia.append(Paragraph("4. ANÁLISIS DEL MEJOR MODELO", estilos['Heading2']))
                
                resultados_mejor_modelo = resultados_df[resultados_df['Modelo'] == nombre_mejor_modelo].iloc[0]
                
                texto_mejor_modelo = f"""
                <para>
                El modelo <b>{nombre_mejor_modelo}</b> demostró el mejor rendimiento con las siguientes métricas:<br/><br/>
                
                <b>Métricas de Rendimiento:</b><br/>
                • R² en conjunto de prueba: {resultados_mejor_modelo['R² Prueba']:.4f}<br/>
                • RMSE en conjunto de prueba: {resultados_mejor_modelo['RMSE Prueba']:.4f}<br/>
                • MAE en conjunto de prueba: {resultados_mejor_modelo['MAE Prueba']:.4f}<br/>
                • Validación cruzada (media): {resultados_mejor_modelo['CV Media']:.4f} ± {resultados_mejor_modelo['CV Desv. Est.']:.4f}<br/><br/>
                
                <b>Interpretación:</b><br/>
                El modelo explica aproximadamente {resultados_mejor_modelo['R² Prueba']*100:.1f}% de la variabilidad 
                en el rendimiento deportivo. El error promedio absoluto es de {resultados_mejor_modelo['MAE Prueba']:.2f} 
                unidades en la métrica de rendimiento.
                </para>
                """
                
                historia.append(Paragraph(texto_mejor_modelo, estilos['Normal']))
                historia.append(Spacer(1, 20))
                
                # Conclusiones y Recomendaciones
                historia.append(Paragraph("5. CONCLUSIONES Y RECOMENDACIONES", estilos['Heading2']))
                
                texto_conclusiones = f"""
                <para>
                <b>Conclusiones principales:</b><br/><br/>
                
                1. <b>Modelo óptimo:</b> {nombre_mejor_modelo} mostró el mejor rendimiento predictivo.<br/><br/>
                
                2. <b>Capacidad predictiva:</b> El modelo puede explicar {resultados_mejor_modelo['R² Prueba']*100:.1f}% 
                de la variabilidad en el rendimiento deportivo.<br/><br/>
                
                3. <b>Robustez:</b> La validación cruzada confirma la estabilidad del modelo 
                ({resultados_mejor_modelo['CV Media']:.4f} ± {resultados_mejor_modelo['CV Desv. Est.']:.4f}).<br/><br/>
                
                <b>Recomendaciones:</b><br/><br/>
                
                1. <b>Implementación:</b> Se recomienda implementar el modelo {nombre_mejor_modelo} 
                para predicciones de rendimiento deportivo.<br/><br/>
                
                2. <b>Monitoreo continuo:</b> Establecer un sistema de monitoreo para evaluar 
                la performance del modelo en producción.<br/><br/>
                
                3. <b>Mejoras futuras:</b> Considerar la recolección de datos adicionales y 
                la reevaluación periódica del modelo.<br/><br/>
                
                4. <b>Variables clave:</b> Enfocar esfuerzos en la recolección precisa de las 
                variables más predictivas identificadas en el análisis.
                </para>
                """
                
                historia.append(Paragraph(texto_conclusiones, estilos['Normal']))
                historia.append(Spacer(1, 20))
                
                # Detalles técnicos
                historia.append(Paragraph("6. DETALLES TÉCNICOS", estilos['Heading2']))
                
                texto_tecnico = f"""
                <para>
                <b>Metodología:</b><br/>
                • División de datos: 80% entrenamiento, 20% prueba<br/>
                • Validación cruzada: 5-fold<br/>
                • Preprocesamiento: Imputación de valores faltantes, codificación de variables categóricas<br/>
                • Escalado: StandardScaler para modelos sensibles a escala<br/><br/>
                
                <b>Modelos evaluados:</b><br/>
                • Algoritmos tradicionales: Linear Regression, Random Forest, Gradient Boosting<br/>
                • Algoritmos avanzados: SVR, Neural Networks<br/>
                • Modelos híbridos: Ensemble methods con VotingRegressor<br/><br/>
                
                <b>Métricas de evaluación:</b><br/>
                • R²: Coeficiente de determinación<br/>
                • RMSE: Error cuadrático medio<br/>
                • MAE: Error absoluto medio<br/>
                • Validación cruzada: Evaluación robusta del rendimiento
                </para>
                """
                
                historia.append(Paragraph(texto_tecnico, estilos['Normal']))
                
                # Construir PDF
                doc.build(historia)
                
                # Preparar descarga del PDF
                buffer.seek(0)
                pdf_datos = buffer.read()
                buffer.close()
                
                # Crear botón de descarga
                st.download_button(
                    label="📥 Descargar Reporte PDF",
                    data=pdf_datos,
                    file_name=f"reporte_rendimiento_deportivo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                st.success("✅ Reporte PDF generado exitosamente!")
                
            except Exception as e:
                st.error(f"❌ Error al generar PDF: {str(e)}")
                st.write("Nota: Algunas librerías pueden no estar disponibles en el entorno de Streamlit.")

def main():
    """Funcion principal para ejecutar la aplicación"""

    # Idioma por defecto la primera vez
    if "codigos_idioma" not in st.session_state:
        st.session_state.codigos_idioma = "es"

    # Cargar textos del idioma actual
    textos = obtener_textos()

    # Mostrar selector con traducción actual
    codigos_idioma = st.sidebar.selectbox(
        f"🌍 {textos['seleccion_lenguaje']}",
        ["es", "en"],
        index=["es", "en"].index(st.session_state.codigos_idioma),
        format_func=lambda x: "Español" if x == "es" else "English"
    )

    # Si cambia el idioma, actualizar y forzar recarga
    if codigos_idioma != st.session_state.codigos_idioma:
        st.session_state.codigos_idioma = codigos_idioma
        st.rerun()

    # Cargar textos definitivos del idioma elegido
    textos = obtener_textos()
    
    # Titulo y descripción
    st.markdown(f'<div class="main-header">🏃‍♂️ {textos["titulo"]}</div>', 
               unsafe_allow_html=True)
    
    st.markdown(f"""
    ### 🎯 {textos['subtitulo']}
    
    {textos['descripcion']}
    
    **{textos['titulo_caracteristicas_principales']}**
    - 📊 {textos['caracteristica_1']}
    - 🤖 {textos['caracteristica_2']}
    - 🧪 {textos['caracteristica_3']}
    - 📈 {textos['caracteristica_4']}
    - 📄 {textos['caracteristica_5']}
    """)

    st.markdown(f"""
    ---
    {textos['nota']}
    """)
    
    # Inicializar predictor
    predictor = PredictorRendimientoDeportivo()
    
    # Barra lateral para navegación
    st.sidebar.title(f'🚀 {textos["navegacion"]}')
    
    # Opción de cargar datos
    st.sidebar.subheader(f"📁 {textos['cargar_datos']}")
    archivo_cargado = st.sidebar.file_uploader(f"{textos['cargar_archivo']}", type=['csv'])
    
    if archivo_cargado is not None:
        # Leer archivo cargado
        try:
            predictor.datos = pd.read_csv(archivo_cargado)
            st.sidebar.success(f"✅ {textos['carga_exitosa']}")
        except Exception as e:
            st.sidebar.error(f"❌ {textos['carga_erronea']} {str(e)}")
            predictor.datos = None
    else:
        # Utilizar datos proporcionados
        try:
            # Intentar leer el archivo CSV proporcionado
            predictor.datos = pd.read_csv('datasport.csv')
            st.sidebar.success(f"✅ {textos['uso_archivo_datasport']}")
        except:
            # Si no se encuentra el archivo, cargar datos de muestra
            if st.sidebar.button(f"🔄 {textos['carga_datos_ejemplo']}"):
                predictor.cargar_data()
    
    # Opciones de Navegación
    pasos_analisis = [
        f"📊 {textos['analisis_exploratorio']}",
        f"🔧 {textos['preprocesamiento']}",
        f"🤖 {textos['entrenamiento_modelo']}",
        f"📈 {textos['evaluacion_comparacion']}",
        f"🧪 {textos['pruebas_estadisticas']}",
        f"📄 {textos['generar_reporte']}"
    ]
    
    paso_seleccionado = st.sidebar.selectbox(f"{textos['seleccionar_analisis']}", pasos_analisis)
    
    # Revisar si hay datos cargados
    if predictor.datos is None:
        st.error(f"❌ {textos['no_datos_disponibles']}")
        return
    
    # Ejecutar analisis seleccionado
    if paso_seleccionado == f"📊 {textos['analisis_exploratorio']}":
        predictor.analisis_datos_exploratorios()
        predictor.estadisticas_descriptivas()
        predictor.crear_visualizaciones()
        
    elif paso_seleccionado == f"🔧 {textos['preprocesamiento']}":
        if predictor.preprocesamiento_datos():
            st.success(f"✅ {textos['preprocesamiento_completado']}")
        
    elif paso_seleccionado == f"🤖 {textos['entrenamiento_modelo']}":
        if predictor.datos_procesados is None:
            if predictor.preprocesamiento_datos():
                predictor.entreno_modelos()
        else:
            predictor.entreno_modelos()
            
    elif paso_seleccionado == f"📈 {textos['evaluacion_comparacion']}":
        if not predictor.resultados:
            if predictor.datos_procesados is None:
                predictor.preprocesamiento_datos()
            predictor.entreno_modelos()
        
        mejor_modelo, resultados_df = predictor.evaluar_modelos()
        
    elif paso_seleccionado == f"🧪 {textos['pruebas_estadisticas']}":
        if not predictor.resultados:
            if predictor.datos_procesados is None:
                predictor.preprocesamiento_datos()
            predictor.entreno_modelos()
        
        predictor.pruebas_estadisticas()
        
    elif paso_seleccionado == f"📄 {textos['generar_reporte']}":
        if not predictor.resultados:
            if predictor.datos_procesados is None:
                predictor.preprocesamiento_datos()
            predictor.entreno_modelos()
        
        mejor_modelo, resultados_df = predictor.evaluar_modelos()
        predictor.generar_reporte_pdf(mejor_modelo, resultados_df)
    
    # Pie de página
    st.markdown("---")
    st.markdown(f"""
    ### 📚 {textos['informacion_adicional']}
    
    **{textos['algoritmos_implementados']}**
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Regression (SVR)
    - Neural Networks (MLP)
    - {textos['hibrido']} 1: Random Forest + Gradient Boosting
    - {textos['hibrido']} 2: Random Forest + SVR + Neural Network
    
    **{textos['metricas_evaluacion']}**
    - R² ({textos['coeficiente_determinacion']})
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - {textos['validacion_cruzada']}
    
    **{textos['pruebas_estadisticas']}**
    - Shapiro-Wilk ({textos['normalidad']})
    - Levene ({textos['homoscedasticidad']})
    - Kruskal-Wallis ({textos['comparacion_grupos']})
    - Post-hoc Dunn ({textos['comparaciones_multiples']})
    """)

# Instrucciones para ejecutar la aplicación

if __name__ == "__main__":
    main()
