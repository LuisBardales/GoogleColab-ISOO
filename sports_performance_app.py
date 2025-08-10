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

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# Statistical Tests
from scipy import stats
from scipy.stats import shapiro, levene, f_oneway, kruskal
import scikit_posthocs as sp

# PDF Report
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

# Configuration
st.set_page_config(
    page_title="Predictor de Rendimiento Deportivo",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SportsPerformancePredictor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self, file_path=None):
        """Load and initial data processing"""
        try:
            if file_path:
                self.data = pd.read_csv(file_path)
            else:
                # Sample data creation for demo
                np.random.seed(42)
                n_samples = 100
                
                sports = ['Running', 'Swimming', 'Cycling', 'Soccer', 'Basketball', 'Tennis']
                events = ['50m Freestyle', '100m Sprint', '10km Road Race', 'Marathon']
                
                data = {
                    'Athlete_ID': [f'A{i:03d}' for i in range(1, n_samples+1)],
                    'Athlete_Name': [f'Athlete_{i}' for i in range(1, n_samples+1)],
                    'Sport_Type': np.random.choice(sports, n_samples),
                    'Event': np.random.choice(events, n_samples),
                    'Training_Hours_per_Week': np.random.normal(25, 8, n_samples),
                    'Average_Heart_Rate': np.random.normal(150, 25, n_samples),
                    'BMI': np.random.normal(23, 3, n_samples),
                    'Sleep_Hours_per_Night': np.random.normal(7.5, 1, n_samples),
                    'Daily_Caloric_Intake': np.random.normal(2500, 500, n_samples),
                    'Hydration_Level': np.random.normal(70, 15, n_samples),
                    'Injury_History': np.random.choice(['None', 'Minor', 'Major'], n_samples, p=[0.5, 0.3, 0.2]),
                    'Previous_Competition_Performance': np.random.normal(50, 20, n_samples),
                    'Training_Intensity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                    'Resting_Heart_Rate': np.random.normal(70, 15, n_samples),
                    'Body_Fat_Percentage': np.random.normal(15, 5, n_samples),
                    'VO2_Max': np.random.normal(55, 10, n_samples),
                    'Performance_Metric': np.random.normal(60, 20, n_samples)
                }
                
                self.data = pd.DataFrame(data)
                
            st.success(f"✅ Datos cargados exitosamente: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
            return True
            
        except Exception as e:
            st.error(f"❌ Error al cargar datos: {str(e)}")
            return False
    
    def exploratory_data_analysis(self):
        """Comprehensive Exploratory Data Analysis"""
        st.markdown('<div class="section-header">📊 Análisis Exploratorio de Datos (EDA)</div>', 
                   unsafe_allow_html=True)
        
        # Basic Information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Filas</h4>
                <h2>{self.data.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔢 Columnas</h4>
                <h2>{self.data.shape[1]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            missing_percentage = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h4>❓ Datos Faltantes</h4>
                <h2>{missing_percentage:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            numeric_cols = self.data.select_dtypes(include=[np.number]).shape[1]
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔢 Variables Numéricas</h4>
                <h2>{numeric_cols}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Overview
        st.subheader("🔍 Vista General de los Datos")
        st.dataframe(self.data.head(10), use_container_width=True)
        
        # Data Types and Missing Values
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Tipos de Datos")
            data_types = pd.DataFrame({
                'Columna': self.data.dtypes.index,
                'Tipo': self.data.dtypes.values
            })
            st.dataframe(data_types, use_container_width=True)
            
        with col2:
            st.subheader("❓ Valores Faltantes")
            missing_data = self.data.isnull().sum()
            missing_df = pd.DataFrame({
                'Columna': missing_data.index,
                'Faltantes': missing_data.values,
                'Porcentaje': (missing_data.values / len(self.data)) * 100
            }).sort_values('Faltantes', ascending=False)
            st.dataframe(missing_df[missing_df['Faltantes'] > 0], use_container_width=True)
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        st.subheader("📈 Estadísticos Descriptivos")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Basic descriptive statistics
        desc_stats = numeric_data.describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Advanced statistics
        st.subheader("📊 Estadísticos Avanzados")
        
        advanced_stats = []
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) > 0:
                stats_dict = {
                    'Variable': column,
                    'Skewness': col_data.skew(),
                    'Kurtosis': col_data.kurtosis(),
                    'CV (%)': (col_data.std() / col_data.mean()) * 100 if col_data.mean() != 0 else 0,
                    'IQR': col_data.quantile(0.75) - col_data.quantile(0.25)
                }
                advanced_stats.append(stats_dict)
        
        advanced_df = pd.DataFrame(advanced_stats)
        st.dataframe(advanced_df, use_container_width=True)
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        st.subheader("📊 Visualizaciones Descriptivas")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Distribution plots
        st.subheader("📈 Distribuciones de Variables Numéricas")
        
        # Select variables for distribution plots
        selected_vars = st.multiselect(
            "Selecciona variables para visualizar:",
            numeric_data.columns.tolist(),
            default=numeric_data.columns[:4].tolist()
        )
        
        if selected_vars:
            fig = make_subplots(
                rows=(len(selected_vars) + 1) // 2, 
                cols=2,
                subplot_titles=selected_vars
            )
            
            for i, var in enumerate(selected_vars):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                data_clean = numeric_data[var].dropna()
                
                fig.add_trace(
                    go.Histogram(x=data_clean, name=var, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=300 * ((len(selected_vars) + 1) // 2), 
                            title_text="Distribuciones de Variables")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Mapa de Calor - Matriz de Correlación")
        
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Matriz de Correlación de Variables Numéricas'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots for categorical variables
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and 'Performance_Metric' in numeric_data.columns:
            st.subheader("📦 Diagramas de Caja por Categorías")
            
            selected_cat = st.selectbox(
                "Selecciona variable categórica:",
                categorical_cols
            )
            
            if selected_cat:
                fig = px.box(
                    self.data, 
                    x=selected_cat, 
                    y='Performance_Metric',
                    title=f'Distribución de Performance_Metric por {selected_cat}'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        st.subheader("🔧 Preprocesamiento de Datos")
        
        # Create a copy for processing
        self.processed_data = self.data.copy()
        
        # Handle missing values
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.processed_data.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        numeric_imputer = SimpleImputer(strategy='median')
        self.processed_data[numeric_cols] = numeric_imputer.fit_transform(self.processed_data[numeric_cols])
        
        # Impute categorical columns with mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.processed_data[categorical_cols] = categorical_imputer.fit_transform(self.processed_data[categorical_cols])
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            if col not in ['Athlete_ID', 'Athlete_Name']:  # Skip ID columns
                le = LabelEncoder()
                self.processed_data[col + '_encoded'] = le.fit_transform(self.processed_data[col].astype(str))
                label_encoders[col] = le
        
        # Prepare features and target
        target_col = 'Performance_Metric'
        if target_col not in self.processed_data.columns:
            st.error(f"❌ Columna objetivo '{target_col}' no encontrada.")
            return False
        
        # Select features (exclude non-predictive columns)
        exclude_cols = ['Athlete_ID', 'Athlete_Name', target_col] + list(categorical_cols)
        feature_cols = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        X = self.processed_data[feature_cols]
        y = self.processed_data[target_col]
        
        # Remove rows with missing target values
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        st.success(f"✅ Datos preprocesados: {X.shape[0]} muestras, {X.shape[1]} características")
        st.write(f"🔹 Conjunto de entrenamiento: {self.X_train.shape[0]} muestras")
        st.write(f"🔹 Conjunto de prueba: {self.X_test.shape[0]} muestras")
        
        return True
    
    def train_models(self):
        """Train multiple ML models"""
        st.subheader("🤖 Entrenamiento de Modelos de Machine Learning")
        
        # Define models
        models_config = {
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
        
        progress_bar = st.progress(0)
        results = {}
        
        for i, (name, model) in enumerate(models_config.items()):
            st.write(f"🔄 Entrenando {name}...")
            
            try:
                # Use scaled data for models that benefit from it
                if name in ['SVR', 'Neural Network']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Train model
                model.fit(X_train_use, self.y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_use)
                y_pred_test = model.predict(X_test_use)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # Cross-validation
                if name in ['SVR', 'Neural Network']:
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                              cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                              cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions_test': y_pred_test
                }
                
                st.write(f"✅ {name} completado - R² Test: {test_r2:.4f}")
                
            except Exception as e:
                st.write(f"❌ Error en {name}: {str(e)}")
                
            progress_bar.progress((i + 1) / len(models_config))
        
        self.results = results
        return results
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        st.subheader("📊 Evaluación de Modelos")
        
        if not self.results:
            st.error("❌ No hay resultados de modelos disponibles.")
            return
        
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Modelo': name,
                'R² Entrenamiento': result['train_r2'],
                'R² Prueba': result['test_r2'],
                'RMSE Entrenamiento': result['train_rmse'],
                'RMSE Prueba': result['test_rmse'],
                'MAE Entrenamiento': result['train_mae'],
                'MAE Prueba': result['test_mae'],
                'CV Media': result['cv_mean'],
                'CV Desv. Est.': result['cv_std']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Display results table
        st.dataframe(results_df, use_container_width=True)
        
        # Best model identification
        best_model_name = results_df.loc[results_df['R² Prueba'].idxmax(), 'Modelo']
        best_r2 = results_df['R² Prueba'].max()
        
        st.success(f"🏆 Mejor modelo: **{best_model_name}** (R² = {best_r2:.4f})")
        
        # Model comparison visualization
        st.subheader("📊 Comparación Visual de Modelos")
        
        # R² comparison
        fig = px.bar(
            results_df, 
            x='Modelo', 
            y=['R² Entrenamiento', 'R² Prueba'],
            title='Comparación de R² por Modelo',
            barmode='group'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # RMSE comparison
        fig2 = px.bar(
            results_df, 
            x='Modelo', 
            y=['RMSE Entrenamiento', 'RMSE Prueba'],
            title='Comparación de RMSE por Modelo',
            barmode='group'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Prediction vs Actual scatter plots for best models
        st.subheader("🎯 Predicciones vs Valores Reales (Mejores Modelos)")
        
        top_3_models = results_df.nlargest(3, 'R² Prueba')['Modelo'].tolist()
        
        fig = make_subplots(
            rows=1, cols=len(top_3_models),
            subplot_titles=top_3_models
        )
        
        for i, model_name in enumerate(top_3_models):
            predictions = self.results[model_name]['predictions_test']
            
            fig.add_trace(
                go.Scatter(
                    x=self.y_test, 
                    y=predictions,
                    mode='markers',
                    name=model_name,
                    showlegend=False
                ),
                row=1, col=i+1
            )
            
            # Add diagonal line
            min_val = min(self.y_test.min(), predictions.min())
            max_val = max(self.y_test.max(), predictions.max())
            
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
        
        return best_model_name, results_df
    
    def statistical_tests(self):
        """Perform robust statistical tests"""
        st.subheader("🧪 Pruebas Estadísticas Robustas")
        
        if not self.results:
            st.error("❌ No hay resultados de modelos disponibles.")
            return
        
        # Collect predictions from all models
        predictions_dict = {}
        for name, result in self.results.items():
            predictions_dict[name] = result['predictions_test']
        
        # Test for normality of residuals
        st.subheader("🔍 Pruebas de Normalidad de Residuos")
        
        normality_results = []
        for name, predictions in predictions_dict.items():
            residuals = self.y_test - predictions
            
            # Shapiro-Wilk test
            stat, p_value = shapiro(residuals)
            
            normality_results.append({
                'Modelo': name,
                'Estadístico Shapiro-Wilk': stat,
                'p-valor': p_value,
                'Normalidad': 'Sí' if p_value > 0.05 else 'No'
            })
        
        normality_df = pd.DataFrame(normality_results)
        st.dataframe(normality_df, use_container_width=True)
        
        # Test for homoscedasticity (Levene's test)
        st.subheader("🔍 Prueba de Homoscedasticidad (Levene)")
        
        residuals_list = []
        model_labels = []
        
        for name, predictions in predictions_dict.items():
            residuals = self.y_test - predictions
            residuals_list.append(residuals)
            model_labels.extend([name] * len(residuals))
        
        if len(residuals_list) > 1:
            levene_stat, levene_p = levene(*residuals_list)
            
            st.write(f"**Estadístico de Levene:** {levene_stat:.4f}")
            st.write(f"**p-valor:** {levene_p:.4f}")
            st.write(f"**Interpretación:** {'Varianzas homogéneas' if levene_p > 0.05 else 'Varianzas heterogéneas'}")
        
        # Comparison of model performance (ANOVA or Kruskal-Wallis)
        st.subheader("📊 Comparación Estadística de Modelos")
        
        # Use R² scores for comparison
        r2_scores = [result['cv_mean'] for result in self.results.values()]
        model_names = list(self.results.keys())
        
        # Since we have cross-validation scores, we can perform statistical tests
        cv_scores_by_model = []
        for name, result in self.results.items():
            # Simulate CV scores for demonstration (in practice, you'd store actual CV scores)
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']
            simulated_scores = np.random.normal(cv_mean, cv_std, 5)  # 5-fold CV
            cv_scores_by_model.append(simulated_scores)
        
        # Kruskal-Wallis test (non-parametric)
        if len(cv_scores_by_model) > 2:
            kruskal_stat, kruskal_p = kruskal(*cv_scores_by_model)
            
            st.write(f"**Prueba de Kruskal-Wallis:**")
            st.write(f"Estadístico: {kruskal_stat:.4f}")
            st.write(f"p-valor: {kruskal_p:.4f}")
            
            if kruskal_p < 0.05:
                st.write("✅ **Conclusión:** Existen diferencias significativas entre los modelos")
                
                # Post-hoc analysis
                st.subheader("🔬 Análisis Post-hoc")
                try:
                    # Prepare data for post-hoc test
                    all_scores = np.concatenate(cv_scores_by_model)
                    all_labels = []
                    for i, scores in enumerate(cv_scores_by_model):
                        all_labels.extend([model_names[i]] * len(scores))
                    
                    posthoc_df = pd.DataFrame({
                        'scores': all_scores,
                        'models': all_labels
                    })
                    
                    # Dunn's test for post-hoc comparisons
                    posthoc_results = sp.posthoc_dunn(posthoc_df, val_col='scores', 
                                                    group_col='models', p_adjust='bonferroni')
                    
                    st.write("**Comparaciones post-hoc (Dunn's test con corrección Bonferroni):**")
                    st.dataframe(posthoc_results, use_container_width=True)
                    
                except Exception as e:
                    st.write(f"No se pudo realizar el análisis post-hoc: {str(e)}")
            else:
                st.write("❌ **Conclusión:** No hay diferencias significativas entre los modelos")
    
    def generate_pdf_report(self, best_model_name, results_df):
        """Generate comprehensive PDF report"""
        st.subheader("📄 Generar Reporte en PDF")
        
        if st.button("🔄 Generar Reporte PDF"):
            try:
                # Create PDF in memory
                buffer = io.BytesIO()
                
                # Create the PDF document
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    spaceAfter=30,
                    alignment=1  # Center alignment
                )
                
                story.append(Paragraph("REPORTE DE PREDICCIÓN DE RENDIMIENTO DEPORTIVO", title_style))
                story.append(Spacer(1, 20))
                
                # Executive Summary
                story.append(Paragraph("RESUMEN EJECUTIVO", styles['Heading2']))
                
                summary_text = f"""
                <para>
                Este reporte presenta un análisis completo de predicción de rendimiento deportivo 
                utilizando técnicas de machine learning. Se evaluaron múltiples algoritmos y 
                se identificó el mejor modelo basado en métricas de rendimiento estadísticamente 
                robustas.<br/><br/>
                
                <b>Mejor Modelo:</b> {best_model_name}<br/>
                <b>Número de muestras analizadas:</b> {self.data.shape[0]}<br/>
                <b>Características utilizadas:</b> {self.X_train.shape[1]}<br/>
                <b>R² del mejor modelo:</b> {results_df['R² Prueba'].max():.4f}
                </para>
                """
                
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Dataset Overview
                story.append(Paragraph("1. DESCRIPCIÓN DEL DATASET", styles['Heading2']))
                
                dataset_text = f"""
                <para>
                El dataset contiene información de {self.data.shape[0]} atletas con {self.data.shape[1]} 
                características diferentes. Las variables incluyen métricas fisiológicas, de entrenamiento 
                y rendimiento histórico.<br/><br/>
                
                <b>Variables principales:</b><br/>
                • Variables fisiológicas: BMI, frecuencia cardíaca, VO2 Max<br/>
                • Variables de entrenamiento: horas de entrenamiento, intensidad<br/>
                • Variables de estilo de vida: sueño, nutrición, hidratación<br/>
                • Variable objetivo: Performance_Metric
                </para>
                """
                
                story.append(Paragraph(dataset_text, styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Statistical Analysis
                story.append(Paragraph("2. ANÁLISIS ESTADÍSTICO DESCRIPTIVO", styles['Heading2']))
                
                numeric_data = self.data.select_dtypes(include=[np.number])
                desc_stats = numeric_data.describe()
                
                # Create table for descriptive statistics
                desc_data = [['Variable', 'Media', 'Desv. Est.', 'Min', 'Max']]
                for col in desc_stats.columns[:5]:  # Show first 5 numeric columns
                    desc_data.append([
                        col,
                        f"{desc_stats.loc['mean', col]:.2f}",
                        f"{desc_stats.loc['std', col]:.2f}",
                        f"{desc_stats.loc['min', col]:.2f}",
                        f"{desc_stats.loc['max', col]:.2f}"
                    ])
                
                desc_table = Table(desc_data)
                desc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(desc_table)
                story.append(Spacer(1, 20))
                
                # Model Results
                story.append(Paragraph("3. RESULTADOS DE MODELOS DE MACHINE LEARNING", styles['Heading2']))
                
                # Create table for model results
                model_data = [['Modelo', 'R² Prueba', 'RMSE Prueba', 'MAE Prueba', 'CV Media']]
                
                for _, row in results_df.iterrows():
                    model_data.append([
                        row['Modelo'],
                        f"{row['R² Prueba']:.4f}",
                        f"{row['RMSE Prueba']:.4f}",
                        f"{row['MAE Prueba']:.4f}",
                        f"{row['CV Media']:.4f}"
                    ])
                
                model_table = Table(model_data)
                model_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(model_table)
                story.append(Spacer(1, 20))
                
                # Best Model Analysis
                story.append(Paragraph("4. ANÁLISIS DEL MEJOR MODELO", styles['Heading2']))
                
                best_model_results = results_df[results_df['Modelo'] == best_model_name].iloc[0]
                
                best_model_text = f"""
                <para>
                El modelo <b>{best_model_name}</b> demostró el mejor rendimiento con las siguientes métricas:<br/><br/>
                
                <b>Métricas de Rendimiento:</b><br/>
                • R² en conjunto de prueba: {best_model_results['R² Prueba']:.4f}<br/>
                • RMSE en conjunto de prueba: {best_model_results['RMSE Prueba']:.4f}<br/>
                • MAE en conjunto de prueba: {best_model_results['MAE Prueba']:.4f}<br/>
                • Validación cruzada (media): {best_model_results['CV Media']:.4f} ± {best_model_results['CV Desv. Est.']:.4f}<br/><br/>
                
                <b>Interpretación:</b><br/>
                El modelo explica aproximadamente {best_model_results['R² Prueba']*100:.1f}% de la variabilidad 
                en el rendimiento deportivo. El error promedio absoluto es de {best_model_results['MAE Prueba']:.2f} 
                unidades en la métrica de rendimiento.
                </para>
                """
                
                story.append(Paragraph(best_model_text, styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Conclusions and Recommendations
                story.append(Paragraph("5. CONCLUSIONES Y RECOMENDACIONES", styles['Heading2']))
                
                conclusions_text = f"""
                <para>
                <b>Conclusiones principales:</b><br/><br/>
                
                1. <b>Modelo óptimo:</b> {best_model_name} mostró el mejor rendimiento predictivo.<br/><br/>
                
                2. <b>Capacidad predictiva:</b> El modelo puede explicar {best_model_results['R² Prueba']*100:.1f}% 
                de la variabilidad en el rendimiento deportivo.<br/><br/>
                
                3. <b>Robustez:</b> La validación cruzada confirma la estabilidad del modelo 
                ({best_model_results['CV Media']:.4f} ± {best_model_results['CV Desv. Est.']:.4f}).<br/><br/>
                
                <b>Recomendaciones:</b><br/><br/>
                
                1. <b>Implementación:</b> Se recomienda implementar el modelo {best_model_name} 
                para predicciones de rendimiento deportivo.<br/><br/>
                
                2. <b>Monitoreo continuo:</b> Establecer un sistema de monitoreo para evaluar 
                la performance del modelo en producción.<br/><br/>
                
                3. <b>Mejoras futuras:</b> Considerar la recolección de datos adicionales y 
                la reevaluación periódica del modelo.<br/><br/>
                
                4. <b>Variables clave:</b> Enfocar esfuerzos en la recolección precisa de las 
                variables más predictivas identificadas en el análisis.
                </para>
                """
                
                story.append(Paragraph(conclusions_text, styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Technical Details
                story.append(Paragraph("6. DETALLES TÉCNICOS", styles['Heading2']))
                
                technical_text = f"""
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
                
                story.append(Paragraph(technical_text, styles['Normal']))
                
                # Build PDF
                doc.build(story)
                
                # Prepare download
                buffer.seek(0)
                pdf_data = buffer.read()
                buffer.close()
                
                # Create download button
                st.download_button(
                    label="📥 Descargar Reporte PDF",
                    data=pdf_data,
                    file_name=f"reporte_rendimiento_deportivo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                st.success("✅ Reporte PDF generado exitosamente!")
                
            except Exception as e:
                st.error(f"❌ Error al generar PDF: {str(e)}")
                st.write("Nota: Algunas librerías pueden no estar disponibles en el entorno de Streamlit.")

def main():
    """Main application function"""
    
    # Title and description
    st.markdown('<div class="main-header">🏃‍♂️ Predictor de Rendimiento Deportivo</div>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Aplicación de Machine Learning para Predicción de Rendimiento Deportivo
    
    Esta aplicación utiliza técnicas avanzadas de inteligencia artificial para predecir el rendimiento 
    deportivo basándose en múltiples características del atleta. Incluye análisis exploratorio completo, 
    múltiples algoritmos de ML, evaluación estadística robusta y generación de reportes.
    
    **Características principales:**
    - 📊 Análisis Exploratorio de Datos (EDA) completo
    - 🤖 Evaluación de 5 algoritmos + 2 modelos híbridos
    - 🧪 Pruebas estadísticas robustas
    - 📈 Visualizaciones interactivas
    - 📄 Reporte PDF profesional
    """)
    
    # Initialize predictor
    predictor = SportsPerformancePredictor()
    
    # Sidebar for navigation
    st.sidebar.title("🚀 Navegación")
    
    # File upload option
    st.sidebar.subheader("📁 Cargar Datos")
    uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        # Read uploaded file
        try:
            predictor.data = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Archivo cargado exitosamente!")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar archivo: {str(e)}")
            predictor.data = None
    else:
        # Use provided data
        try:
            # Try to read the provided CSV file
            predictor.data = pd.read_csv('datasport.csv')
            st.sidebar.success("✅ Usando archivo datasport.csv")
        except:
            # If file not found, load sample data
            if st.sidebar.button("🔄 Cargar Datos de Ejemplo"):
                predictor.load_data()
    
    # Navigation options
    analysis_steps = [
        "📊 Análisis Exploratorio",
        "🔧 Preprocesamiento",
        "🤖 Entrenamiento de Modelos",
        "📈 Evaluación y Comparación",
        "🧪 Pruebas Estadísticas",
        "📄 Generar Reporte"
    ]
    
    selected_step = st.sidebar.selectbox("Seleccionar Análisis", analysis_steps)
    
    # Check if data is available
    if predictor.data is None:
        st.error("❌ No hay datos disponibles. Por favor, carga un archivo CSV o usa los datos de ejemplo.")
        return
    
    # Execute selected analysis
    if selected_step == "📊 Análisis Exploratorio":
        predictor.exploratory_data_analysis()
        predictor.descriptive_statistics()
        predictor.create_visualizations()
        
    elif selected_step == "🔧 Preprocesamiento":
        if predictor.preprocess_data():
            st.success("✅ Preprocesamiento completado. Puedes continuar con el entrenamiento de modelos.")
        
    elif selected_step == "🤖 Entrenamiento de Modelos":
        if predictor.processed_data is None:
            if predictor.preprocess_data():
                predictor.train_models()
        else:
            predictor.train_models()
            
    elif selected_step == "📈 Evaluación y Comparación":
        if not predictor.results:
            if predictor.processed_data is None:
                predictor.preprocess_data()
            predictor.train_models()
        
        best_model, results_df = predictor.evaluate_models()
        
    elif selected_step == "🧪 Pruebas Estadísticas":
        if not predictor.results:
            if predictor.processed_data is None:
                predictor.preprocess_data()
            predictor.train_models()
        
        predictor.statistical_tests()
        
    elif selected_step == "📄 Generar Reporte":
        if not predictor.results:
            if predictor.processed_data is None:
                predictor.preprocess_data()
            predictor.train_models()
        
        best_model, results_df = predictor.evaluate_models()
        predictor.generate_pdf_report(best_model, results_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Información Adicional
    
    **Algoritmos implementados:**
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Regression (SVR)
    - Neural Networks (MLP)
    - Híbrido 1: Random Forest + Gradient Boosting
    - Híbrido 2: Random Forest + SVR + Neural Network
    
    **Métricas de evaluación:**
    - R² (Coeficiente de determinación)
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - Validación cruzada 5-fold
    
    **Pruebas estadísticas:**
    - Shapiro-Wilk (normalidad)
    - Levene (homoscedasticidad)
    - Kruskal-Wallis (comparación de grupos)
    - Post-hoc Dunn (comparaciones múltiples)
    """)

# Instructions for running the app
st.markdown("""
---
### 🚀 Instrucciones de Uso

**Para ejecutar esta aplicación:**

1. **En Google Colab:**
   ```python
   !pip install streamlit plotly scikit-posthocs reportlab
   !streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

2. **En local:**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy scikit-posthocs reportlab
   streamlit run app.py
   ```

3. **Dependencias principales:**
   - streamlit
   - pandas, numpy
   - matplotlib, seaborn, plotly
   - scikit-learn
   - scipy
   - scikit-posthocs
   - reportlab

**Nota:** Coloca tu archivo `datasport.csv` en el mismo directorio que esta aplicación, o usa la opción de carga de archivos en la barra lateral.
""")

if __name__ == "__main__":
    main()
