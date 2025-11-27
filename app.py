import streamlit as st
import pandas as pd
from joblib import load

# Cargar modelo y preprocesadores
pipeline = load("output/GradientBoosting_optimized_hybrid.joblib")
scaler = load("output/scaler.joblib")
encoder = load("output/encoder.joblib")
column_info = load("output/column_info.joblib")

cat_cols = column_info['cat_cols']
num_cols = column_info['num_cols']

st.title("Predicción de Marketing Bancario")
st.write("Predice si un cliente contratará un depósito a plazo usando Gradient Boosting.")

st.sidebar.header("Ingrese los datos del cliente")

age = st.sidebar.number_input("Edad", min_value=18, max_value=100, value=30)
balance = st.sidebar.number_input("Balance", value=1000)
day = st.sidebar.number_input("Día del mes de contacto", min_value=1, max_value=31, value=15)
duration = st.sidebar.number_input("Duración de la llamada (segundos)", min_value=1, value=200)
campaign = st.sidebar.number_input("Número de contactos durante esta campaña", min_value=1, value=1)
pdays = st.sidebar.number_input("Días desde último contacto (-1 si nunca contactado)", value=-1)
previous = st.sidebar.number_input("Número de contactos previos", min_value=0, value=0)
emp_var_rate = st.sidebar.number_input("Tasa de variación del empleo", value=1.0)
cons_price_idx = st.sidebar.number_input("Índice de precio al consumidor", value=93.0)
cons_conf_idx = st.sidebar.number_input("Índice de confianza del consumidor", value=-40.0)
euribor3m = st.sidebar.number_input("Tasa Euribor 3 meses", value=4.0)
nr_employed = st.sidebar.number_input("Número de empleados", value=5000.0)

job = st.sidebar.selectbox("Trabajo", ["admin.", "technician", "services", "management", "retired",
                                        "blue-collar", "unemployed", "entrepreneur", "housemaid",
                                        "self-employed", "student", "unknown"])
marital = st.sidebar.selectbox("Estado civil", ["married", "single", "divorced"])
education = st.sidebar.selectbox("Educación", ["primary", "secondary", "tertiary", "unknown"])
default = st.sidebar.selectbox("Tiene deuda?", ["yes", "no"])
housing = st.sidebar.selectbox("Crédito hipotecario?", ["yes", "no"])
loan = st.sidebar.selectbox("Préstamo personal?", ["yes", "no"])
contact = st.sidebar.selectbox("Tipo de contacto", ["cellular", "telephone", "unknown"])
month = st.sidebar.selectbox("Mes de contacto", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
day_of_week = st.sidebar.selectbox("Día de la semana", ["mon","tue","wed","thu","fri"])
poutcome = st.sidebar.selectbox("Resultado de campaña previa", ["failure","nonexistent","success"])

if st.button("Predecir"):
    try:
        # Crear DataFrame con los datos de entrada
        input_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed,
            "day_of_week": day_of_week
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Aplicar el mismo preprocesamiento
        # 1) Separar numéricas y categóricas
        X_num = input_df[num_cols]
        X_cat = input_df[cat_cols]
        
        # 2) Escalar numéricas
        X_num_scaled = pd.DataFrame(scaler.transform(X_num), columns=num_cols)
        
        # 3) Codificar categóricas
        X_cat_encoded = pd.DataFrame(encoder.transform(X_cat),
                                      columns=encoder.get_feature_names_out(cat_cols))
        
        # 4) Concatenar
        X_processed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
        
        # 5) Hacer predicción
        prediction = pipeline.predict(X_processed)
        proba = pipeline.predict_proba(X_processed)

        st.subheader("Resultado de la predicción:")
        if prediction[0] == 1:
            st.success("✅ Sí - El cliente probablemente contratará el depósito")
        else:
            st.warning("❌ No - El cliente probablemente no contratará el depósito")
        
        st.subheader("Probabilidades:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("No contratará", f"{proba[0][0]:.1%}")
        with col2:
            st.metric("Sí contratará", f"{proba[0][1]:.1%}")
        
        # Mostrar confianza
        confidence = max(proba[0])
        st.info(f"Nivel de confianza: {confidence:.1%}")
            
    except Exception as e:
        st.error(f"❌ Error en la predicción")
        with st.expander("Ver detalles del error"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())