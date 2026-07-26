#Librerías
import joblib
import pandas as pd
import streamlit as st

#Configuración de la pestaña de la página
st.set_page_config(
    page_title='Detección temprana de diabetes',
    page_icon='🩺',
    layout='wide'
)

#Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load('Modelo_deteccion_diabetes_svc.pkl') #Ruta del archivo

modelo = cargar_modelo()

#Título
st.title('Detección temprana de diabetes 🩺')

#Descripción de la página
st.write(
    """
    Calculadora de probabilidad de padecer diabetes a partir de indicadores clínicos.
    """
)

#Advertencia
st.warning(
    """
    Esta aplicación tiene fines informativos y de prevención, NO sustituye un diagnóstico médico profesional.
    """
)

#Información del modelo
with st.expander('Información sobre el modelo'):
    st.write(
        """
        Modelo: Máquina de Soporte Vectorial.

        **Métricas de evaluación:**

        - Accuracy: 0.72
        - Precision: 0.58
        - Recall: 0.74
        - F1-score: 0.65
        - ROC-AUC: 0.807
        """
    )

#Forrmulario para ingresar los datos del paciente
with st.form('formulario_paciente'):
    
    #Subtítulo
    st.subheader('Ingrese la información clínica del paciente')
    st.write('Los datos con los que no cuente pueden ser ingresados como 0 pero, considere que esto afectará la credibilidad del resultado')
    
    #Dividiendo en dos columnas el formulario
    col1, col2 = st.columns(2)

    #Variables de la columna 1
    with col1:
        embarazos = st.number_input(
            'Número de embarazos',
            min_value=0,
            max_value=10,
            value=1
        )

        glucosa = st.number_input(
            'Glucosa',
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        presion_arterial = st.number_input(
            'Presión arterial diastólica',
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        espesor_cutaneo = st.number_input(
            'Espesor del pliegue cutáneo',
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

    #Variables de la columna 2
    with col2:
        insulina = st.number_input(
            'Insulina',
            min_value=0.0,
            max_value=900.0,
            value=80.0
        )

        imc = st.number_input(
            'Índice de masa corporal',
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )

        pedigri = st.number_input(
            'Pedigrí de función de diabetes',
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        edad = st.number_input(
            'Edad (de 18 a 100 años)',
            min_value=18,
            max_value=100,
            value=30
        )

    #Botón para realizar la predicción
    boton_predecir = st.form_submit_button(
        'Evaluar',
        use_container_width=True
    )

#Lógica del botón
if boton_predecir:

    #Relación de columnas
    info_paciente = pd.DataFrame({
        'Embarazos': [embarazos],
        'Glucosa': [glucosa],
        'Presión Arterial': [presion_arterial],
        'Espesor Cutáneo': [espesor_cutaneo],
        'Insulina': [insulina],
        'IMC': [imc],
        'Pedigrí de Función de Diabetes': [pedigri],
        'Edad': [edad]
    })

    #Predicción y probabilidad
    prediccion = modelo.predict(info_paciente)[0]
    probabilidad_diabetes = modelo.predict_proba(info_paciente)[0, 1]

    st.divider()
    st.subheader('Resultado')

    st.metric(
        'Probabilidad estimada de padecer diabetes:',
        f'{probabilidad_diabetes:.1%}'
    )

    st.progress(float(probabilidad_diabetes))

    #Riesgo Bajo
    if probabilidad_diabetes < 0.40:
        st.success(
            'Se identificó un riesgo BAJO de padecer diabetes.'
        )

    #Riesgo Medio
    elif probabilidad_diabetes < 0.65:
        st.warning(
            'Se identificó un riesgo MEDIO de padecer diabetes, se recomienda consultar a su médico.'
        )

    #Riesgo Alto
    else:
        st.error(
            'Se detectó un riesgo ALTO de padecer diabetes, por favor, consulte a su médico.'
        )

    #DF con el resumen de los datos usados
    with st.expander('Datos utilizados para la predicción'):
        st.dataframe(
            info_paciente,
            use_container_width=True,
            hide_index=True
        )