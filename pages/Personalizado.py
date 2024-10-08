import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Iemanjá Web", layout="wide")

with st.container():
    st.subheader('Iemanja Web')
    st.title('Dashboards Personalizados para Monitoramento da Maré.')

    st.page_link("Inicio.py", label="Voltar", icon="↩️")

with st.container():
    st.write('---')
    st.write('Analise seus próprios dados aqui!')

    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
   
    data = pd.read_csv(uploaded_file)

    
    st.subheader("Dados do CSV")
    st.write(data.head())  

    
    date_column = st.selectbox("Selecione a coluna de data", data.columns)

   
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    
    if data[date_column].isnull().any():
        st.error("A coluna de data contém valores inválidos ou não é uma data válida.")
    else:
        
        start_date = st.date_input("Data de início", data[date_column].min())
        end_date = st.date_input("Data de fim", data[date_column].max())

        if start_date < end_date:
            filtered_data = data[(data[date_column] >= pd.to_datetime(start_date)) &
                                 (data[date_column] <= pd.to_datetime(end_date))]

            
            value_column = st.selectbox("Selecione a coluna de valor", data.columns.drop(date_column))

            
            st.subheader("Dados Filtrados")
            st.write(filtered_data[[date_column, value_column]].head())

            st.subheader("Gráfico de Série Temporal")
            fig = px.line(filtered_data, x=date_column, y=value_column, title='Série Temporal')
            
            st.plotly_chart(fig, use_container_widht=False)

        else:
            st.error('Erro: A data de início deve ser anterior à data de fim.')
else:
    st.info("Por favor, faça o upload de um arquivo CSV para visualizar os dados.")
