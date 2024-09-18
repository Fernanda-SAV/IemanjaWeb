import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime

st.set_page_config(page_title="Iemanja Web", layout="wide")


st.subheader('Iemanja Web')
st.title('Previsão Personalizada.')

with st.container():
    st.write('---')
    st.write('Previsão Personalizada do Mar na Baía de São Marcos')
    st.page_link("Inicio.py", label="Voltar", icon="↩️")



@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    return df



def predict_sea_level(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]



def plot_with_plotly(time, y_test, y_test_pred):
    df_results = pd.DataFrame({
        'Tempo': time,
        'Valores Reais': y_test,
        'Previsões': y_test_pred
    })

    fig = px.line(df_results, x='Tempo', y=['Valores Reais', 'Previsões'],
                  title="Previsões do Nível do Mar ao Longo do Tempo",
                  labels={'Tempo': 'Tempo', 'value': 'Nível do Mar (m)'})
    return fig



file_path = ('dados_atualizados_final.csv')  
df=load_data(file_path)



time_column = 'datetime'  
X = df[['coef_mare_instantaneo', 's4', 'pressao_atm', 'veloc_vento']].values
y = df['Mare_medida'].values
time = df[time_column]  


s4_mean = df['s4'].mean()
veloc_vento_mean = df['veloc_vento'].mean()


df_filled = df.copy()
df_filled['s4'].fillna(s4_mean, inplace=True)
df_filled['veloc_vento'].fillna(veloc_vento_mean, inplace=True)


X_filled = df_filled[['coef_mare_instantaneo', 's4', 'pressao_atm', 'veloc_vento']].values
y_filled = df_filled['Mare_medida'].values



X_train, X_temp, y_train, y_temp, time_train, time_temp = train_test_split(X, y, time, test_size=0.6, random_state=42)


X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
time_val, time_test = train_test_split(time_temp, test_size=0.5, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


mlp = MLPRegressor(hidden_layer_sizes=(10), tol = 0.0000001, learning_rate_init = 0.1, solver='sgd', activation='relu', learning_rate='constant', verbose=2, max_iter=1000, random_state=42)


mlp.fit(X_train_scaled, y_train)


y_test_pred = mlp.predict(X_test_scaled)


st.title('Visualização das Previsões do Nível do Mar')

plotly_fig = plot_with_plotly(time_test, y_test, y_test_pred)
st.plotly_chart(plotly_fig)


st.subheader('Previsão do Nível do Mar com Dados Parciais')


coef_mare_instantaneo = st.number_input('Coeficiente Mare Instantâneo', value=0.0)
pressao_atm = st.number_input('Pressão Atmosférica', value=0.0)


input_data = [coef_mare_instantaneo, s4_mean, pressao_atm, veloc_vento_mean]

input_dia = st.date_input('Data:', datetime.date(2022,8,8))

if st.button('Prever Nível do Mar com Dados Parciais para o dia escolhido', key='predict_button'):
    prediction = predict_sea_level(mlp, scaler, input_data)
    st.write(f'Previsão do nível do mar para o dia {input_dia} com coef_mare_instantaneo={coef_mare_instantaneo} e pressão_atm={pressao_atm}(hPa): {prediction:.2f} metros')

