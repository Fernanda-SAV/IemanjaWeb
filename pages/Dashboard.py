import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Iemanjá Web", layout="wide")

with st.container():
    st.subheader('Iemanja Web')
    st.title('Dashboards para Monitoramento e Previsão de Maré.')

    st.page_link("Inicio.py", label="Voltar", icon="↩️")


with st.container():
    st.write('---')
    st.write('Dashboards de Monitoramento do mar na Baía de São Marcos.')

@st.cache_data
def load_data():
    data = pd.read_csv('maregrafo_semoutlier_08_2022a08_2024.csv')  
    return data

data = load_data()


date_column = 'Datetime'  
data[date_column] = pd.to_datetime(data[date_column], errors='coerce')


if data[date_column].isnull().any():
    st.error("A coluna de data contém valores inválidos ou não é uma data válida.")
else:
    # Filtros de data
    start_date = st.date_input("Data de início", data[date_column].min())
    end_date = st.date_input("Data de fim", data[date_column].max())

    if start_date < end_date:
        filtered_data = data[(data[date_column] >= pd.to_datetime(start_date)) &
                             (data[date_column] <= pd.to_datetime(end_date))]

        
        value_column = 'Mare_medida'  

        
        st.subheader("Dados Filtrados")
        st.write(filtered_data[[date_column, value_column]].head())

        st.subheader("Gráfico de Série Temporal")
        fig = px.line(filtered_data, x=date_column, y=value_column, title='Série Temporal')
        
        st.plotly_chart(fig)
    else:
        st.error('Erro: A data de início deve ser anterior à data de fim.')



with st.container():
    st.write('---')
    st.write('Dashboards de Previsão do Mar na Baía de São Marcos.')



@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    return df


file_path = ('dados_atualizados_final.csv') 
df=load_data(file_path)



st.subheader("Dados do CSV")
st.write(data.head())


X = df[['coef_mare_instantaneo', 's4', 'pressao_atm', 'veloc_vento']].values
y = df['Mare_medida'].values
time_column = 'datetime'  
time = df[time_column]




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


def plot_with_plotly(y_test, y_test_pred):
    df_results = pd.DataFrame({
        'Valores Reais': y_test,
        'Previsões': y_test_pred
    })

    fig = px.scatter(df_results, x='Valores Reais', y='Previsões', title="Comparação: Valores Reais vs Previsões",
                     labels={'Valores Reais': 'Valores Reais (m)', 'Previsões': 'Previsões (m)'})

    fig.add_shape(
        type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
        line=dict(color="Red", dash="dash"),
    )
    return fig


st.title('Visualização dos Resultados da Rede Neural')

plotly_fig = plot_with_plotly(y_test, y_test_pred)
st.plotly_chart(plotly_fig)



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


st.title('Visualização das Previsões do Nível do Mar')

plotly_fig = plot_with_plotly(time_test, y_test, y_test_pred)
st.plotly_chart(plotly_fig)


with st.container():
    st.subheader('Gostaria de saber o nível do mar para um dia e hora específicos? [Clique aqui!](Previsao)')
