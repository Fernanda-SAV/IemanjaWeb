import streamlit as st

st.set_page_config(page_title="Iemanjá Web", layout="wide")



with st.container():
    st.subheader('Iemanjá Web')
    st.title("Sobre")
    st.page_link("Inicio.py", label="Voltar", icon="↩️")

with st.container():
    st.write('---')
st.write("""
Nossa aplicação web oferece uma **interface fácil e interativa** para o monitoramento e previsão do nível do mar. Desenvolvida para pesquisadores, profissionais e empresas, esta ferramenta proporciona uma análise eficiente e acessível dos dados marinhos.
""")
st.write('---')
col1, col2 = st.columns(2)
with col1:
    with st.container():

        st.markdown("""
        ### Características Principais:
- **Interface Amigável:** Navegue com facilidade e acesse informações detalhadas sem a necessidade de conhecimentos técnicos avançados.
- **Visualizações Dinâmicas:** Explore gráficos interativos com Matplotlib e Plotly. Veja gráficos estáticos e, ao clicar em um botão, acesse visualizações interativas para uma análise mais aprofundada.
- **Previsões Avançadas:** Utilize redes neurais perceptron multicamadas para obter previsões precisas do nível do mar com base em variáveis como coeficiente de maré e pressão atmosférica.
- **Análise de Tendências:** Monitore a variação do nível do mar ao longo do tempo e identifique padrões importantes.
- **Previsões Personalizadas:** Insira dados específicos para obter previsões adaptadas a datas e horários futuros.
        """)

with col2:
    with st.container():
        st.markdown("""
### Benefícios:

- **Eficiência e Precisão:** Receba análises detalhadas e previsões precisas rapidamente.
- **Acessibilidade:** Interface intuitiva que facilita o uso, mesmo para usuários com pouca experiência técnica.
- **Flexibilidade:** Personalize análises e previsões de acordo com suas necessidades específicas.
""")

st.write()
st.write('---')
st.write('Experimente nossa aplicação e transforme a forma como você interage com os dados do nível do mar!')