import streamlit as st

st.set_page_config(page_title="Iemanja Web", layout="wide")

with st.container():
    st.subheader('Iemanjá Web')
    st.title('Sistema Web para Monitoramento e Previsão de Maré.')
    st.write('Monitore o nível do Mar e tenha previsões de como ele estará no dia de seu interesse.')
    st.write('Quer um monitoramento personalizado? [Clique aqui.](Personalizado)')

    st.write('Faça seu cadastro para ter acesso à previsões para sua região. [Clique aqui.](Cadastro)')

with st.container():
    st.write("---")
    st.write("Navegue pelo site.")
    st.page_link("Inicio.py", label="Home", icon="🏠")
    st.page_link("pages/Dashboard.py", label="Dashboards", icon="📈")
    st.page_link("pages/Sobre.py", label="Sobre", icon="🔎")
    st.page_link("pages/Contato.py", label="Contato", icon="📲")


