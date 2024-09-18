import streamlit as st

with st.container():
    st.subheader('Iemanjá Web')
    st.title("Contato")
    st.page_link("Inicio.py", label="Voltar", icon="↩️")

with st.container():
    st.write("---")
    st.write("Preencha o formulário abaixo para entrar em contato conosco.")

    with st.form(key='contact_form'):
        name = st.text_input("Nome")
        email = st.text_input("Email")
        message = st.text_area("Mensagem")

        # Botão para enviar o formulário
        submit_button = st.form_submit_button(label='Enviar')

        if submit_button:
            if name and email and message:
                st.success(f"Obrigado, {name}! Sua mensagem foi enviada com sucesso.")
                st.write("Nome:", name)
                st.write("Email:", email)
                st.write("Mensagem:", message)
            else:
                st.error("Por favor, preencha todos os campos.")

