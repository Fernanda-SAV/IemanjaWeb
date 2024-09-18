
import sqlite3
import streamlit as st

st.set_page_config(page_title="Iemanja Web", layout="wide")


# Conectando ao banco de dados SQLite (cria o banco de dados se não existir)
conn = sqlite3.connect('cadastro.db')
c = conn.cursor()

# Criar a tabela de usuários se ela não existir
def create_table():
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  password TEXT)''')

# Função para salvar o usuário no banco de dados
def add_user(name, email, password):
    c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, password))
    conn.commit()

# Chamar a função para criar a tabela no banco
create_table()


st.title("Cadastro de Usuário")

st.write("Preencha os campos abaixo para criar sua conta.")

# Formulário de cadastro
with st.form(key='registration_form'):
    name = st.text_input("Nome Completo")
    email = st.text_input("Email")
    password = st.text_input("Senha", type="password")
    confirm_password = st.text_input("Confirme sua Senha", type="password")

    # Botão para enviar o formulário
    submit_button = st.form_submit_button(label='Cadastrar')

if submit_button:
    if name and email and password and confirm_password:
        if password == confirm_password:
            add_user(name, email, password)
            st.success(f"Obrigado, {name}! Seu cadastro foi realizado com sucesso.")
        else:
            st.error("As senhas não coincidem. Tente novamente.")
    else:
        st.error("Por favor, preencha todos os campos.")


st.page_link("Inicio.py", label="Voltar", icon="↩️")

# Função para buscar todos os usuários no banco de dados
def get_users():
    c.execute('SELECT * FROM users')
    return c.fetchall()

# Exibir os usuários cadastrados
st.subheader("Usuários Cadastrados")
users = get_users()
for user in users:
    st.write(f"ID: {user[0]}, Nome: {user[1]}, Email: {user[2]}")
