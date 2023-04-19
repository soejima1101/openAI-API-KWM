import streamlit as st
import streamlit_authenticator as stauth

from openai_app import openai_app_main

import yaml
from yaml.loader import SafeLoader



with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    openai_app_main()
    
    # with st.sidebar:
    #     authenticator.logout('Logout', 'main')
    #     st.write(f'Welcome *{name}*')
    
    
elif authentication_status is False:
    st.error('Username/password is incorrect')
# elif authentication_status is None:
#     st.warning('Please enter your username and password')
    
if st.session_state["authentication_status"]:
    with st.sidebar:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{st.session_state["name"]}*')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] is None:
#     st.warning('Please enter your username and password')