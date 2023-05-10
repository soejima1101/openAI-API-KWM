import streamlit as st
import streamlit_authenticator as stauth

from main import openai_app_main

import yaml
from yaml.loader import SafeLoader



# ===========================================================================================================================================
#
# 認証処理（ログイン画面生成）
#
# ===========================================================================================================================================
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"]
)

name, authentication_status, username = authenticator.login("OpenAI App Login", "main")

if authentication_status:
    openai_app_main()
    
# elif authentication_status is False:
#     st.error('Username/password is incorrect')

try:
    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.write(f'Welcome *{st.session_state["name"]}* !!')
            authenticator.logout('Logout', 'main')

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        
except:
    st.error("Session Error")
