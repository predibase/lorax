import streamlit as st
import requests

LORAX_PORT = 8000
HOST = f"http://localhost:{LORAX_PORT}"

txt = st.text_area("Enter prompt", "Type Here ...")

data = {
    "inputs": txt,
    "parameters": {
        "max_new_tokens": 64
    }
}

if st.button("Generate"):
    response = requests.post(f"{HOST}/generate", json=data)
    st.write(response.json()["text"])
