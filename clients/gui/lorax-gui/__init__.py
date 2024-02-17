import streamlit as st
import requests
import streamlit_pydantic as sp
from LORAX_TYPES import Parameters

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

LORAX_PORT = 8080
HOST = f"http://localhost:{LORAX_PORT}"

data = sp.pydantic_form(key="my_form", model=Parameters)
txt = st.text_area("Enter prompt", "Type Here ...")

data = {
    "inputs": txt,
    "parameters": {
        "max_new_tokens": 64
    }
}

if data:
    st.json(data.json())

if st.button("Generate"):
    response = requests.post(f"{HOST}/generate", json=data)
    response.raise_for_status()
    resp_data = response.json()
    st.write(resp_data['generated_text'])
