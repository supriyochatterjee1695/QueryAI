import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from langchain_community.llms import Ollama
from pandasai import SmartDataframe
import seaborn



st.set_page_config(page_title="SupDATA AI V1.0", layout= "wide")


st.title("SupDATA AI V1.0")


llm = Ollama(
    model='mistral'
)

st.header("Select your csv file for analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(10))


    df = SmartDataframe(data, config={"llm": llm})
    prompt = st.text_area("Enter your query")

    if st.button("Analyse"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))
                