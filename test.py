import pandas as pd
import streamlit as st
df = pd.read_csv('label_df.csv')

st.title("Testing")
num = 5
num = st.text_input("Input the number of rows")
num = int(num)
st.write(df.head(num))