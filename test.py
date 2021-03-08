import pandas as pd
import streamlit as st
df = pd.read_csv('label_df.csv')
st.title("Testing")
st.write(df.head())