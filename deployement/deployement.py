import time
import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('/app/results/results.csv')
df['date'] = pd.to_datetime(df['date'])

st.title('Real and predicted power consumption')

# Tworzenie wykresu za pomocą funkcji line_chart w Streamlit
st.line_chart(df.set_index('date'))

while True:
    time.sleep(1)