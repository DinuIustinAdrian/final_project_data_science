"""
# My first app
Here's our first attempt at using data to create a table:
"""

import numpy as np
import pandas as pd
import streamlit as st
import os

def lista_file(directory, inizia_con=None, finisce_con=None):
    lista = []
    for file in os.listdir(directory):
        if inizia_con is not None and finisce_con is not None:
            if file.startswith(inizia_con) and file.endswith(finisce_con):
                lista.append(file)
        elif inizia_con is not None:
            if file.startswith(inizia_con):
                lista.append(file)
        elif finisce_con is not None:
            if file.endswith(finisce_con):
                lista.append(file)
        else:
            lista.append(file)
    return lista

def rinomina_file(nome_file):
    nome_file = nome_file.replace("plot_", "")
    nome_file = nome_file.replace("_vs_", " vs ")
    nome_file = nome_file.replace(".svg", "")

    return nome_file


pathFile = 'Student_Depression_Dataset.csv'
pathLog = 'data/debug/'
pathPlots= 'data/plots/'

## LISTA PLOT ###
listaPlot = lista_file(pathPlots, finisce_con='svg')
#listaPlot = [rinomina_file(file) for file in listaPlot]

df1 = pd.read_csv(filepath_or_buffer=pathFile)

if st.checkbox(label='Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

if st.checkbox("Prova 1"):
    st.write("CIAO")

def creaTabella():
    return None

if st.checkbox("Vedi Plot Salvati"):
    st.write(listaPlot)

scelta = st.menu_button("Vedi Plot Salvati", options=listaPlot)
str(scelta)

### Display di un plot svg
if scelta != None:
    st.image(pathPlots+scelta, caption=scelta)
    




