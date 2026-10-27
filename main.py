import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Caricamento dati

dataset = 'Student_Depression_Dataset.csv'
df = pd.read_csv(dataset)

print(df.columns.values)