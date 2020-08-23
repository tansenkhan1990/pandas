import pandas as pd
import numpy as np

poke=pd.read_csv('pokemon_data.csv')

poke.drop('HP')

print(poke.head())