import pandas as pd
import matplotlib.pyplot as plt

# reading data from csv file
df = pd.read_csv('./data/data.csv')

print(df)
df.plot()
plt.show() # show the plot
