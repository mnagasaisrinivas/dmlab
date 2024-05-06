import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Iris_data = pd.read_csv("E:\Iris.csv")
Iris_data.info()
Iris_data.head(3)
Iris_data.describe()
Iris_data.Species.value_counts()

plt.scatter(Iris_data['SepalLengthCm'],Iris_data['SepalWidthCm'])
plt.show()

sns.set_style('whitegrid')
sns.FacetGrid(Iris_data,hue ='Species') \
.map(plt.scatter,'SepalLengthCm','SepalWidthCm') \
.add_legend() 
plt.show()

sns.pairplot(Iris_data.drop(['Id'],axis=1),hue='Species')
plt.show()
