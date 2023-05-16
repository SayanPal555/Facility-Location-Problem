from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("D:\\FacilityLocationProblem\data.xlsx")

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['X','Y']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of square error')
plt.plot(k_rng,sse)
plt.show()