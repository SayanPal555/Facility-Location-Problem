from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


n = input('No of Clusters by seeing elbow methods : ')

df = pd.read_excel("D:\\FacilityLocationProblem\data.xlsx")
print(df)

numClus = int(n) 
km = KMeans(n_clusters = numClus)  
y_predicted = km.fit_predict(df[['X','Y']])
df['cluster']=y_predicted

print('Centers of clusters are: ')
print(km.cluster_centers_)

cl=[]
for i in range(numClus):
    c = km.cluster_centers_[i]
    cl.append(c)


X = km.cluster_centers_[:,0] 
Y = km.cluster_centers_[:,1] 


dfrm = df[df.cluster==0]

for j in range(1, numClus):
    data = df[df.cluster==j]
    dfrm = pd.concat([dfrm, data], ignore_index=True)


for j in range(numClus):
    print(dfrm[dfrm['cluster'] == j])

avgPop = []
for i in range(numClus):
    a = sum(df[df.cluster==i].Population) / len(df[df.cluster==i].Population)
    avgPop.append(a)
print('Average population of each cluster: ', avgPop)


xCd = 0
for m in range(numClus):
    xCd = xCd + cl[m][0]*avgPop[m]
xFlp = xCd/sum(avgPop)


yCd = 0
for m in range(numClus):
    yCd = yCd + cl[m][1]*avgPop[m]
yFlp = yCd/sum(avgPop)


print('Facility location is: ( ', xFlp, ', ', yFlp, ' )')

color = ['green', 'black', 'magenta', 'orange', 'brown', 'yellow', 'purple', 'cyan', 'gold', 'gray']

col = []
for i in range(numClus):
    col.append(color[i])


for i in range(numClus):
    plt.scatter(df[df.cluster==i].X, df[df.cluster==i].Y, color = col[i], marker='o')

plt.scatter(X, Y, color='blue',marker='X',label='centroid of each cluster')

plt.scatter(xFlp, yFlp, color='red', marker='*', label='Facility location')


plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
