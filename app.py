# import pandas as pd
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# # Cargar datos
# data = pd.read_csv('temperaturas.csv')

# # Convertir la columna 'DATE' a datetime
# data['DATE'] = pd.to_datetime(data['DATE'])

# # Extraer componentes de la fecha, como el año y el día del año
# data['Year'] = data['DATE'].dt.year
# data['DayOfYear'] = data['DATE'].dt.dayofyear

# # Preparar la matriz de características para el clustering
# X = data[['DayOfYear', 'TMAX']].values
# # Graficar los datos antes del dendrograma
# plt.figure(figsize=(10, 7))
# plt.scatter(data['DayOfYear'], data['TMAX'], alpha=0.6, edgecolors='w', s=80)
# plt.title('Dispersión de Temperaturas Máximas a lo Largo del Año')
# plt.xlabel('Día del Año')
# plt.ylabel('Temperatura Máxima')
# plt.grid(True)
# plt.show()

# # Realizar el clustering jerárquico
# linked = linkage(X, 'average')

# # Crear el dendrograma
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#             orientation='top',
#             labels=data['DATE'].astype(str).values,
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.title('Dendrograma Jerárquico con Fechas y Temperatura Máxima')
# plt.show()

# # Decidir el número de clústeres basado en el dendrograma y aplicar el modelo
# clustering = AgglomerativeClustering(n_clusters=4, linkage='average')
# clustering.fit(X)

# # Añadir la asignación de clústeres al DataFrame
# data['Cluster'] = clustering.labels_

# # Calcular y mostrar las métricas de evaluación
# silhouette_avg = silhouette_score(X, clustering.labels_)
# davies_bouldin = davies_bouldin_score(X, clustering.labels_)
# calinski_harabasz = calinski_harabasz_score(X, clustering.labels_)

# print(f"Coeficiente de Silueta: {silhouette_avg}")
# print(f"Índice de Davies-Bouldin: {davies_bouldin}")
# print(f"Índice Calinski-Harabasz: {calinski_harabasz}")



import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Cargar datos
data = pd.read_csv('temperaturas.csv')

# Convertir la columna 'DATE' a datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Extraer componentes de la fecha, como el año y el día del año
data['Year'] = data['DATE'].dt.year
data['DayOfYear'] = data['DATE'].dt.dayofyear

# Convertir la fecha a un número entero de días desde la fecha mínima
data['DaysSinceMin'] = (data['DATE'] - data['DATE'].min()).dt.days

# Preparar la matriz de características para el clustering
X = data[['DaysSinceMin', 'TMAX']].values


# Realizar el clustering jerárquico
linked = linkage(X, 'average')

# Crear el dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=data['DATE'].astype(str).values,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendrograma Jerárquico con Fechas y Temperatura Máxima')
plt.show()

# Decidir el número de clústeres basado en el dendrograma y aplicar el modelo
clustering = AgglomerativeClustering(n_clusters=4, linkage='average')
clustering.fit(X)

# Añadir la asignación de clústeres al DataFrame
data['Cluster'] = clustering.labels_

# Calcular y mostrar las métricas de evaluación
silhouette_avg = silhouette_score(X, clustering.labels_)
davies_bouldin = davies_bouldin_score(X, clustering.labels_)
calinski_harabasz = calinski_harabasz_score(X, clustering.labels_)

print(f"Coeficiente de Silueta: {silhouette_avg}")
print(f"Índice de Davies-Bouldin: {davies_bouldin}")
print(f"Índice Calinski-Harabasz: {calinski_harabasz}")
