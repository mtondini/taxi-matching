import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV
file_path = '/resultados_punto4.csv'

# Gráfico de barras para comparar las distancias greedy y LP
plt.figure(figsize=(14, 8))
sns.barplot(data=df_punto4, x='Instancia', y='Distancia greedy', color='blue', label='Distancia greedy')
sns.barplot(data=df_punto4, x='Instancia', y='Distancia LP', color='red', alpha=0.6, label='Distancia LP')
plt.xlabel('Instancia')
plt.ylabel('Distancia')
plt.title('Comparación de Distancias: Greedy vs LP')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('comparacion_distancias_greedy_vs_lp.png')
plt.show()

# Gráfico de líneas para la mejora relativa
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_punto4, x='Instancia', y='Mejora relativa', marker='o', label='Mejora relativa (%)')
plt.xlabel('Instancia')
plt.ylabel('Mejora relativa (%)')
plt.title('Mejora Relativa entre Greedy y LP')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('mejora_relativa_greedy_vs_lp.png')
plt.show()

# Gráfico de dispersión para el tiempo de ejecución
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df_punto4, x='Instancia', y='Tiempo LP', s=100, color='green', label='Tiempo LP (s)')
plt.xlabel('Instancia')
plt.ylabel('Tiempo LP (s)')
plt.title('Tiempo de Ejecución del Modelo LP por Instancia')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('tiempo_ejecucion_lp.png')
plt.show()

