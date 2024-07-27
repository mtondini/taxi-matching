import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los resultados
df_punto6_85 = pd.read_csv('/Users/luanagiusto/PycharmProjects/TP Modelos/resultados_punto6_85.csv')
df_punto6_80 = pd.read_csv('/Users/luanagiusto/PycharmProjects/TP Modelos/resultados_punto6_80.csv')
df_punto6_90 = pd.read_csv('/Users/luanagiusto/PycharmProjects/TP Modelos/resultados_punto6_90.csv')

# Combinar los resultados de los diferentes thresholds
df_punto6_combined = pd.concat([df_punto6_80, df_punto6_85, df_punto6_90])

# Identificar las asignaciones no factibles (Distancia LP es inf)
df_punto6_combined['Distancia LP'] = pd.to_numeric(df_punto6_combined['Distancia LP'], errors='coerce')
df_punto6_combined['No Factible'] = df_punto6_combined['Distancia LP'] == float('inf')

# Calcular el número de asignaciones no factibles
no_factibles_df = df_punto6_combined.groupby(['Instancia', 'Percentil'])['No Factible'].sum().reset_index()
no_factibles_df.columns = ['Instancia', 'Percentil', 'Numero de asignaciones no factibles']

# Filtrar las filas factibles para calcular la mejora relativa
df_factible = df_punto6_combined[~df_punto6_combined['No Factible']].copy()
df_factible['Mejora relativa'] = 100 * (df_factible['Distancia greedy'] - df_factible['Distancia LP']) / df_factible['Distancia LP']

# Agrupar los resultados por instancia y percentil para analizar las métricas
grouped = df_factible.groupby(['Instancia', 'Percentil']).agg({
    'Distancia greedy': 'mean',
    'Distancia LP': 'mean',
    'Mejora relativa': 'mean',
    'Tiempo LP': 'mean'
}).reset_index()

# Merge con el número de asignaciones no factibles
final_df = pd.merge(grouped, no_factibles_df, on=['Instancia', 'Percentil'], how='left')

# Asegurarse de que las asignaciones no factibles sean tratadas como 0 en el DataFrame final
final_df['Numero de asignaciones no factibles'] = final_df['Numero de asignaciones no factibles'].fillna(0).astype(int)

# Guardar los resultados en un archivo CSV
final_df.to_csv('resultados_comparativos.csv', index=False)

# Mostrar los resultados
print(final_df)

# Visualizar los resultados comparativos con matplotlib
def visualize_results(df):
    plt.figure(figsize=(14, 8))

    for percentil in [80, 85, 90]:
        subset = df[df['Percentil'] == percentil]
        plt.plot(subset['Instancia'], subset['Mejora relativa'], label=f'Percentil {percentil}')

    plt.xlabel('Instancia')
    plt.ylabel('Mejora relativa (%)')
    plt.title('Mejora relativa por percentil')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    for percentil in [80, 85, 90]:
        subset = df[df['Percentil'] == percentil]
        plt.plot(subset['Instancia'], subset['Numero de asignaciones no factibles'], label=f'Percentil {percentil}')

    plt.xlabel('Instancia')
    plt.ylabel('Numero de asignaciones no factibles')
    plt.title('Asignaciones no factibles por percentil')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Visualizar los resultados
visualize_results(final_df)

# Analizar los resultados comparativos
def interpret_results(df):
    print("Análisis de Resultados Comparativos:")
    for percentil in [80, 85, 90]:
        subset = df[df['Percentil'] == percentil]
        avg_mejora = subset['Mejora relativa'].mean()
        avg_no_factibles = subset['Numero de asignaciones no factibles'].mean()
        print(f"Percentil {percentil}:")
        print(f"  - Promedio Mejora %: {avg_mejora:.2f}")
        print(f"  - Promedio Asignaciones No Factibles: {avg_no_factibles:.2f}")

# Interpretar los resultados comparativos
interpret_results(final_df)
