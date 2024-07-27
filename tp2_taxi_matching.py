import pandas as pd
from Instance import Instance
import cplex
import numpy as np
import time

EPS = 1e-6  # Pequeño valor para evitar problemas de precisión numérica

################## Solucion FCFS greedy ######################
def solve_instance_greedy(inst):
    n = inst.n
    dist_matrix = inst.dist_matrix

    T = set(range(n))  # Generamos el diccionario de tuplas
    solution = []
    total_distance = 0

    for j in range(n):
        min_distance = float('inf')
        best_taxi = -1
        for i in T:
            if dist_matrix[i][j] < min_distance:
                min_distance = dist_matrix[i][j]
                best_taxi = i

        solution.append((best_taxi, j))
        total_distance += min_distance
        T.remove(best_taxi)

    return total_distance, solution

################## Solucion LP ################################
def generate_variables(inst, myprob): 
    n = inst.n
    dist_matrix = inst.dist_matrix

    obj = [dist_matrix[i][j] for i in range(n) for j in range(n)]
    var_names = ["x_{}_{}".format(i, j) for i in range(n) for j in range(n)]
    var_types = [myprob.variables.type.binary] * (n * n)
    myprob.variables.add(obj=obj, types=var_types, names=var_names)

def generate_constraints(inst, myprob, threshold=None): # Generamos las restricciones 
    n = inst.n

    for j in range(n): # Generamos la resticcion de que cada cliente solo pueda tener un vehiculo asignado
        indices = ["x_{}_{}".format(i, j) for i in range(n)]
        values = [1.0] * n
        myprob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],
            rhs=[1.0]
        )

    for i in range(n): # Generamos la resticcion de que cada vehiculo solo pueda tener un cliente asignado
        indices = ["x_{}_{}".format(i, j) for j in range(n)]
        values = [1.0] * n
        myprob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],
            rhs=[1.0]
        )

    if threshold is not None: # Agregamos una restriccion de threshold para el punto 5
        for i in range(n):
            for j in range(n):
                if inst.dist_matrix[i][j] > threshold:
                    myprob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=["x_{}_{}".format(i, j)], val=[1.0])],
                        senses=["E"],
                        rhs=[0.0]
                    )

def populate_by_row(inst, myprob, threshold=None):
    generate_variables(inst, myprob)
    generate_constraints(inst, myprob, threshold)

def solve_instance_lp(inst, threshold=None): #Generamos nuestro modelo de programación lineal 
    myprob = cplex.Cplex()
    myprob.set_problem_type(cplex.Cplex.problem_type.LP)

    populate_by_row(inst, myprob, threshold)

    start_time = time.time()
    myprob.solve()
    solve_time = time.time() - start_time

    # Validamos que la solución sea óptima
    if myprob.solution.get_status() not in [1, 101, 102]:
        return float('inf'), [], solve_time

    total_distance = myprob.solution.get_objective_value()

    solution = []
    values = myprob.solution.get_values()
    n = inst.n
    for index, value in enumerate(values):
        if value > EPS:  # Usamos EPS para evitar problemas de precisión numérica
            i = index // n
            j = index % n
            solution.append((i, j))

    return total_distance, solution, solve_time

###############################################################

def main():
    inst_types = ['small', 'medium', 'large', 'xl']
    n_inst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    percentiles = [80, 85, 90]  # Percentiles a utilizar

    # Creamos un vector vacío para guardar los resultados para el punto 4
    results_punto4 = []

    # Resultados para el punto 5, evaluamos el threshold con distintos percentiles
    results_punto6 = {80: [], 85: [], 90: []}

    for t in inst_types:
        for n in n_inst:
            inst_file = 'input/' + t + '_' + n + '.csv'
            inst = Instance(inst_file)

            f_greedy, x_greedy = solve_instance_greedy(inst)
            f_lp, x_lp, lp_time = solve_instance_lp(inst)

            results_punto4.append({
                'Instancia': inst_file,
                'Distancia greedy': f_greedy,
                'Distancia LP': f_lp,
                'Mejora relativa': (f_greedy - f_lp) / f_lp * 100 if f_lp > 0 else float('inf'),
                'Tiempo LP': lp_time
            })

            # Calculamos percentiles
            distances = [inst.dist_matrix[i][j] for i in range(inst.n) for j in range(inst.n)]
            for percentile in percentiles:
                threshold = np.percentile(distances, percentile)
                f_lp_threshold, x_lp_threshold, lp_threshold_time = solve_instance_lp(inst, threshold)

                results_punto6[percentile].append({
                    'Instancia': inst_file,
                    'Percentil': percentile,
                    'Threshold': threshold,
                    'Distancia greedy': f_greedy,
                    'Distancia LP': f_lp_threshold,
                    'Mejora relativa': (f_greedy - f_lp_threshold) / f_lp_threshold * 100 if f_lp_threshold > 0 else float('inf'),
                    'Numero de asignaciones no factibles': len([1 for i, j in x_lp_threshold if inst.dist_matrix[i][j] > threshold]),
                    'Tiempo LP': lp_threshold_time
                })

    # Guardar resultados en archivos CSV
    df_punto4 = pd.DataFrame(results_punto4)
    df_punto4.to_csv('resultados_punto4.csv', index=False)
    print(df_punto4)

    for percentile in percentiles: # Guardar resultados en archivos CSV para cada percentil
        df_punto6 = pd.DataFrame(results_punto6[percentile])
        df_punto6.to_csv(f'resultados_punto6_{percentile}.csv', index=False)
        print(df_punto6)

if __name__ == '__main__':
    main()
