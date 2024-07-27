import pandas as pd
import numpy as np
from Instance import Instance
import cplex

EPS = 1e-6


################## Solucion FCFS greedy ######################
def solve_instance_greedy(inst):
    n = inst.n
    taxis_longlat = inst.taxis_longlat
    paxs_longlat = inst.paxs_longlat
    dist_matrix = inst.dist_matrix

    T = set(range(n))
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


def generate_constraints(inst, myprob, dist_matrix, threshold=None):
    n = inst.n

    for j in range(n):
        indices = ["x_{}_{}".format(i, j) for i in range(n)]
        values = [1.0] * n
        myprob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],
            rhs=[1.0]
        )

    for i in range(n):
        indices = ["x_{}_{}".format(i, j) for j in range(n)]
        values = [1.0] * n
        myprob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],
            rhs=[1.0]
        )

    if threshold is not None:
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][j] > threshold:
                    myprob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=["x_{}_{}".format(i, j)], val=[1.0])],
                        senses=["L"],
                        rhs=[0.0]
                    )


def populate_by_row(inst, myprob, threshold=None):
    dist_matrix = inst.dist_matrix
    generate_variables(inst, myprob)
    generate_constraints(inst, myprob, dist_matrix, threshold)


def solve_instance_lp(inst, threshold=None):
    try:
        myprob = cplex.Cplex()
        myprob.set_problem_type(cplex.Cplex.problem_type.LP)

        populate_by_row(inst, myprob, threshold)

        myprob.solve()

        total_distance = myprob.solution.get_objective_value()

        solution = []
        values = myprob.solution.get_values()
        n = inst.n
        for index, value in enumerate(values):
            if value == 1.0:
                i = index // n
                j = index % n
                solution.append((i, j))

        return total_distance, solution

    except cplex.exceptions.CplexError as exc:
        print(exc)
        return None, None


def calculate_threshold(dist_matrix, percentile=90):
    # Calcular el threshold como el percentil especificado de las distancias
    flattened_distances = [dist_matrix[i][j] for i in range(len(dist_matrix)) for j in range(len(dist_matrix[i]))]
    threshold = np.percentile(flattened_distances, percentile)
    return threshold


###############################################################

def main():
    inst_types = ['small', 'medium', 'large', 'xl']
    n_inst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    percentile = 90  # Percentil para calcular el threshold

    results = []

    for t in inst_types:
        for n in n_inst:
            inst_file = 'input/' + t + '_' + n + '.csv'
            inst = Instance(inst_file)
            threshold = calculate_threshold(inst.dist_matrix, percentile)

            f_greedy, x_greedy = solve_instance_greedy(inst)
            f_lp, x_lp = solve_instance_lp(inst)
            f_lp_threshold, x_lp_threshold = solve_instance_lp(inst, threshold)

            results.append({
                'Instancia': inst_file,
                'Distancia Greedy': f_greedy,
                'Distancia LP': f_lp,
                'Distancia LP con Threshold': f_lp_threshold,
                'Mejor Método': 'Greedy' if f_greedy < f_lp else 'LP',
                'Mejor Distancia': min(f_greedy, f_lp),
                'Mejora Relativa (%)': (f_greedy - f_lp) / f_lp * 100 if f_lp != 0 else None
            })

            print(f'Instancia: {inst_file} | Greedy: {f_greedy} | LP: {f_lp} | LP con Threshold: {f_lp_threshold}')

    df = pd.DataFrame(results)
    print(df)

    best_solution = df.loc[df['Mejor Distancia'].idxmin()]
    print('Mejor solución:')
    print('Instancia:', best_solution['Instancia'])
    print('Método:', best_solution['Mejor Método'])
    print('Distancia total:', best_solution['Mejor Distancia'])


if __name__ == '__main__':
    main()


