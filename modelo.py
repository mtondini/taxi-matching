import cplex

from Instance import Instance
import cplex

EPS = 1e-6


################## Solucion FCFS greedy ######################
def solve_instance_greedy(inst):
    n = inst.n
    taxis_longlat = inst.taxis_longlat
    paxs_longlat = inst.paxs_longlat
    dist_matrix = inst.dist_matrix

    # Inicialmente, T = {1, . . . , n} es el conjunto de vehículos disponibles y T̄ = ∅ los vehículos ya utilizados.
    T = set(range(n))
    solution = []
    total_distance = 0

    # Recorrer los pasajeros de a uno en el orden en el que están definidos.
    for j in range(n):
        # Sea la j-ésima iteración, que considera el pasajero j. Buscamos el vehículo i* ∈ T más cercano al pasajero j.
        min_distance = float('inf')
        best_taxi = -1
        for i in T:
            if dist_matrix[i][j] < min_distance:
                min_distance = dist_matrix[i][j]
                best_taxi = i

        # Asignar el vehículo i* al pasajero j.
        solution.append((best_taxi, j))
        total_distance += min_distance

        # Actualizar T = T\{i*} y T̄ = T̄ ∪ {i*}.
        T.remove(best_taxi)

    return total_distance, solution


###############################################################

################## Solucion LP ################################
def generate_variables(inst, myprob):
    n = inst.n
    dist_matrix = inst.dist_matrix

    # Definir la función objetivo
    obj = []
    for i in range(n):
        for j in range(n):
            obj.append(dist_matrix[i][j])

    # Definir las variables de decisión
    var_names = ["x_{}_{}".format(i, j) for i in range(n) for j in range(n)]
    var_types = [myprob.variables.type.binary] * (n * n)
    myprob.variables.add(obj=obj, types=var_types, names=var_names)


def generate_constraints(inst, myprob):
    n = inst.n

    # Restricciones de asignación de vehículos a pasajeros
    for j in range(n):
        indices = ["x_{}_{}".format(i, j) for i in range(n)]
        values = [1.0] * n
        myprob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],
            rhs=[1.0]
        )

    # Restricciones de asignación de pasajeros a vehículos
    for i in range(n):
        indices = ["x_{}_{}".format(i, j) for j in range(n)]
        values = [1.0] * n
        myprob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
            senses=["E"],
            rhs=[1.0]
        )


def populate_by_row(inst, myprob):
    ''' Genera el modelo.'''
    generate_variables(inst, myprob)
    generate_constraints(inst, myprob)


def solve_instance_lp(inst):
    ''' Dada una instancia del problema, retorna la solucion general resolviendo un LP.
    La funcion idealmente devuelve una tupla de parametros: funcion objetivo y solucion.'''
    try:
        # Crear un problema CPLEX
        myprob = cplex.Cplex()
        myprob.set_problem_type(cplex.Cplex.problem_type.LP)

        # Generar el modelo
        populate_by_row(inst, myprob)

        # Resolver el problema
        myprob.solve()

        # Obtener el valor de la función objetivo
        total_distance = myprob.solution.get_objective_value()

        # Obtener la solución
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


###############################################################

#### Implementar funciones auxiliares necesarias para analizar resultados y proponer mejoras.

def main():
    inst_types = ['small', 'medium', 'large', 'xl']
    n_inst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Esquema para ejecutar las soluciones directamente sobre las 40 instancias.
    for t in inst_types:
        for n in n_inst:
            inst_file = 'input/' + t + '_' + n + '.csv'
            inst = Instance(inst_file)

            # Solucion greedy.
            f_greedy, x_greedy = solve_instance_greedy(inst)

            # Solucion lp
            f_lp, x_lp = solve_instance_lp(inst)

            # Modificar para ajustar el formato segun la conveninencia del grupo, agregando
            # o quitando informacion.
            print(inst_file, f_greedy, f_lp)


if __name__ == '__main__':
    main()
