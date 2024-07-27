from Instance import Instance
import cplex

EPS = 1e-6

################## Solucion FCFS greedy ######################
def solve_instance_greedy(inst):
	''' Dada una instancia del problema, retorna la solucion FCFS con criterio greedy.
	La funcion idealmente devuelve una tupla de parametros: funcion objetivo y solucion.'''
	pass

###############################################################

################## Solucion LP ################################
def generate_variables(inst, myprob):
	''' Genera la matriz de restricciones sobre myprob. Reemplazar pass por el codigo correspondiente.'''
	pass
			

def generate_constraints(inst, myprob):
	''' Genera la matriz de restricciones sobre myprob. Reemplazar pass por el codigo correspondiente.'''
	pass

def populate_by_row(inst, myprob):
	''' Genera el modelo.'''
	generate_variables(inst, myprob)
	generate_constraints(inst, myprob)

def solve_instance_lp(inst):
	''' Dada una instancia del problema, retorna la solucion general resolviendo un LP.
	La funcion idealmente devuelve una tupla de parametros: funcion objetivo y solucion.'''
	pass
###############################################################

#### Implementar funciones auxiliares necesarias para analizar resultados y proponer mejoras.

def main():
	inst_types = ['small','medium','large','xl']
	n_inst = ['0','1','2','3','4','5','6','7','8','9']

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
			print(inst_file,f_greedy, f_lp)

if __name__ == '__main__':
	main()
