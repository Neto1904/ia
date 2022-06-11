from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_performance_indicator, get_problem, get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
import sys

def execute(problem_name):
    problem = get_problem(problem_name)
    pareto_front = problem.pareto_front()

    pop_size = 100
    crossover = get_crossover('real_sbx', prob=0.9)
    mutation = get_mutation('real_pm', prob=0.2)
    n_gen = 250

    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=crossover,
        mutation=mutation
    )

    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen)
    )

    plot = Scatter()
    plot.add(result.F, facecolor="none", edgecolor="red")
    plot.add(pareto_front, plot_type="line", color="black", alpha=0.7)
    plot.save(f'plots/{problem_name}')

    igd = get_performance_indicator("igd", pareto_front)
    hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))

    print_results(problem_name, igd, hv, result.F)

    print(f'Inverted Generational Distance - {problem_name}', igd.do(result.F))
    print(f'Hypervolume - {problem_name}', hv.do(result.F))
    print('-------------------------------------------------------------------------------')

def print_results(problem_name, igd, hv, result_f):
    source_file =  open('results/quality_metrics.txt', 'a')
    print(f'Inverted Generational Distance | {problem_name} | {igd.do(result_f)}', file=source_file)
    print('-------------------------------------------------------------------------------', file=source_file)
    print(f'Hypervolume                    | {problem_name} | {hv.do(result_f)}', file=source_file)
    print('-------------------------------------------------------------------------------', file=source_file)
    source_file.close()
    
def clear_file():
    open('results/quality_metrics.txt', 'w').close()

def create_header():
    source_file =  open('results/quality_metrics.txt', 'a')
    print(f'Metric                         | Problem        | Value', file=source_file)
    print('-------------------------------------------------------------------------------', file=source_file)
    source_file.close()

def main():
    clear_file()
    create_header()
    execute('zdt1')
    execute('zdt2')
    execute('zdt3')

main()
