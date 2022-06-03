from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT2, ZDT1, ZDT3
from jmetal.util.comparator import GDominanceComparator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

def execute(problem_name):
    problem = get_problem(problem_name)
    reference_front = read_solutions(filename=f'fronts/{problem_name}.pf')

    problem.reference_front = reference_front
    reference_point = [0.2, 0.5]
    max_evaluations = 25000

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables, 
            distribution_index=20
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        dominance_comparator=GDominanceComparator(reference_point),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(
            reference_front=problem.reference_front, 
            reference_point=reference_point
        )
    )

    algorithm.run()
    result = algorithm.get_result()

    plot_front = InteractivePlot(
        title="Pareto front approximation. Problem: " + problem.get_name(),
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(result, label=algorithm.label, filename=f'plots/{algorithm.get_name()}-{problem_name}')



def get_problem(problem_name):
    problems = {
        'zdt1':  ZDT1(),
        'zdt2': ZDT2(),
        'zdt3': ZDT3()
    }
    return problems[problem_name]


def main():
    execute('zdt1')
    execute('zdt2')
    execute('zdt3')

main()
