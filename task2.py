import math
from pulp import *

# import / export restrictions
a = [[5000., 9000., 3000., 2000.],
     [6000., 8000., 5000., 8000.],
     [9000., 6000., 3000., 4000.],
     [5000., 1000., 4000., 6000.]]

# prices
c = [[40., 36., 14., 15.],
     [50., 40., 30., 40.],
     [45., 35., 40., 40.],
     [50., 10., 14., 35.]]

# production restrictions
d = [3000, 20000, 10000]
# production costs
res = [1, 3, 2]

# countries
dim_i = len(a)
# products
dim_j = len(d) + 1


alpha_val = 0.8
alpha_plus = 1 + math.sqrt((1 - alpha_val) / (2*alpha_val))
alpha_minus = 1 - math.sqrt((1 - alpha_val) / (2*alpha_val))


def add_restrictions(prob, c_matr):

    for j in range(dim_j-1):
        prob += pulp.lpSum([x_matr[i, j] for i in range(dim_i)]) <= d[j], "d" + str(j)

    prob += pulp.lpSum([x_matr[i, j] * c_matr[i][j] for i in range(dim_i) for j in range(dim_j-1)]) - \
            pulp.lpSum([x_matr[i, dim_j-1] * c_matr[i][dim_j-1] for i in range(dim_i)]) == 0, "bal"

    for i in range(dim_i):
        for j in range(dim_j):
            prob += x_matr[i, j] - a[i][j] <= 0, "c" + str(i) + "_" + str(j)


def print_values(values1, prob2):

    for v1, v2 in zip(values1, prob2.variables()):
        print(v2.name, "=", v1, " \t=> ", v2.varValue)


def get_values(prob):
    values = []
    for v in prob.variables():
        values.append(v.varValue)

    return values


def get_costs(values):
    cost_sum = 0
    for j in range(dim_j-1):
        cost_sum += res[j]*sum([values[i*dim_j + j] for i in range(dim_i)])
    return cost_sum


def solve_model(c_matrix):

    problem_1 = LpProblem("LPproblem_1", LpMaximize)

    problem_1 += pulp.lpSum([x_matr[i, dim_j-1] for i in range(dim_i)]), "obj"

    add_restrictions(problem_1, c_matrix)
    problem_1.solve(PULP_CBC_CMD(msg=False))
    values1 = get_values(problem_1)

    problem_2 = LpProblem("LPproblem2", LpMinimize)

    problem_2 += sum([res[j]*pulp.lpSum([x_matr[i, j] for i in range(dim_i)]) for j in range(dim_j-1)]), "obj"

    add_restrictions(problem_2, c_matrix)
    problem_2 += pulp.lpSum([x_matr[i, dim_j-1] for i in range(dim_i)]) >= int(value(problem_1.objective)), "goal"
    problem_2.solve(PULP_CBC_CMD(msg=False))
    values2 = get_values(problem_2)

    print_values(values1, problem_2)
    print("Maximized T3 = ", sum([values1[i*dim_j+dim_j-1] for i in range(dim_i)]))
    print("Costs = ", get_costs(values1))
    print("Minimized costs = ", get_costs(values2))
    print("T3 = ", sum([values2[i*dim_j+dim_j-1] for i in range(dim_i)]))
    del problem_1
    del problem_2


x_matr = pulp.LpVariable.dicts("x_matr",
                               ((i, j) for i in range(dim_i) for j in range(dim_j)),
                               lowBound=0,
                               cat='Continuous')

print("\nGeneral problem:\n")
solve_model(c)

c_opt = [[0 for j in range(dim_j)] for i in range(dim_i)]
for i in range(dim_i):
    for j in range(dim_j-1):
        c_opt[i][j] = int(c[i][j]*alpha_plus)

for i in range(dim_i):
    c_opt[i][dim_j-1] = int(c[i][dim_j-1]*alpha_minus)

print("\nOptimist problem:\n")
solve_model(c_opt)

c_pes = [[0 for j in range(dim_j)] for i in range(dim_i)]
for i in range(dim_i):
    for j in range(dim_j-1):
        c_pes[i][j] = int(c[i][j]*alpha_minus)

for i in range(dim_i):
    c_pes[i][dim_j-1] = int(c[i][dim_j-1]*alpha_plus)

print("\nPessimist problem:\n")
solve_model(c_pes)
