import math
from pulp import *
from pandas import Series
import matplotlib.pyplot as plt

# productivity of each factory for each type of parts
A = [[300., 200., 50., 300., 600.],
     [60., 100., 35, 280., 400.],
     [100., 150., 200., 2.5, 350]]
# factories of each type
ent = [8., 4., 70., 10., 4.]
# parts of each type in the product
prod = [1., 4., 1.]

# types of factories
ent_dim = len(ent)
# types of parts
prod_dim = len(prod)

sum_dim = ent_dim + prod_dim
mul_dim = ent_dim * prod_dim + 1

alpha = 0.8


def alpha_supp(alpha_val):
    return math.sqrt((1 - alpha_val) / alpha_val)


def form_limits(new_alpha_val):

    a_new = [[0 for col in range(mul_dim)] for row in range(sum_dim)]

    for i in range(prod_dim):
        a_new[i][0] = prod[i]
    for i in range(prod_dim):
        for j in range(ent_dim):
            a_new[i][ent_dim * i + j + 1] = -(A[i][j] * new_alpha_val)
    for i in range(ent_dim):
        for j in range(prod_dim):
            a_new[i + prod_dim][i + j*ent_dim + 1] = 1.

    return a_new


b = []
for i in range(prod_dim):
    b.append(0.0)
for i in range(ent_dim):
    b.append(ent[i])


def solve_model(prob, A_matr):

    prob += x, "obj"

    for line in range(sum_dim):
        prob += A_matr[line][0] * x + \
                pulp.lpSum([A_matr[line][j + i*ent_dim + 1] * x_matr[i, j] for i in range(prod_dim) for j in range(ent_dim)]) \
                <= b[line], "c" + str(line + 1)

    prob.solve(PULP_CBC_CMD(msg=False))


def print_results(prob):
    print("Status:", LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    print("objective = %s " % value(prob.objective))


x = LpVariable("X", 0, None, 'Integer')
x_matr = pulp.LpVariable.dicts("x_matr",
                               ((i, j) for i in range(prod_dim) for j in range(ent_dim)),
                               lowBound=0,
                               cat='Integer')


print("General Problem:")
A_matr = form_limits(1)
problem = LpProblem("problem", LpMaximize)
solve_model(problem, A_matr)
print_results(problem)

print("Optimistic Solution(alpha = 0.8):")
A_matr = form_limits(1 + alpha_supp(alpha))
prob_optimist = LpProblem("prob_optimist", LpMaximize)
solve_model(prob_optimist, A_matr)
print_results(prob_optimist)

print("Pessimistic Solution(alpha = 0.8):")
A_matr = form_limits(1 - alpha_supp(alpha))
prob_pessimist = LpProblem("prob_pessimist", LpMaximize)
solve_model(prob_pessimist, A_matr)
print_results(prob_pessimist)


alpha_test_val = [x/10 for x in range(1, 11)]
alpha_test = [str(alph) for alph in alpha_test_val]

opt = []
pes = []

for alph in alpha_test_val:
    A_matr = form_limits(1 + alpha_supp(alph))
    problem = LpProblem("Problem", LpMaximize)
    solve_model(problem, A_matr)
    opt.append(value(problem.objective))
    del problem

    A_matr = form_limits(1 - alpha_supp(alph))
    problem = LpProblem("Problem", LpMaximize)
    solve_model(problem, A_matr)
    pes.append(value(problem.objective))
    del problem


ser1 = Series(pes, index=alpha_test, name="pessimist")
ser2 = Series(opt, index=alpha_test, name="optimist")
ser1.plot()
ser2.plot()
plt.legend()
plt.show()
