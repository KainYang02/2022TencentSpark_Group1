import tensorcircuit as tc
from random import random

K = tc.set_backend('tensorflow')

def u_beta(c, beta, n):
    for i in range(n):
        c.rx(i, theta = beta[i])

def u_gamma(c, gamma, edges):
    for i, e in enumerate(edges):
        x, y, w = e
        c.rzz(x, y, theta = w * gamma[i])

def loss_cell(betas, gammas, n, edges, n_layer):
    c = tc.Circuit(n);
    for i in range(n):
        c.h(i)
    for i in range(n_layer):
        u_gamma(c, gammas[i], edges)
        u_beta(c, betas[i], n)
    return c

def loss_val(betas, gammas, n, edges, n_layer):
    c = loss_cell(betas, gammas, n, edges, n_layer)
    loss = 0.;
    for e in edges:
        x, y, w = e
        loss += w * K.real(c.expectation_ps(z = [x, y]))
    return loss

gloss = K.jit(tc.backend.vvag(loss_val, argnums = [0, 1], vectorized_argnums = [0, 1]))


n, m = map(int, input().split())
edges = []
for _ in range(m):
    x, y, w = input().split()
    edges.append([int(x), int(y), float(w)])

option_file = open('options.txt')
n_layer, n_iteration, n_parallel, advance_count = map(int, option_file.readline().split())
advance_rate, speed = map(float, option_file.readline().split())
log_frequency = int(option_file.readline())
option_file.close()

betas = K.implicit_randn(shape = [n_parallel, n_layer, n], stddev = 1.)
gammas = K.implicit_randn(shape = [n_parallel, n_layer, m], stddev = 1.)

loss, [beta_grads, gamma_grads] = gloss(betas, gammas, n, edges, n_layer)
for iteration in range(n_iteration):
    if iteration % log_frequency == 0:
        print('On iteration', iteration, ': ', min(loss))
    betas -= beta_grads * speed
    gammas -= gamma_grads * speed
    curloss, [beta_grads, gamma_grads] = gloss(betas, gammas, n, edges, n_layer)
    if sum(int(a > b) for a, b in zip(curloss, loss)) >= advance_count:
        speed *= advance_rate
    loss = curloss

print(betas, gammas)

fin_id = K.argmin(loss)
c = loss_cell(betas[fin_id], gammas[fin_id], n, edges, n_layer)
print(c.measure(*range(n), with_prob = True))

