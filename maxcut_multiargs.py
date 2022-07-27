import tensorcircuit as tc
from random import random

K = tc.set_backend('tensorflow')

def u_beta(c, beta, n):
    for i in range(n):
        c.rx(i, theta = beta[i])

def u_gamma(c, gamma, edges):
    for i, e in enumerate(edges):
        x, y = e
        c.rzz(x, y, theta = gamma[i])

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
        loss += K.real(c.expectation_ps(z = e))
    return loss

gloss = K.jit(tc.backend.vvag(loss_val, argnums = [0, 1], vectorized_argnums = [0, 1]))


n, m = map(int, input().split())
edges = []
for _ in range(m):
    edges.append([int(i) for i in input().split()])

n_layer, n_iteration, n_parallel, advance_count = map(int, input().split())
advance_rate, speed = map(float, input().split())
log_frequency = int(input())

betas = K.implicit_randn(shape = [n_parallel, n_layer, n], stddev = 0.1)
gammas = K.implicit_randn(shape = [n_parallel, n_layer, m], stddev = 0.1)
#params = K.convert_to_tensor([[random()*0.3+(1 if i<n_layer else 0.5) for i in range(n_layer*2)] for _ in range(n_parallel)])

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

