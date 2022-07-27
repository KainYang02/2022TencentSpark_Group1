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

layer_beta_mask = []
layer_gamma_mask = []
for layer in range(n_layer):
    mask = K.tensordot(K.ones(n_parallel, dtype = 'float32'), [ float(i == layer) for i in range(n_layer) ], 0)
    layer_beta_mask.append(K.tensordot(mask, K.ones(n, dtype = 'float32'), 0))
    layer_gamma_mask.append(K.tensordot(mask, K.ones(m, dtype = 'float32'), 0))

loss, [beta_grads, gamma_grads] = gloss(betas, gammas, n, edges, n_layer)
speed = [speed for _ in range(n_layer)]
for iteration in range(n_iteration):
    if iteration % log_frequency == 0:
        print('On iteration', iteration, ': ', min(loss))
    for layer in range(n_layer):
        betas -= beta_grads * layer_beta_mask[layer] * speed[layer]
        gammas -= gamma_grads * layer_gamma_mask[layer] * speed[layer]
        pre_loss = loss
        loss, [beta_grads, gamma_grads] = gloss(betas, gammas, n, edges, n_layer)
        if sum(int(a > b) for a, b in zip(loss, pre_loss)) >= advance_count:
            speed[layer] *= advance_rate

print(betas, gammas)

fin_id = K.argmin(loss)
c = loss_cell(betas[fin_id], gammas[fin_id], n, edges, n_layer)
print(c.measure(*range(n), with_prob = True))

