import tensorcircuit as tc
from random import random

K = tc.set_backend('tensorflow')

def u_beta(c, beta, n):
    for i in range(n):
        c.rx(i, theta = beta)

def u_gamma(c, gamma, edges):
    for e in edges:
        x, y = e
        c.rzz(x, y, theta = gamma)

def loss_cell(params, n, edges, n_layer):
    c = tc.Circuit(n);
    for i in range(n):
        c.h(i)
    for i in range(n_layer):
        u_gamma(c, params[i + n_layer], edges)
        u_beta(c, params[i], n)
    return c

def loss_val(params, n, edges, n_layer):
    c = loss_cell(params, n, edges, n_layer)
    loss = 0.;
    for e in edges:
        loss += K.real(c.expectation_ps(z = e))
    return loss

gloss = K.jit(tc.backend.vvag(loss_val))


n, m = map(int, input().split())
edges = []
for _ in range(m):
    edges.append([int(i) for i in input().split()])

n_layer, n_iteration, n_parallel, advance_count = map(int, input().split())
advance_rate, speed = map(float, input().split())
log_frequency = int(input())

params = K.implicit_randn(shape = [n_parallel, n_layer * 2], stddev = 0.1)
#params = K.convert_to_tensor([[random()*0.3+(1 if i<n_layer else 0.5) for i in range(n_layer*2)] for _ in range(n_parallel)])

loss, grads = gloss(params, n, edges, n_layer)
for iteration in range(n_iteration):
    if iteration % log_frequency == 0:
        print('On iteration', iteration, ': ', min(loss))
    params -= grads * speed
    curloss, grads = gloss(params, n, edges, n_layer)
    if sum(int(a > b) for a, b in zip(curloss, loss)) >= advance_count:
        speed *= advance_rate
    loss = curloss

print(params)
c = loss_cell(params[K.argmin(loss)], n, edges, n_layer)
print(c.measure(*range(n), with_prob = True))

