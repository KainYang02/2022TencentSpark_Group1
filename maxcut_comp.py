import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend('tensorflow')

edges = []

n, m = map(int, input().split())
for _ in range(m):
    edges.append([int(i) for i in input().split()])

p = int(input().split()[0])

# n,m=4,3
# edges.append((0,1));
# edges.append((1,2));
# edges.append((2,3));
# edges.append((3,0));
# p=4;

def u_beta(c, beta):
    for i in range(n):
        c.rx(i, theta = beta)

def u_gamma(c, gamma):
    for e in edges:
        x, y = e
        c.rzz(x, y, theta = gamma)

def loss_cell(params):
    betas, gammas = params[:p],params[p:]
    c = tc.Circuit(n);
    for i in range(n):
        c.h(i)
    for i in range(p):
        u_gamma(c, gammas[i])
        u_beta(c, betas[i])
    return c

def loss(params):
    c = loss_cell(params)
    loss = K.real(0.);
    for e in edges:
        loss += K.real(c.expectation_ps(z = [e[0], e[1]]))
    return loss

QAOA_vvag = K.jit(tc.backend.vvag(loss))

params = K.implicit_randn(
    shape=[n, 2 * p], stddev=0.1
)
opt = K.optimizer(tf.keras.optimizers.Adagrad())

for i in range(10000):
    loss, grads = QAOA_vvag(params)
    if i % 100 == 0:
    	print(K.numpy(loss))
    params = opt.update(grads, params)

loss, grads = QAOA_vvag(params)
print(min(loss))
# print(params)
# c = loss_cell(params)
# print(c.measure(*range(n), with_prob = True))

