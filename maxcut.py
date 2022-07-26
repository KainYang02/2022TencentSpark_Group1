import tensorcircuit as tc

K = tc.set_backend('tensorflow')

edges = []

n, m = map(int, input().split())
for _ in range(m):
    edges.append([int(i) for i in input().split()])

p = int(input())

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

betas = [ 1. for _ in range(p) ]
gammas = [ 1. for _ in range(p) ]

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
    Loss = K.real(0.);
    for e in edges:
        Loss += K.real(c.expectation_ps(z = [e[0], e[1]]))
    return Loss

# speed = 0.01
speed = 0.001
gloss = K.grad(loss)
for iteration in range(1000):
    grad = gloss(K.convert_to_tensor(betas+gammas))
#    bef = loss(betas + gammas)
#    if iteration % 5 == 0:
#        print('On iteration', iteration, ': ', loss(betas + gammas))
    if grad == None or sum(np.abs(x) for x in grad) < 0.01:
        print("break on ",iteration)
        break
    for i in range(p):
        betas[i] -= grad[i] * speed
        gammas[i] -= grad[i + p] * speed
    speed *= 0.99
c=loss_cell(betas + gammas)

print(c.measure(*range(n),with_prob = True))

