import tensorcircuit as tc
import tensorflow as tf
import math
from random import randint
import numpy as np
from matplotlib import pyplot as plt

K = tc.set_backend('tensorflow')

def optmize(n,m,edges,p):

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
        shape=[1, 2 * p], stddev=0.1
    )
    opt = K.optimizer(tf.keras.optimizers.SGD())

    ALL=[]
    for i in range(1000):
        loss, grads = QAOA_vvag(params)
        if i % 200 == 0 :
            tmp_params = params.numpy().tolist()
            for t in range(2 * p) :
                tmp_params[0][t] += math.pi * 3 / 2
    #             if tmp_params[0][t] > math.pi :
    #                tmp_params[0][t] -= math.pi
            ALL+=tmp_params[0]
        params = opt.update(grads, params)
    loss, grads = QAOA_vvag(params)
    return ALL

def gen(n,d):
    if n * d % 2 == 1:
        print("n x d is odd")
        return -1

    v = [d for _ in range(n)]

    edge=[]
    for _ in range(n * d // 2):
        x, y = 0, 0
        while x == y or v[x] == 0 or v[y] == 0:
            x = randint(0, n - 1)
            y = randint(0, n - 1)
        edge.append([x, y])
    return (n,len(edge),edge)

def draw(a,layer):
    print(a)
    for i in range(len(a)//(layer*2)):
        for j in range(i*(layer*2),i*(layer*2)+(layer)):
            if j%(layer*2)==0:
                a[j]=a[j]+math.pi
            if a[j]>=0.5*math.pi or a[j]<0:
                a[j]=a[j]-int(a[j]/(math.pi))*math.pi
        plt.plot([i for i in range(1,layer+1)],a[i*(layer*2):i*(layer*2)+(layer)],label=i);
    plt.legend()
    plt.show()

    plt.clf()
    for i in range(len(a)//(layer*2)):
        for j in range(i*(layer*2)+(layer),i*(layer*2)+(layer*2)):
            a[j]=a[j]-int(a[j]/(math.pi))*math.pi
        plt.plot([i for i in range(1,layer+1)],a[i*(layer*2)+(layer):i*(layer*2)+(layer*2)],label=i);
    plt.legend()
    plt.show()

n,d,p=6,3,7
draw(optmize(*gen(n,d),p),p)
