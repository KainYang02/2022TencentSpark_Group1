### 优化效果改善

#### 逐层优化

每轮逐层顺序做一次梯度下降，但优化效果不显著。例如，在 $11$ 个点随机稠密图上，优化前后都需要 $\sim70$ 次迭代才解保证解的错误率不超过 $5\%$；如果要求解的错误率不超过 $0.5\%$，那么优化前后都需要 $\sim120$ 次迭代。

```python
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
```

### 多参数扩展

将每层的参数限制放宽，使得每一层中各点与各边的参数值可以不同。这个改造是容易的：

```python
def u_beta(c, beta, n):
    for i in range(n):
        c.rx(i, theta = beta[i])

def u_gamma(c, gamma, edges):
    for i, e in enumerate(edges):
        x, y = e
        c.rzz(x, y, theta = gamma[i])
```

实验发现，在各参数相同时，在点数较少的图与稀疏图上，以上改造影响不大；而在点数较多的稠密图上，以上改造可以显著提升迭代效果。在本机生成的一个 $11$ 个点 $40$ 条边的随机图上，改造前 $1000$ 次迭代后得到的解的错误率为 $3.7\%$，改造后 $1000$ 次迭代后得到的解的错误率为 $0.09\%$。

### 有权重图

重新定义

$$
H_c=\sum_{(i,j,w)\in E}wZ_iZ_j.
$$

其余推导相同。

```python
def u_gamma(c, gamma, edges):
    for i, e in enumerate(edges):
        x, y, w = e
        c.rzz(x, y, theta = w * gamma[i])

def loss_val(betas, gammas, n, edges, n_layer):
    c = loss_cell(betas, gammas, n, edges, n_layer)
    loss = 0.;
    for e in edges:
        x, y, w = e
        loss += w * K.real(c.expectation_ps(z = [x, y]))
    return loss
```