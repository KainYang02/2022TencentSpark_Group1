import random as rd

n, d = map(int, input().split())

if n * d % 2 == 1:
    print("n x d is odd")
    exit()

v = [d for _ in range(n)]

print(n, n * d // 2)
for _ in range(n * d // 2):
    x, y = 0, 0
    while x == y or v[x] == 0 or v[y] == 0:
        x = rd.randint(0, n - 1)
        y = rd.randint(0, n - 1)
    print(x, y)
