import numpy as np


buffer = [100, 10, 10, 10, 10, 10]
counts = []
reaction = 2

for n in range(1, len(buffer) + 1):
    # counts.append(1) # fixed
    # counts.append(round(self.reaction/((n+1)))) # parabola
    a = round((n + 1))
    counts.append( a / reaction)  # linier reverse
counts = counts[::-1]

ls = []
for n, i in enumerate(counts):
    # print(n,i)
    if i == 0: break
    while counts[n] != 0:
        ls.append(buffer[n])
        counts[n] -= 1


print(counts)
print(ls)
print(int(np.average(ls, axis=0)))
