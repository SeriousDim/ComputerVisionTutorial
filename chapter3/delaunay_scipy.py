import matplotlib.pyplot as plt
import scipy.spatial as s
import numpy as np

# Триангуляция по Делоне. Scipy

p = []
"""
for i in range(50):
    p.append(np.random.randint(0, 300, 2))
"""

for i in range(10):
    for j in range(20):
        p.append([i*20, j*20])

p = np.array(p)

tri = s.Delaunay(p)
print(tri.simplices) # [[ind1, ind2, ind3], [...], [...]]

plt.triplot(p[:,0], p[:,1], tri.simplices)
plt.plot(p[:,0], p[:,1], 'o')
plt.show()
