import numpy as np
import matplotlib.pyplot as plt
#y = ax+c
rng = (np.random)
x = rng.rand(50)*10
c = rng.randn(50)
y = 2*x+c
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
