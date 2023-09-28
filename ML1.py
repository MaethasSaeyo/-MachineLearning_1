import numpy as np
import matplotlib.pyplot as plt
a = 2
b = 1
x = np.linspace(-5,5,100)
# y = ax+b
y = a*x+b

plt.plot(x,y,'-r',label='y=2x+1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")
plt.title("Display y = 2x+1")
plt.grid()
plt.show()