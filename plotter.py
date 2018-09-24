import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt

cod = input("What sample do u want? (0..63)")
filename ="sensore"+str(cod)+".csv"
A = genfromtxt(filename, delimiter=',')
plt.plot(A)
plt.show()
