import matplotlib.pyplot as plt

import numpy as np

cpu = np.transpose(np.loadtxt("../Lab1/cmake-build-debug/write.txt"))
gpu = np.transpose(np.loadtxt("cmake-build-debug-visual-studio/Debug/write.txt"))

x = np.arange(0, 500)
plt.xlabel("N")
plt.ylabel("T, s")
plt.plot(x, cpu, label='CPU')
plt.plot(x, gpu, label='GPU')
plt.legend()
plt.grid()
plt.show()
