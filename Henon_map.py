import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义Henon Map超混沌系统的参数
a = 1.4
b = 0.3

# 定义Henon Map超混沌系统的初始状态
x0 = 0.1
y0 = 0.1

# 定义Henon Map超混沌系统的迭代函数
def henon_map(state, a, b):
    x, y, z = state
    xn = 1 - a * x**2 + y
    yn = b * x
    zn = 0.1 * x + z
    return [xn, yn, zn]

# 计算Henon Map超混沌系统的状态随时间的变化
num_iter = 5000
states = np.zeros((num_iter, 3))
states[0, 0] = x0
states[0, 1] = y0
sequence=[]
for i in range(1, num_iter):
    states[i, :] = henon_map(states[i-1, :], a, b)
    sequence += states[i, :].tolist()



sequence = np.floor(np.mod((np.abs(sequence) - np.floor(np.abs(sequence))) * (10 ** 14), 256))
sequence=sequence[1000:]
keys = []
for i in range(1, 17):
    keys.append(sequence[512*256*i-512*256:256*512*i])

keys = np.concatenate(keys)

print(keys)



# 绘制Henon Map超混沌系统的3D相空间图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2], '.', markersize=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Phase Space')
plt.show()
