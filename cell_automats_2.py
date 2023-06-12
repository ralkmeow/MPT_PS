import math
import numpy as np
import numexpr as ne

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation

def func_val(T, expr):
    return ne.evaluate(expr)

def printMat(matrix):
    for i in range(0, len(matrix)):
        str1 = ""
        for j in range(0, len(matrix[i])):
            str1 += str(matrix[i][j]) + "\t"
        print(str1)
    print()

N = 21
M = 21

Temp = [[0.0 for i in range(N)] for j in range(M)]
lam = [[1 for i in range(N)] for j in range(M)]
C = [[1000 for i in range(N)] for j in range(M)]
ro = [[1500 for i in range(N)] for j in range(M)]

h = 0.001
delt = 0.005

Nt = int(20/delt)

alf = [[delt / (C[i][j] * ro[i][j] * h * h) for i in range(N)] for j in range(M)]
C.clear()
ro.clear()

# gamma = "0.9 * T"
# gamma = "0.1 * (T ** 1.2)"
gamma = "0.15 * T - 0.01 * T * T * T"
Temp[10][10] = 5

anim = []
anim.append(Temp)

for t in range(Nt):
    TempNew = [[0 for i in range(N)] for j in range(M)]

    TempNew[0][0] = Temp[0][0] + alf[0][0] * \
                    (lam[1][0] * (Temp[1][0] - Temp[0][0])
                     + lam[0][1] * (Temp[0][1] - Temp[0][0]) + func_val(Temp[0][0], gamma))

    for y in range(1, N - 1):
        TempNew[0][y] = Temp[0][y] + alf[0][y] * \
                    (lam[0][y - 1] * (Temp[0][y - 1] - Temp[0][y])
                     + lam[1][y] * (Temp[1][y] - Temp[0][y])
                     + lam[0][y + 1] * (Temp[0][y + 1] - Temp[0][y]) + func_val(Temp[0][y], gamma))

    TempNew[0][N - 1] = Temp[0][N - 1] + alf[0][N - 1] * \
                    (lam[1][N - 1] * (Temp[1][N - 1] - Temp[0][N - 1])
                     + lam[0][N - 2] * (Temp[0][N - 2] - Temp[0][N - 1]) + func_val(Temp[0][N - 1], gamma))

    for x in range(1, M - 1):

        TempNew[x][0] = Temp[x][0] + alf[x][0] * \
                        (lam[x - 1][0] * (Temp[x - 1][0] - Temp[x][0])
                         + lam[x][1] * (Temp[x][1] - Temp[x][0])
                         + lam[x + 1][0] * (Temp[x + 1][0] - Temp[x][0]) + func_val(Temp[x][0], gamma))

        for y in range(1, N - 1):
            TempNew[x][y] = Temp[x][y] + alf[x][y] * \
                            (lam[x - 1][y] * (Temp[x - 1][y] - Temp[x][y])
                             + lam[x][y - 1] * (Temp[x][y - 1] - Temp[x][y])
                             + lam[x + 1][y] * (Temp[x + 1][y] - Temp[x][y])
                             + lam[x][y + 1] * (Temp[x][y + 1] - Temp[x][y]) + func_val(Temp[x][y], gamma))

        TempNew[x][N - 1] = Temp[x][N - 1] + alf[x][N - 1] * \
                        (lam[x - 1][N - 1] * (Temp[x - 1][N - 1] - Temp[x][N - 1])
                         + lam[x][N - 2] * (Temp[x][N - 2] - Temp[x][N - 1])
                         + lam[x + 1][N - 1] * (Temp[x + 1][N - 1] - Temp[x][N - 1]) + func_val(Temp[x][N - 1], gamma))

    TempNew[M - 1][0] = Temp[M - 1][0] + alf[M - 1][0] * \
                    (lam[M - 2][0] * (Temp[M - 2][0] - Temp[M - 1][0])
                     + lam[M - 1][1] * (Temp[M - 1][1] - Temp[M - 1][0]) + func_val(Temp[M - 1][0], gamma))

    for y in range(1, N - 1):
        TempNew[M - 1][y] = Temp[M - 1][y] + alf[M - 1][y] * \
                        (lam[M - 1][y - 1] * (Temp[M - 1][y - 1] - Temp[M - 1][y])
                         + lam[M - 2][y] * (Temp[M - 2][y] - Temp[M - 1][y])
                         + lam[M - 1][y + 1] * (Temp[M - 1][y + 1] - Temp[M - 1][y]) + func_val(Temp[M - 1][y], gamma))

    TempNew[M - 1][N - 1] = Temp[M - 1][N - 1] + alf[M - 1][N - 1] * \
                        (lam[M - 2][N - 1] * (Temp[M - 2][N - 1] - Temp[M - 1][N - 1])
                         + lam[M - 1][N - 2] * (Temp[M - 1][N - 2] - Temp[M - 1][N - 1]) + func_val(Temp[M - 1][N - 1], gamma))


    Temp = TempNew

    for i in range(M):
        for j in range(N):
            lam[i][j] = Temp[i][j]

    anim.append(Temp)

fig = plt.figure()
ax = plt.axes(projection = "3d")
plt.xlabel('ось y')
plt.ylabel('ось x')
plt.title('Прогрев пластины')
x_arr = np.array([i for i in range(N)])
y_arr = np.array([j for j in range(M)])
X, Y = np.meshgrid(x_arr, y_arr)

# frames = []
#
# for f in range(Nt):
#     Z = np.array(anim[f])
#     line = ax.plot_surface(X, Y, Z, cmap = 'plasma')
#     frames.append([line])
#
# ani = animation.ArtistAnimation(fig, frames, interval= 20, blit= False, repeat= True)

# ani.save('kursovaya_1.png')

Z = np.array(Temp)
ax.plot_surface(X, Y, Z, cmap='plasma')

ax.view_init(15, -135)
plt.show()