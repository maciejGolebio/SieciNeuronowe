import numpy as np
import random as rand
import math


class OneLayeNet:

    # S - wejscie
    # K - wyjscia
    # W - macierz wag sieci
    @staticmethod
    def init1(S, K):
        W = (np.random.random(size=(S, K)) / 5) - 0.1
        return W

    # W - macierz wag sieci
    # X wektor sygnałow wejsciowych
    # Y wyjscie sieci jednowastwowej
    # U - pobudzenie neuronu
    @staticmethod
    def dzialaj1(W, X):
        B = 2
        WT = W.transpose()
        U = WT.dot(X)
        return 1 / (1 + np.exp(-U * B))

    @staticmethod
    def ucz1(W, Pyrzklady, Testy, epoki):
        wiersze, kolumny = Pyrzklady.shape
        wsp_uczenia = 0.1
        Wpo = W
        for i in range(epoki):
            x = rand.randint(0, kolumny - 1)
            X = Pyrzklady[:, x]
            Y = OneLayeNet.dzialaj1(Wpo, X)
            D = (Testy[:, x] - Y.transpose()).transpose()
            dW = wsp_uczenia * X * D.transpose()
            Wpo = Wpo + dW
        return Wpo


W = OneLayeNet.init1(5, 3)
X = OneLayeNet.init1(5, 1)
P = np.matrix('4.0 0.01 0.01 1 1.5; 2.0 1.0 2.0 2.5 2.0; 1 3.5 0.1 2.0 1.5').transpose()
T = np.identity(3)
print('zbior uczacy')
print(P)

Wpo = OneLayeNet.ucz1(W, P, T, 100)


print('\n\nPO UCZENIU\n')
odp = OneLayeNet.dzialaj1(Wpo, P[:, 1])
print(odp)
czlowieki = np.\
    matrix('2.0 0.0 0.0 0.0 0.0; 2.0 0.2 0.01 0.5 0.0; 2 0.5 0.1 0.0 0.5')\
    .transpose()

czlowiek = OneLayeNet.dzialaj1(Wpo, czlowieki[:, 2])
print('\n\nCZLOWIEK\n')
print(czlowieki)
print('\n')
print(czlowiek)
