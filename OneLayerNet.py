import random as rand

import numpy as np
import matplotlib.pyplot as plt


class OneLayerNet:

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
        B = 1
        WT = W.transpose()
        U = WT.dot(X)
        return 1 / (1 + np.exp(-U * B))

    @staticmethod
    def ucz1(W, Pyrzklady, Testy, m):
        _, kolumny = Pyrzklady.shape
        wsp_uczenia = 0.01
        Wpo = W
        Err = []
        for _ in range(m):
            x = rand.randint(0, kolumny - 1)
            X = Pyrzklady[:, x]
            Y = OneLayerNet.dzialaj1(Wpo, X)
            D = (Testy[:, x] - Y.transpose()).transpose()
            Err.append(D.transpose())
            dW = wsp_uczenia * X * D.transpose()
            Wpo = Wpo + dW
        return Wpo, Err

    @staticmethod
    def mse(Err):
        err_value = []
        for i in Err:
            tmp = 0
            for j in range(i.size):
                tmp += i.item(j) ** 2
            err_value.append(tmp)

        x = []
        for i in range(len(err_value)):
            x.append(i)

        plt.plot(x, err_value)
        plt.show()


def main():
    W = OneLayerNet.init1(5, 3)
    P = np.matrix('4.0 0.01 0.01 0.02 -0.1; 2.0 1.0 2.0 2.5 2.0; 1 3.5 0.1 2.0 1.5').transpose()
    T = np.identity(3)
    print('zbior uczacy')
    print(P)
    epoki = 1000
    Wpo, Err = OneLayerNet.ucz1(W, P, T, epoki)
    OneLayerNet.mse(Err)
    print('\n\nPO UCZENIU\n')
    odp = OneLayerNet.dzialaj1(Wpo, P[:, 1])
    print(odp)
    czlowieki = np. \
        matrix('2.0 0.0 0.0 0.0 0.0; 2.0 0.2 0.01 0.5 0.0; 2 0.5 0.1 0.0 0.5'). \
        transpose()

    czlowiek = OneLayerNet.dzialaj1(Wpo, czlowieki[:, 0])
    czlowiek1 = OneLayerNet.dzialaj1(Wpo, czlowieki[:, 1])
    czlowiek2 = OneLayerNet.dzialaj1(Wpo, czlowieki[:, 2])
    print('\n\nCZLOWIEK\n')
    print(czlowieki)
    print('\n')
    print(czlowiek)
    print(czlowiek1)
    print(czlowiek2)


if __name__ == '__main__':
    main()
