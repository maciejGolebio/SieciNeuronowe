from OneLayerNet import OneLayerNet
import numpy as np
import random as rand


class TwoLayerNet:

    @staticmethod
    def aktywacja(x, b):
        return 1 / (1 + np.exp(-x * b))

    @staticmethod
    def pochodna_aktywacji(y, b):
        return b * (1 - y) * y

    @staticmethod
    def init2(S, K1, K2):
        W1 = OneLayerNet.init1(S, K1)
        W2 = OneLayerNet.init1(K1, K2)
        bias1 = np.ones((1, K1)) * -1
        bias2 = np.ones((1, K2)) * -1
        W1 = np.vstack([bias1, W1])
        W2 = np.vstack([bias2, W2])
        return W1, W2

    @staticmethod
    def dzialaj2(W1, W2, X):
        x = np.vstack([[-1], X])
        # print(x)
        Y1 = OneLayerNet.dzialaj1(W1, x)
        y1 = np.vstack([[-1], Y1])
        # print(Y1)
        Y2 = OneLayerNet.dzialaj1(W2, y1)
        # print(Y2)
        return Y1, Y2

    @staticmethod
    def ucz2(W1p, W2p, Przyklady, Test, m):
        wiersze, kolumny = Przyklady.shape
        B = 5
        W1 = W1p
        W2 = W2p
        Wsp_uczenia = 0.01
        for i in range(m):
            x = rand.randint(0, kolumny - 1)
            X = Przyklady[:, x]
            Y1, Y2 = TwoLayerNet.dzialaj2(W1, W2, X)

            Y2_blad = Y2 - Test
            Y2_delta = Y2_blad * TwoLayerNet.pochodna_aktywacji(Y2, B)

            Y1_blad = Y2_delta.dot(W2.transpose())
            Y1_delta = Y1_blad * TwoLayerNet.pochodna_aktywacji(Y1, B)


W1, W2 = TwoLayerNet.init2(4, 4, 2)
X = np.ones((4, 1))
Y1, Y2 = TwoLayerNet.dzialaj2(W1, W2, X)

print(Y1)
print()
print(Y2)
