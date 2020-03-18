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
        """"
            W1 - Wagi warstwy wejsciowej
            W2 - wagi warstyw 2giej - ukrytej
            B1 - biasy warstyw wejsciowej
            B2 - biasy warswy ukrytej
        """
        W1 = np.random.uniform(size=(S, K1))
        W2 = np.random.uniform(size=(K1, K2))
        B1 = np.ones((1, K1))
        B2 = np.ones((1, K2))
        return W1, W2, B1, B2

    @staticmethod
    def dzialaj2(W1, B1, W2, B2, wej, Beta):
        Y1 = np.dot(wej, W1)
        Y1 = Y1 + B1
        Y1 = TwoLayerNet.aktywacja(Y1, Beta)
        Y2 = np.dot(Y1, W2)
        Y2 += B2
        Y2 = TwoLayerNet.aktywacja(Y2, Beta)
        return Y1, Y2

    @staticmethod
    def ucz2(W1p, W2p, b1p, b2p, P, T, n):
        """
        :param W1p: 1 warstwa wagi pocz
        :param W2p: 2 warstwa wagi pocz
        :param b1p: biasy 1 wars pocz
        :param b2p: biasy 2 wars pocz
        :param P: Przyklady
        :param T: test
        :param n: ilosc epok
        :return: zwraca nauczone wagi i biasy
        """
        B = 5
        b1 = b1p
        b2 = b2p
        W1 = W1p
        W2 = W2p
        # print(W2)
        # krok gradientu
        wsp_uczenia = 0.1
        # wiersze i kolumny potrzebne do losowania
        w, k = P.shape
        for i in range(n):
            # losowanie przykladu
            x = rand.randint(0, w - 1)
            X = P[x, :]
            Y1, Y2 = TwoLayerNet.dzialaj2(W1, b1, W2, b2, X, B)
            # propagacja wsteczna
            # delety - gradienty kierunkowe
            blad2 = T[x] - Y2
            delta2 = blad2 * TwoLayerNet.pochodna_aktywacji(Y2, B)
            blad1 = delta2.dot(W2.T)
            delta1 = blad1 * TwoLayerNet.pochodna_aktywacji(Y1, B)

            # aktualizacja wag i biasów
            W2 = W2 + Y1.T.dot(delta2) * wsp_uczenia
            b2 = np.sum(Y2, axis=0, keepdims=True) * wsp_uczenia
            W1 = W1 + X.T * delta1 * wsp_uczenia
            b1 = b1 + np.sum(delta1, axis=0, keepdims=True) * wsp_uczenia
        return W1, W2, b1, b2


w1, w2, b1, b2 = TwoLayerNet.init2(2, 2, 1)
# dane uczace
wejscia = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
wyjscia = np.array([[0], [1], [1], [0]])
w1, w2, b1, b2 = TwoLayerNet.ucz2(w1, w2, b1, b2, wejscia, wyjscia, 50000)
B = 5
for i in range(4):
    y1, y2 = TwoLayerNet.dzialaj2(w1, b1, w2, b2, wejscia[i, :], B)
    print('X')
    print(wejscia[i, :])
    print('Y')
    print(*y2)
