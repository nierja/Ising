import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
import math


class mrizka:

    def __init__(self):
        """Vytvoří čtvercovou matici náhodných spinů."""

        self.matice = np.ones((delkahrany, delkahrany))
        for a in range(delkahrany):
            for b in range(delkahrany):
                if random.random() < 0.5:
                    self.matice[a, b] *= -1


    def sousede(self, x, y):
        """Vrátí matici obsahující hodnoty sousedních spinů"""

        horni = ((x - 1) % delkahrany, y)
        dolni = ((x + 1) % delkahrany, y)
        pravy = (x, (y + 1) % delkahrany)
        levy  = (x, (y - 1) % delkahrany)

        return [self.matice[horni[0], horni[1]],
                self.matice[dolni[0], dolni[1]],
                self.matice[pravy[0], pravy[1]],
                self.matice[levy[0], levy[1]]]


    def energie(self, x, y):
        """Vrátí energii konkrétního spinu na souřadnicích [x, y]"""

        return -J * self.matice[x, y] * sum(self.sousede(x, y))


    def MC(self, beta):
        """Provede MC simulaci"""

        E = np.zeros(pocet_kroku + ralaxacni_kroky)             #pole nul
        pocet_prijatych = 0                                     #counter pro spočtení zlomku přijatých MC kroků
        E_celkova = 0

        for a in range(delkahrany):                             #Spočte počáteční energii mřížky
            for b in range(delkahrany):
                E_celkova += self.energie(a, b)

        for krok in range(pocet_kroku + ralaxacni_kroky):
            a = np.random.randint(0, delkahrany)
            b = np.random.randint(0, delkahrany)
            e0 = self.energie(a, b)

            if np.exp((2*e0)*beta) > random.random():
                self.matice[a, b] *= -1
                pocet_prijatych += 1
                E_celkova += -4 * (e0)                          #update celkové energie mřížky
            E[krok] = E_celkova                                 #uložení energií jednolivých kroků, následně jde do fce. statistika()

        prumerna_E, yerr, variance = self.statistika(E[ralaxacni_kroky:], pocet_kroku)
        zlomek_prijatych = pocet_prijatych / (pocet_kroku + ralaxacni_kroky)

        return zlomek_prijatych, prumerna_E, yerr, variance


    def mereni_magnetizace(self, beta):
        """Provede MC simulaci, při které měří celkovou magnetizaci"""

        M = np.zeros(kroky_magnetizace)
        M_celkova = 0

        for a in range(delkahrany):                             #Spočte počáteční energii mřížky
            for b in range(delkahrany):
                M_celkova += self.matice[a, b]

        for krok in range(kroky_magnetizace):
            a = np.random.randint(0, delkahrany)
            b = np.random.randint(0, delkahrany)
            e0 = self.energie(a, b)

            if np.exp((2*e0)*beta) > random.random():
                if self.matice[a, b] == -1:
                    M_celkova += 2
                else:
                    M_celkova -= 2

                self.matice[a, b] *= -1

            M[krok] = M_celkova

        kroky = [i+1 for i in range(kroky_magnetizace)]
        plt.figure()
        plt.plot(kroky, M)
        plt.xlabel("krok")
        plt.ylabel("M")
        plt.title("M=M(krok)")
        plt.savefig("graf_M.pdf")
        plt.show()
        plt.close()
        return


    def statistika(self, E, pocetkroku):
        """Statistické zpracování"""

        yerr = math.sqrt(statistics.variance(E) / pocetkroku)           #spočte počáteční chybu
        prumer = sum(E) / pocetkroku
        variance = 0
        try:
            variance = statistics.variance(E)
        except statistics.StatisticsError:
            pass

        while len(E) > 1:
            novadelka = len(E) // 2
            noveE = []
            for i in range(novadelka):                                  #provede zblokování po sobě jdoucích dvojic energií
                noveE.append((E[2*i] + E[2*i+1])/2)

            try:                                                        #spočte novou chybu zblokovaných hodnot energií
                new_yerr = math.sqrt(statistics.variance(noveE) / novadelka)
            except statistics.StatisticsError:
                pass

            if new_yerr <= yerr or len(E) <= 3:                         #provádí se bloková metoda, dokud chyba roste
                #print("Std. error výsl:", yerr)
                break
            else:
                E, yerr = noveE, new_yerr

        return prumer, yerr, variance


    def graf_E_beta(self, beta, prumerneE, yerr):
        """Vynese s chybami E=E(beta)"""

        plt.figure()
        plt.errorbar(beta,prumerneE, yerr=yerr)
        plt.xlabel("\u03B2")
        plt.ylabel("E(\u03B2)")
        plt.title("E=E(\u03B2)")
        plt.savefig("graf_E_beta.pdf")
        plt.show()
        plt.close()
        return


    def graf_dE_dbeta(self, dE, beta, yerr):
        """Vynese s chybami C=dE/dBeta=Var(E)*beta^2"""

        plt.figure()
        plt.errorbar(beta,dE, yerr=yerr)
        plt.xlabel("\u03B2")
        plt.ylabel("dE/d\u03B2")
        plt.title("C=(dE/d\u03B2)")
        plt.savefig("graf_dE_dbeta.pdf")
        plt.show()
        plt.close()
        return


    def graf_E_kroky(self, kroky,E):
        """Vynese celkovou energii po sobě jdoucích MC kroků; E=E(krok)"""

        plt.plot(kroky,E, "ro", color='black',
             markersize=1)
        plt.xlabel("\u03B2")
        plt.ylabel("E(\u03B2)")
        plt.title("E=E(\u03B2)")
        plt.show()
        plt.savefig("ising.pdf")
        plt.close()
        return


    def graf_M_kroky(self, kroky, M):
        """Vynese celkovou energii po sobě jdoucích MC kroků; E=E(krok)"""

        plt.plot(kroky,M, "ro", color='black',
             markersize=1)
        plt.xlabel("\u03B2")
        plt.ylabel("E(\u03B2)")
        plt.title("E=E(\u03B2)")
        plt.show()
        plt.savefig("ising.pdf")
        plt.close()
        return


    def simulace(self):
        """Provede simulace pro různé teploty"""

        # inicializuje seznamy hodnot pro jednotlivé osy
        beta = []
        prumerneE = []
        yerror = []
        C = []
        krok = 1

        print("Spouštím simulaci na mřížce o rozměrech {}*{}, J={}, počet kroků/počet relax. kroků = {}/{}\n"
              "----------------------------------------------------------------------------------------------\n"
              "".format(delkahrany, delkahrany, J, pocet_kroku, ralaxacni_kroky))

        for i in [x/100 for x in range(1, 101)]:
            prijate, prumer, yerr, variance = self.MC(i)
            print("Krok č. {}/100: beta={}, \tE: {:.3f}, \tpřijatých: {:.3f} % , \tstd. error: {:.3f}"
                  "".format(krok, i,prumer, prijate * 100, yerr))

            beta.append(i)
            yerror.append(yerr)
            prumerneE.append(prumer)
            C.append(variance*(i**2))                   #počítá C = dE/dBeta = Var(E)*beta^2
            krok += 1

        #print(beta)
        #print(prumerneE)
        #print(yerror)
        self.graf_E_beta(beta, prumerneE, yerror)            #vynese požadované grafy
        self.graf_dE_dbeta(C, beta, yerror)


#---Konsanty-a-parametry-simulace---------------------------------------------------------------------------------------
delkahrany = 8
J = 1
pocet_kroku = 10000
ralaxacni_kroky = 10000
kroky_magnetizace = 300000


#---Spuštění-simulace---------------------------------------------------------------------------------------------------
mrizka = mrizka()
print("Počáteční konfigurace mříky:")
print(mrizka.matice)
mrizka.mereni_magnetizace(0.45)
mrizka.simulace()
