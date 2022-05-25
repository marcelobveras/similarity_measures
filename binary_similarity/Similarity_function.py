import numpy as np
import operator
import math

class similarity_score:

    def __init__(self, x, y, U=None, data_type='set', abcdn=None):
        """
        x and y is the two inputed sets that will
        be evaluated.

        U is the optional universe set, if its
        not inputed then it will be the same as
        x union y.

        'a' is the number of features where
        the values for both x and y are 1,

        'b' and 'c' are the number of features
        where the value for x is 0 and y is 1 and
        vice versa, respectively

        'd' is the number of features where the
        values for both x and y are 0 ."""
        if abcdn is not None:
            self.a, self.b, self.c, self.d, self.n = abcdn
        elif data_type == 'set':
            self.a = len(x&y)
            self.b = len(x-y)
            self.c = len(y-x)
            if not U:
                self.d = 0;
                self.n = self.a + self.b + self.c + self.d
            else:
                self.d = len(U.difference(x|y))
                self.n = len(U)
        elif data_type == 'array':
            self.a = sum(x & y)
            self.b = sum(x - (x & y))
            self.c = sum(y - (x & y))
            self.n = len(x)
            if not U:
                self.d = 0
            else:
                self.d = sum(U.difference(x | y))
        else:
            raise Exception('Type of data not supported')

    def evaluate(self,similarity_function=None):
        if (similarity_function == None):
            return getattr(self, 'jaccard')()
        else:
            return getattr(self, similarity_function)()

    def evaluateAll(self):
        method_list = [func for func in dir(self) if callable(getattr(self, func))]
        excludedMethods = ['evaluate','evaluateAll','setNewFunction','listSimFunctions','listFunctions','get_constants']
        method_list = [func for func in method_list if '__' not in func and func not in excludedMethods]
        def exec_or_die(method):
            try:
                result = float(method())
                if np.isfinite(result):
                    return result
                else:
                    return np.nan
            except:
                # print("Error in " + method.__name__ + "\na = "+str(self.a)+ "\nb = "+str(self.b)+ "\nc = "+str(self.c)+ "\nd = "+str(self.d)+ "\nn = "+str(self.n))
                return np.nan
        result = {method: exec_or_die(getattr(self,method)) for method in method_list}
        return result

    def get_constants(self):
        return [self.a, self.b, self.c, self.d, self.n]

    @classmethod
    def listFunctions(cls):
        method_list = [func for func in dir(cls) if callable(getattr(cls, func))]
        excludedMethods = ['evaluate', 'evaluateAll', 'setNewFunction',
                           'listSimFunctions', 'listFunctions', 'get_constants', 'setNewFunction']
        method_list = [func for func in method_list if '__' not in func and func not in excludedMethods]
        return method_list

    def listSimFunctions(self):
        method_list = [func for func in dir(self) if callable(getattr(self, func))]
        excludedMethods = ['evaluate', 'evaluateAll', 'setNewFunction',
                           'listSimFunctions', 'listFunctions', 'get_constants', 'setNewFunction']
        method_list = [func for func in method_list if '__' not in func and func not in excludedMethods]
        return method_list

    def Jaccard(self):
        # Eq. ID: 1
        return self.a / (self.a + self.b + self.c)

    def Dice_2(self):
        # Eq. ID: 2
        return self.a / (2 * self.a + self.b + self.c)

    def Dice_1(self):
        # Eq. ID: 3
        return 2 * self.a / (2 * self.a + self.b + self.c)

    def Jaccard_3w(self):
        # Eq. ID: 4
        return 3 * self.a / (3 * self.a + self.b + self.c)

    def NeiLi(self):
        # Eq. ID: 5
        return 2 * self.a / ((self.a + self.b) + (self.a + self.c))

    def SokalSneath_1(self):
        # Eq. ID: 6
        return self.a / (self.a + 2 * self.b + 2 * self.c)

    def SokalMichener(self):
        # Eq. ID: 7
        return (self.a + self.d) / (self.a + self.b + self.c + self.d)

    def SokalSneath_2(self):
        # Eq. ID: 8
        return 2 * (self.a + self.d) / (2 * self.a + self.b + self.c + 2 * self.d)

    def RogerTanimoto(self):
        # Eq. ID: 9
        return (self.a + self.d) / (self.a + 2 * (self.b + self.c) + self.d)

    def Faith(self):
        # Eq. ID: 10
        return (self.a + (0.5 * self.d)) / (self.a + self.b + self.c + self.d)

    def GowerLegendre(self):
        # Eq. ID: 11
        return (self.a + self.d) / (self.a + 0.5 * (self.b + self.c) + self.d)

    def Intersection(self):
        # Eq. ID: 12
        return self.a

    def InnerProduct(self):
        # Eq. ID: 13
        return self.a + self.d

    def RusselRao(self):
        # Eq. ID: 14
        return self.a / (self.a + self.b + self.c + self.d)

    def Cosine(self):
        # Eq. ID: 31
        return self.a / np.sqrt((self.a + self.b) * (self.a + self.c))

    def GilbertWells(self):
        # Eq. ID: 32
        return np.log(self.a) - np.log(self.n) - np.log((self.a + self.b) / self.n) - np.log((self.a + self.c) / self.n)

    def Ochiai_1(self):
        # Eq. ID: 33
        return self.a / np.sqrt((self.a + self.b) * (self.a + self.c))

    def Forbes_1(self):
        # Eq. ID: 34
        return (self.n * self.a) / (self.a + self.b) * (self.a + self.c)

    def Fossum(self):
        # Eq. ID: 35
        return (self.n * ((self.a - 0.5) ** 2)) / (self.a + self.b) * (self.a + self.c)

    def Sorgenfiel(self):
        # Eq. ID: 36
        return (self.a ** 2) / (self.a + self.b) * (self.a + self.c)

    def Mountford(self):
        # Eq. ID: 37
        return self.a / (0.5 * ((self.a * self.b) + (self.a * self.c)) + (self.b * self.c))

    def Otsuka(self):
        # Eq. ID: 38
        return self.a / np.sqrt((self.a + self.b) * (self.a + self.c))

    def Mcconnaughey(self):
        # Eq. ID: 39
        return ((self.a ** 2) - (self.b * self.c)) / ((self.a + self.b) * (self.a + self.c))

    def Tarwid(self):
        # Eq. ID: 40
        return ((self.n * self.a) - ((self.a + self.b) * (self.a + self.c))) / (
                    (self.n * self.a) + ((self.a + self.b) * (self.a + self.c)))

    def Kulczynski_2(self):
        # Eq. ID: 41
        return ((self.a / 2) * (2 * self.a + self.b + self.c)) / ((self.a + self.b) * (self.a + self.c))

    def DriverKroeber(self):
        # Eq. ID: 42
        return (self.a / 2) * ((1 / (self.a + self.b)) + (1 / (self.a + self.c)))

    def Johnson(self):
        # Eq. ID: 43
        return (self.a / (self.a + self.b)) + (self.a / (self.a + self.c))

    def Dennis(self):
        # Eq. ID: 44
        return ((self.a * self.d) - (self.b * self.c)) / np.sqrt(self.n * ((self.a + self.b) * (self.a + self.c)))

    def Simpson(self):
        # Eq. ID: 45
        return self.a / np.min([(self.a + self.b), (self.a + self.c)])

    def BraunBanquet(self):
        # Eq. ID: 46
        return self.a / np.max([(self.a + self.b), (self.a + self.c)])

    def FagerMcGowan(self):
        # Eq. ID: 47
        return (self.a / np.sqrt((self.a + self.b) * (self.a + self.c))) - (
                    np.max([(self.a + self.b), (self.a + self.c)]) / 2)

    def Forbes_2(self):
        # Eq. ID: 48
        p1 = (self.n * self.a) - ((self.a + self.b) * (self.a + self.c))
        p2 = self.n * min([(self.a + self.b), (self.a + self.c)])
        p3 = (self.a + self.b) * (self.a + self.c)
        return p1 / (p2 - p3)

    def SokalSneath_4(self):
        # Eq. ID: 49
        return ((self.a / (self.a + self.b)) + (self.a / (self.a + self.c)) + (self.d / (self.b + self.d)) + (
                    self.d / (self.c + self.d))) / 4

    def Gower(self):
        # Eq. ID: 50
        return (self.a + self.d) / np.sqrt(
            (self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d))

    def Pearson_1(self):
        # Eq. ID: 51
        return self.n * (((self.a * self.d) - (self.b * self.c)) ** 2) / (
                    (self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d))

    def Pearson_2(self):
        # Eq. ID: 52
        X = self.Pearson_1()
        return np.sqrt((X ** 2) / (self.n + (X ** 2)))

    def PearsonHeron_1(self):
        # Eq. ID: 54
        return ((self.a * self.d) - (self.b * self.c)) / np.sqrt(
            (self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d))

    def Pearson_3(self):
        # Eq. ID: 53
        p = self.PearsonHeron_1()
        if p >= 0:
            return np.sqrt(p / (self.n + p))
        else:
            return 0

    def PearsonHeron_2(self):
        # Eq. ID: 55
        return np.cos((np.pi * np.sqrt(self.b * self.c)) / (np.sqrt(self.a * self.d) + np.sqrt(self.b * self.c)))

    def SokalSneath_3(self):
        # Eq. ID: 56
        return (self.a + self.d) / (self.b + self.c)

    def SokalSneath_5(self):
        # Eq. ID: 56
        return (self.a * self.d) / (
                    ((self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d)) ** 0.5)

    def Stiles(self):
        # Eq. ID: 59
        return np.log10((self.n * ((np.abs(((self.a * self.d) - (self.b * self.c))) - (self.n / 2)) ** 2))) / (
                    (self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d))

    def Ochiai_2(self):
        # Eq. ID: 60
        return (self.a * self.d) / np.sqrt(
            (self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d))

    def Yuleq(self):
        # Eq. ID: 61
        return ((self.a * self.d) - (self.b * self.c)) / ((self.a * self.d) + (self.b * self.c))

    def Yulew(self):
        # Eq. ID: 63
        return (np.sqrt(self.a * self.d) - np.sqrt(self.b * self.c)) / (
                    np.sqrt(self.a * self.d) + np.sqrt(self.b * self.c))

    def Kulczynski_1(self):
        # Eq. ID: 64
        return self.a / (self.b + self.c)

    def Tanimoto(self):
        # Eq. ID: 65
        return self.a / ((self.a + self.b) + (self.a + self.c) - self.a)

    def Disperson(self):
        # Eq. ID: 66
        return ((self.a * self.d) - (self.b * self.c)) / ((self.a * self.d + self.b * self.c) ** 2)

    def Hamann(self):
        # Eq. ID: 67
        return ((self.a + self.d) - (self.b + self.c)) / (self.a * self.d + self.b * self.c)

    def Michael(self):
        # Eq. ID: 68
        return (4 * ((self.a * self.d) - (self.b * self.c))) / (((self.a + self.d) ** 2) + ((self.b + self.c) ** 2))

    def GoodmanKruskal(self):
        # Eq. ID: 69
        sigma = np.max([self.a, self.b]) + \
                np.max([self.c, self.d]) + \
                np.max([self.a, self.c]) + \
                np.max([self.b, self.d])
        sigma_lin = np.max([self.a + self.c,
                            self.b + self.d]) + \
                    np.max([self.a + self.b,
                            self.c + self.d])
        return (sigma - sigma_lin) / ((2 * self.n) - sigma_lin)

    def Anderberg(self):
        # Eq. ID: 70
        sigma = np.max([self.a, self.b]) + np.max([self.c, self.d]) + np.max([self.a, self.c]) + np.max(
            [self.b, self.d])
        sigma_lin = np.max([self.a + self.c, self.b + self.d]) + np.max([self.a + self.b, self.c + self.d])
        return (sigma - sigma_lin) / (2 * self.n)

    def Baroni_UrbaniBuser_1(self):
        # Eq. ID: 71
        return (np.sqrt(self.a * self.d) + self.a) / (np.sqrt(self.a * self.d) + self.a + self.b + self.c)

    def Baroni_UrbaniBuser_2(self):
        # Eq. ID: 72
        return (np.sqrt(self.a * self.d) + self.a - (self.b + self.c)) / (
                    np.sqrt(self.a * self.d) + self.a + self.b + self.c)

    def Peirce(self):
        # Eq. ID: 73
        return ((self.a * self.b) + (self.b * self.c)) / ((self.a * self.b) + (2 * self.b * self.c) + (self.c * self.d))

    def Eyraud(self):
        # Eq. ID: 74
        return ((self.n ** 2) * ((self.n * self.a) - ((self.a + self.b) * (self.a + self.c)))) / (
                    (self.a + self.b) * (self.a + self.c) * (self.c + self.d) * (self.b + self.d))

    def Tarantula(self):
        # Eq. ID: 75
        return (self.a * (self.a + self.b)) / (self.c * (self.c + self.d))

    def Ample(self):
        # Eq. ID: 76
        return np.abs(self.Tarantula())

    def Derived_RusellRao(self):
        # Eq. ID: 77
        return np.log(1 + self.a) / np.log(1 + self.n)

    def Derived_Jaccard(self):
        # Eq. ID: 78
        return np.log(1 + self.a) / np.log(1 + self.a + self.b + self.c)

    def Var_of_Correlation(self):
        # Eq. ID: 79
        return (np.log(1 + (self.a * self.d)) - np.log(1 + (self.b * self.c))) / np.log((1 + (self.n ** 2)) / 4)

    def Derived_SokalMichener(self):
        # Eq. Ref T1
        return np.log(1 + self.a + self.d) / np.log(1 + self.n)

    def Derived_logSokalMichener(self):
        # Eq. Ref T2
        return (np.log(1 + self.n) - np.log(1 + self.b + self.c)) / np.log(1 + self.n)

    """
        OUTROS MÉTODOS DE SIMILARIDADE ABAIXO
    """


    """
        OUTROS MÉTODOS DE SIMILARIDADE ACIMA
    """

    def setNewFunction(self, function):
        def newFunction(function):
            return lambda: function(self)
        setattr(self, function.__name__, newFunction(function))

    # @staticmethod
    # def insert_function_lib(lib, function, name):
    #     setattr(lib, name, function)

