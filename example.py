import binary_similarity as sim

"""Exemplo de utilização"""

#Criação dos conjuntos
a = set(['a','b','e'])
b = set(['a','b','c','d'])
u = set(a.union(b).union(set(['h','f'])))
# Instanciação do modelo
model = sim.similarity_score(a,b,U=u)

#List of all available similarity functions
functions = model.listSimFunctions()

#Métodos de similaridades dos conjuntos
jaccard_score = model.evaluate('Jaccard')
dice1_score = model.evaluate('Dice_1')

##Criação e chamada de um método personalizado
##fora do modelo
#def outsideSimFunc(self):
#    return 2*self.a/(2*self.a+self.b+self.c)

#model.setNewFunction(outsideSimFunc)

#teste = model.evaluate('outsideSimFunc')

testeAll = model.evaluateAll()