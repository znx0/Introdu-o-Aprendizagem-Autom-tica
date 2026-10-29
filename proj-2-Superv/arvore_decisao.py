import numpy as np
#IMPORTE AS BIBLIOTECAS QUE PRECISAR

class ArvoreDecisao:
    class Node:
        def __init__(self, feature=None, threshold=None, esquerda=None, direita=None, *, valor=None):
            self.feature = feature
            self.limiar = threshold
            self.esquerda = esquerda
            self.direita = direita
            self.valor = valor

        def is_leaf_node(self):
            return self.value is not None

    def __init__(self, profundidade_maxima=None):
        self.profundidade_maxima = profundidade_maxima
        self.raiz = None

    def _entropy(self, y):
    #TO DO - CÁLCULO DA ENTROPIA

    def _mutual_information(self, y, y_esquerda, y_direita):
    #TO DO - CALCULO DA INFORMAÇÃO MUTUA

    def _best_split(self, X, y):
    #TO DO - ENCONTRAR O MELHOR SPLIT

    def _build_tree(self, X, y, profundidade=0):
    # TO DO - CONSTRUIR A ARVORE

    def fit(self, X, y):
    #TO DO - TREINAR A ARVORE

    def predict(self, X):
    #TO DO -  FAZER A PREDIÇÃO