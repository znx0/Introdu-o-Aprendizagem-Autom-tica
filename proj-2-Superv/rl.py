import numpy as np
#ACRESCENTAR BIBLIOTECAS QUE SEJAM NECESSÁRIAS

class MulticlassLogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, num_epocas=100):
        self.lr = learning_rate
        self.epocas = num_epocas
        self.pesos = None
        self.offset = None
        self.loss_history = []

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def fit(self, X, y, num_classes):
    #TO DO - IMPLEMENTAR O TREINO DA REGRESSÃO LOGÍSTICA

    def predict(self, X):
    #TO DO - IMPLEMENTAR A PREDIÇÃO DA REGRESSÃO LOGÍSTICA