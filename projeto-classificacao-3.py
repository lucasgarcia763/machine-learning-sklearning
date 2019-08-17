from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

map = {
    "expected_hours" : "horas_esperadas",
    "unfinished" : "inacabado",
    "price" : "preco"
}

dados = dados.rename(columns = map)

trocar = {
    0 : 1,
    1 : 0
}

dados["finalizado"] = dados.inacabado.map(trocar)

#sns.scatterplot(x = "horas_esperadas", y = "preco", hue="finalizado", data = dados)

#sns.relplot(x = "horas_esperadas", y = "preco", col="finalizado", data = dados)
#plt.show()

x = dados[["horas_esperadas", "preco"]]
y = dados[["finalizado"]]

model = LinearSVC()

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

model.fit(treino_x, treino_y)

previsoes = model.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acurácia foi %.2f%%" % acuracia)

previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) * 100

print("A acurácia do algoritmo de baseline foi %.2f%%" % acuracia)

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

pixel = 100

eixo_x = np.arange(x_min, x_max, (x_max - x_min) / 100)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / 100)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(pontos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_x, s=1)
plt.show()