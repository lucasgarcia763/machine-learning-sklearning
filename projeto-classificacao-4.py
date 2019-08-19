import pandas as pd
import numpy as np
import graphviz
from datetime import datetime
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#from sklearn.dummy import DummyClassifier
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC

# # ***** Importação dos dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
dados = pd.read_csv(uri)

# # ***** Tratamento dos dados
renomear = {
    "mileage_per_year" : "milhas_por_ano",
    "model_year" : "ano_modelo",
    "price" : "preco",
    "sold" : "vendido"
}

dados = dados.rename(columns = renomear)

trocar = {
    "yes" : 1,
    "no" : 0
}

dados.vendido = dados.vendido.map(trocar)

ano_atual = datetime.today().year

dados["idade_do_modelo"] = ano_atual - dados.ano_modelo
dados["km_por_ano"] = dados.milhas_por_ano * 1.60934

dados = dados.drop(columns = ["Unnamed: 0", "milhas_por_ano", "ano_modelo"], axis = 1)

x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados[["vendido"]]


# # ***** Fixando a seed de aleatoriedade
SEED = 5
np.random.seed(SEED)

# # ***** Treino com algoritmo linear
# treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)
#
# print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))
#
# modelo = LinearSVC()
# modelo.fit(treino_x, treino_y)
# previsoes = modelo.predict(teste_x)
#
# acuracia = accuracy_score(teste_y, previsoes) * 100
# print("A acurácia foi %.2f%%" % acuracia)
#
# # ***** Treino com dummy stratified, para gerar um score baseline
# dummy_stratified = DummyClassifier()
# dummy_stratified.fit(treino_x, treino_y)
# acuracia = dummy_stratified.score(teste_x, teste_y) * 100
# print("A acurácia do dummy_stratified foi %.2f%%" % acuracia)
#
# # ***** Treino com dummy most frequent, para gerar um score baseline
# dummy_most_frequent = DummyClassifier(strategy="most_frequent")
# dummy_most_frequent.fit(treino_x, treino_y)
# acuracia = dummy_most_frequent.score(teste_x, teste_y) * 100
# print("A acurácia do dummy_most_frequent foi %.2f%%" % acuracia)
#
# # ***** Alterando o algoritmo de classificação para o não linear
# raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)
# print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x), len(raw_teste_x)))
#
# scaler = StandardScaler()
# scaler.fit(raw_treino_x)
# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)
#
# model = SVC()
# model.fit(treino_x, treino_y)
#
# previsoes = model.predict(teste_x)
#
# acuracia = accuracy_score(teste_y, previsoes) * 100
#
# print("A acurácia foi %.2f%%" % acuracia)

# # ***** Utilizando Árvore de decisão
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x), len(raw_teste_x)))

model = DecisionTreeClassifier(max_depth=2)
model.fit(raw_treino_x, treino_y)

previsoes = model.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acurácia foi %.2f%%" % acuracia)

features = x.columns
dot_data = export_graphviz(model, out_file=None, feature_names=features, filled=True, rounded=True, class_names=["Não", "Sim"])
grafico = graphviz.Source(dot_data)
grafico.view()