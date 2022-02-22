import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# um classificador linear que utiliza o Gradiente Descendente Estocástico como método de treino. Por padrão, utiliza o estimador SVM.
from sklearn.linear_model import SGDClassifier
# Uma rede neural Perceptron Multicamadas
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
import matplotlib.pyplot as plt

DIR_DATASET = r'C:\Users\Daniel.Vale\Personal\PBDDC\TCC\Projeto\datasets\\'
ARQUIVO_PROPOSICOES = f'{DIR_DATASET}\\proposicoes-2021.csv'
ARQUIVO_TEMAS = f'{DIR_DATASET}\\proposicoesTemas-2021.csv'

if __name__ == '__main__':

  colunasPreposicoes = ['id', 'uri', 'ementa']
  colunasTemas = ['uriProposicao', 'siglaTipo', 'codTema', 'tema']

  dfPreposicoes = pd.read_csv(ARQUIVO_PROPOSICOES, sep=';', low_memory=False, usecols=colunasPreposicoes)
  dfTemas = pd.read_csv(ARQUIVO_TEMAS, sep=';', low_memory=False, usecols=colunasTemas)
  print(dfPreposicoes.describe())
  print(dfTemas.describe())

  dfPreposicoesTemas = pd.merge(dfTemas, dfPreposicoes, left_on='uriProposicao', right_on='uri')
  print(dfPreposicoesTemas.describe())

  dfClassificacao = dfPreposicoesTemas[['ementa', 'tema']]
  print(dfClassificacao.describe())

  ementas = dfPreposicoesTemas['ementa'].values
  temas = dfPreposicoesTemas['tema'].values

  ementas_treino, ementas_teste, temas_treino, temas_test = train_test_split(ementas, temas, test_size=0.5)

  classificador = MLPClassifier(max_iter=100, random_state=1, verbose=True)

  vetorizador = TfidfVectorizer()
  vetor_ementas_treino = vetorizador.fit_transform(ementas_treino)

  classificador.fit(vetor_ementas_treino, temas_treino)

  vetor_ementas_teste = vetorizador.transform(ementas_teste)
  predicao = classificador.predict(vetor_ementas_teste)

  print(metrics.classification_report(temas_test, predicao, target_names=pd.unique(temas), labels=pd.unique(temas)))

  print(classificador.classes_)

  matriz_confusao = metrics.confusion_matrix(temas_test, predicao).ravel()
  matriz_confusao.ravel()
  print(matriz_confusao)

  plt.matshow(matriz_confusao)
  plt.title("Matriz de confusão")
  plt.colorbar()
  plt.ylabel("Classificações corretas")
  plt.xlabel("Classificações obtidas")
  plt.show()
