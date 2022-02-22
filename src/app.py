import logging
from time import time

import matplotlib as plt
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def prepararDados(usarHashing=True):
  DIR_DATASET = r'C:\Users\Daniel.Vale\Personal\PBDDC\TCC\Projeto\datasets\\'
  ARQUIVO_PROPOSICOES = f'{DIR_DATASET}\\proposicoes.csv'
  ARQUIVO_TEMAS = f'{DIR_DATASET}\\temas.csv'
  colunasPreposicoes = ['id', 'uri', 'ementa']
  colunasTemas = ['uriProposicao', 'siglaTipo', 'codTema', 'tema']

  print('_' * 80)
  print(f'Carregando dados datasets: \nproposições ==> {ARQUIVO_PROPOSICOES}\ntemas==> {ARQUIVO_TEMAS} ')

  print('_' * 80)
  print(f'Criando dataframes:')

  dfPreposicoes = pd.read_csv(ARQUIVO_PROPOSICOES, sep='|', low_memory=False, usecols=colunasPreposicoes)
  print(f'Qtd. de proposições: {len(dfPreposicoes)}')

  dfTemas = pd.read_csv(ARQUIVO_TEMAS, sep='|', low_memory=False, usecols=colunasTemas)
  print(f'Qtd. de proposições classificadas: {len(dfTemas)}')

  print('_' * 80)
  print(f'Fazendo merge das preposições com as preposicões com temas')

  dfPreposicoesTemas = pd.merge(dfTemas, dfPreposicoes, left_on='uriProposicao', right_on='uri')
  print(f'Qtd. de proposições com temas: {len(dfPreposicoesTemas)}')

  print('_' * 80)
  print('Criando dataframe para o modelo de aprendizado')

  dfClassificacao = dfPreposicoesTemas[['ementa', 'tema']]
  print(dfClassificacao.describe())

  print("." * 80)
  print('Lista de temas')
  print(pd.unique(dfClassificacao['tema']))

  print('_' * 80)
  print('Dividindo dados de treino e de teste para o modelo de aprendizado')
  ementas = dfPreposicoesTemas['ementa'].values.astype('U')
  temas = dfPreposicoesTemas['tema'].values.astype('U')
  target_names = pd.unique(temas)

  ementas_treino, ementas_teste, temas_treino, temas_test = train_test_split(ementas, temas, test_size=0.33)

  print(f'Divisão dos dados: Treino={len(ementas_treino) / len(ementas)} | Teste={len(ementas_teste) / len(ementas)}')
  print(f'Dados de treino: Ementas={len(ementas_treino)} | Temas={len(temas_treino)}')
  print(f'Dados de teste: Ementas={len(ementas_teste)} | Temas={len(temas_test)}')

  print('_' * 80)
  print(f'Vetorizando o texto das ementas [utilizando Hashing={usarHashing}]')
  stopWords = pd.read_csv(f'{DIR_DATASET}\\stop_words.csv', usecols=['stop_words']) \
    .replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""])
  listaStopWords = stopWords['stop_words'].values.tolist()
  print(f'Total de stopwords: {len(stopWords)}')

  inicio = time()
  if (usarHashing):
    vetorizador = HashingVectorizer(stop_words=listaStopWords, alternate_sign=False, n_features=2 ** 4)
    X_train = vetorizador.transform(ementas_treino)
  else:
    vetorizador = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=listaStopWords)
    X_train = vetorizador.fit_transform(ementas_treino)

  duracao = time() - inicio

  print(f'Vetorização dados de treino finalizada em {duracao}0.2f s')
  print("No. de exemplos (n_samples): %d, No. Features (n_features): %d" % X_train.shape)

  print('.' * 80)
  inicio = time()
  X_test = vetorizador.transform(ementas_teste)
  duracao = time() - inicio
  print(f'Vetorização dados de teste finalizada em {duracao}0.2f s')
  print("No. de exemplos (n_samples): %d, No. Features (n_features): %d" % X_test.shape)

  return X_train, temas_treino, X_test, temas_test, target_names


def executarModeloClassificacao(classificador, X_treino, y_treino, X_teste, y_teste, target_names, imprimirRelatorio=False,
                                imprimirMatrizConfusao=False):
  nomeClassificador = str(classificador).split("(")[0]
  print('.' * 80)
  print(f'Treinando o modelo: {classificador}')
  inicio = time()
  classificador.fit(X_treino, y_treino)
  tempoTreino = time() - inicio
  print(f'Tempo de treinamento: {tempoTreino:.2f} segundos')

  inicio = time()
  predicao = classificador.predict(X_teste)
  tempoTeste = time() - inicio
  print(f'Tempo de teste: {tempoTeste:.2f} segundos')

  acuracia = metrics.accuracy_score(y_teste, predicao)
  matrizConfusao = metrics.confusion_matrix(y_teste, predicao)
  relatorioClassificao = metrics.classification_report(y_teste,
                                                       predicao,
                                                       target_names=target_names,
                                                       labels=target_names,zero_division=0)

  print(f'Acurácia = {acuracia * 100:.2f}')

  if hasattr(classificador, "coef_"):
    print(f'Dimensionaliade: {classificador.coef_.shape[1]}')
    print(f'Densidade: {density(classificador.coef_)}')

  if imprimirRelatorio:
    print("RELATÓRIO DE CLASSIFICAÇÃO:")
    print(relatorioClassificao)

  if imprimirMatrizConfusao:
    print('MATRIZ DE CONFUSAO')
    plt.matshow(matrizConfusao)
    plt.title(f'Matriz de confusão {nomeClassificador}')
    plt.colorbar()
    plt.ylabel("Classificações corretas")
    plt.xlabel("Classificações obtidas")
    plt.show()

  print('.' * 80)

  return nomeClassificador, acuracia, tempoTreino, tempoTeste, relatorioClassificao, matrizConfusao

def imprimirRelatorioClassificacao(nomeClassificador, acuracia, tempoTreino, tempoTeste, relatorioClassificao, matrizConfusao):
  print(f'Tempo de treinamento: {tempoTreino:.2f} segundos')
  print(f'Acurácia = {acuracia * 100:.2f}')
  print(f'Acurácia = {acuracia * 100:.2f}')


if __name__ == '__main__':
  resultados = []
  X_train, temas_treino, X_test, temas_test, target_names = prepararDados(False)

  for classificador, nome in (
      (SGDClassifier(alpha=0.0001, max_iter=50, penalty='l2'), 'SGD panalidade l2'),
      # (SGDClassifier(alpha=0.0001, max_iter=50, penalty='l1'), 'SGD panalidade l1'),
      # (LinearSVC(penalty='l2', dual=False, tol=1e-3), 'Linear SVC penalidade l2'),
      # (LinearSVC(penalty='l1', dual=False, tol=1e-3), 'Linear SVC penalidade l1'),
      # (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
      # (Perceptron(max_iter=50), "Perceptron"),
      # (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
      # (KNeighborsClassifier(n_neighbors=10), "kNN"),
      # (RandomForestClassifier(), "Random forest"),
      # (MLPClassifier(max_iter=100, random_state=1, verbose=False), 'MLP Classifier'),
      (NearestCentroid(), 'NearestCentroid '),
      (MultinomialNB(alpha=0.01), ' Naive Bayes - MultinomialNB'),
      (BernoulliNB(alpha=0.01), ' Naive Bayes - BernoulliNB'),
      (ComplementNB(alpha=0.01), ' Naive Bayes - ComplementNB'),
      (Pipeline(
        [
          (
              "feature_selection",
              SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3)),
          ),
          ("classification", LinearSVC(penalty="l2")),
        ]
      ), 'LinearSVC c/ feature selection')
  ):
    print("=" * 80)
    print(nome)
    resultados.append(executarModeloClassificacao(classificador, X_train, temas_treino, X_test, temas_test, target_names))
