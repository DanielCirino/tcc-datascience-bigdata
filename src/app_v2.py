import PyPDF2
import pandas as pd
import requests
from requests.auth import HTTPProxyAuth

DIR_DATASET = r'C:\Users\Daniel.Vale\Personal\PBDDC\TCC\Projeto\datasets\\'
ARQUIVO_PROPOSICOES = f'{DIR_DATASET}\\despesas_cota_parlamentar.csv'
proxies = {'https': 'http://bmg%5Cdaniel.vale:Dnl%3D210504@proxybmg:8080',
           'http': 'http://bmg%5Cdaniel.vale:Dnl%3D210504@proxybmg:8080'}


def normalizarTeorPreposicao(texto: str):
  pontuacao = ['(', ')', ';', ':', '[', ']', ',', '-', '.', '$', '!', '°', '§', 'º', '/', '"', '\'']
  texto = texto.replace('\\n', ' ')
  for ponto in pontuacao:
    texto = texto.replace(ponto, ' ')

  for numero in range(0, 10):
    texto = texto.replace(str(numero), ' ')

  stopWords = pd.read_csv(f'{DIR_DATASET}\\stop_words.csv', usecols=['stop_words']) \
    .replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""])
  listaStopWords = stopWords['stop_words'].values.tolist()
  tokens = texto.strip().split()
  palavrasChave = [palavra for palavra in tokens if not palavra.lower() in listaStopWords and not palavra in pontuacao]

  return ' '.join(palavrasChave)


def lerDados():
  filename = f'{DIR_DATASET}proposicoes\\inteiro_teor\\inteiroTeor-712981.pdf'
  # open allows you to read the file.

  sesssao = requests.Session()
  sesssao.proxies = proxies
  sesssao.auth = HTTPProxyAuth('daniel.vale','Dnl%3D10504')

  response = sesssao.get('https://www.camara.leg.br/proposicoesWeb/prop_mostrarintegra?codteor=1862095',
                          proxies=proxies,auth=('daniel.vale','Dnl%3D210504'))

  pdfFileObj = open(filename, 'rb')
  # The pdfReader variable is a readable object that will be parsed.
  pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
  # Discerning the number of pages will allow us to parse through all the pages.
  num_pages = pdfReader.numPages
  count = 0
  text = ""
  # The while loop will read each page.
  while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count += 1
    text += pageObj.extractText()

  textoNormalizado = normalizarTeorPreposicao(text)
  print(textoNormalizado)


if __name__ == '__main__':
  lerDados()
