#from nltk.corpus import PlaintextCorpusReader
import re
import pprint
import nltk
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
'''
1. Lectura de texto como cadena
2. Limpiar de HTML
3. TOkenizar: word_tokenize(), PlainTextCorpusReader, split()
...
4. (CARACTERÍSTICAS QUE MEJOR DESCRIBAN EL TEXTO) Selección de los tokens relevantes
------

Bag of words:
Se trata del texto aún con palabras. Todo el texto se ve como un conjunto de palabras donde no hay orden.
La hipótesis es que la presencia o ausencia de palabras tiene información suficiente para hacer conclusiones según
el contenido del texto. Es decir, dichas palabras sin orden pueden reflejar la semántica de un texto o de una palabra.
Tags: Contexto derecho, contexto izquierdo, ventanaN

My [cat] is dying
Big [cat] is dying

Documento:
Cualquier cosade nuestro interés. Documento puede ser texto completo, una palabra. En este caso el documento
será un diccionario que incluya las palabras del contexto. No tiene una estructura y podría llamársele Pseudo Documento. 

Bag of Words -> Vector Space Model
		  word
word1		|	d1 = (x1,x2,x3...xn) n = vocabulary size	PSEUDO DOC ("cat")
			|  /
			| /	   / d2 = (y1,y2...tn) PSEUDO DOC ("dog")
____________|/____/______	word
		   /|\
		  /	| \
		 /  |  \
word2			wordN

Cada una de estas palabras tiene diferente frecuencia, denotado por el eje X
Cada eje es una palabra y dependiendo su magnitud será la cantidad que se repite.
Si los vectores están cerca entonces las palabras son similares. Aquí se mide la similitud como el ángulo. 
Dicha similitud se mide utilizando el coseno.


=== Capítulo 2: STOPWORDS ===

Palabras auxiliares, artículos, verbos, que se usan para conjugar otros verbos y construir formas
de tiempo. Por sí mismas tienen pequeño o ningún significado. Usualmente se eliminan durante
el procesamiento del texto para tratar de obtener el máximo significado y contexto

Normalización de texto:
- Normalización de texto:
- Eliminar las etiquetas HTML
- Eliminar todo caracter que no es letra
- Eliminar STOPWORDS
- Obtener vocabulario
-> La frecuencia representa la semántica de la forma más precisa posible, ya sea en pasado o 
en presente (stemming)
- Lematización, lema es la forma de la palabra que aparece en el diccionario



TAREA:
Text Analytics with Python
P131
P150

- Hacer vocabulario de las palabras del texto
-- sin caracteres especiales
-- sin stopwords
-- ordenado
____________________

Vocabulario: Ejemplo
 Abrir | comprar | decir | | |

 Texto sin stopwords -> lista de palabras normalizada
 Vocabulario

 _ _ _ _ gato _ _ _ _
 _ _ _ _ gato _ _ _ _
 _ _ _ _ gato _ _ _ _
 _ _ _ _ gato _ _ _ _

 Tomar todas las palabras del vocabulario y tomar 4 a la izquierda y a la derecha

[abrir, comprar, decir, hablar]
Vectores de frecuencias
 gato_vector = ([cantidadde repeticiones de abrir en es])


'''


#wordlists = PlaintextCorpusReader(corpus_root, '.*')
#print(wordlists)
#print(wordlists.fileids())

def remove_html_tokens(filename):
	with open(filename) as fd:
		text = fd.read()
		#Eliminar tags de html
		soup = BeautifulSoup(text, 'lxml')
		text = soup.get_text()
		text = text.lower()

	#with open('text_string.txt','w') as spit_fd:
	#	spit_fd.write(text)

	tokens = nltk.word_tokenize(text)
	text = nltk.Text(tokens)
	print("Token count: {0}".format(len(text)))
	return text
	#print(text[:100])
	#print("\nConcordancia\n")
	#text.concordance("actividad")
	#print("\nPalabras similares\n")
	#text.similar("actividad")

def clean_tokens(text):
	clean_tokens_arr = []
	for token in text:
		t = []
		for char in token:
			if re.match(r'[a-záéíóúüñ]', char):		t.append(char)
		whole = ''.join(t)
		#if (whole := ''.join(t)) != '':
		if whole != '':
			clean_tokens_arr.append(whole)

	print("Text without strange chars count: {0}".format(len(clean_tokens_arr)))
	return clean_tokens_arr

def remove_stopwords(text):
	stpwords = stopwords.words('spanish')
	text_without_stopwords = [token for token in text if token not in stpwords]
	print("Text without stopwords count: {0}".format(len(text_without_stopwords)))
	return text_without_stopwords


def get_unique(text):
	result = sorted(set(text))
	print("Vocabulary count: {0}".format(len(result)))
	return result

def save_text(text, filename):
	print(type(text))
	with open(filename, 'w') as fd:
		concatenated_text = ' '.join(text)
		fd.write(concatenated_text)

def get_window(vocabulary, normalized, size=4):
	index = 0
	vocabulary_window_list = {}
	for word in normalized:
		for vocabulary_item in vocabulary:
			if vocabulary_item not in vocabulary_window_list:
				vocabulary_window_list[vocabulary_item] = []
			if vocabulary_item==word:
				start = 0 if (index-size <0 ) else index-size	
				vocabulary_window_list[vocabulary_item].append(normalized[start:index+size+1])
		index+=1
	return vocabulary_window_list
	
if __name__ == '__main__':
	pp = pprint.PrettyPrinter(indent=4)
	curr_path = os.path.dirname(os.path.realpath(__file__))
	nltk.data.path.append(curr_path + '/nltk_data')
	path = curr_path+'/e961024.htm'
	clean_text = remove_html_tokens(path)
	clean_text = clean_tokens(clean_text)
	normalized = remove_stopwords(clean_text)
	vocabulary = get_unique(normalized)
	
	#normalized = ['5', '1', '2', '3', '4', '5', '6', '7', '8', '9','5', '50']
	#vocabulary = ['5']


	contextlist = get_window(vocabulary,normalized)
	pp.pprint(contextlist)
	#index = 0
	#for item in contextlist:
#		if index < 51:
#			pp.pprint(item, contextlist[item])
#			index+=1