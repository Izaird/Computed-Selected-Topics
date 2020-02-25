#from nltk.corpus import PlaintextCorpusReader
import re
import nltk
import os
import numpy as np
from pathlib import Path
from progress.bar import ChargingBar as Bar
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp as cess
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

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
- Normalización de texto
- Eliminar STOPWORDS
- Eliminar las etiquetas HTML
- Eliminar todo caracter que no es letra
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
	total = len(normalized)
	bar = Bar('Building vocabulary windows', max=total, suffix='%(percent)d%% %(eta)ds left')
	for word in normalized:
		for vocabulary_item in vocabulary:
			if vocabulary_item not in vocabulary_window_list:
				vocabulary_window_list[vocabulary_item] = []
			if vocabulary_item==word:
				start = 0 if index-size<0 else index-size
				vocabulary_window_list[vocabulary_item].extend(normalized[start:index+size+1])
		index+=1
		bar.next()
	bar.finish()
	print("Vocabulary windows: Done.")
	return vocabulary_window_list

	
def save_data(data, filename):
	with open(filename, 'wb') as handle:	handle.write(pickle.dumps(data))

def retrieve_data(filename):
	with open(filename, 'rb') as handle:	return pickle.loads(handle.read())

def vectorize(windows, vocabulary):
	vectorized = {}
	total = len(windows)
	bar = Bar('Building vectors', max=total, suffix='%(percent)d%% %(eta)ds left')
	for word in windows:
		words_window = windows[word]
		vectorized[word] = []
		for vocabulary_word in vocabulary:
			vectorized[word].append(words_window.count(vocabulary_word))
		vectorized[word] = np.array(vectorized[word])
		bar.next()
	bar.finish()
	print("Vectorize: Done")
	return vectorized

def generate_cosines(vectors, vect_against):
	cosines = {}
	for vector in vectors:
		it_vector = vectors[vector]
		cosine = np.dot(it_vector, vect_against)/\
			(np.sqrt(np.sum(it_vector ** 2)) *  np.sqrt(np.sum(vect_against ** 2)))
		cosines[vector] = cosine
	import operator
	cosines = sorted(cosines.items(), key = operator.itemgetter(1), reverse = True)
	return cosines

def save_cosines_into_file(vectors, word):
	cosines = generate_cosines(vectors, vectors[word])
	#cosines = cosines.items()
	with open("cosines.txt", "w") as handle:
		bar = Bar(f'Writing cosines for [{word}]', max=len(cosines), suffix='%(percent)d%% %(eta)ds left')
		for cosine in cosines:
			string = "{}\t{}\n".format(cosine[0],cosine[1])
			handle.write(string)
			bar.next()
		bar.finish()
	print("Cosine file has been written.")
	#print(cosines[:10])			
	return cosines

def build_combined_tagger():
	print("Building tagger")
	default_tagger = nltk.DefaultTagger('V')
	pattern = [
	(r'.*os$', 'PM'),                  # adverbs
	(r'.*as$','PF'),
	(r'.*o$','SM'),
	(r'.*a$','SF')
	]
	cess_sents = cess.tagged_sents()
	regex_tagger = nltk.RegexpTagger(pattern, backoff=default_tagger)
	tagger = nltk.UnigramTagger(cess_sents, backoff=regex_tagger)
	return tagger

def probab(frequencies, vocabulary):
	v_probab = {}
	res = list(frequencies.keys())[0] 
	contex_prob = np.empty_like(frequencies.get(res))
	v_probab = {}
	for contex in frequencies:
		contex_prob = frequencies.get(contex) / np.sum(frequencies.get(contex))
		v_probab.update({contex : contex_prob})
	return v_probab


def v_tf(frequencies, vocabulary):
	k = 1.2
	v_tf = {}
	res = list(frequencies.keys())[0] 
	contex_tf = np.empty_like(frequencies.get(res))
	v_tf = {}
	for contex in frequencies:
		contex_tf = (((k+1) * frequencies.get(contex)) /  (frequencies.get(contex) + k))
		v_tf.update({contex : contex_tf})
	return v_tf

def stem_with_snowball(normalized):
	total = len(normalized)
	spanishStemmer=SnowballStemmer("spanish")
	stems = []
	bar = Bar('Building stemmers', max=total, suffix='%(percent)d%% %(eta)ds left')
	for word in normalized:
		stems.append(spanishStemmer.stem(word))
		bar.next()
	bar.finish()	
	return stems

if __name__ == '__main__':
	nltk.data.path.append('/sdcard/nltk_data/nltk_data')
	curr_path = os.path.dirname(os.path.realpath(__file__))
	path = curr_path+'/Corpus/e961024.htm'
	clean_text = remove_html_tokens(path)
	clean_text = clean_tokens(clean_text)
	normalized = remove_stopwords(clean_text)
	vocabulary = get_unique(normalized)
	if not Path('windows.pkl').is_file():
		windows = get_window(vocabulary,normalized)
		save_data(windows, "windows.pkl")
	else:
		windows = retrieve_data("windows.pkl")
		print("Window file: OK")

	if not Path('vectors.pkl').is_file():
		vectors = vectorize(windows,vocabulary)
		save_data(vectors, "vectors.pkl")
	else:
		vectors = retrieve_data("vectors.pkl")
		print("Vector file: OK")
	
	if not Path('tagger.pkl').is_file():
		tagger = build_combined_tagger()
		save_data(tagger, "tagger.pkl")
	else:
		tagger = retrieve_data("tagger.pkl")
		print("Tagger file: OK")

	if not Path('snowball.pkl').is_file():
		stems = stem_with_snowball(normalized)
		save_data(stems, "snowball.pkl")
	else:
		stems = retrieve_data("snowball.pkl")
		print("Stems file: OK")

	#Using sb prefix to refer to snowball
	vocabulary_sb =  get_unique(stems)

	if not Path('windows_sb.pkl').is_file():
		windows_sb = get_window(vocabulary_sb,stems)
		save_data(windows_sb, "windows_sb.pkl")
	else:
		windows_sb = retrieve_data("windows_sb.pkl")
		print("Window_sb file: OK")

	if not Path('vectors_sb.pkl').is_file():
		vectors_sb = vectorize(windows_sb,vocabulary_sb)
		save_data(vectors_sb, "vectors_sb.pkl")
	else:
		vectors_sb = retrieve_data("vectors_sb.pkl")
		print("Vector_sb file: OK")

	if not Path('v_probab_sb.pkl').is_file():
		v_probab_sb = probab(vectors_sb ,vocabulary_sb)
		save_data(v_probab_sb, "v_probab_sb.pkl")
	else:
		v_probab_sb = retrieve_data("v_probab_sb.pkl")
		print("v_probab_sb file: OK")

	if not Path('v_tf_sb.pkl').is_file():
		v_tf_sb = v_tf(vectors_sb ,vocabulary_sb)
		save_data(v_tf_sb, "v_tf_sb.pkl")
	else:
		v_tf_sb = retrieve_data("v_tf_sb.pkl")
		print("v_tf_sb file: OK")
	# v_tf_sb = v_tf(vectors_sb ,vocabulary_sb)

	cosines = save_cosines_into_file(vectors, "grande")
	tags = tagger.tag(vocabulary)
	print(cosines[:10])
	print()

