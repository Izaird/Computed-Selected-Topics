import pickle
import re
import gzip
import nltk
import os
import numpy as np
from math import log
from math import e as eulerval
from pathlib import Path
from progress.bar import IncrementalBar as Bar
from progress.spinner import Spinner
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp as cess
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

K_VALUE = 1.2

def remove_html_tokens(filename):
	with open(filename) as fd:
		text = fd.read()
		#Eliminar tags de html
		soup = BeautifulSoup(text, 'lxml')
		text = soup.get_text()
		text = text.lower()

	tokens = nltk.word_tokenize(text)
	text = nltk.Text(tokens)
	print("Token count: {0}".format(len(text)))
	return text

def removeHtmlTokens(text):
	soup = BeautifulSoup(text, 'lxml')
	text = soup.get_text()
	text = text.lower()
	tokens = nltk.word_tokenize(text)
	text = nltk.Text(tokens)
	return text

def clean_tokens(text):
	clean_tokens_arr = []
	for token in text:
		t = []
		for char in token:
			if re.match(r'[a-záéíóúüñ]', char):		t.append(char)
		whole = ''.join(t)
		if whole != '':
			clean_tokens_arr.append(whole)

	# print("Text without strange chars count: {0}".format(len(clean_tokens_arr)))
	return clean_tokens_arr

def remove_stopwords(text):
	stpwords = stopwords.words('spanish')
	text_without_stopwords = [token for token in text if token not in stpwords]
	print("Text without stopwords count: {0}".format(len(text_without_stopwords)))
	return text_without_stopwords

def snowball_stemmer(normalized_text):
	stemmer = SpanishStemmer()
	normalized_stems = []
	bar = Bar('Stemming', max=len(normalized_text), suffix='%(percent)d%% %(eta)ds', width=20)
	for word in normalized_text:
		stem = stemmer.stem(word)
		normalized_stems.append(stem)
		bar.next()
	bar.finish()
	return normalized_stems

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
	bar = Bar('Building windows', max=total, suffix='%(percent)d%% %(eta)ds', width=20)
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
	with gzip.open(filename, 'wb') as handle:	handle.write(pickle.dumps(data))

def retrieve_data(filename):
	with gzip.open(filename, 'rb') as handle:	return pickle.loads(handle.read())

def vectorize(windows, vocabulary):
	vectorized = {}
	total = len(windows)
	bar = Bar('Building vectors', max=total, suffix='%(percent)d%% %(eta)ds', width=20)
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
		cosine = np.dot(it_vector, vect_against)/(np.linalg.norm(it_vector) * np.linalg.norm(vect_against))
		cosines[vector] = cosine
	import operator
	cosines = sorted(cosines.items(), key = operator.itemgetter(1), reverse = True)
	return cosines

def generate_lemmas():
	with open("generate.txt", encoding='iso-8859-1') as handle:
		lemma_dict = {}
		contents = handle.readlines()		#Read line by line
		bar = Bar('Building lemmas', max=len(contents), suffix='%(percent)d%% %(eta)ds', width=20)
		for line in contents:
			separated = line.split("#")		#Split with delimiter
			if not separated[0].isalnum():	#For non-alphanumeric characters
				bar.next()
				continue
			suffix = separated[1]			#Suffix + identifier
			word = separated[0]				
			lemma = separated[-1].strip().split(" ")[-1] #Lemma is at the last position of the list
			if suffix[0] != " ":
				suffix = suffix.split(" ")[0]
				word = word + suffix
				lemma_dict[word] = lemma
			bar.next()
		bar.finish()
	return lemma_dict

def save_cosines_into_file(vectors, word, filename, to_stem=False, lemmas=[]):
	if word not in vectors and to_stem:
		#Try to stem the word if it wasn't found as is in the vector argument
		stemmer = SpanishStemmer()
		word = stemmer.stem(word)
		if word not in vectors:
			#If it wasn't found, return and raise an exception
			raise KeyError("Not even the stemmed word was found in your vectors. Exiting...")	
	else:
		#Instead, try to lemmatize the word or use the original word
		word = get_lemma_for_word(lemmas, word)
			
	cosines = generate_cosines(vectors, vectors[word])
	#cosines = cosines.items()
	with open(filename, "w") as handle:
		bar = Bar(f'{filename}', max=len(cosines), suffix='%(percent)d%% %(eta)ds', width=20)
		for cosine in cosines:
			string = "{0:15} {1:15}\n".format(cosine[0],cosine[1])
			handle.write(string)
			bar.next()
		bar.finish()
	#print(cosines[:10])			
	return cosines

def get_lemma_for_word(lemma_dict, word):
	return lemma_dict[word] if word in lemma_dict else word 

def lemmatizer(normalized_text, lemmas):
	normalized = []
	bar = Bar('Lemmas-Normalized', max=len(normalized_text), suffix='%(percent)d%% %(eta)ds', width=20)
	for word in normalized_text:
		normalized.append(get_lemma_for_word(lemmas,word))
		bar.next()
	bar.finish()
	return normalized

def build_combined_tagger():
	print('Building tagger...')
	default_tagger = nltk.DefaultTagger('V')
	patterns = [(r'.*o$', 'NMS'), # noun masculine singular
				(r'.*os$', 'NMP'), # noun masculine plural
				(r'.*a$', 'NFS'),  # noun feminine singular
				(r'.*as$', 'NFP')  # noun feminine singular
				]
	cess_sents = cess.tagged_sents()
	regex_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
	tagger = nltk.UnigramTagger(cess_sents, backoff=regex_tagger)
	return tagger

def vectors_tf(vectors):
	tf_vect = {}
	bar = Bar(f'Building tfVect', max=len(vectors), suffix='%(percent)d%% %(eta)ds', width=20)
	KVAL_PLUS_ONE = K_VALUE+1
	for vect in vectors:
		this_vector = vectors[vect]
		new_vect = []
		for component in this_vector:
			new_vect.append( ((KVAL_PLUS_ONE)*component)/(component+K_VALUE) )
		tf_vect[vect] = np.array(new_vect)
		bar.next()
	bar.finish()
	return tf_vect

def vectors_idf(vectors):
	total = len(vectors)
	idf = np.empty(total)
	idf.fill(0)
	#idf = [0 for index in range(total)]
	bar = Bar(f'Building idfV', max=total, suffix='%(percent)d%% %(eta)ds', width=20)
	for vect in vectors:
		this_vector = vectors[vect]
		bar.next()
		for index, component in enumerate(this_vector):
			if component:
				idf[index] += 1
	bar.finish()
	return np.array([log((total+1)/item,eulerval) for item in idf])

def vectors_by_frequency(vectors):
	vectors_by_freq = {}
	bar = Bar(f'Building freqVectors', max=len(vectors), suffix='%(percent)d%% %(eta)ds', width=20)
	for vector in vectors:
		this_vector = vectors[vector]
		sum = np.sum(this_vector)
		vectors_by_freq[vector] = np.array([component/sum for component in this_vector])
		bar.next()
	bar.finish()
	return vectors_by_freq

def tfidf(tf,idf):
	tfidf = {}
	bar = Bar(f'Bulding tfidf', max=len(tf), suffix='%(percent)d%% %(eta)ds', width=20)
	for vect in tf:
		this_vector = tf[vect]
		tfidf[vect] = np.multiply(this_vector, idf)
		bar.next()
	bar.finish()
	return tfidf


def getArticles(dname):
	f = open(dname, encoding='utf-8')
	text_string = f.read()
	f.close()

	article_segments = re.split('<h3>',text_string)
	articles = []
	articles_normalized = []
	if not Path('lemmas.pkl').is_file():
		lemmas = generate_lemmas()
		save_data(lemmas, "lemmas.pkl")
	else:
		lemmas = retrieve_data("lemmas.pkl")
		print("Lemmatizer file: OK, len:",len(lemmas))

	for art in article_segments:
		soup = BeautifulSoup(art, 'lxml')
		text = soup.get_text()
		articles.append(text)


	for art in articles:
		clean_text = removeHtmlTokens(art)		#Eliminar tags html
		clean_text = clean_tokens(clean_text)		#Eliminar signos de puntuación, etc
		art = remove_stopwords(clean_text)
		art = lemmatizer(art, lemmas)
		articles_normalized.append(art)

	return articles_normalized 

def getNounsList(normalized):
	is_noun = lambda pos: pos[:2] == 'NN'
	# do the nlp stuff
	nouns = [word for (word, pos) in nltk.pos_tag(normalized) if is_noun(pos)] 
	
	nounFreq = {}
	for noun in nouns:
		nounFreq.update({noun : nouns.count(noun)})
	
	nounFreq = sorted(nounFreq.items(), key=lambda x: x[1], reverse=True)

	return nounFreq


def getNounsListIDFTF(normalized):
	number_of_words = len(normalized) 
	is_noun = lambda pos: pos[:2] == 'NN'
	# do the nlp stuff
	nouns = [word for (word, pos) in nltk.pos_tag(normalized) if is_noun(pos)] 
	
	nounFreq = {}
	for noun in nouns:
		nounFreq.update({noun : nouns.count(noun)})
	
	for noun in nounFreq:
		x = nounFreq[noun]
		IDFDF = (((K_VALUE+1)*x)/ (x * K_VALUE))  * log(number_of_words/x)
		nounFreq.update({noun : IDFDF})
	nounFreq = sorted(nounFreq.items(), key=lambda x: x[1], reverse=True)

	return nounFreq



def getFullNormalize(path):
	clean_text = remove_html_tokens(path)		#Eliminar tags html
	clean_text = clean_tokens(clean_text)		#Eliminar signos de puntuación, etc
	normalized = remove_stopwords(clean_text)	#Eliminar stopwords
	if not Path('lemmas.pkl').is_file():
		lemmas = generate_lemmas()
		save_data(lemmas, "lemmas.pkl")
	else:
		lemmas = retrieve_data("lemmas.pkl")
		print("Lemmatizer file: OK, len:",len(lemmas))
	###############################
	normalized = lemmatizer(normalized, lemmas)

	return normalized

def topic_covarage(topics, articles):
	j = 0 #article number

	topic_covarage = np.zeros([len(topics), len(articles)])
	for article in articles:
		k = 0 #topic number 
		for topic in topics:
			topic_covarage[k][j] = article.count(topic)
			k += 1 
		
		j += 1


	j = 0 #article number
	k = 0 #topic number 

	sums = np.sum(topic_covarage, axis=0)

	for j in range(len(articles)):
		for k in range(len(topics)):
			if topic_covarage[k][j] and sums[j]:
				topic_covarage[k][j] = (topic_covarage[k][j] / sums[j]) * 100

	return topic_covarage