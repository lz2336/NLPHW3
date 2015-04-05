from xml.dom import minidom
import json
import codecs
import sys
import unicodedata
from sklearn import svm, neighbors
import nltk
import string

def remove_punctuation(input_str):
	for c in string.punctuation:
		input_str = input_str.replace(c, '')
	return input_str

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def apply_features(input_str):
	input_str = remove_punctuation(input_str)
	input_str = replace_accented(input_str)
	input_str = input_str.lower()
	return input_str

def remove_stopwords(language, words):
	language = language.lower()
	if language == 'catalan':
		# Using google stop word list for catalan: http://meta.wikimedia.org/wiki/Stop_word_list/google_stop_word_list#Catalan
		stopwords = ['de', 'es', 'i', 'a', 'o', 'un', 'una', 'unes', 'uns', 'un', 'tot', 'tambe', 'altre', 'algun', 'alguna', 'alguns', 'algunes', 'ser', 'es', 'soc', 'ets', 'som', 'estic', 'esta', 'estem', 'esteu', 'estan', 'com', 'en', 'per', 'perque', 'per', 'que', 'estat', 'estava', 'ans', 'abans', 'essent', 'ambdos', 'pero', 'per', 'poder', 'potser', 'puc', 'podem', 'podeu', 'poden', 'vaig', 'va', 'van', 'fer', 'faig', 'fa', 'fem', 'feu', 'fan', 'cada', 'fi', 'inclos', 'primer', 'des', 'de', 'conseguir', 'consegueixo', 'consigueix', 'consigueixes', 'conseguim', 'consigueixen', 'anar', 'haver', 'tenir', 'tinc', 'te', 'tenim', 'teniu', 'tene', 'el', 'la', 'les', 'els', 'seu', 'aqui', 'meu', 'teu', 'ells', 'elles', 'ens', 'nosaltres', 'vosaltres', 'si', 'dins', 'sols', 'solament', 'saber', 'saps', 'sap', 'sabem', 'sabeu', 'saben', 'ultim', 'llarg', 'bastant', 'fas', 'molts', 'aquells', 'aquelles', 'seus', 'llavors', 'sota', 'dalt', 'us', 'molt', 'era', 'eres', 'erem', 'eren', 'mode', 'be', 'quant', 'quan', 'on', 'mentre', 'qui', 'amb', 'entre', 'sense', 'jo', 'aquell']
	else:
		stopwords_accented = nltk.corpus.stopwords.words(language)
		stopwords = [replace_accented(w) for w in stopwords_accented]
	removed = [w for w in words if w not in stopwords]
	return removed

def porter_stem(words):
	pstemmer = nltk.stem.porter.PorterStemmer()
	pstemmed = [pstemmer.stem(w) for w in words]
	return pstemmed

def lancaster_stem(words):
	lstemmer = nltk.stem.lancaster.LancasterStemmer()
	lstemmed = [lstemmer.stem(w) for w in words]
	return lstemmed

def snowball_stem(language, words):
	if language == 'English':
		stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=False)
	elif language == 'Spanish':
		stemmer = nltk.stem.snowball.SpanishStemmer(ignore_stopwords=False)
	else:
		return words
	stemmed = [stemmer.stem(w) for w in words]
	return stemmed

# def parse_data(input_file):
# 	'''
# 	Parse the .xml dev data file

# 	param str input_file: The input data file path
# 	return dict: A dictionary with the following structure
# 		{
# 			lexelt: [(instance_id, context), ...],
# 			...
# 		}
# 	'''
# 	xmldoc = minidom.parse(input_file)
# 	data = {}
# 	lex_list = xmldoc.getElementsByTagName('lexelt')
# 	for node in lex_list:
# 		lexelt = format_str(node.getAttribute('item'))
# 		data[lexelt] = []
# 		inst_list = node.getElementsByTagName('instance')
# 		for inst in inst_list:
# 			instance_id = inst.getAttribute('id')
# 			l = inst.getElementsByTagName('context')[0]
# 			context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
# 			data[lexelt].append((instance_id, context))
# 	return data

def build_train_vectors(language):
	'''
	##############
	'''
	data = {} 
	input_file = 'data/' + language + '-train.xml'
	xmldoc = minidom.parse(input_file)
	lex_list = xmldoc.getElementsByTagName('lexelt')
	for node in lex_list:
		lexelt = replace_accented(node.getAttribute('item'))
		data[lexelt] = ()
		inst_list = node.getElementsByTagName('instance')
		sense_ids = []
		contexts = []

		for inst in inst_list:
			#instance_id = inst.getAttribute('id')
			sense_id = replace_accented(inst.getElementsByTagName('answer')[0].getAttribute('senseid'))
			
			#FEAT: skip senses with senseid "U" (English only)
			if sense_id == 'U':
				continue
			
			if language == 'English':
				l = inst.getElementsByTagName('context')[0]
			else:
				l = inst.getElementsByTagName('context')[0].getElementsByTagName('target')[0]
			
			left_str = l.childNodes[0].nodeValue.replace('\n', '')
			right_str = l.childNodes[2].nodeValue.replace('\n', '')

			left = nltk.word_tokenize(apply_features(left_str))
			right = nltk.word_tokenize(apply_features(right_str))

			left_k = left[-10:]
			right_k = right[0:10]
			context = []
			context = left_k + right_k

			# FEAT: remove stopwords
			context = remove_stopwords(language, context)

			# FEAT: stemming
			# if language == 'English':
				# context = porter_stem(context)
				# context = lancaster_stem(context)
			context = snowball_stem(language, context)


			print context
			
			sense_ids.append(sense_id.encode('utf-8', 'ignore'))
			contexts.append(context)

			# print lexelt
			# print context
		
		# remove duplicate items in all_context_words to create s
		s = []
		for each_context in contexts:
			for each_word in each_context:
				if each_word not in s:
					s.append(each_word)

		# # Calculate context vectors with respect to s
		context_vectors = build_context_vectors(s, contexts)

		data[lexelt] = (s, sense_ids, context_vectors)
		# print lexelt
		# print sense_ids
		# print context_vectors

	return data

def build_dev_data(language):
	'''
	##############
	'''
	data = {} 
	input_file = 'data/' + language + '-dev.xml'
	xmldoc = minidom.parse(input_file)
	lex_list = xmldoc.getElementsByTagName('lexelt')
	for node in lex_list:
		lexelt = replace_accented(node.getAttribute('item'))
		data[lexelt] = []
		inst_list = node.getElementsByTagName('instance')

		for inst in inst_list:
			instance_id = replace_accented(inst.getAttribute('id'))
			if language == 'English':
				l = inst.getElementsByTagName('context')[0]
			else:
				l = inst.getElementsByTagName('context')[0].getElementsByTagName('target')[0]

			left_str = l.childNodes[0].nodeValue.replace('\n', '')
			right_str = l.childNodes[2].nodeValue.replace('\n', '')

			left = nltk.word_tokenize(apply_features(left_str))
			right = nltk.word_tokenize(apply_features(right_str))

			left_k = left[-10:]
			right_k = right[0:10]
			context = []
			context = left_k + right_k

			# FEAT: remove stopwords
			context = remove_stopwords(language, context)

			# FEAT: stemming
			# if language == 'English':
			# 	context = porter_stem(context)
			# 	context = lancaster_stem(context)
			context = snowball_stem(language, context)

		# # Calculate context vectors with respect to s
		# context_vectors = []
		# for each_context in contexts:
		# 	# Initialize context vector: all zeros
		# 	context_vector = [0] * len(s)
		# 	for each_word in each_context:
		# 		if each_word in s:
		# 			idx = s.index(each_word)
		# 			context_vector[idx] += 1

		# 	context_vectors.append(context_vector)

			data[lexelt].append((instance_id.encode('utf-8', 'ignore'), context))

	return data

def build_context_vectors(s, contexts):
	context_vectors = []
	for each_context in contexts:
		context_vector = [0] * len(s)
		for each_word in each_context:
			if each_word in s:
				idx = s.index(each_word)
				context_vector[idx] +=1
		context_vectors.append(context_vector)
	return context_vectors

def train_svm(data, targets):
	svm_clf = svm.LinearSVC()
	svm_clf.fit(data, targets)
	return svm_clf

def train_kneighbors(data, targets):
	kneighbors_clf = neighbors.KNeighborsClassifier()
	kneighbors_clf.fit(data, targets)
	return kneighbors_clf

def build_dict(train_data):
	model_dict = {}
	for lexelt in train_data:
		sense_ids = train_data[lexelt][1]
		context_vectors = train_data[lexelt][2]
		svm_clf = train_svm(context_vectors, sense_ids)
		kneighbors_clf = train_kneighbors(context_vectors, sense_ids)
		model_dict[lexelt] = (svm_clf, kneighbors_clf)
	return model_dict

def disambiguate(language, model, train_data, dev_data):
	outfile_svm = codecs.open(language + '-svm.answer', encoding = 'utf-8', mode = 'w')
	outfile_kneighbors = codecs.open(language + '-k.answer', encoding = 'utf-8', mode = 'w')
	for lexelt, id_ctxt_tuples in sorted(dev_data.iteritems(), key = lambda d: replace_accented(d[0].split('.')[0])):
		for each_id_ctxt_tuple in sorted(id_ctxt_tuples, key = lambda d: int(d[0].split('.')[-1])):
			instance_id = each_id_ctxt_tuple[0]
			context = each_id_ctxt_tuple[1]
			# Build context vector
			s = train_data[lexelt][0]
			context_vector = build_context_vectors(s, [context])[0]
			# print s
			# print context_vector
			print lexelt
			print instance_id
			# Predict
			svm_predict_sense_id = model[lexelt][0].predict(context_vector)[0]
			kneighbors_predict_sense_id = model[lexelt][1].predict(context_vector)[0]

			print svm_predict_sense_id
			print kneighbors_predict_sense_id

			# output
			# outfile_svm.write(replace_accented(lexelt) + ' ' + replace_accented(instance_id) + ' ' + replace_accented(svm_predict_sense_id) + '\n')
			# outfile_kneighbors.write(replace_accented(lexelt) + ' ' + replace_accented(instance_id) + ' ' + replace_accented(kneighbors_predict_sense_id) + '\n')
			outfile_svm.write(lexelt + ' ' + instance_id + ' ' + svm_predict_sense_id + '\n')
			outfile_kneighbors.write(lexelt + ' ' + instance_id + ' ' + kneighbors_predict_sense_id + '\n')
	outfile_svm.close()
	outfile_kneighbors.close()

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print 'Usage: python main.py [language]'
		sys.exit(0)
	
	language = sys.argv[1]
	train_data = build_train_vectors(language)
	model = build_dict(train_data)
	dev_data = build_dev_data(language)
	disambiguate(language, model, train_data, dev_data)
	







	# for a_lexelt, a_tuple in data.iteritems():
	# 	print a_lexelt
	# 	print a_tuple
	# 	print '\n'


