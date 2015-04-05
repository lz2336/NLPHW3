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
		lexelt = node.getAttribute('item')
		data[lexelt] = ()
		inst_list = node.getElementsByTagName('instance')
		sense_ids = []
		contexts = []

		for inst in inst_list:
			#instance_id = inst.getAttribute('id')
			sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
			
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
			
			sense_ids.append(sense_id)
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
		lexelt = node.getAttribute('item')
		data[lexelt] = []
		inst_list = node.getElementsByTagName('instance')

		for inst in inst_list:
			instance_id = inst.getAttribute('id')
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

			data[lexelt].append((instance_id, context))

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
			print s
			print context_vector
			# Predict
			svm_predict_sense_id = model[lexelt][0].predict(context_vector)[0]
			kneighbors_predict_sense_id = model[lexelt][1].predict(context_vector)[0]

			# output
			outfile_svm.write(replace_accented(lexelt) + ' ' + replace_accented(instance_id) + ' ' + replace_accented(svm_predict_sense_id) + '\n')
			outfile_kneighbors.write(replace_accented(lexelt) + ' ' + replace_accented(instance_id) + ' ' + replace_accented(kneighbors_predict_sense_id) + '\n')
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


