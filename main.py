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

def format_str(input_str):
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

def calculate_context_vectors(language):
	'''
	##############
	'''
	data = {} 
	input_file = 'data/' + language + '-train.xml'
	xmldoc = minidom.parse(input_file)
	lex_list = xmldoc.getElementsByTagName('lexelt')
	for node in lex_list:
		all_context_words = []
		lexelt = format_str(node.getAttribute('item'))
		data[lexelt] = ()
		sense_ids = []
		contexts = []
		inst_list = node.getElementsByTagName('instance')

		for inst in inst_list:
			#instance_id = inst.getAttribute('id')
			sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
			l = inst.getElementsByTagName('context')[0]
			left = nltk.word_tokenize(format_str(l.childNodes[0].nodeValue))
			right = nltk.word_tokenize(format_str(l.childNodes[2].nodeValue.replace('\n', '')))
			left_k = left[-10:]
			right_k = right[0:10]
			context = []
			context = left_k + right_k
			
			sense_ids.append(sense_id)
			contexts.append(context)
		
		# remove duplicate items in all_context_words to create s
		s = []
		for each_context in contexts:
			for each_word in each_context:
				if each_word not in s:
				s.append(each_word)

		# Calculate context vectors with respect to s
		for each_context in contexts:
			# Initialize context vector: all zeros
			context_vector = [0] * len(s)
			for each_word in each_context:
				idx = s.index(each_word)
				context_vector[idx] += 1

			context_vectors.append(context_vector)

		data[lexelt] = (s, sense_ids, context_vectors)

	return data


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print 'Usage: python main.py [language]'
		sys.exit(0)
	data = calculate_context_vectors('data/' + sys.argv[1] + '-train.xml')
	for a_lexelt, a_tuple in data.iteritems():
		print a_lexelt
		print a_tuple
		print '\n'


