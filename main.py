from xml.dom import minidom
import json
import codecs
import sys
import unicodedata
import nltk
import string

def remove_punctuation(input_str):
	table = string.maketrans('', '')
	return input_str.translate(table, string.punctuation)

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def calculate_context_vectors(input_file):
	'''
	Parse the .xml dev data file

	param str input_file: The input data file path
	return dict: A dictionary with the following structure
		{
			lexelt: [(instance_id, context), ...],
			...
		}
	'''
	xmldoc = minidom.parse(input_file)
	data = {} 
	lex_list = xmldoc.getElementsByTagName('lexelt')
	for node in lex_list:
		all_context_words = []
		lexelt = node.getAttribute('item')
		data[lexelt] = ()
		sense_ctxt_dict = {}
		inst_list = node.getElementsByTagName('instance')

		for inst in inst_list:
			#instance_id = inst.getAttribute('id')
			sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
			l = remove_punctuation(inst.getElementsByTagName('context')[0])
			left = nltk.word_tokenize(l.childNodes[0].nodeValue)
			right = nltk.word_tokenize(l.childNodes[2].nodeValue.replace('\n', ''))
			left_k = left[-10:]
			right_k = right[0:10]
			context = []
			context = left_k + right_k
			# Append words in current context to all_context_words
			all_context_words += context
			
			if sense_id in sense_ctxt_dict:
				sense_ctxt_dict[sense_id] += context
			else:
				sense_ctxt_dict[sense_id] = context
		
		# remove duplicate items in all_context_words to create s
		s = []
		for each_word in all_context_words:
			if each_word not in s:
				s.append(each_word)

		# Calculate context vectors with respect to s
		sense_ids = []
		context_vectors = []
		for a_sense_id, a_context in sense_ctxt_dict.iteritems():
			# Initialize context vector: all zeros
			context_vector = [0] * len(s)
			for each_word in a_context:
				idx = s.index(each_word)
				context_vector[idx] += 1

			sense_ids.append(a_sense_id)
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


