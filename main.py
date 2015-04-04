from xml.dom import minidom
import json
import codecs
import sys
import unicodedata
import nltk

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
		s = []
		lexelt = node.getAttribute('item')
		data[lexelt] = ()
		sense_ctxt_dict = {}
		inst_list = node.getElementsByTagName('instance')

		for inst in inst_list:
			#instance_id = inst.getAttribute('id')
			sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
			l = inst.getElementsByTagName('context')[0]
			left = nltk.word_tokenize(l.childNodes[0].nodeValue)
			right = nltk.word_tokenize(l.childNodes[2].nodeValue.replace('\n', ''))
			left_k = left[-10:]
			right_k = right[0:10]
			context = []
			context.append(left_k)
			context.append(right_k)
			print context
			# Append words in current context to s
			s.append(context)
			
			if sense_id in sense_ctxt_dict:
				sense_ctxt_dict[sense_id].append(context)
			else:
				sense_ctxt_dict[sense_id] = context
		
		# remove duplicate items in s
		s = list(set(s))

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
		print 'Usage: python baseline.py [language]'
		sys.exit(0)
	data = calculate_context_vectors('data/' + sys.argv[1] + '-train.xml')
	for a_lexelt, a_tuple in data.iteritems():
		print a_lexelt
		print a_tuple
		print '\n'


