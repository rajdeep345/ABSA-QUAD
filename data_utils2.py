# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
					'food prices',
					'food quality',
					'food general',
					'ambience general',
					'service general',
					'restaurant prices',
					'drinks prices',
					'restaurant miscellaneous',
					'drinks quality',
					'drinks style_options',
					'restaurant general',
					'food style_options']


def read_line_examples_from_file(data_path, data_type, task):
	"""
	If the task is 'aste', then read data from two files:
	1. data_type.sent: Contains each sentence in a separate line
	2. data_type.tup: Contains all triplets corresponding to a sentence in a separate line

	If the task is 'asqp', then read data from file, each line is: sent####labels
		
	Return List[List[word]], List[Tuple]
	"""
	sents, labels = [], []

	if task == 'aste':	
		with open(f'{data_path}/{data_type}.sent', 'r', encoding='UTF-8') as fp:
			for line in fp:
				line = line.strip()
				if line != '':
					sents.append(line.split())		
		with open(f'{data_path}/{data_type}.tup', 'r', encoding='UTF-8') as fp:
			for line in fp:
				line = line.strip()
				if line != '':
					triplets = []
					_triplets = line.split('|')
					for t in _triplets:
						triplet = t.split(';')
						triplets.append(triplet)				
					labels.append(triplets)	
	elif task == 'asqp' or task == 'acos':
		with open(f'{data_path}/{data_type}.txt', 'r', encoding='UTF-8') as fp:
			words, labels = [], []
			for line in fp:
				line = line.strip()
				if line != '':
					words, tuples = line.split('####')
					sents.append(words.split())
					labels.append(eval(tuples))
	elif task == 'caves':
		with open(f'{data_path}/{data_type}.json') as fp:
			lines = json.load(fp)
			for line in enumerate(lines):
				_, values = line
				sents.append(values['tweet'].strip().split())
				tuples = []
				for k,v in values['labels'].items():
					tuples.append([k, v[0]['terms'].strip()])
				labels.append(tuples)

	assert len(sents) == len(labels)	
	print(f"Total examples = {len(sents)}")
	return sents, labels


def get_para_caves_targets(sents, labels, target_mode):
	targets = []
	for label in labels:
		all_target_sentences = []
		for _tuple in label:
			reason, explanation = [elem.strip() for elem in _tuple]

			if target_mode == 'para':
				if reason == 'none':  # if no specific anti-vax concern is mentioned
					sent = "reason for not taking vaccine is not mentioned explicitly"
				else:
					sent = f"reason for not taking vaccine is {reason} because {explanation}"
			
			all_target_sentences.append(sent)

		target = ' [SSEP] '.join(all_target_sentences)
		targets.append(target)
	return targets


def get_para_aste_targets(sents, labels, target_mode):
	targets = []
	for label in labels:
		all_tri_sentences = []
		for triplet in label:
			at, ot, sp = [elem.strip() for elem in triplet]
			
			if at == 'NULL':  # for implicit aspect term
				at = 'it'
			if ot == 'NULL':  # for implicit opinion term
				ot = 'implied'
			
			if target_mode == 'para':
				sp = senttag2opinion[sp]	# 'POS' -> 'good'
				one_tri = f"It is {sp} because {at} is {ot}"
			
			elif target_mode == 'temp':
				sp = senttag2word[sp]	# 'POS' -> 'positive'
				# one_tri = f"[ASPECT] {at} [OPINION] {ot} [SENTIMENT] {sp}"
				one_tri = f"<aspect> {at} <opinion> {ot} <sentiment> {sp}"
			
			all_tri_sentences.append(one_tri)

		target = ' [SSEP] '.join(all_tri_sentences)
		targets.append(target)
	return targets


def get_para_asqp_targets(sents, labels, target_mode):
	"""
	Obtain the target sentence under the "target_mode"" paradigm
	"""
	targets = []
	for label in labels:
		all_quad_sentences = []
		for quad in label:
			at, ac, sp, ot = quad

			if at == 'NULL':  # for implicit aspect term
				at = 'it'			

			if target_mode == 'para':
				man_ot = sentword2opinion[sp]  # 'positive': 'great'
				one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"

			elif target_mode == 'temp':
				# added here because this is not a contribution by PARAPHRASE paper.
				if ot == 'NULL':  # for implicit opinion term
					ot = 'implied'
				one_quad_sentence = f"<aspect> {at} <category> {ac} <opinion> {ot} <sentiment> {sp}"

			all_quad_sentences.append(one_quad_sentence)

		target = ' [SSEP] '.join(all_quad_sentences)
		targets.append(target)
	return targets


def get_transformed_io(data_path, data_type, task, target_mode):
	"""
	The main function to transform input & target according to the task
	"""
	sents, labels = read_line_examples_from_file(data_path, data_type, task)

	# the input is just the raw sentence
	inputs = [s.copy() for s in sents]

	if task == 'aste':
		targets = get_para_aste_targets(sents, labels, target_mode)
	elif task == 'asqp' or task == 'acos':
		targets = get_para_asqp_targets(sents, labels, target_mode)
	elif task == 'caves':
		targets = get_para_caves_targets(sents, labels, target_mode)
	else:
		raise NotImplementedError

	return inputs, targets


class ABSADataset(Dataset):
	def __init__(self, tokenizer, data_dir, data_type, task, target_mode, max_len=128):
		self.data_path = f'data_{task}' if task == 'caves' else f'data_{task}/{data_dir}'
		self.data_type = data_type
		self.task = task
		self.target_mode = target_mode
		self.max_len = max_len
		self.tokenizer = tokenizer
		self.data_dir = data_dir

		self.inputs = []
		self.targets = []

		self._build_examples()

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, index):
		source_ids = self.inputs[index]["input_ids"].squeeze()
		target_ids = self.targets[index]["input_ids"].squeeze()

		src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
		target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

		return {"source_ids": source_ids, "source_mask": src_mask, 
				"target_ids": target_ids, "target_mask": target_mask}

	def _build_examples(self):

		inputs, targets = get_transformed_io(self.data_path, self.data_type, self.task, self.target_mode)

		for i in range(len(inputs)):
			# change input and target to two strings
			input = ' '.join(inputs[i])
			target = targets[i]

			tokenized_input = self.tokenizer.batch_encode_plus(
			  [input], max_length=self.max_len, padding="max_length",
			  truncation=True, return_tensors="pt"
			)
			tokenized_target = self.tokenizer.batch_encode_plus(
			  [target], max_length=self.max_len, padding="max_length",
			  truncation=True, return_tensors="pt"
			)

			self.inputs.append(tokenized_input)
			self.targets.append(tokenized_target)
