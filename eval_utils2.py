# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
						  'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}

def extract_spans_para(task, target_mode, seq, seq_type):
	tuples = []
	sents = [s.strip() for s in seq.split('[SSEP]')]
	if task == 'aste':
		if target_mode == 'para':
			for s in sents:
				# It is bad because editing is problem.
				try:
					sp, atot = s.split(' because ')
					sp = opinion2word.get(sp[6:], 'nope')    # 'good' -> 'positive'
					at, ot = atot.split(' is ')					
				except ValueError:
					# print(f'In {seq_type} seq, cannot decode: {s}')
					at, ot, sp = '', '', ''
				tuples.append((at, ot, sp))
		elif target_mode == 'temp':
			for s in sents:
				# <aspect> It/pizza <opinion> over cooked <sentiment> negative
				if '<aspect>' in s and '<opinion>' in s and '<sentiment>' in s:
					at = s.split('<opinion>')[0].split('<aspect>')[1].strip()
					ot = s.split('<opinion>')[1].split('<sentiment>')[0].strip()
					sp = s.split('<sentiment>')[1].strip()
					# if the aspect term is implicit
					if at.lower() == 'it':
						at = 'NULL'
					# if the opinion term is implicit
					if ot.lower() == 'implied':
						ot = 'NULL'
				else:
					print(f'Cannot decode: {s}')
					at, ot, sp = '', '', ''
				tuples.append((at, ot, sp))
	elif task == 'asqp' or task == 'acos':
		if target_mode == 'para':
			for s in sents:
				# food quality is bad because pizza is over cooked.
				try:
					ac_sp, at_ot = s.split(' because ')
					ac, sp = ac_sp.split(' is ')
					at, ot = at_ot.split(' is ')

					# if the aspect term is implicit
					if at.lower() == 'it':
						at = 'NULL'
				except ValueError:
					try:
						# print(f'In {seq_type} seq, cannot decode: {s}')
						pass
					except UnicodeEncodeError:
						# print(f'In {seq_type} seq, a string cannot be decoded')
						pass
					ac, at, sp, ot = '', '', '', ''
				tuples.append((ac, at, sp, ot))
		elif target_mode == 'temp':
			for s in sents:
				# <aspect> It/pizza <category> food quality <opinion> over cooked <sentiment> negative
				if '<aspect>' in s and '<category>' in s and '<opinion>' in s and '<sentiment>' in s:
					at = s.split('<category>')[0].split('<aspect>')[1].strip()
					ac = s.split('<category>')[1].split('<opinion>')[0].strip()
					ot = s.split('<opinion>')[1].split('<sentiment>')[0].strip()
					sp = s.split('<sentiment>')[1].strip()
					# if the aspect term is implicit
					if at.lower() == 'it':
						at = 'NULL'
					# if the opinion term is implicit
					if ot.lower() == 'implied':
						ot = 'NULL'
				else:
					print(f'Cannot decode: {s}')
					ac, at, sp, ot = '', '', '', ''
				tuples.append((ac, at, sp, ot))

	elif task == 'caves':
		if target_mode == 'para':
			for s in sents:
				# reason for not taking vaccine is {unnecessary} because {Covid cases are mild/assymptomatic}
				if s.startswith('reason for not taking vaccine is'):
					reason = s.strip().split()[6]
					if reason == 'none' or reason == 'not':
						reason = 'none'
					if reason == 'none':
						expln = ''
					else:
						expln = ' '.join(s.strip().split()[8:])			
				else:
					print(f'Cannot decode: {s}')
					reason, expln = '', ''
				tuples.append((reason, expln))

	elif task == 'hateXplain':
		if target_mode == 'para':
			for s in sents:
				# the expressed stance is {hatespeech} because {disgusting kike language exterminate the goyim}
				if s.startswith('the expressed stance is'):
					op = s.strip().split()[4]
					if op == 'normal':
						expln = ''
					else:
						expln = ' '.join(s.strip().split()[6:])			
				else:
					print(f'Cannot decode: {s}')
					op, expln = '', ''
				tuples.append((op, expln))
	else:
		raise NotImplementedError
	return tuples


def compute_f1_scores(pred_pt, gold_pt):
	"""
	Function to compute F1 scores with pred and gold quads
	The input needs to be already processed
	"""
	# number of true postive, gold standard, predictions
	n_tp, n_gold, n_pred = 0, 0, 0

	for i in range(len(pred_pt)):
		n_gold += len(gold_pt[i])
		n_pred += len(pred_pt[i])

		for t in gold_pt[i]:
			if t in pred_pt[i]:
				n_tp += 1

	print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
	precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
	recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
	f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
	scores = {'precision': precision, 'recall': recall, 'f1': f1}

	return scores


def compute_scores(pred_seqs, gold_seqs, sents, task, target_mode):
	"""
	Compute model performance
	"""
	assert len(pred_seqs) == len(gold_seqs)
	num_samples = len(gold_seqs)

	all_labels, all_preds = [], []

	for i in range(num_samples):
		gold_list = extract_spans_para(task, target_mode, gold_seqs[i], 'gold')
		pred_list = extract_spans_para(task, target_mode, pred_seqs[i], 'pred')

		all_labels.append(gold_list)
		all_preds.append(pred_list)

	print("\nResults:")
	scores = compute_f1_scores(all_preds, all_labels)
	print(scores)

	return scores, all_labels, all_preds
