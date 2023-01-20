# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils2 import ABSADataset
from data_utils2 import read_line_examples_from_file
from eval_utils2 import compute_scores


logger = logging.getLogger(__name__)


def custom_print(*msg):
	for i in range(0, len(msg)):
		if i == len(msg) - 1:
			print(msg[i])
			custom_logger.write(str(msg[i]) + '\n')
		else:
			print(msg[i], ' ', end='')
			custom_logger.write(str(msg[i]))


def init_args():
	parser = argparse.ArgumentParser()
	# basic settings
	parser.add_argument("--task", default='asqp', type=str, required=True,
						help="The name of the task, selected from: [aste, asqp, acos]")
	parser.add_argument("--target_mode", default='para', type=str, required=True,
						help="The mode in which the target is to be framed, selected from: [para, temp]")
	parser.add_argument("--dataset", default='rest15', type=str, required=True,
						help="The name of the dataset, selected from: [lap14, rest14, rest15, rest16, laptop, rest]")
	parser.add_argument("--model_name_or_path", default='t5-base', type=str,
						help="Path to pre-trained model or shortcut name")
	parser.add_argument("--do_train", action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true',
						help="Whether to run eval on the dev/test set.")
	parser.add_argument("--do_direct_eval", action='store_true', 
						help="Whether to run eval on the dev/test set.")
	parser.add_argument("--do_inference", action='store_true', 
						help="Whether to run inference with trained checkpoints")

	# other parameters
	parser.add_argument("--max_seq_length", default=128, type=int)
	parser.add_argument("--n_gpu", default=0)
	parser.add_argument("--train_batch_size", default=16, type=int,
						help="Batch size per GPU/CPU for training.")
	parser.add_argument("--eval_batch_size", default=16, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--learning_rate", default=3e-4, type=float)
	parser.add_argument("--num_train_epochs", default=20, type=int, 
						help="Total number of training epochs to perform.")
	parser.add_argument('--seed', type=int, default=42,
						help="random seed for initialization")

	# multi-task details
	parser.add_argument("--use_tagger", default = False, type = bool) 
	parser.add_argument("--use_reg", default = False, type = bool)
	parser.add_argument("--alpha", default=1.0, type=float)
	parser.add_argument("--beta", default=0.2, type=float)

	# k-shot experimentation
	parser.add_argument('--k_shot', default=False, type=bool, help="whether to perform low-resource k-shot experiments")

	# training details
	parser.add_argument("--weight_decay", default=0.0, type=float)
	parser.add_argument("--adam_epsilon", default=1e-8, type=float)
	parser.add_argument("--warmup_steps", default=0.0, type=float)

	args = parser.parse_args()

	# set up output dir which looks like './outputs/rest15/'
	if not os.path.exists('./outputs'):
		os.mkdir('./outputs')

	task_dir = f"./outputs/{args.task}"
	if not os.path.exists(task_dir):
		os.mkdir(task_dir)

	output_dir = f"{task_dir}/{args.dataset}"
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	args.output_dir = output_dir

	return args


def get_dataset(tokenizer, type_path, args):
	return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=type_path,
					   task=args.task, target_mode=args.target_mode, max_len=args.max_seq_length)

def load_model_weights(model, new_checkpoint):
	model.load_state_dict(torch.load(new_checkpoint))
	model.train() 
	return model

class T5FineTuner(pl.LightningModule):
	"""
	Fine tune a pre-trained T5 model
	"""
	def __init__(self, hparams, tfm_model, tokenizer):
		super(T5FineTuner, self).__init__()
		self.hparams = hparams
		self.model = tfm_model
		self.tokenizer = tokenizer

		if args.use_tagger:
			### Tagger
			self.classifier = nn.Linear(768, 3)  ## 3 labels, B, I, and O.
			self.softmax = nn.Softmax(dim=2)
			self.tag_criterion = nn.CrossEntropyLoss(ignore_index=-100)
			self.token_dropout = nn.Dropout(0.1)
			self.alpha = args.alpha

		if args.use_reg:
			### Regressor
			self.regressor_layer = nn.Linear(768, 128)
			self.relu1 = nn.ReLU()
			self.ff1 = nn.Linear(128, 64)
			self.tanh1 = nn.Tanh()
			self.ff2 = nn.Linear(64, 1)
			self.regressor_criterion = nn.MSELoss()
			self.beta = args.beta

	def is_logger(self):
		return True

	def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
				decoder_attention_mask=None, labels=None):
		return self.model(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			labels=labels,
		)

	def _step(self, batch):
		lm_labels = batch["target_ids"]
		lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

		outputs = self(
			input_ids=batch["source_ids"],
			attention_mask=batch["source_mask"],
			labels=lm_labels,
			decoder_attention_mask=batch['target_mask']
		)
		# print(outputs)

		loss = outputs[0]
		# print(loss, "loss before tagger")

		if args.use_tagger:
			encoder_states = outputs.encoder_last_hidden_state
			logits = self.classifier(self.token_dropout(encoder_states))			
			tag_loss = self.tag_criterion(logits.view(-1, 3), batch['op_tags'].view(-1))		
			loss += self.alpha * tag_loss
			# print(loss, "loss after tagger")

		if args.regressor:
			encoder_states = outputs.encoder_last_hidden_state
			mask_position = torch.tensor(np.where(batch["source_ids"].cpu().numpy() == 1, 1, 0)).to(device)
			print(sum(mask_position))
			masked_embeddings = encoder_states * mask_position.unsqueeze(2)
			sentence_embedding = torch.sum(masked_embeddings, axis=1)
			normalized_sentence_embeddings = sentence_embedding

			outs = self.regressor_layer(self.token_dropout(normalized_sentence_embeddings))
			outs = self.relu1(outs)
			outs = self.ff1(outs)
			outs = self.tanh1(outs)
			outs = self.ff2(outs)

			regressor_loss = self.regressor_criterion(outs, batch['triplet_count'].view(-1).type_as(outs))
			loss += self.beta * regressor_loss
			# print(loss, "loss after regression")

		return loss

	def _generate(self, batch):

		"""
		Compute scores given the predictions and gold labels
		"""
		device = torch.device(f'cuda:{args.n_gpu}')
		model.model.to(device)

		model.model.eval()

		outputs, targets = [], []

		for batch in tqdm(data_loader):
			# need to push the data to device
			outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
										attention_mask=batch['source_mask'].to(device), 
										max_length=128)  # num_beams=8, early_stopping=True)

			dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
			target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
			
			outputs.extend(dec)
			targets.extend(target)

		outs = self.model.generate(input_ids=batch['source_ids'], 
							attention_mask=batch['source_mask'], 
							max_length=128)
		outputs = []
		targets = []
		#print(outs)
		for i in range(len(outs)):

			dec = tokenizer.decode(outs[i], skip_special_tokens=False)
			labels = np.where(batch["target_ids"][i].cpu().numpy() != -100, batch["target_ids"][i].cpu().numpy(), tokenizer.pad_token_id)
			target = tokenizer.decode(torch.tensor(labels), skip_special_tokens=False)

			outputs.append(dec)
			targets.append(target)

		decoded_labels = correct_spaces(targets)
		decoded_preds = correct_spaces(outputs)
		# print('decoded_preds', decoded_preds)
		# print('decoded_labels', decoded_labels)

		linearized_triplets = {}
		linearized_triplets['predictions'] = decoded_preds
		linearized_triplets['labels'] = decoded_labels

		return linearized_triplets

	def training_step(self, batch, batch_idx):
		loss = self._step(batch)
		tensorboard_logs = {"train_loss": loss}
		return {"loss": loss, "log": tensorboard_logs}

	def training_epoch_end(self, outputs):
		avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
		tensorboard_logs = {"avg_train_loss": avg_train_loss}
		return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

	def validation_step(self, batch, batch_idx):
		loss = self._step(batch)
		return {"val_loss": loss}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
		tensorboard_logs = {"val_loss": avg_loss}
		return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

	def configure_optimizers(self):
		""" Prepare optimizer and schedule (linear warmup and decay) """
		model = self.model
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self.hparams.weight_decay,
			},
			{
				"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
				"weight_decay": 0.0,
			},
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
		self.opt = optimizer
		return [optimizer]

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
		if self.trainer.use_tpu:
			xm.optimizer_step(optimizer)
		else:
			optimizer.step()
		optimizer.zero_grad()
		self.lr_scheduler.step()

	def get_tqdm_dict(self):
		tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
		return tqdm_dict

	def train_dataloader(self):
		train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
		dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
								drop_last=True, shuffle=True, num_workers=4)
		t_total = (
			(len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
			// self.hparams.gradient_accumulation_steps
			* float(self.hparams.num_train_epochs)
		)
		scheduler = get_linear_schedule_with_warmup(
			self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
		)
		self.lr_scheduler = scheduler
		return dataloader

	def val_dataloader(self):
		val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
		return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
	def on_validation_end(self, trainer, pl_module):
		logger.info("***** Validation results *****")
		if pl_module.is_logger():
			metrics = trainer.callback_metrics
		# Log results
		for key in sorted(metrics):
			if key not in ["log", "progress_bar"]:
				logger.info("{} = {}\n".format(key, str(metrics[key])))

	def on_test_end(self, trainer, pl_module):
		logger.info("***** Test results *****")

		if pl_module.is_logger():
			metrics = trainer.callback_metrics

		# Log and save results to file
		output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
		with open(output_test_results_file, "w") as writer:
			for key in sorted(metrics):
				if key not in ["log", "progress_bar"]:
					logger.info("{} = {}\n".format(key, str(metrics[key])))
					writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents, task, target_mode):
	"""
	Compute scores given the predictions and gold labels
	"""
	device = torch.device(f'cuda:{args.n_gpu}')
	model.model.to(device)

	model.model.eval()

	outputs, targets = [], []

	for batch in tqdm(data_loader):
		# need to push the data to device
		outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
									attention_mask=batch['source_mask'].to(device), 
									max_length=128)  # num_beams=8, early_stopping=True)

		dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
		target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
		
		outputs.extend(dec)
		targets.extend(target)

	
	print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
	for i in [1, 5, 25, 42, 50]:
		try:
			print(f'>>Target    : {targets[i]}')
			print(f'>>Generation: {outputs[i]}')
		except UnicodeEncodeError:
			print('Unable to print due to coding error')
	print()

	scores, all_labels, all_preds = compute_scores(outputs, targets, sents, task, target_mode)
	results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
	# pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))

	return scores



# initialization
args = init_args()
print("\n", "="*30, f"NEW EXP: {args.task} on {args.dataset}", "="*30, "\n")

# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
if args.target_mode == 'temp':
	tokenizer.add_tokens(['<aspect>', '<category>', '<opinion>', '<sentiment>'], special_tokens=True)
print(f"Here is an example (from the dev set):")
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='dev',
					   task=args.task, target_mode=args.target_mode, max_len=args.max_seq_length)
data_sample = dataset[10]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=False))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=False))


# training process
if args.do_train:	
	print("\n****** Conduct Training ******")

	# initialize the T5 model
	tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
	if args.target_mode == 'temp':
		tfm_model.resize_token_embeddings(len(tokenizer))
	model = T5FineTuner(args, tfm_model, tokenizer)

	# checkpoint_callback = pl.callbacks.ModelCheckpoint(
	#     filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
	# )

	# prepare for trainer
	train_params = dict(
		default_root_dir=args.output_dir,
		accumulate_grad_batches=args.gradient_accumulation_steps,
		gpus=args.n_gpu,
		gradient_clip_val=1.0,
		max_epochs=args.num_train_epochs,
		callbacks=[LoggingCallback()],
	)

	trainer = pl.Trainer(**train_params)
	trainer.fit(model)

	# save the final model
	# model.model.save_pretrained(args.output_dir)
	# tokenizer.save_pretrained(args.output_dir)

	print("Finish training and saving the model!")


# evaluation
if args.do_direct_eval:
	print("\n****** Conduct Evaluating with the last state ******")

	# model = T5FineTuner(args)

	# print("Reload the model")
	# model.model.from_pretrained(args.output_dir)

	data_path = f'data_{args.task}/{args.dataset}'
	sents, _ = read_line_examples_from_file(data_path, 'test', args.task)

	print()
	test_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='test',
					   task=args.task, target_mode=args.target_mode, max_len=args.max_seq_length)
	test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
	# print(test_loader.device)

	# compute the performance scores
	scores = evaluate(test_loader, model, sents, args.task, args.target_mode)

	# write to file
	log_file_path = f"results_log/{args.dataset}.txt"
	local_time = time.asctime(time.localtime(time.time()))

	exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
	exp_results = f"F1 = {scores['f1']:.4f}"

	log_str = f'============================================================\n'
	log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

	if not os.path.exists('./results_log'):
		os.mkdir('./results_log')

	with open(log_file_path, "a+") as f:
		f.write(log_str)


if args.do_inference:
	print("\n****** Conduct inference on trained checkpoint ******")

	# initialize the T5 model from previous checkpoint
	print(f"Load trained model from {args.output_dir}")
	print('Note that a pretrained model is required and `do_true` should be False')
	tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
	tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

	model = T5FineTuner(args, tfm_model, tokenizer)

	data_path = f'data_{args.task}/{args.dataset}'
	sents, _ = read_line_examples_from_file(data_path, 'test', args.task)

	print()
	test_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type='test',
					   task=args.task, target_mode=args.target_mode, max_len=args.max_seq_length)
	test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
	# print(test_loader.device)

	# compute the performance scores
	scores = evaluate(test_loader, model, sents, agrs.task, args.target_mode)

	# write to file
	log_file_path = f"results_log/{args.dataset}.txt"
	local_time = time.asctime(time.localtime(time.time()))

	exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
	exp_results = f"F1 = {scores['f1']:.4f}"

	log_str = f'============================================================\n'
	log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

	if not os.path.exists('./results_log'):
		os.mkdir('./results_log')

	with open(log_file_path, "a+") as f:
		f.write(log_str)
