'''
Quora Paraphrase Detection with GPT-2 cloze-style yes/no prediction.

This script uses only train/dev for training, prompt selection, threshold tuning,
and error analysis. The test split is loaded only after the best dev setting is
fixed, to create the submission prediction file.
'''

import argparse
import csv
import json
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data,
  paraphrase_prompt,
)
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = True


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """GPT-2 next-token model restricted to the answer tokens "no" and "yes"."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)
    self.yes_token_id = args.yes_token_id
    self.no_token_id = args.no_token_id

    for param in self.gpt.parameters():
      param.requires_grad = True

  def full_vocab_logits(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids, attention_mask)
    return self.gpt.hidden_state_to_token(outputs['last_token'])

  def forward(self, input_ids, attention_mask):
    vocab_logits = self.full_vocab_logits(input_ids, attention_mask)
    no_logits = vocab_logits[:, self.no_token_id]
    yes_logits = vocab_logits[:, self.yes_token_id]
    return torch.stack([no_logits, yes_logits], dim=1)


def ensure_dir(path):
  os.makedirs(path, exist_ok=True)


def json_safe_args(args):
  result = {}
  for key, value in vars(args).items():
    if isinstance(value, (str, int, float, bool)) or value is None:
      result[key] = value
  return result


def write_json(path, obj):
  with open(path, 'w') as f:
    json.dump(obj, f, indent=2, sort_keys=True)


def write_prediction_csv(path, rows):
  fieldnames = ['id', 'q1', 'q2', 'gold_label', 'pred_label', 'yes_score', 'no_score', 'correct']
  with open(path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_submission_csv(path, sent_ids, preds):
  ensure_dir(os.path.dirname(path) or '.')
  with open(path, 'w') as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for sent_id, pred in zip(sent_ids, preds):
      f.write(f"{sent_id}, {int(pred)} \n")


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  yes_ids = tokenizer.encode('yes', add_special_tokens=False)
  no_ids = tokenizer.encode('no', add_special_tokens=False)
  if len(yes_ids) != 1 or len(no_ids) != 1:
    raise ValueError(f"Expected single-token yes/no ids, got yes={yes_ids}, no={no_ids}")
  args.yes_token_id = yes_ids[0]
  args.no_token_id = no_ids[0]
  return args


def maybe_debug_subset(data, debug_subset):
  if debug_subset is None or debug_subset <= 0:
    return data
  return data[:debug_subset]


def add_bidirectional_examples(data):
  augmented = []
  for q1, q2, label, sent_id in data:
    augmented.append((q1, q2, label, sent_id))
    augmented.append((q2, q1, label, f"{sent_id}__rev"))
  return augmented


def word_tokens(text):
  return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_hard_negative(q1, q2, overlap_threshold):
  words1 = word_tokens(q1)
  words2 = word_tokens(q2)
  overlap = len(words1 & words2) / max(1, len(words1 | words2))
  negations = {'not', 'never', 'no'}
  negation_mismatch = bool(words1 & negations) != bool(words2 & negations)
  numbers1 = set(re.findall(r"\b\d+(?:\.\d+)?\b", q1))
  numbers2 = set(re.findall(r"\b\d+(?:\.\d+)?\b", q2))
  number_mismatch = bool(numbers1 or numbers2) and numbers1 != numbers2
  return overlap >= overlap_threshold or negation_mismatch or number_mismatch


def oversample_hard_negatives(data, factor, overlap_threshold):
  if factor <= 1:
    return data
  augmented = list(data)
  hard_count = 0
  for q1, q2, label, sent_id in data:
    if label == 0 and is_hard_negative(q1, q2, overlap_threshold):
      hard_count += 1
      for copy_idx in range(1, factor):
        augmented.append((q1, q2, label, f"{sent_id}__hard{copy_idx}"))
  print(f"Oversampled {hard_count} train-only hard negatives by factor {factor}")
  return augmented


def linear_warmup_decay_lambda(current_step, warmup_steps, total_steps):
  if current_step < warmup_steps:
    return float(current_step + 1) / max(1, warmup_steps)
  remaining_steps = max(0, total_steps - current_step)
  decay_steps = max(1, total_steps - warmup_steps)
  return float(remaining_steps) / decay_steps


def encode_pairs(tokenizer, q1, q2, args, device):
  prompts = [paraphrase_prompt(a, b, args.template_id) for a, b in zip(q1, q2)]
  encoding = tokenizer(
    prompts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=args.max_length
  )
  return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)


@torch.no_grad()
def score_batch(model, batch, device, args, tokenizer):
  b_ids = batch['token_ids'].to(device)
  b_mask = batch['attention_mask'].to(device)
  logits = model(b_ids, b_mask)
  probs = torch.softmax(logits, dim=1)
  scores = probs[:, 1]

  if args.bidirectional:
    rev_ids, rev_mask = encode_pairs(tokenizer, batch['q2'], batch['q1'], args, device)
    rev_probs = torch.softmax(model(rev_ids, rev_mask), dim=1)
    scores = (scores + rev_probs[:, 1]) / 2.0

  return scores.detach().cpu().numpy()


@torch.no_grad()
def evaluate_paraphrase(dataloader, model, device, args, tokenizer, threshold):
  model.eval()
  rows = []
  y_true, y_pred = [], []
  for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
    scores = score_batch(model, batch, device, args, tokenizer)
    preds = (scores >= threshold).astype(int)
    labels = batch['labels'].cpu().numpy().astype(int)
    for i, sent_id in enumerate(batch['sent_ids']):
      gold = int(labels[i])
      pred = int(preds[i])
      row = {
        'id': sent_id,
        'q1': batch['q1'][i],
        'q2': batch['q2'][i],
        'gold_label': gold,
        'pred_label': pred,
        'yes_score': float(scores[i]),
        'no_score': float(1.0 - scores[i]),
        'correct': int(gold == pred),
      }
      rows.append(row)
      y_true.append(gold)
      y_pred.append(pred)

  acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
  return acc, rows


@torch.no_grad()
def predict_test(dataloader, model, device, args, tokenizer, threshold):
  model.eval()
  sent_ids, preds = [], []
  for batch in tqdm(dataloader, desc='test', disable=TQDM_DISABLE):
    scores = score_batch(model, batch, device, args, tokenizer)
    batch_preds = (scores >= threshold).astype(int)
    sent_ids.extend(batch['sent_ids'])
    preds.extend([int(x) for x in batch_preds])
  return sent_ids, preds


def tune_threshold(dev_rows):
  best_threshold = 0.5
  best_acc = -1.0
  thresholds = [round(x, 2) for x in np.arange(0.30, 0.701, 0.01)]
  gold = np.array([int(row['gold_label']) for row in dev_rows])
  scores = np.array([float(row['yes_score']) for row in dev_rows])
  for threshold in thresholds:
    preds = (scores >= threshold).astype(int)
    acc = float(np.mean(preds == gold))
    if acc > best_acc:
      best_acc = acc
      best_threshold = threshold
  return best_threshold, best_acc


def word_overlap(q1, q2):
  w1 = set(re.findall(r"[a-z0-9]+", q1.lower()))
  w2 = set(re.findall(r"[a-z0-9]+", q2.lower()))
  if not w1 and not w2:
    return 0.0
  return len(w1 & w2) / max(1, len(w1 | w2))


def error_analysis(dev_rows, output_dir):
  errors = [row for row in dev_rows if int(row['correct']) == 0]
  categories = {
    'long_question_pair': [],
    'high_overlap_different_meaning': [],
    'low_overlap_same_meaning': [],
    'number_date_place_proper_noun_difference': [],
    'negation': [],
    'ambiguous_yes_score': [],
  }
  neg_words = {'not', 'never', 'no'}
  proper_or_number = re.compile(r"\b\d+\b|[A-Z][a-z]+")

  for row in errors:
    q1, q2 = row['q1'], row['q2']
    overlap = word_overlap(q1, q2)
    gold = int(row['gold_label'])
    score = float(row['yes_score'])
    if len(q1.split()) + len(q2.split()) >= 40:
      categories['long_question_pair'].append(row)
    if overlap >= 0.5 and gold == 0:
      categories['high_overlap_different_meaning'].append(row)
    if overlap <= 0.2 and gold == 1:
      categories['low_overlap_same_meaning'].append(row)
    if proper_or_number.search(q1) or proper_or_number.search(q2):
      categories['number_date_place_proper_noun_difference'].append(row)
    if neg_words & set(re.findall(r"[a-z]+", f"{q1} {q2}".lower())):
      categories['negation'].append(row)
    if 0.45 <= score <= 0.55:
      categories['ambiguous_yes_score'].append(row)

  summary = {
    'num_dev_examples': len(dev_rows),
    'num_errors': len(errors),
    'error_rate': float(len(errors) / max(1, len(dev_rows))),
    'category_counts': {name: len(rows) for name, rows in categories.items()},
  }
  write_json(os.path.join(output_dir, 'error_analysis.json'), summary)

  example_rows = []
  for name, rows in categories.items():
    for row in rows[:20]:
      item = dict(row)
      item['error_type'] = name
      example_rows.append(item)
  if example_rows:
    fieldnames = ['error_type', 'id', 'q1', 'q2', 'gold_label', 'pred_label', 'yes_score', 'no_score', 'correct']
    with open(os.path.join(output_dir, 'error_analysis_examples.csv'), 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(example_rows)
  else:
    write_prediction_csv(os.path.join(output_dir, 'error_analysis_examples.csv'), [])
  return summary


def train(args):
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  ensure_dir(args.output_dir)
  ensure_dir('predictions')

  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)
  para_train_data = maybe_debug_subset(para_train_data, args.debug_subset)
  para_dev_data = maybe_debug_subset(para_dev_data, args.debug_subset)
  para_train_data = oversample_hard_negatives(
    para_train_data,
    args.hard_negative_oversample_factor,
    args.hard_negative_overlap
  )
  if args.bidirectional:
    para_train_data = add_bidirectional_examples(para_train_data)

  train_dataset = ParaphraseDetectionDataset(para_train_data, args)
  dev_dataset = ParaphraseDetectionDataset(para_dev_data, args)
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
  dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

  tokenizer = train_dataset.tokenizer
  model = ParaphraseGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
  total_update_steps = updates_per_epoch * args.epochs
  scheduler = None
  if args.scheduler == 'linear':
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = LambdaLR(
      optimizer,
      lambda step: linear_warmup_decay_lambda(step, warmup_steps, total_update_steps)
    )

  best_dev_acc = -1.0
  best_threshold = args.threshold
  metrics = {
    'epochs': [],
    'best_dev_accuracy': None,
    'best_threshold': None,
    'test_used_for_training_or_tuning': False,
  }
  write_json(os.path.join(args.output_dir, 'config.json'), json_safe_args(args))

  for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0
    num_steps = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f'train-{epoch}', disable=TQDM_DISABLE), start=1):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)

      if args.restricted_yes_no_loss:
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels, reduction='mean')
      else:
        vocab_logits = model.full_vocab_logits(b_ids, b_mask)
        target_ids = torch.where(
          labels == 1,
          torch.full_like(labels, args.yes_token_id),
          torch.full_like(labels, args.no_token_id)
        )
        loss = F.cross_entropy(vocab_logits, target_ids, reduction='mean')

      (loss / args.grad_accum_steps).backward()
      if step % args.grad_accum_steps == 0 or step == len(train_loader):
        if args.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
          scheduler.step()
        optimizer.zero_grad()

      train_loss += loss.item()
      num_steps += 1

    train_loss = train_loss / max(1, num_steps)
    dev_acc, dev_rows = evaluate_paraphrase(dev_loader, model, device, args, tokenizer, args.threshold)
    epoch_threshold = args.threshold
    threshold_acc = dev_acc
    if args.threshold_tuning:
      epoch_threshold, threshold_acc = tune_threshold(dev_rows)

    metrics['epochs'].append({
      'epoch': epoch,
      'train_loss': train_loss,
      'dev_accuracy_at_default_threshold': dev_acc,
      'dev_accuracy': threshold_acc,
      'threshold': epoch_threshold,
      'learning_rate': optimizer.param_groups[0]['lr'],
    })

    if threshold_acc > best_dev_acc:
      best_dev_acc = threshold_acc
      best_threshold = epoch_threshold
      save_model(model, optimizer, args, args.filepath)
      final_acc, final_rows = evaluate_paraphrase(dev_loader, model, device, args, tokenizer, best_threshold)
      write_prediction_csv(os.path.join(args.output_dir, 'dev_predictions.csv'), final_rows)
      write_prediction_csv(os.path.join(args.output_dir, 'dev_errors.csv'), [row for row in final_rows if int(row['correct']) == 0])

    print(
      f"Epoch {epoch}: train loss :: {train_loss:.3f}, "
      f"dev acc :: {threshold_acc:.3f}, threshold :: {epoch_threshold:.2f}"
    )

  metrics['best_dev_accuracy'] = best_dev_acc
  metrics['best_threshold'] = best_threshold
  write_json(os.path.join(args.output_dir, 'metrics.json'), metrics)
  write_json(os.path.join(args.output_dir, 'threshold.json'), {'threshold': best_threshold, 'source': 'dev'})

  saved = torch.load(args.filepath, map_location=device)
  model.load_state_dict(saved['model'])
  final_acc, final_rows = evaluate_paraphrase(dev_loader, model, device, args, tokenizer, best_threshold)
  write_prediction_csv(os.path.join(args.output_dir, 'dev_predictions.csv'), final_rows)
  write_prediction_csv(os.path.join(args.output_dir, 'dev_errors.csv'), [row for row in final_rows if int(row['correct']) == 0])
  analysis = error_analysis(final_rows, args.output_dir)
  print(f"Best dev acc :: {final_acc:.3f}, threshold :: {best_threshold:.2f}")
  print(f"Dev errors :: {analysis['num_errors']} / {analysis['num_dev_examples']}")

  return best_threshold


@torch.no_grad()
def test(args, threshold):
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  saved = torch.load(args.filepath, map_location=device)
  saved_args = saved['args']
  model = ParaphraseGPT(saved_args)
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")
  print(f"Using dev-selected threshold {threshold:.2f}")

  para_test_data = load_paraphrase_data(args.para_test, split='test')
  para_test_data = maybe_debug_subset(para_test_data, args.debug_subset)
  test_dataset = ParaphraseDetectionTestDataset(para_test_data, args)
  test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

  sent_ids, preds = predict_test(test_loader, model, device, args, test_dataset.tokenizer, threshold)
  write_submission_csv(args.para_test_out, sent_ids, preds)
  print(f"Wrote test predictions to {args.para_test_out}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true', default=torch.cuda.is_available())
  parser.add_argument("--no_use_gpu", action='store_false', dest='use_gpu')

  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--grad_accum_steps", type=int, default=1)
  parser.add_argument("--scheduler", choices=['none', 'linear'], default='none')
  parser.add_argument("--warmup_ratio", type=float, default=0.0)
  parser.add_argument("--max_grad_norm", type=float, default=0.0)
  parser.add_argument("--max_length", type=int, default=128)
  parser.add_argument("--hard_negative_oversample_factor", type=int, default=1)
  parser.add_argument("--hard_negative_overlap", type=float, default=0.5)
  parser.add_argument("--debug_subset", type=int, default=None)
  parser.add_argument("--save_dir", type=str, default="runs/paraphrase")
  parser.add_argument("--run_name", type=str, default=None)

  parser.add_argument("--template_id", type=int, choices=[1, 2], default=1)
  parser.add_argument("--bidirectional", action='store_true')
  parser.add_argument("--restricted_yes_no_loss", action='store_true')
  parser.add_argument("--threshold_tuning", action='store_true')
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument("--skip_test", action='store_true',
                      help="Train and evaluate on dev without loading or predicting the test split.")

  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  if args.grad_accum_steps < 1:
    raise ValueError("--grad_accum_steps must be >= 1")
  if args.hard_negative_oversample_factor < 1:
    raise ValueError("--hard_negative_oversample_factor must be >= 1")
  if not 0.0 <= args.warmup_ratio < 1.0:
    raise ValueError("--warmup_ratio must be in [0, 1)")
  if args.run_name is None:
    args.run_name = f"template{args.template_id}_lr{args.lr}_ep{args.epochs}"
  args.output_dir = os.path.join(args.save_dir, args.run_name)
  args.filepath = os.path.join(args.output_dir, "best_model.pt")
  return args


if __name__ == "__main__":
  args = add_arguments(get_args())
  seed_everything(args.seed)
  threshold = train(args)
  if not args.skip_test:
    test(args, threshold)
