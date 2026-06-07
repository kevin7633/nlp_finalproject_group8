"""Tune a two-model paraphrase ensemble on dev, then optionally predict test."""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import ParaphraseDetectionTestDataset, load_paraphrase_data
from paraphrase_detection import (
  ParaphraseGPT,
  score_batch,
  write_submission_csv,
)


def load_dev_predictions(path):
  with open(path, newline='') as f:
    return {row['id']: row for row in csv.DictReader(f)}


def tune_ensemble(rows_a, rows_b, weight_step, threshold_step):
  ids = list(rows_a)
  if set(ids) != set(rows_b):
    raise ValueError("The two dev prediction files contain different ids.")

  gold = np.array([int(rows_a[item_id]['gold_label']) for item_id in ids])
  scores_a = np.array([float(rows_a[item_id]['yes_score']) for item_id in ids])
  scores_b = np.array([float(rows_b[item_id]['yes_score']) for item_id in ids])
  thresholds = np.arange(0.30, 0.7001, threshold_step)

  best = None
  for weight_a in np.arange(0.0, 1.0001, weight_step):
    scores = weight_a * scores_a + (1.0 - weight_a) * scores_b
    for threshold in thresholds:
      preds = (scores >= threshold).astype(int)
      accuracy = float(np.mean(preds == gold))
      candidate = (accuracy, float(weight_a), float(threshold))
      if best is None or candidate > best:
        best = candidate

  accuracy, weight_a, threshold = best
  scores = weight_a * scores_a + (1.0 - weight_a) * scores_b
  preds = (scores >= threshold).astype(int)
  return ids, gold, scores, preds, {
    'dev_accuracy': accuracy,
    'weight_a': weight_a,
    'weight_b': 1.0 - weight_a,
    'threshold': threshold,
  }


def write_dev_outputs(output_dir, output_path, ids, gold, scores, preds):
  rows = []
  for item_id, label, score, pred in zip(ids, gold, scores, preds):
    rows.append({
      'id': item_id,
      'gold_label': int(label),
      'pred_label': int(pred),
      'yes_score': float(score),
      'no_score': float(1.0 - score),
      'correct': int(label == pred),
    })

  fields = ['id', 'gold_label', 'pred_label', 'yes_score', 'no_score', 'correct']
  with open(os.path.join(output_dir, 'dev_predictions.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  with open(os.path.join(output_dir, 'dev_errors.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(row for row in rows if not row['correct'])

  with open(output_path, 'w') as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for item_id, pred in zip(ids, preds):
      f.write(f"{item_id}, {int(pred)} \n")


@torch.no_grad()
def checkpoint_test_scores(checkpoint, test_data, device):
  saved = torch.load(checkpoint, map_location=device)
  args = saved['args']
  dataset = ParaphraseDetectionTestDataset(test_data, args)
  loader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dataset.collate_fn
  )
  model = ParaphraseGPT(args).to(device)
  model.load_state_dict(saved['model'])
  model.eval()

  sent_ids, scores = [], []
  for batch in loader:
    batch_scores = score_batch(model, batch, device, args, dataset.tokenizer)
    sent_ids.extend(batch['sent_ids'])
    scores.extend(float(score) for score in batch_scores)
  return sent_ids, np.array(scores)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_a", required=True)
  parser.add_argument("--run_b", required=True)
  parser.add_argument("--name_a", default="model_a")
  parser.add_argument("--name_b", default="model_b")
  parser.add_argument("--output_dir", default="runs/paraphrase/ensemble")
  parser.add_argument("--dev_out", default="predictions/para-dev-output.csv")
  parser.add_argument("--test_out", default="predictions/para-test-output.csv")
  parser.add_argument("--para_test", default="data/quora-test-student.csv")
  parser.add_argument("--weight_step", type=float, default=0.01)
  parser.add_argument("--threshold_step", type=float, default=0.005)
  parser.add_argument("--skip_test", action='store_true')
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  dev_a = load_dev_predictions(os.path.join(args.run_a, 'dev_predictions.csv'))
  dev_b = load_dev_predictions(os.path.join(args.run_b, 'dev_predictions.csv'))
  ids, gold, scores, preds, result = tune_ensemble(
    dev_a, dev_b, args.weight_step, args.threshold_step
  )
  result.update({
    'name_a': args.name_a,
    'name_b': args.name_b,
    'run_a': args.run_a,
    'run_b': args.run_b,
    'test_labels_used': False,
  })
  write_dev_outputs(args.output_dir, args.dev_out, ids, gold, scores, preds)

  if not args.skip_test:
    test_data = load_paraphrase_data(args.para_test, split='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids_a, scores_a = checkpoint_test_scores(
      os.path.join(args.run_a, 'best_model.pt'), test_data, device
    )
    ids_b, scores_b = checkpoint_test_scores(
      os.path.join(args.run_b, 'best_model.pt'), test_data, device
    )
    if ids_a != ids_b:
      raise ValueError("The two test score streams contain different ids.")
    final_scores = result['weight_a'] * scores_a + result['weight_b'] * scores_b
    test_preds = (final_scores >= result['threshold']).astype(int)
    write_submission_csv(args.test_out, ids_a, test_preds)

  with open(os.path.join(args.output_dir, 'ensemble_config.json'), 'w') as f:
    json.dump(result, f, indent=2, sort_keys=True)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
