"""Tune an N-model paraphrase ensemble on dev, then optionally predict test.

This script only uses dev labels for weight and threshold selection. The test
split is loaded only when --predict_test is set, after the dev-selected ensemble
configuration is fixed.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import ParaphraseDetectionTestDataset, load_paraphrase_data
from paraphrase_detection import ParaphraseGPT, score_batch, write_submission_csv


def load_dev_predictions(path):
  with open(path, newline='') as f:
    return {row['id']: row for row in csv.DictReader(f)}


def aligned_dev_arrays(run_dirs):
  dev_maps = [load_dev_predictions(os.path.join(run_dir, 'dev_predictions.csv')) for run_dir in run_dirs]
  ids = list(dev_maps[0])
  id_set = set(ids)
  for dev_map in dev_maps[1:]:
    if set(dev_map) != id_set:
      raise ValueError("Dev prediction files contain different ids.")

  gold = np.array([int(dev_maps[0][item_id]['gold_label']) for item_id in ids])
  scores = np.stack([
    np.array([float(dev_map[item_id]['yes_score']) for item_id in ids])
    for dev_map in dev_maps
  ], axis=0)
  return ids, gold, scores


def threshold_search(gold, scores, threshold_step):
  best = None
  for threshold in np.arange(0.30, 0.7001, threshold_step):
    preds = (scores >= threshold).astype(int)
    accuracy = float(np.mean(preds == gold))
    candidate = (accuracy, float(threshold))
    if best is None or candidate > best:
      best = candidate
  return best


def greedy_weight_search(gold, model_scores, weight_step, threshold_step):
  num_models = model_scores.shape[0]
  single_results = []
  for model_idx in range(num_models):
    accuracy, threshold = threshold_search(gold, model_scores[model_idx], threshold_step)
    single_results.append((accuracy, threshold, model_idx))

  best_accuracy, best_threshold, best_model = max(single_results)
  weights = np.zeros(num_models)
  weights[best_model] = 1.0
  current_scores = model_scores[best_model].copy()
  remaining = set(range(num_models)) - {best_model}
  history = [{
    'action': 'start',
    'model_index': int(best_model),
    'dev_accuracy': float(best_accuracy),
    'threshold': float(best_threshold),
  }]

  improved = True
  while improved and remaining:
    improved = False
    best_candidate = None
    for model_idx in sorted(remaining):
      for keep_weight in np.arange(0.0, 1.0001, weight_step):
        candidate_scores = keep_weight * current_scores + (1.0 - keep_weight) * model_scores[model_idx]
        accuracy, threshold = threshold_search(gold, candidate_scores, threshold_step)
        candidate = (accuracy, threshold, keep_weight, model_idx, candidate_scores)
        if best_candidate is None or candidate[:4] > best_candidate[:4]:
          best_candidate = candidate

    accuracy, threshold, keep_weight, model_idx, candidate_scores = best_candidate
    if accuracy > best_accuracy:
      weights *= keep_weight
      weights[model_idx] = 1.0 - keep_weight
      current_scores = candidate_scores
      best_accuracy = accuracy
      best_threshold = threshold
      remaining.remove(model_idx)
      improved = True
      history.append({
        'action': 'add',
        'model_index': int(model_idx),
        'keep_existing_weight': float(keep_weight),
        'dev_accuracy': float(best_accuracy),
        'threshold': float(best_threshold),
      })

  preds = (current_scores >= best_threshold).astype(int)
  return current_scores, preds, {
    'dev_accuracy': float(best_accuracy),
    'threshold': float(best_threshold),
    'weights': [float(x) for x in weights],
    'history': history,
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
  parser.add_argument("--runs", nargs='+', required=True)
  parser.add_argument("--names", nargs='+', default=None)
  parser.add_argument("--output_dir", default="runs/paraphrase/ensemble_n")
  parser.add_argument("--dev_out", default="predictions/para-dev-output.csv")
  parser.add_argument("--test_out", default="predictions/para-test-output.csv")
  parser.add_argument("--para_test", default="data/quora-test-student.csv")
  parser.add_argument("--weight_step", type=float, default=0.01)
  parser.add_argument("--threshold_step", type=float, default=0.005)
  parser.add_argument("--predict_test", action='store_true')
  args = parser.parse_args()

  if args.names is not None and len(args.names) != len(args.runs):
    raise ValueError("--names must have the same length as --runs")

  names = args.names or [os.path.basename(run_dir) for run_dir in args.runs]
  os.makedirs(args.output_dir, exist_ok=True)
  ids, gold, model_scores = aligned_dev_arrays(args.runs)
  scores, preds, result = greedy_weight_search(
    gold, model_scores, args.weight_step, args.threshold_step
  )
  result.update({
    'names': names,
    'runs': args.runs,
    'test_labels_used': False,
    'search': 'greedy_convex_addition',
  })
  write_dev_outputs(args.output_dir, args.dev_out, ids, gold, scores, preds)

  if args.predict_test:
    test_data = load_paraphrase_data(args.para_test, split='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ids = None
    weighted_scores = None
    for weight, run_dir in zip(result['weights'], args.runs):
      if weight == 0.0:
        continue
      ids_i, scores_i = checkpoint_test_scores(os.path.join(run_dir, 'best_model.pt'), test_data, device)
      if test_ids is None:
        test_ids = ids_i
        weighted_scores = weight * scores_i
      elif test_ids != ids_i:
        raise ValueError("Test score streams contain different ids.")
      else:
        weighted_scores += weight * scores_i
    test_preds = (weighted_scores >= result['threshold']).astype(int)
    write_submission_csv(args.test_out, test_ids, test_preds)

  with open(os.path.join(args.output_dir, 'ensemble_config.json'), 'w') as f:
    json.dump(result, f, indent=2, sort_keys=True)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
