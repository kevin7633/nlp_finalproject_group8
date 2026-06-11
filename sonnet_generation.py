'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 최종 실험에서는 GPT-2 전체 파라미터를 fine-tuning한다.
    # 작은 소네트 데이터셋에서는 partial fine-tuning도 실험했지만,
    # 최종 선택 모델은 전체 fine-tuning에서 더 안정적인 생성 결과를 보였다.
    for param in self.gpt.parameters():
      param.requires_grad = True

    trainable_params = sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in self.gpt.parameters())
    print(f"Trainable GPT parameters: {trainable_params:,} / {total_params:,}")

  def forward(self, input_ids, attention_mask):
    """
    입력 시퀀스의 모든 위치에 대해 vocabulary logits를 반환한다.

    Sonnet Generation은 자기회귀 언어 모델링 과제이므로,
    모델은 t번째 위치까지의 토큰을 보고 t+1번째 토큰을 예측해야 한다.
    따라서 마지막 hidden state 하나만 사용하는 분류 모델과 달리,
    시퀀스 전체 위치의 hidden state를 vocabulary logits로 변환해야 한다.

    Args:
      input_ids: [batch_size, seq_len] 형태의 입력 토큰 ID
      attention_mask: [batch_size, seq_len] 형태의 attention mask

    Returns:
      logits: [batch_size, seq_len, vocab_size] 형태의 다음 토큰 예측 점수
    """
    # GPT-2에 전체 입력 시퀀스를 통과시켜 각 위치의 hidden state를 얻는다.
  # 출력 hidden state의 크기는 [batch_size, seq_len, hidden_dim]이다.
    outputs = self.gpt(input_ids, attention_mask)
    sequence_output = outputs["last_hidden_state"]

    # 각 위치의 hidden state를 vocabulary logits로 변환한다.
    # 이렇게 해야 학습 시 t번째 토큰으로 t+1번째 토큰을 예측하는
    # next-token prediction loss를 계산할 수 있다.
    logits = self.gpt.hidden_state_to_token(sequence_output)

    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128,
               target_lines=14, repetition_penalty=1.15, no_repeat_ngram_size=3,
               min_tokens_per_line=5, max_tokens_per_line=12, newline_bias=3.0):
    """
    top-p sampling을 기반으로 새로운 소네트를 생성한다.

    기본 sampling 방식은 자연스러운 문장을 만들 수 있지만,
    소네트의 14줄 구조를 명시적으로 보장하지는 못한다.
    따라서 이 함수는 다음과 같은 생성 제약을 추가한다.

      1. 14줄 형식 제어:
         생성 중 현재 줄 수를 확인하고, 전체 소네트가 14줄에 도달하면 생성을 종료한다.

      2. 줄바꿈 제어:
         한 줄이 너무 길어지면 newline token의 확률을 높이거나 강제로 newline을 생성한다.
         이를 통해 긴 산문처럼 이어지는 출력을 줄이고, 14줄 소네트 형식을 유도한다.

      3. 반복 억제:
         repetition penalty와 no-repeat n-gram을 적용하여 같은 표현이 반복되는 현상을 줄인다.

    Args:
      encoding: prompt token ids. 일반적으로 held-out sonnet의 첫 3줄이 들어온다.
      temperature: softmax 분포의 sharpness를 조절하는 값
      top_p: nucleus sampling에서 사용할 누적 확률 기준
      max_length: 최대 생성 토큰 수
      target_lines: 목표 소네트 줄 수. 셰익스피어식 소네트는 14줄이다.
      repetition_penalty: 반복 토큰 억제 강도
      no_repeat_ngram_size: 반복을 금지할 n-gram 크기
      min_tokens_per_line: 이 길이 전까지는 줄바꿈을 강하게 유도하지 않는다.
      max_tokens_per_line: 이 길이를 넘으면 강제로 줄바꿈을 생성한다.
      newline_bias: 줄바꿈 token의 logit을 높이는 정도

    Returns:
      token_ids: prompt와 생성 토큰을 모두 포함한 token id
      generated_output: prompt 이후 새로 생성된 텍스트
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    prompt_length = token_ids.shape[1]

    newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
    newline_token_id = newline_ids[0] if newline_ids else None

    def get_banned_tokens(tokens, ngram_size):
      """
      이미 등장한 n-gram을 기준으로, 현재 위치에서 생성하면 반복 n-gram이 되는 토큰을 찾는다.
      """
      if ngram_size <= 0 or len(tokens) + 1 < ngram_size:
        return []

      prefix = tuple(tokens[-(ngram_size - 1):])
      banned = []

      for i in range(len(tokens) - ngram_size + 1):
        prev_ngram = tuple(tokens[i:i + ngram_size])
        if prev_ngram[:-1] == prefix:
          banned.append(prev_ngram[-1])

      return banned

    def count_current_line_tokens(tokens):
      """
      마지막 newline 이후 현재 줄에 몇 개의 token이 생성되었는지 계산한다.
      줄이 너무 길어지는 것을 막기 위해 사용한다.
      """
      if newline_token_id is None:
        return 0

      last_newline_idx = -1
      for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx] == newline_token_id:
          last_newline_idx = idx
          break

      return len(tokens) - last_newline_idx - 1

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :]

      temperature = max(float(temperature), 1e-8)
      logits_last_token = logits_last_token / temperature

      current_tokens = token_ids[0].detach().cpu().tolist()

      # 줄 수 판단은 prompt를 제외한 generated 부분만 기준으로 한다.
      # held-out prompt에는 이미 앞 3행이 들어 있으므로,
      # prompt까지 포함해서 줄 수를 세면 남은 행을 충분히 생성하기 전에 종료될 수 있다.
      generated_tokens_for_check = token_ids[0, prompt_length:].detach().cpu().tolist()
      decoded_text = self.tokenizer.decode(generated_tokens_for_check, skip_special_tokens=True)
      current_lines = [
        line.strip()
        for line in decoded_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
      ]

      current_line_len = count_current_line_tokens(current_tokens)

      # 목표 행 수에 도달했더라도 마지막 행이 너무 짧으면 종료하지 않는다.
      if len(current_lines) >= target_lines and current_line_len >= min_tokens_per_line:
        break

      if repetition_penalty is not None and repetition_penalty > 1.0:
        generated_tokens = set(current_tokens)
        for token_id in generated_tokens:
          if logits_last_token[0, token_id] < 0:
            logits_last_token[0, token_id] *= repetition_penalty
          else:
            logits_last_token[0, token_id] /= repetition_penalty

      if no_repeat_ngram_size is not None and no_repeat_ngram_size > 1:
        banned_tokens = get_banned_tokens(current_tokens, no_repeat_ngram_size)
        if banned_tokens:
          logits_last_token[:, banned_tokens] = -float("inf")

      # 줄바꿈 제어를 적용한다.
      # 현재 줄이 충분히 길어졌다면 newline token의 logit을 높여 줄바꿈을 유도하고,
      # 너무 길어진 경우에는 newline을 강제로 선택하여 산문처럼 길게 이어지는 출력을 막는다.
      force_newline = (
        newline_token_id is not None
        and current_line_len >= max_tokens_per_line
        and len(current_lines) < target_lines
      )

      if force_newline:
        sampled_token = torch.tensor([[newline_token_id]], dtype=torch.long, device=self.get_device())
      else:
        if (
          newline_token_id is not None
          and current_line_len >= min_tokens_per_line
          and len(current_lines) < target_lines
        ):
          logits_last_token[:, newline_token_id] += newline_bias

        # 목표 행 수를 채우기 전에는 EOS token을 샘플링 후보에서 제거한다.
        # EOS가 너무 일찍 선택되면 남은 행을 다 생성하기 전에 소네트가 끊긴다.
        eos_allowed = (
          len(current_lines) >= target_lines
          and current_line_len >= min_tokens_per_line
        )
        if not eos_allowed and self.tokenizer.eos_token_id is not None:
          logits_last_token[:, self.tokenizer.eos_token_id] = -float("inf")

        probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        top_p_mask = cumulative_probs <= top_p
        top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
        top_p_mask[..., 0] = True

        filtered_probs = sorted_probs * top_p_mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        sampled_index = torch.multinomial(filtered_probs, 1)
        sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_token_ids = token_ids[0, prompt_length:].detach().cpu().tolist()
    generated_output = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return token_ids, generated_output


def clean_sonnet(text, max_lines=14):
  """
  생성된 소네트를 저장하기 전에 정리한다.

  생성 결과에는 GPT-2 특수 토큰, 빈 줄, 불필요한 공백,
  14줄을 초과한 문장이 포함될 수 있다. chrF는 character-level
  n-gram 기반 평가 지표이므로, 이런 표면적 노이즈가 점수에
  영향을 줄 수 있다.

  이 함수는 생성 결과를 14줄 소네트 형식에 가깝게 정리하여
  제출 파일의 안정성과 생성 결과의 가독성을 높인다.
  """
  text = text.replace("<|endoftext|>", "")
  text = text.replace("\r\n", "\n").replace("\r", "\n")

  cleaned_lines = []
  for line in text.split("\n"):
    line = " ".join(line.strip().split())
    if not line:
      continue

    cleaned_lines.append(line)

    if len(cleaned_lines) >= max_lines:
      break

  return "\n".join(cleaned_lines)


def get_last_word(line):
  """
  한 줄의 마지막 단어를 추출한다.

  소네트의 운율은 각 줄의 마지막 단어에서 강하게 드러난다.
  이후 운율 단서를 만들기 위해 줄 끝 단어를 소문자로 정리하고,
  마침표나 쉼표 같은 단순 구두점을 제거한다.
  """
  if not line:
    return ""

  tokens = line.strip().split()
  if not tokens:
    return ""

  word = tokens[-1].lower()
  word = word.strip(".,;:!?\"()[]{}")
  return word


def get_rhyme_key(line, suffix_len=3):
  """
  줄 끝 단어에서 간단한 운율 단서를 추출한다.

  완전한 영어 운율 판단에는 발음 사전이나 음운 정보가 필요하지만,
  이 프로젝트에서는 추가 라이브러리 없이 사용할 수 있는
  character suffix 기반 방법을 사용한다.

  chrF 자체가 character-level 평가 지표이므로, 마지막 단어의 suffix는
  생성 결과의 표면적 운율 유사성을 측정하는 가벼운 기준으로 사용할 수 있다.
  """
  word = get_last_word(line)
  if not word:
    return ""

  if len(word) <= suffix_len:
    return word
  return word[-suffix_len:]


def rhyme_consistency_score(lines):
  """
  생성된 소네트가 ABAB CDCD EFEF GG 운율 구조를 얼마나 따르는지 계산한다.

  셰익스피어식 소네트에서는 일반적으로 다음 줄 쌍들이 서로 운율을 이룬다.
    1행-3행, 2행-4행,
    5행-7행, 6행-8행,
    9행-11행, 10행-12행,
    13행-14행

  각 줄의 마지막 단어 suffix가 같으면 해당 운율 쌍이 맞았다고 보고,
  전체 운율 쌍 중 맞은 비율을 점수로 반환한다.
  """
  if len(lines) < 2:
    return 0.0

  rhyme_pairs = [
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (8, 10),
    (9, 11),
    (12, 13),
  ]

  total = 0
  matched = 0

  for i, j in rhyme_pairs:
    if i >= len(lines) or j >= len(lines):
      continue

    key_i = get_rhyme_key(lines[i])
    key_j = get_rhyme_key(lines[j])

    if not key_i or not key_j:
      continue

    total += 1
    if key_i == key_j:
      matched += 1

  if total == 0:
    return 0.0
  return matched / total


def repetition_score(lines):
  """
  생성된 소네트에서 반복이 얼마나 많은지 계산한다.

  작은 데이터셋으로 GPT-2를 fine-tuning하면 같은 줄이나 비슷한 줄 끝 단어가
  반복되는 문제가 생길 수 있다. 반복은 정성적 가독성을 떨어뜨리고,
  원본 소네트와 다른 문자열 패턴을 과도하게 늘려 chrF에도 불리할 수 있다.

  이 함수는 완전히 같은 줄의 반복과 줄 끝 단어 반복을 함께 고려해
  반복 penalty를 계산한다.
  """
  if not lines:
    return 0.0

  normalized = [line.lower().strip() for line in lines if line.strip()]
  unique_lines = set(normalized)

  repeated_line_penalty = len(normalized) - len(unique_lines)

  endings = [get_last_word(line) for line in normalized]
  endings = [word for word in endings if word]
  repeated_ending_penalty = len(endings) - len(set(endings))

  return float(repeated_line_penalty + 0.5 * repeated_ending_penalty)


def is_complete_sonnet_candidate(text, target_lines=14):
  """
  생성 후보가 완결된 14행 소네트 형태인지 검사한다.

  이 함수는 정답 reference를 보지 않고 생성 텍스트 자체만 확인한다.
  best-of-N 후보 선택 과정에서 마지막 줄이 너무 짧거나 끊긴 후보보다
  형식적으로 완결된 후보를 우선 선택하기 위한 bonus 기준으로 사용된다.
  """
  cleaned = clean_sonnet(text, max_lines=target_lines)
  lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

  if len(lines) != target_lines:
    return False

  bad_final_words = {
    "a", "an", "the", "and", "or", "but", "nor", "so",
    "of", "to", "from", "in", "on", "at", "by", "with", "for",
    "that", "which", "who", "whom", "where", "when",
    "is", "are", "was", "were", "be", "been",
    "do", "doth", "did", "does",
    "shall", "will", "would", "should", "could", "may", "might",
    "my", "thy", "thine", "your", "his", "her", "their",
    "not", "left", "prove", "purpose", "happy", "wit", "mak",
    "d", "e", "th", "whe", "suff",
  }

  last_line = lines[-1].strip()
  last_word = get_last_word(last_line)

  if len(last_line.split()) < 5:
    return False
  if last_word in bad_final_words:
    return False
  if len(last_word) <= 2:
    return False
  if last_line.startswith(".") or last_line.startswith(",") or last_line.startswith("?"):
    return False

  # 마지막 행은 완결된 문장처럼 끝나는 후보만 complete로 인정한다.
  if not (last_line.endswith(".") or last_line.endswith("!") or last_line.endswith("?") or last_line.endswith("--")):
    return False

  noisy_patterns = [
    "chapter", "edition", "[pg", "page", "--------", "......",
    "♦", "®", "►", "�", "■",
  ]
  lower = cleaned.lower()
  for pattern in noisy_patterns:
    if pattern in lower:
      return False

  return True


def candidate_quality_score(text, target_lines=14):
  """
  정답 소네트를 보지 않고 생성 후보의 품질을 계산한다.

  best-of-N generation에서 여러 후보 중 더 소네트 형식에 가까운 출력을
  고르기 위한 점수 함수이다. reference sonnet은 사용하지 않고,
  생성 텍스트 자체에서 확인 가능한 형식적 기준만 사용한다.

  주요 기준:
    1. 목표 행 수인 14행에 가까운 후보를 선호한다.
    2. 반복되는 줄이나 줄 끝 단어를 감점한다.
    3. ABAB CDCD EFEF GG 운율 구조에 가까우면 가점한다.
    4. 마지막 줄이 너무 짧거나 조각난 단어로 끝나면 감점한다.
    5. 페이지 표기, 문서 노이즈, 특수문자, 현대적 표현, 부적절 표현을 감점한다.
    6. 한 줄이 너무 길어 산문처럼 보이는 후보를 감점한다.
  """
  cleaned = clean_sonnet(text, max_lines=target_lines)
  lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

  line_count_penalty = abs(len(lines) - target_lines)
  rep_penalty = repetition_score(lines)
  rhyme_bonus = rhyme_consistency_score(lines)

  bad_end_words = {
    "a", "an", "the",
    "and", "or", "but", "nor", "yet", "so",
    "of", "to", "from", "in", "on", "at", "by", "with", "for", "as",
    "than", "through", "into", "unto", "upon", "within", "without",
    "that", "which", "who", "whom", "whose", "where", "when", "while",
    "if", "though", "because", "since", "till", "until",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "doth", "did", "does",
    "shall", "will", "would", "should", "could", "may", "might", "must", "can",
    "my", "thy", "thine", "your", "his", "her", "their", "our",
  }

  document_noise_words = [
    "chapter", "edition", "page", "[pg", "copyright", "volume",
    "scan", "unmasked", "substance", "country of gods",
  ]

  modern_or_bad_words = [
    "cunt", "penis", "balls", "thrust", "sexiest",
    "chapter", "edition", "mr.", "mrs.", "dr.", "socrates",
    "homer", "rome", "america", "lincoln", "orange county",
  ]

  weird_chars = [
    "♦", "®", "►", "�", "ō", "ʿ", "‎", "■", "|",
  ]

  fragment_penalty = 0.0
  weird_penalty = 0.0
  length_penalty = 0.0
  style_penalty = 0.0

  lower_text = cleaned.lower()

  for noise in document_noise_words:
    if noise in lower_text:
      weird_penalty += 3.0

  for bad in modern_or_bad_words:
    if bad in lower_text:
      style_penalty += 4.0

  for ch in weird_chars:
    if ch in cleaned:
      weird_penalty += 2.0

  if "--------" in cleaned or "......" in cleaned:
    weird_penalty += 3.0
  if "__" in cleaned:
    weird_penalty += 3.0

  for idx, line in enumerate(lines):
    words = line.split()
    last_word = get_last_word(line)
    lower_line = line.lower()

    # 소네트 한 행으로 보기 어려운 너무 짧은 줄 감점
    if len(words) <= 2:
      fragment_penalty += 1.5
    elif len(words) <= 4:
      fragment_penalty += 0.6

    # 너무 긴 산문형 줄 감점
    if len(words) >= 17:
      length_penalty += 1.0
    elif len(words) >= 14:
      length_penalty += 0.4

    # 마지막 행은 완결성이 중요 더 강하게 감점
    if idx == len(lines) - 1:
      if len(words) < 6:
        fragment_penalty += 3.0
      if last_word in bad_end_words:
        fragment_penalty += 3.0
      if len(last_word) <= 2:
        fragment_penalty += 3.0
      if line.endswith("-") or line.endswith("--"):
        fragment_penalty += 2.0
      if line.endswith(",") or line.endswith(":") or line.endswith(";"):
        fragment_penalty += 1.5

      # 최종 후 제출용 마지막 행의 완결성을 강하게 본다.
      # 마침표/물음/느낌표/긴 dash로 끝나지 않는 경우 조각난 문장일 가능성이 높다.
      if not (line.endswith(".") or line.endswith("!") or line.endswith("?") or line.endswith("--")):
        fragment_penalty += 3.0

    # 모든 행에 대해 조각난 끝 단어 감점
    if last_word in bad_end_words:
      fragment_penalty += 0.5
    if last_word and len(last_word) <= 2:
      fragment_penalty += 0.8

    # 괄호/따옴표/대괄호가 깨진                                   감점
    if line.count("[") + line.count("]") >= 1:
      weird_penalty += 1.0
    if line.count("(") != line.count(")"):
      weird_penalty += 1.0
    if line.count('"') % 2 == 1:
      weird_penalty += 0.7

    # 문서 표기나 산문 표식 감점
    if any(noise in lower_line for noise in document_noise_words):
      weird_penalty += 2.0
    if any(bad in lower_line for bad in modern_or_bad_words):
      style_penalty += 3.0

  score = 0.0
  score -= 2.0 * line_count_penalty
  score -= 1.0 * rep_penalty
  score += 2.0 * rhyme_bonus
  score -= 1.4 * fragment_penalty
  score -= 1.2 * weird_penalty
  score -= 0.8 * length_penalty
  score -= 1.5 * style_penalty

  return score


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


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # 마지막 위치는 다음 토큰 label이 없으므로 제외한다.
      labels = b_ids[:, 1:].contiguous().flatten()  # 첫 번째 토큰은 이전 문맥이 없으므로 label에서 제외한다.

      # attention mask 기반 next-token loss를 계산한다.
      # 길이가 다른 소네트를 batch로 묶으면 padding token이 포함될 수 있다.
      # padding token은 실제 시 텍스트가 아니므로, loss에 포함되면 학습 신호가 불필요하게 흐려질 수 있다.
      # 따라서 각 토큰 위치별 loss를 먼저 계산한 뒤, attention_mask가 1인 실제 토큰 위치만 평균한다.
      loss_per_token = F.cross_entropy(logits, labels, reduction='none')
      loss_mask = b_mask[:, 1:].contiguous().flatten().float()
      loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p,
        target_lines=args.target_lines,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_tokens_per_line=args.min_tokens_per_line,
        max_tokens_per_line=args.max_tokens_per_line,
        newline_bias=args.newline_bias,
      )
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: 소넷의 작은 테이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    prompt_text = batch[1]

    # held-out prompt에는 이미 소네트의 앞 3행이 들어 있다.
    # 따라서 생성 단계에서는 전체 14행을 다시 만드는 것이 아니라,
    # prompt 이후의 남은 행만 생성하도록 목표 행 수를 조정한다.
    prompt_lines = [
      line.strip()
      for line in prompt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
      if line.strip()
    ]
    remaining_lines = max(1, args.target_lines - len(prompt_lines))

    # prompt 마지막 행과 생성 첫 행이 붙어버리지 않도록 newline을 명시적으로 추가한다.
    prompt_for_generation = "\n".join(prompt_lines).strip() + "\n"
    encoding = model.tokenizer(prompt_for_generation, return_tensors='pt', padding=False, truncation=True).to(device)

    best_score = -float("inf")
    best_sonnet = None

    # best-of-N generation을 적용한다.
    # 하나의 prompt에 대해 여러 후보를 생성한 뒤, 정답 소네트를 보지 않고
    # 생성 결과 자체의 형식적 품질만으로 가장 좋은 후보를 선택한다.
    #
    # 이 방식은 test reference를 사용하지 않으므로 test set 부정 사용에 해당하지 않는다.
    # 후보 선택 기준은 candidate_quality_score()에 정의되어 있으며,
    # 14줄 형식, 반복 정도, ABAB CDCD EFEF GG 운율 일관성을 함께 고려한다.
    for _ in range(args.num_candidates):
      _, generated_text = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=220,
        target_lines=remaining_lines,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_tokens_per_line=args.min_tokens_per_line,
        max_tokens_per_line=args.max_tokens_per_line,
        newline_bias=args.newline_bias,
      )

      # prompt와 generated_text를 단순 문자열 덧셈으로 합치면,
      # prompt 마지막 행과 생성 첫 행이 붙거나  행이 잘릴 수 있다.
      # 따라서 행 단위로 분리한 뒤 prompt  + 생성 행을 합쳐 최종 14행을 구성한다.
      generated_lines = [
        line.strip()
        for line in generated_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
      ]
      candidate_lines = prompt_lines + generated_lines
      candidate_text = "\n".join(candidate_lines[:args.target_lines])
      candidate_text = clean_sonnet(candidate_text, max_lines=args.target_lines)
      score = candidate_quality_score(candidate_text, target_lines=args.target_lines)

      # 완결성이 있는 14행 후보를 우선 선택한다.
      # complete 후보에는 보너스를 부여하여,
      # 마지막 줄이 끊긴 후보보다 선택될 가능성을 높인다.
      if is_complete_sonnet_candidate(candidate_text, target_lines=args.target_lines):
        score += 5.0

      if score > best_score:
        best_score = score
        best_sonnet = candidate_text

    full_sonnet = f'{best_sonnet}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'[sonnet_id={sonnet_id}] best_candidate_score={best_score:.3f}')
    print(f'{best_sonnet}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--target_lines", type=int, default=14,
                      help="생성 결과가 맞춰야 하는 소네트 줄 수. 셰익스피어식 소네트는 14줄이다.")
  parser.add_argument("--repetition_penalty", type=float, default=1.15,
                      help="이미 등장한 토큰의 반복 생성을 억제하는 강도.")
  parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                      help="같은 n-gram 반복을 막기 위한 n-gram 크기.")
  parser.add_argument("--num_candidates", type=int, default=5,
                      help="각 prompt마다 생성할 후보 수. 여러 후보 중 형식, 반복, 운율 점수가 가장 좋은 후보를 선택한다.")
  parser.add_argument("--min_tokens_per_line", type=int, default=5,
                      help="이 token 수 이후부터 줄바꿈 생성을 유도한다.")
  parser.add_argument("--max_tokens_per_line", type=int, default=12,
                      help="한 줄이 이 token 수를 넘으면 강제로 줄바꿈을 생성한다.")
  parser.add_argument("--newline_bias", type=float, default=3.0,
                      help="줄바꿈 token의 logit을 높이는 정도.")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
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
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  generate_submission_sonnets(args)
