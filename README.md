# 자연어처리 2026-1 기말 프로젝트: GPT-2 구현

이 프로젝트는 GitHub 원본 저장소 `https://github.com/kikim6114/nlp2026-final`의 GPT-2 기반 과제 코드입니다.  
제공된 원래 코드는 GPT-2 모델 구조, 데이터 로더, 학습/평가 스크립트, 제출 파일 생성 스크립트를 포함하고 있으며, 일부 핵심 함수의 빈 코드 블록을 직접 구현하도록 구성되어 있습니다.

본 README는 아직 비어 있는 다른 과제 블록은 제외하고, 원래 제공 코드의 구성과 이번에 작성한 `modules/attention.py`, `optimizer.py` 구현 내용을 중심으로 설명합니다.

## 프로젝트 구성

### 모델 관련 코드

- `models/base_gpt.py`
  - GPT 계열 모델의 공통 부모 클래스인 `GPTPreTrainedModel`을 정의합니다.
  - 선형층, 임베딩층, LayerNorm의 기본 초기화 방식을 제공합니다.
  - 모델 파라미터의 dtype을 확인하는 공통 속성을 제공합니다.

- `models/gpt2.py`
  - GPT-2 전체 모델 골격을 정의합니다.
  - 단어 임베딩과 위치 임베딩을 만들고, 여러 개의 `GPT2Layer`를 순차적으로 통과시킨 뒤 마지막 LayerNorm을 적용하는 구조입니다.
  - Hugging Face `GPT2Model`에서 사전학습 가중치를 읽어와 이 프로젝트의 GPT-2 구현에 복사하는 `from_pretrained()` 메서드를 포함합니다.
  - 마지막 non-padding 토큰의 hidden state를 `last_token`으로 반환하여 분류 작업에서 사용할 수 있게 합니다.

- `modules/gpt2_layer.py`
  - GPT-2의 Transformer block 한 층을 정의합니다.
  - 한 층은 causal self-attention, attention 이후의 residual/dropout/dense 처리, feed-forward network, 두 개의 LayerNorm으로 구성됩니다.
  - 이 파일의 빈 구현 부분은 현재 README의 검토 대상에서 제외했습니다.

- `modules/attention.py`
  - GPT-2의 multi-head causal self-attention을 담당합니다.
  - 입력 hidden state를 query, key, value로 선형 변환한 뒤 head 단위로 나누고, attention 결과를 다시 `[batch_size, seq_len, hidden_size]` 형태로 합칩니다.
  - 이번에 직접 구현한 핵심 파일입니다.

### 학습 및 태스크 코드

- `classifier.py`
  - SST와 CFIMDB 감정 분류를 위한 GPT-2 기반 분류기 학습/평가 스크립트입니다.
  - tokenizer, dataloader, train/eval loop, 모델 저장 및 테스트 출력 파일 생성을 포함합니다.
  - `AdamW` optimizer를 사용합니다.

- `paraphrase_detection.py`
  - Quora 문장쌍 데이터로 paraphrase detection을 수행하는 스크립트입니다.
  - GPT-2 출력을 paraphrase 여부 분류에 사용하도록 설계되어 있습니다.

- `sonnet_generation.py`
  - Shakespeare sonnet 데이터로 GPT-2를 fine-tuning하고, held-out prompt에서 sonnet을 생성하는 스크립트입니다.
  - top-p sampling과 temperature를 이용한 생성 루프가 포함되어 있습니다.

### 데이터 및 평가 코드

- `datasets.py`
  - Quora paraphrase 데이터와 sonnet 데이터를 PyTorch `Dataset` 형태로 감싸는 코드입니다.
  - GPT-2 tokenizer를 사용하여 입력 문장을 token id와 attention mask로 변환합니다.

- `evaluation.py`
  - paraphrase detection의 accuracy/F1 계산 함수와 sonnet 생성 결과의 chrF 평가 함수를 제공합니다.

- `utils.py`
  - Hugging Face 모델 파일 다운로드/캐시 관련 유틸리티와 attention mask 확장 함수를 포함합니다.
  - `get_extended_attention_mask()`는 `[batch_size, seq_len]` mask를 attention 계산에 사용할 `[batch_size, 1, 1, seq_len]` 형태로 바꿉니다.

- `prepare_submit.py`
  - 제출에 필요한 Python 파일, 모델 파일, 모듈 파일, 예측 파일을 zip으로 묶는 스크립트입니다.

## 직접 작성한 구현

### 1. `modules/attention.py`의 causal self-attention

`CausalSelfAttention.attention()`의 빈 코드 블록을 구현했습니다.

구현 흐름은 다음과 같습니다.

1. query와 key의 내적으로 attention score를 계산합니다.
2. head dimension의 제곱근으로 나누어 scaled dot-product attention을 적용합니다.
3. lower triangular causal mask를 만들어 현재 토큰이 미래 토큰을 보지 못하게 합니다.
4. padding token을 무시하기 위해 외부에서 전달된 `attention_mask`를 더합니다.
5. softmax와 dropout을 적용해 attention probability를 만듭니다.
6. attention probability와 value를 곱해 context vector를 계산합니다.
7. 여러 attention head를 다시 합쳐 `[batch_size, seq_len, hidden_size]` 형태로 반환합니다.

이 구현은 GPT-2의 autoregressive 성질에 맞게 미래 토큰을 차단하며, 프로젝트의 `utils.get_extended_attention_mask()`가 만드는 padding mask와도 호환됩니다.

### 2. `optimizer.py`의 AdamW

`AdamW.step()`의 빈 코드 블록을 구현했습니다.

구현 흐름은 다음과 같습니다.

1. 각 파라미터별 optimizer state를 초기화합니다.
   - `step`
   - `exp_avg`
   - `exp_avg_sq`
2. gradient의 1차 모멘트와 2차 모멘트를 Adam 방식으로 누적합니다.
3. `correct_bias=True`인 경우 bias correction을 적용합니다.
4. `addcdiv_()`를 사용해 Adam 업데이트를 수행합니다.
5. 메인 gradient 업데이트 이후 decoupled weight decay를 적용합니다.

이 방식은 과제 설명의 AdamW 요구사항인 moment update, bias correction, parameter update, weight decay 적용 순서를 만족합니다.

## 구현 검토 결과

작성한 두 파일은 원본 저장소 기준으로 비어 있던 코드 블록만 채운 상태입니다.

- `modules/attention.py`
  - tensor shape 흐름이 기존 `transform()` 및 `GPT2Layer` 호출 방식과 일치합니다.
  - causal mask와 padding mask가 모두 적용됩니다.
  - 간단한 로컬 검증에서 출력 shape가 정상이고, 첫 번째 토큰 출력이 미래 토큰 변화에 영향을 받지 않음을 확인했습니다.

- `optimizer.py`
  - 제공된 `optimizer_test.py`를 통과했습니다.
  - 테스트 결과 reference weight와 실제 학습 후 weight가 허용 오차 안에서 일치했습니다.

실행한 검증:

```bash
python optimizer_test.py
```

결과:

```text
Optimizer test passed!
```

## 실행 안내

환경은 제공된 `env.yml`을 기준으로 생성합니다.

```bash
conda env create -f env.yml
conda activate nlp_final
```

optimizer 구현만 확인하려면 다음 명령을 실행합니다.

```bash
python optimizer_test.py
```

전체 GPT-2 sanity check는 `models/gpt2.py`, `modules/gpt2_layer.py` 등 다른 빈 코드 블록까지 구현된 뒤 실행해야 합니다.

```bash
python sanity_check.py
```

## 주의 사항

이 README는 현재 작성 완료된 `modules/attention.py`, `optimizer.py` 검토에 초점을 맞췄습니다.  
`classifier.py`, `models/gpt2.py`, `modules/gpt2_layer.py`, `paraphrase_detection.py`, `sonnet_generation.py` 등에 남아 있는 빈 코드 블록은 이번 검토 범위에 포함하지 않았습니다.
