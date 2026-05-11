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