# 자연어처리 2026-1 지정주제 기말 프로젝트: GPT-2 구축

8조의 GPT-2 구현 및 후속 태스크 확장 프로젝트입니다.

## 프로젝트 구성

- PART-I: GPT-2, causal self-attention, GPT-2 layer, AdamW 구현
- Sentiment Analysis: SST와 CFIMDB의 last-linear-layer/full-model 학습
- PART-II: GPT-2 기반 cloze-style Paraphrase Detection
- PART-II: prompt-aware Sonnet Generation과 형식 제약 decoding

## 주요 결과

- Paraphrase Detection: 4개 모델 가중 앙상블, dev accuracy 90.739%
- Sonnet Generation: 3행 prompt 보존, 14행 구조 제어, best-of-N 후보 선택
- 최종 예측 파일은 `predictions/`에 포함

## 실행 환경

과제에서 제공된 환경을 사용합니다.

```bash
conda env create -f env.yml
conda activate nlp_final
```

환경 이름이 다르게 생성되면 `conda env list`로 확인한 이름을 사용하십시오.

## 주요 실행

PART-I 구현 검사:

```bash
python sanity_check.py
python optimizer_test.py
```

감정 분석:

```bash
python classifier.py --fine-tune-mode last-linear-layer
python classifier.py --fine-tune-mode full-model
```

Paraphrase Detection의 전체 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python paraphrase_detection.py --help
python paraphrase_ensemble_n.py --help
```

Sonnet Generation:

```bash
python sonnet_generation.py --help
```

## 산출물

- `predictions/*sst*-out.csv`: SST dev/test 예측
- `predictions/*cfimdb*-out.csv`: CFIMDB dev/test 예측
- `predictions/para-dev-output.csv`: Quora dev 예측
- `predictions/para-test-output.csv`: Quora test 예측
- `predictions/generated_sonnets.txt`: 생성된 최종 소네트
- `final_report_revised_20260614.docx`: 프로젝트 완료 보고서

모델 체크포인트(`*.pt`)와 실험 디렉터리(`runs/`)는 용량 때문에 저장소에서
제외했습니다. 코드와 최종 예측 결과는 체크포인트 없이도 검토할 수 있습니다.

## 기준모델 제출 ZIP 생성

```bash
python prepare_submit.py
```

명세에 맞는 `nlp2026-final-outputs.zip`이 생성됩니다. ZIP 파일은 Git 저장소에
포함하지 않습니다.
