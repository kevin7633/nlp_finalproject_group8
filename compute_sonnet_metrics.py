import argparse
import math
import re

import torch
import torch.nn.functional as F
from sacrebleu.metrics import CHRF

from sonnet_generation import SonnetGPT


def load_numbered_sonnets(path):
    """
    번호가 붙은 sonnet 파일을 읽어서 {id: text} 형태로 반환한다.

    예시 형식:
    132
    line1
    line2
    ...
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r"\n\s*(\d+)\s*\n", text)

    sonnets = {}

    for i in range(1, len(parts) - 1, 2):
        sonnet_id = parts[i].strip()
        body = parts[i + 1]

        lines = [
            line.strip()
            for line in body.split("\n")
            if line.strip()
        ]

        if len(lines) > 0:
            sonnets[sonnet_id] = "\n".join(lines)

    return sonnets


def compute_chrf(generated_path, reference_path):
    generated = load_numbered_sonnets(generated_path)
    reference = load_numbered_sonnets(reference_path)

    common_ids = sorted(
        set(generated.keys()) & set(reference.keys()),
        key=lambda x: int(x)
    )

    if len(common_ids) == 0:
        raise ValueError(
            "generated 파일과 reference 파일 사이에 공통 sonnet id가 없습니다. "
            "dev 생성 파일과 dev 정답 파일을 비교하고 있는지 확인하세요."
        )

    hypotheses = [generated[i] for i in common_ids]
    references = [reference[i] for i in common_ids]

    chrf = CHRF()
    score = chrf.corpus_score(hypotheses, [references]).score

    return score, common_ids


@torch.no_grad()
def compute_perplexity(ckpt_path, reference_path, use_gpu=False):
    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SonnetGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()

    reference = load_numbered_sonnets(reference_path)
    texts = [reference[i] for i in sorted(reference.keys(), key=lambda x: int(x))]

    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        encoding = model.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)

        # t번째 위치의 logits로 t+1번째 token을 예측
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        ).view(shift_labels.size())

        masked_loss = loss_per_token * shift_mask

        total_nll += masked_loss.sum().item()
        total_tokens += shift_mask.sum().item()

    avg_loss = total_nll / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    chrf_score, common_ids = compute_chrf(args.generated, args.reference)
    loss, ppl = compute_perplexity(args.ckpt, args.reference, use_gpu=args.use_gpu)

    print("===== Sonnet Generation Evaluation =====")
    print(f"Compared sonnet ids: {common_ids}")
    print(f"CHRF: {chrf_score:.4f}")
    print(f"Perplexity loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()