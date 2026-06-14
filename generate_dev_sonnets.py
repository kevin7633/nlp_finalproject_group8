import argparse
import torch

from datasets import SonnetsDataset
from sonnet_generation import (
    SonnetGPT,
    seed_everything,
    clean_sonnet,
    candidate_quality_score,
    is_complete_sonnet_candidate,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="9_10-1.1e-05-sonnet.pt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
    parser.add_argument("--sonnet_out", type=str, default="generated_sonnets_dev.txt")
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--target_lines", type=int, default=14)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.88)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--min_tokens_per_line", type=int, default=5)
    parser.add_argument("--max_tokens_per_line", type=int, default=12)
    parser.add_argument("--newline_bias", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=11711)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")

    saved = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = SonnetGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()

    held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []

    for idx, prompt_text in held_out_dataset:
        prompt_lines = [
            line.strip()
            for line in prompt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]

        remaining_lines = max(1, args.target_lines - len(prompt_lines))
        prompt_for_generation = "\n".join(prompt_lines).strip() + "\n"

        encoding = model.tokenizer(
            prompt_for_generation,
            return_tensors="pt",
            padding=False,
            truncation=True
        )

        input_ids = encoding["input_ids"].to(device)

        best_score = -float("inf")
        best_sonnet = None

        for _ in range(args.num_candidates):
            _, generated_text = model.generate(
                input_ids,
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

            generated_lines = [
                line.strip()
                for line in generated_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                if line.strip()
            ]

            candidate_lines = prompt_lines + generated_lines
            candidate_text = "\n".join(candidate_lines[:args.target_lines])
            candidate_text = clean_sonnet(candidate_text, max_lines=args.target_lines)

            score = candidate_quality_score(candidate_text, target_lines=args.target_lines)

            if is_complete_sonnet_candidate(candidate_text, target_lines=args.target_lines):
                score += 5.0

            if score > best_score:
                best_score = score
                best_sonnet = candidate_text

        # dev true file id가 132~143인 경우를 맞추기 위해 132부터 시작
        sonnet_id = str(132 + idx)
        generated_sonnets.append((sonnet_id, best_sonnet))

        print(f"sonnet_id={sonnet_id}, best_score={best_score:.4f}")

    with open(args.sonnet_out, "w", encoding="utf-8") as f:
        f.write("--Generated Sonnets--\n\n")
        for sonnet_id, sonnet_text in generated_sonnets:
            f.write(f"{sonnet_id}\n")
            f.write(sonnet_text.strip())
            f.write("\n\n")

    print(f"Saved to {args.sonnet_out}")


if __name__ == "__main__":
    main()