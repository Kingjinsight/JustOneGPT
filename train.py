"""Train a small GPT on Chinese text, or sample from its checkpoint."""

import argparse
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import ChineseTokenizer, DEFAULT_TOKENIZER


CHAPTER = re.compile(r"(?m)^[ \t]*第[零〇一二三四五六七八九十百千万两0-9]+[章节回篇][^\n]*")


def parse_args() -> argparse.Namespace:
    """Read command-line overrides for the defaults below."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="path to the source text")
    parser.add_argument("--encoding", default="gb18030", help="source text encoding")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Chinese tokenizer model ID or local directory")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)

    # These values are passed to GPTConfig, overriding the defaults in model.py.
    model = parser.add_argument_group("model")
    model.add_argument("--block-size", type=int, default=1024)
    model.add_argument("--n-layer", type=int, default=6)
    model.add_argument("--n-head", type=int, default=4)
    model.add_argument("--n-embd", type=int, default=256)
    model.add_argument("--dropout", type=float, default=0.1)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=32)
    # Total parameter updates, including steps already completed when resuming.
    training.add_argument("--steps", type=int, default=10000)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--eval-interval", type=int, default=100)
    training.add_argument("--eval-iters", type=int, default=20)
    training.add_argument("--log-interval", type=int, default=10)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue from out-dir/last.pt")
    mode.add_argument("--sample", type=str, help="generate from out-dir/best.pt")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)

    args = parser.parse_args()
    if args.sample is None and args.data is None:
        parser.error("--data is required for training")
    if min(args.block_size, args.n_layer, args.n_head, args.n_embd, args.batch_size,
           args.steps, args.eval_interval, args.eval_iters, args.log_interval) <= 0:
        parser.error("model sizes, batch size, steps and intervals must be positive")
    if args.temperature <= 0 or args.top_k <= 0 or args.max_new_tokens < 0:
        parser.error("temperature and top-k must be positive; max-new-tokens cannot be negative")
    return args


def select_device(name: str) -> torch.device:
    """Automatically prefer CUDA, then Apple's MPS, then CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def load_text(path: Path, encoding: str) -> tuple[str, str, int]:
    """Read the book and reserve its final 10% of chapters for validation."""
    text = path.read_text(encoding=encoding)
    matches = list(CHAPTER.finditer(text))
    if len(matches) < 2:
        raise ValueError("source text needs at least two chapter headings")

    # Split whole chapters so a chapter never appears in both sets.
    # Text before the first chapter heading is not included.
    chapters = [text[match.start():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
                for i, match in enumerate(matches)]
    split = int(0.9 * len(chapters))
    return "".join(chapters[:split]), "".join(chapters[split:]), len(chapters)


def encode_text(train_text: str, val_text: str,
                tokenizer: ChineseTokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Use the same tokenizer to turn both text splits into arrays of token IDs."""
    train_ids = np.asarray(tokenizer.encode(train_text), dtype=np.int64)
    val_ids = np.asarray(tokenizer.encode(val_text), dtype=np.int64)
    return train_ids, val_ids


def get_batch(data: np.ndarray, batch_size: int, block_size: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows instead of traversing the dataset in fixed epochs."""
    starts = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    # Predict one token ahead: a stream [A, B, C, D] gives X=[A, B, C], Y=[B, C, D].
    x = torch.from_numpy(np.stack([data[i:i + block_size] for i in starts]))
    y = torch.from_numpy(np.stack([data[i + 1:i + block_size + 1] for i in starts]))
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, train_data: np.ndarray, val_data: np.ndarray,
                  args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    """Average several batches without computing gradients or updating weights."""
    # Evaluation disables dropout, making train and validation estimates comparable.
    model.eval()
    losses = {}
    for name, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, model.config.block_size, device)
            _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / args.eval_iters
    model.train()
    return losses


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer,
                    step: int, best_val_loss: float) -> None:
    """Save weights, architecture and optimizer history so training can continue."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    # Finish writing before replacing the previous checkpoint.
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def sample(args: argparse.Namespace, device: torch.device) -> None:
    """Load the best validation checkpoint and continue the supplied text opening."""
    checkpoint = torch.load(args.out_dir / "best.pt", map_location="cpu", weights_only=True)
    # Recreate the saved architecture, regardless of today's command-line defaults.
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    # Token IDs must have the same meaning as they did during training.
    tokenizer = ChineseTokenizer(args.out_dir / "tokenizer")
    prompt = torch.tensor([tokenizer.encode(args.sample)], device=device)
    output = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist()))


def train(args: argparse.Namespace, device: torch.device) -> None:
    """Prepare data and model, then alternate weight updates with evaluation."""
    train_text, val_text, chapter_count = load_text(args.data, args.encoding)
    tokenizer = ChineseTokenizer(args.out_dir / "tokenizer" if args.resume else args.tokenizer)
    train_data, val_data = encode_text(train_text, val_text, tokenizer)
    print(f"chapters: {chapter_count}, train tokens: {len(train_data):,}, "
          f"val tokens: {len(val_data):,}, vocab size: {tokenizer.vocab_size:,}")

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.out_dir / "last.pt", map_location="cpu", weights_only=True)
        # Resume keeps the saved layers and context length; model flags do not resize it.
        config = GPTConfig(**checkpoint["model_config"])
    else:
        config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                           n_layer=args.n_layer, n_head=args.n_head,
                           n_embd=args.n_embd, dropout=args.dropout)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary differs from the checkpoint")
    if min(len(train_data), len(val_data)) <= config.block_size:
        raise ValueError("train and validation sets must exceed block_size")

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    step, best_val_loss = 0, float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Keep AdamW's history, but use the learning rate requested for this run.
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        step, best_val_loss = checkpoint["step"], checkpoint["best_val_loss"]
    if step >= args.steps:
        raise ValueError("--steps must exceed the step stored in the checkpoint")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        # Save the tokenizer locally so future sampling and resume use the same IDs.
        tokenizer.save(args.out_dir / "tokenizer")
    print(f"device: {device}, parameters: {sum(p.numel() for p in model.parameters()):,}, "
          f"starting step: {step}")
    model.train()
    # Elapsed time measures this invocation, not earlier training sessions.
    started = time.perf_counter()
    while step < args.steps:
        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        # Each update uses this batch only, so discard the previous batch's gradients.
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        # Limit the gradient norm to prevent an unusually large parameter update.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1

        if step == 1 or step % args.log_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}, elapsed {time.perf_counter() - started:.1f}s")
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, args, device)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            # best.pt is for sampling; last.pt always holds the latest saved progress.
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(args.out_dir / "best.pt", model, optimizer, step, best_val_loss)
            save_checkpoint(args.out_dir / "last.pt", model, optimizer, step, best_val_loss)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    # A supplied text opening selects generation; otherwise run training.
    if args.sample is not None:
        sample(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
