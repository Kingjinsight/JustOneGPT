"""Use the Chinese WordPiece tokenizer from a GPT-2 project."""

import re
from pathlib import Path

from transformers import BertTokenizerFast


DEFAULT_TOKENIZER = "uer/gpt2-chinese-cluecorpussmall"


class ChineseTokenizer:
    """Wrap one vocabulary shared by data preparation and text generation."""

    def __init__(self, source: str | Path = DEFAULT_TOKENIZER):
        # Load tokenizer files only; this does not load pretrained GPT weights.
        self.tokenizer = BertTokenizerFast.from_pretrained(str(source))

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        # Encode raw text without BERT's special start and end markers.
        return self.tokenizer.encode(text, add_special_tokens=False, verbose=False)

    def decode(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        # WordPiece inserts spaces around Chinese characters when decoding.
        text = re.sub(r"(?<=[\u3400-\u9fff]) (?=[\u3400-\u9fff，。！？；：、])", "", text)
        return re.sub(r"(?<=[，。！？；：、]) (?=[\u3400-\u9fff])", "", text)

    def save(self, directory: Path) -> None:
        # Keep the vocabulary with its checkpoint instead of depending on a later download.
        self.tokenizer.save_pretrained(directory)
