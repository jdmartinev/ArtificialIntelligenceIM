from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from typing import List

class BPETokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL = [PAD, UNK, BOS, EOS]

    def __init__(self, num_merges: int = 200):
        self.num_merges = num_merges
        self._tok: Tokenizer | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, corpus: str) -> None:
        """Train BPE on `corpus` (plain string)."""
        model = BPE(unk_token=self.UNK)
        self._tok = Tokenizer(model)
        self._tok.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size     = 1000,
            min_frequency  = 1,
            special_tokens = self.SPECIAL,
            show_progress  = False,
        )

        self._tok.train_from_iterator([corpus], trainer=trainer)

        # Wrap sequences with BOS / EOS automatically
        bos_id = self._tok.token_to_id(self.BOS)
        eos_id = self._tok.token_to_id(self.EOS)
        self._tok.post_processor = TemplateProcessing(
            single        = f"{self.BOS} $A {self.EOS}",
            special_tokens = [(self.BOS, bos_id), (self.EOS, eos_id)],
        )

        print(f"[BPE] Training complete. Vocab size: {self.vocab_size}")

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        assert self._tok is not None, "Call train() or load() first."
        ids = self._tok.encode(text).ids
        if not add_special_tokens:
            if ids and ids[0]  == self.bos_id: ids = ids[1:]
            if ids and ids[-1] == self.eos_id: ids = ids[:-1]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        assert self._tok is not None, "Call train() or load() first."
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(self.PAD)

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id(self.BOS)

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(self.EOS)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        assert self._tok is not None, "Nothing to save — call train() first."
        self._tok.save(path)
        print(f"[BPE] Tokenizer saved to {path}")

    def load(self, path: str) -> None:
        self._tok = Tokenizer.from_file(path)
        print(f"[BPE] Tokenizer loaded from {path}  vocab_size={self.vocab_size}")