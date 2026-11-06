#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   tokenizer.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/10/13 14:14:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/21 15:02:42 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
import regex
from pathlib import Path
from typing import Any, cast


class Tokenizer:
    """Load the vocabulary and configuration of a BPE tokenizer."""

    def __init__(
        self,
        vocab_path: str,
        tokenizer_path: str,
        merges_path: str,
    ) -> None:
        """Initialize the tokenizer from its configuration files."""
        self.vocab_path = Path(vocab_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.merges_path = Path(merges_path)

        self.vocab = self._load_json(self.vocab_path)
        self.config = self._load_json(self.tokenizer_path)
        self.merges = self._load_merges(self.merges_path)

        self.byte_encoder = self._build_byte_encoder()
        self.byte_decoder = {
            char: byte_value
            for byte_value, char in self.byte_encoder.items()
        }

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Load a JSON file."""
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")

        return data

    def _load_merges(self, path: Path) -> list[tuple[str, str]]:
        """Load BPE merge rules in their original order."""
        merges: list[tuple[str, str]] = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid BPE merge rule: {line!r}"
                    )

                merges.append((parts[0], parts[1]))

        return merges

    def inspect_merges(self, limit: int = 10) -> None:
        """Display the first BPE merge rules."""
        print(f"Total merge rules: {len(self.merges)}")

        for rank, (first, second) in enumerate(self.merges[:limit]):
            print(
                f"Rank {rank}: "
                f"{first!r} + {second!r}"
            )

    def apply_bpe(self, symbols: list[str]) -> list[str]:
        """Merge symbols according to BPE ranks."""
        symbols = symbols.copy()

        while len(symbols) > 1:
            pairs = [
                (symbols[i], symbols[i + 1])
                for i in range(len(symbols) - 1)
            ]

            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in pairs
                if pair in self.merge_ranks
            ]

            if not ranked_pairs:
                break

            _, best_pair = min(ranked_pairs)

            merged_symbols: list[str] = []
            i = 0

            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and (symbols[i], symbols[i + 1]) == best_pair
                ):
                    merged_symbols.append(
                        symbols[i] + symbols[i + 1]
                    )
                    i += 2
                else:
                    merged_symbols.append(symbols[i])
                    i += 1

            symbols = merged_symbols

        return symbols

    def _build_byte_encoder(self) -> dict[int, str]:
        """Map every byte to its ByteLevel Unicode symbol."""
        visible_bytes = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        visible_chars = visible_bytes.copy()
        extra = 0

        for byte_value in range(256):
            if byte_value not in visible_bytes:
                visible_bytes.append(byte_value)
                visible_chars.append(256 + extra)
                extra += 1

        return {
            byte_value: chr(char_code)
            for byte_value, char_code in zip(
                visible_bytes, visible_chars
            )
        }

    def encode_bytelevel(self, text: str) -> str:
        """Convert UTF-8 bytes into ByteLevel Unicode symbols."""
        return "".join(
            self.byte_encoder[byte_value]
            for byte_value in text.encode("utf-8")
        )

    def decode_bytelevel(self, text: str) -> str:
        """Convert ByteLevel Unicode symbols back to text."""

        raw_bytes = bytes(
            self.byte_decoder[char]
            for char in text
        )
        return raw_bytes.decode("utf-8", errors="replace")

    def pre_tokenize(self, text: str) -> list[str]:
        """Split text using the Qwen tokenizer's Unicode pattern."""
        pattern = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            r"[^\r\n\p{L}\p{N}]?\p{L}+|"
            r"\p{N}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|"
            r"\s+(?!\S)|"
            r"\s+"
        )
        return cast(list[str], regex.findall(pattern, text))

    def tokenize_to_pieces(self, text: str) -> list[str]:
        """Convert text into ByteLevel BPE pieces."""
        pieces = []

        for part in self.pre_tokenize(text):
            bytelevel_text = self.encode_bytelevel(part)
            symbols = list(bytelevel_text)
            pieces.extend(self.apply_bpe(symbols))

        return pieces

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        pieces = self.tokenize_to_pieces(text)
        token_ids = []

        for piece in pieces:
            token_id = self.vocab.get(piece)
            if token_id is None:
                raise ValueError(f"Token not found in vocabulary: {piece!r}")
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""
        pieces = []

        for token_id in token_ids:
            piece = self.id_to_token.get(token_id)
            if piece is None:
                raise ValueError(f"Unknown token ID: {token_id}")
            pieces.append(piece)

        bytelevel_text = "".join(pieces)
        return self.decode_bytelevel(bytelevel_text)
