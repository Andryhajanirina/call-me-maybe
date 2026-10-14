#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   tokenizer.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/10/13 14:14:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/13 14:28:03 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from pathlib import Path
from typing import Any


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
