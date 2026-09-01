import json


class TokenVocabulary:
    def __init__(self, vocab_path: str) -> None:
        with open(vocab_path, "r", encoding="utf-8") as file:
            self.token_to_id: dict[str, int] = json.load(file)

        self.id_to_token: dict[int, str] = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

    def get_token_id(self, token: str) -> int | None:
        return self.token_to_id.get(token)

    def get_token(self, token_id: int) -> str | None:
        return self.id_to_token.get(token_id)

    def __len__(self) -> int:
        return len(self.token_to_id)
