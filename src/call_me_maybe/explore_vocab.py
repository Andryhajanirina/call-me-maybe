import json

from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    print("Vocabulary type:", type(vocab))
    print("Vocabulary size:", len(vocab))

    id_to_token = {token_id: token for token, token_id in vocab.items()}

    print("Token 0:", id_to_token[0])
    print("Token 32:", id_to_token[32])
    print("Token 64:", id_to_token[64])

    texts = [
        "Hello",
        " Hello",
        "{",
        "}",
        '"name"',
        '"parameters"',
        ":",
        ",",
    ]

    for text in texts:
        ids = model.encode(text)[0].tolist()
        print(text, "->", ids)


if __name__ == "__main__":
    main()
