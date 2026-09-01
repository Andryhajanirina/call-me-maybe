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

    json_chars = ["{", "}", "[", "]", ":", ",", '"']

    for char in json_chars:
        print(f"\nTokens contenant {char!r}:")
        for token_id, token in id_to_token.items():
            if char in token and len(token) <= 4:
                print(token_id, repr(token))

    # print("Token 0:", id_to_token[0])
    # print("Token 32:", id_to_token[32])
    # print("Token 64:", id_to_token[64])

    # texts = [
    #     "Hello",
    #     " Hello",
    #     "{",
    #     "}",
    #     '"name"',
    #     '"parameters"',
    #     ":",
    #     ",",
    # ]
    texts = [
        "{",
        '{"',
        '{"name"',
        '{"name":',
        '{"name":"',
        '{"name":"test"',
        '{"name":"test"}',
        '{"parameters":',
    ]

    # for text in texts:
    #     ids = model.encode(text)[0].tolist()
    #     print(text, "->", ids)
    for text in texts:
        ids = model.encode(text)[0].tolist()
        print(repr(text), "->", ids)

        for token_id in ids:
            print("   ", token_id, repr(id_to_token.get(token_id)))


if __name__ == "__main__":
    main()
