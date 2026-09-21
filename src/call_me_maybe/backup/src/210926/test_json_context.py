from llm_sdk import Small_LLM_Model
from call_me_maybe.json_decoder import JSONDecoder
from call_me_maybe.json_context import JSONContext


def main() -> None:
    # model = Small_LLM_Model()
    # decoder = JSONDecoder()
    context = JSONContext()

    # vocab = model._tokenizer.get_vocab()

    # print("Vocabulary size:", len(vocab))

    # test_token_ids = [90, 314, 341, 515, 0, 92]

    # # for token_id in test_token_ids:
    # #     for token_text, vocab_id in vocab.items():
    # #         if vocab_id == token_id:
    # #             print(token_id, repr(token_text))
    # #             break

    # for token_id in test_token_ids:
    #     token_text = model.decode([token_id])
    #     valid = decoder.consume_token(token_text)
    #     if valid:
    #         context.add_key_character(token_text)

    #     # print(
    #     #     token_id,
    #     #     repr(token_text),
    #     #     "->",
    #     #     valid,
    #     #     decoder.state,
    #     # )
    # print(token_id, repr(context.finish_key()))

    key = "age"

    context.start_key()
    for char in key:
        context.add_key_character(char)

    assert context.finish_key() == "age"
    print("JSONContext test: OK")

    context.start_key()

    key = "name"

    for char in key:
        context.add_key_character(char)

    assert context.finish_key() == "name"

    print("JSONContext test: OK")


if __name__ == "__main__":
    main()
