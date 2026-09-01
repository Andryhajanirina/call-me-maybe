from llm_sdk import Small_LLM_Model
from .vocabulary import TokenVocabulary


# def main() -> None:
#     model = Small_LLM_Model()

#     input_ids = model.encode("Hello")
#     input_ids_list = input_ids[0].tolist()

#     # print("Input IDs:", input_ids)
#     # print("Input IDs list:", input_ids_list)
#     # print("Decoded:", model.decode(input_ids_list))

#     # logits = model.get_logits_from_input_ids(input_ids_list)

#     # print("Number of logits:", len(logits))
#     # print("Max logit:", max(logits))

#     # # Get index of max logit for getting the corresponding token
#     # max_token_id = logits.index(max(logits))
#     # print("Best token ID:", max_token_id)

#     # print("Best token:", model.decode([max_token_id]))

#     for _ in range(10):
#         logits = model.get_logits_from_input_ids(input_ids_list)

#         token_id = logits.index(max(logits))

#         input_ids_list.append(token_id)

#         print(f"{model.decode([input_ids_list])}")

#     vocab_path = model.get_path_to_vocab_file()

#     print("Vocabulary:", vocab_path)

def main() -> None:
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    vocab = TokenVocabulary(vocab_path)

    print("Vocabulary size:", len(vocab))

    print("ID of '{':", vocab.get_token_id("{"))
    print("Token 90:", repr(vocab.get_token(90)))

    print("ID of '}':", vocab.get_token_id("}"))
    print("Token 92:", repr(vocab.get_token(92)))
