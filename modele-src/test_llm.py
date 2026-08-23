from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    input_ids = model.encode("Hello")
    input_ids_list = input_ids[0].tolist()

    print("Input IDs:", input_ids)
    print("Input IDs list:", input_ids_list)
    print("Decoded:", model.decode(input_ids_list))

    logits = model.get_logits_from_input_ids(input_ids_list)

    print("Number of logits:", len(logits))
    print("Max logit:", max(logits))