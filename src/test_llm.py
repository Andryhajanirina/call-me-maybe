from llm_sdk.llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
input_ids = model.encode("Hello")
print(input_ids)