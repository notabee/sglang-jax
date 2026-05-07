from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-FP8", trust_remote_code=True)

tokens = ["[gMASK]", "<sop>", "<|user|>", "<|assistant|>"]
for t in tokens:
    ids = tokenizer.encode(t, add_special_tokens=False)
    print(f"Token: {t}, IDs: {ids}")

print(f"Special tokens map: {tokenizer.special_tokens_map}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
