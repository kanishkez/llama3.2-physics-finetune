from unsloth import FastLanguageModel
import torch
import time

MODEL_PATH = "final_model"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
DTYPE = None
MAX_HISTORY_TOKENS = 1800
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)
model = model.to(DEVICE)

conversation_history = [
    {"role": "system", "content": "You are a physics expert. Answer clearly and accurately."}
]

def truncate_history():
    global conversation_history
    encoded = tokenizer.apply_chat_template(conversation_history, tokenize=False)
    if len(encoded) > MAX_HISTORY_TOKENS:
        conversation_history = conversation_history[:1] + conversation_history[-4:]

def chat(user_message):
    conversation_history.append({"role": "user", "content": user_message})
    truncate_history()
    prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        use_cache=True
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded[len(prompt):].strip()
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

def main():
    print("="*60)
    print("Physics Expert Chatbot")
    print("="*60)
    print("Type 'quit' to exit or 'reset' to clear conversation.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_input.lower() == 'reset':
            conversation_history.clear()
            conversation_history.append({"role": "system", "content": "You are a physics expert. Answer clearly and accurately."})
            print("Conversation reset!\n")
            continue
        if not user_input:
            continue
        print("\nAssistant: ", end="", flush=True)
        start = time.time()
        answer = chat(user_input)
        print(answer)
        print(f"\n--- Response Time: {time.time() - start:.2f}s ---\n")

if __name__ == "__main__":
    main()
