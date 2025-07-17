import os
import csv
import time
import json
from tqdm import tqdm
from datetime import datetime
from run_fusion import get_llama, get_llava

def load_dataset(file_path, args):
    questions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if len(row) < 6:
                    continue
                answer = row[5].strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                question = {
                    "question": row[0].strip(),
                    "choices": [row[i].strip() for i in range(1,5)],
                    "answer": answer
                }
                questions.append(question)
                if len(questions) >= args.max_questions:
                    break
    except Exception as e:
        print(f"Failed to load file: {file_path}")
    return questions


def mmlu_single_question(model, tokenizer, question, shots):
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for shot in range(shots):
        prompt += f"Question: {shot['question']}\nChoices: {shot["choices"]}\nAnswer: {shot['answer']}\n\n"
    prompt += f"Question: {question['question']}\nChoices: {question["choices"]}\nAnswer: {question['answer']}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False
    )
    prediction = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
    return prediction.strip() == question["answer"].strip()

def mmlu_evaluate(args):
    questions = load_dataset(args.dataset, args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if "llava" in args.model.lower():
        model = get_llava(args.model)
    else:
        model = get_llama(args.model)

    correct = 0
    shots_num = args.shot_number
    shots = questions[:shots_num]
    print("MMLU start")
    for question in tqdm(questions, desc= 'Processing...'):
        if mmlu_single_question(model, tokenizer, question, shots):
            correct += 1
    accuracy = correct / 50
    return accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str,
        help='The path of the dataset.'
    )
    parser.add_argument(
        '--shot-number',
        type=int, default=5, help='The number of shots.'
    )
    parser.add_argument(
        '--max-questions',
        type=int, default=500, help='The max question number.'
    )
    args = parser.parse_args()

    print(f"Accuracy: {mmlu_evaluate(args)}")