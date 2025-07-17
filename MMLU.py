import os
import csv
import random
import time
import json
from tqdm import tqdm
from datetime import datetime
from run_fusion import get_llama, get_llava

def load_mmlu(args):
    from datasets import load_dataset
    test_data = load_dataset("cais/mmlu", "all", split="test")

    random.seed(args.seed)
    indices = random.sample(range(len(test_data)), args.shot_number + args.nsamples)
    
    shots = []
    for i in indices[:args.shot_number]:
        choices = test_data[i]["choices"]
        shot = {
            "question": test_data[i]["question"],
            "choices": f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
            "answer": test_data[i]["answer"]
        }
        shots.append(shot)
    
    questions = []
    for i in indices[args.shot_number:]:
        choices = test_data[i]["choices"]
        question = {
            "question": test_data[i]["question"],
            "choices": f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
            "answer": test_data[i]["answer"]
        }
        questions.append(question)
    
    return shots, questions


def mmlu_single_question(model, tokenizer, question, shots):
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for shot in shots:
        prompt += f"Question: {shot['question']}\nChoices: {shot['choices']}\nAnswer: {shot['answer']}\n\n"
    prompt += f"Question: {question['question']}\nChoices: {question['choices']}\nAnswer: {question['answer']}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False
    )
    prediction = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
    print(f"prediction: {prediction}, answer: {question['answer']}")
    return prediction.strip() == "ABCD"[question["answer"]].strip()

def mmlu_evaluate(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if "llava" in args.model.lower():
        model = get_llava(args.model)
    else:
        model = get_llama(args.model)

    shots, questions = load_mmlu(args)
    correct = 0
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
        help='Model to load; pass location of huggingface converted checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--shot-number',
        type=int, default=5, help='The number of shots.'
    )
    parser.add_argument(
        '--nsamples',
        type=int, default=500, help='The number of samples.'
    )
    args = parser.parse_args()

    print(f"Accuracy: {mmlu_evaluate(args)}")