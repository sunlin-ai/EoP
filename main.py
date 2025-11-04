import os
import json
import time
import jinja2
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from openai import AzureOpenAI
from utils.dataset import data_reader
from utils.helper import answer_clean_all, parse_json_response

prompt_pec = """
Extract premises and clear question from input, output a dictionary with 'premise' and 'question' as keys.

[Demonstration]
Input:
There are 96 fourth-graders at Small Tree School. 43 of them are girls. On Friday, 5 fourth-grade
girls and 4 fourth grade boys were absent. How many fourth grade boys were at Small Tree School on
Friday?
Output:
```json
{
"premises":[
"Small Tree School has a total of 96 fourth-graders.",
"Out of these, 43 are girls.",
"On Friday, 5 girls and 4 boys from the fourth grade were absent."
],
"question": How many fourth-grade boys were present at Small Tree School on Friday?
}
```

[Question to be answered]
Input:
{{question}}
Output:
```json
{
"premises": ["string", ...], // all premises extracted from input
"question": string, //core question from input
}
```
"""

prompt_qr = """
Revise and improve the given question while retaining all its original premises and final result:

Original question:
{{question}}

New question:
"""

prompt_reasoning = """Follow the given demonstration and answer the question.
[Demonstration]
{{demonstrations}}

[Question to be answered]
Question: {{question}}

[Note]
{% if dataset == "aqua" -%}
The final answer in the format of "the answer is ANSWER" should be included, where
ANSWER is one from the options [(a), (b), (c), (d), (e)]. For example, 'the answer is (a)',
'the answer is (b)', 'the answer is (c)'... If the answer is not in the options, select the most possible
option.
{%- else -%}
The final answer in the format of "the answer is" should be included.
{%- endif %}

Answer:
"""


def create_response_chat(prompt_input, eng='gpt-3.5-turbo', max_tokens=256, temperature=0.0, stop="Question"):
    if eng == "gpt-3.5-turbo":
        api_version = "2023-07-01-preview"
        model = "git-35-turbo-chat"
    else:
        api_version = "2024-02-15-preview"
        model = "gpt-4"

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    response = client.chat.completions.create(
        model=model,
        messages=prompt_input,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=[stop]
    )
    return response


def get_answer_from_gpt_sample(prompt, example, eng, max_tokens=256, temperature=0.0):
    stop = "Question:"
    prompt_input = jinja2.Template(prompt).render(example)
    response = create_response_chat([
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt_input},
    ], eng, max_tokens, temperature, stop)
    resp = response.choices[0].message.content.strip()
    return resp


def concat_question_aug(question_org, question_aug):
    text = parse_json_response(question_aug)
    premises = text['premises']
    question = text['question']

    result = ""
    for premise in premises:
        result += premise + " "
    result += question

    if "Answer Choices" in question_org:
        answer_choice = question_org.split("Answer Choices:")[-1]
        result += " Answer Choices:" + answer_choice
    return result


def get_demonstration(file, shot_num=None):
    with open(file, "r", encoding='utf-8') as f:
        prompt = f.read().strip()
    if shot_num is not None:
        prompt_list = prompt.split('\n\n')
        text = ""
        for prompt in prompt_list[:shot_num]:
            text += prompt + "\n\n"
    else:
        text = prompt
    return text


def main():
    # load data
    dataset_path = {"addsub": "AddSub/AddSub.json",
                    "aqua": "AQuA/AQuA.json",
                    "gsm8k": "gsm8k/gsm8k.jsonl",
                    "multiarith": "MultiArith/MultiArith.json",
                    "singleeq": "SingleEq/SingleEq.json",
                    "svamp": "SVAMP/SVAMP.json",
                    "algebra": "math/algebra",
                    "counting_and_probability": "math/counting_and_probability",
                    "geometry": "math/geometry",
                    "intermediate_algebra": "math/intermediate_algebra",
                    "number_theory": "math/number_theory",
                    "prealgebra": "math/prealgebra",
                    "precalculus": "math/precalculus",
                    }

    args.dataset_path = "dataset/{}".format(dataset_path[args.dataset])
    questions, answers = data_reader(args)
    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions))]
    print("loading dataset complete. altogether", len(qa_pairs), "questions")
    NUM_TEST = args.num_test
    if NUM_TEST == -1:
        qa_pairs_test = qa_pairs
    else:
        if args.test_ind is None:
            np.random.seed(args.seed)
            rand_indices = np.random.choice(len(qa_pairs), NUM_TEST, replace=False)
            qa_pairs_test = [qa_pairs[i] for i in rand_indices]
        else:
            with open(args.test_ind, "r") as f:
                test_ind = json.load(f)
            assert len(test_ind) == NUM_TEST
            qa_pairs_test = [qa_pairs[i] for i in test_ind]

    # load demonstrations
    demonstrations = get_demonstration(args.prompt_dir, args.shot_num)
    demonstrations_hint = get_demonstration(args.hint, args.shot_num)

    if args.augment_method == 'pec':
        prompt_augment = prompt_pec
    else:
        prompt_augment = prompt_qr

    # Store org and aug results
    file_name_org = f"results/{args.augment_method}/{args.type}/{args.dataset}_{args.eng}_org_{args.shot_num}.jsonl"
    file_name_aug = f"results/{args.augment_method}/{args.type}/{args.dataset}_{args.eng}_aug_{args.shot_num}.jsonl"
    os.makedirs(os.path.dirname(file_name_org), exist_ok=True)

    count = 0
    correct_org = 0
    correct_aug = 0
    result_org_list = [{}]
    result_aug_list = [{}]
    max_tokens = args.max_tokens

    tbar = tqdm(qa_pairs_test)
    for id, (question, answer) in enumerate(tbar):
        count += 1

        # redefine original question
        try:
            example = {'question': question}
            question_aug = get_answer_from_gpt_sample(
                prompt_augment, example, eng=args.eng, max_tokens=max_tokens, temperature=args.temp)
            if args.augment_method == 'pec':
                question_aug = concat_question_aug(question, question_aug)
        except:
            question_aug = question

        result_org = {'id': id, 'question': question, 'answer': answer}
        result_aug = {'id': id, 'question': question_aug, 'answer': answer}

        count_this = 0
        answer_previous_number_array_org, answer_previous_number_array_aug = [], []
        answer_previous_number_array_org_hint, answer_previous_number_array_aug_hint = [], []

        while True:
            if count_this == 0:
                try:
                    # try to get answer
                    example = {"demonstrations": demonstrations, "question": question, "dataset": args.dataset}
                    answer_org = get_answer_from_gpt_sample(
                        prompt_reasoning, example, eng=args.eng, max_tokens=max_tokens, temperature=args.temp)

                    example_aug = {"demonstrations": demonstrations, "question": question_aug, "dataset": args.dataset}
                    answer_aug = get_answer_from_gpt_sample(
                        prompt_reasoning, example_aug, eng=args.eng, max_tokens=max_tokens, temperature=args.temp)

                    count_this += 1

                    print(f"round {count_this} done!")

                    # clean answer.
                    answer_this_number_check_org, answer_this_hint_org = answer_clean_all(args, [answer_org])
                    answer_previous_number_array_org.append(answer_this_number_check_org)
                    answer_previous_number_array_org_hint.append(answer_this_hint_org)  # used for hint

                    result_org[f'solution_{count_this}'] = answer_org
                    result_org[f'pred_{count_this}'] = answer_this_number_check_org
                    result_org_list.append(result_org)

                    answer_this_number_check_aug, answer_this_hint_aug = answer_clean_all(args, [answer_aug])
                    answer_previous_number_array_aug.append(answer_this_number_check_aug)
                    answer_previous_number_array_aug_hint.append(answer_this_hint_aug)  # used for hint

                    result_aug[f'solution_{count_this}'] = answer_aug
                    result_aug[f'pred_{count_this}'] = answer_this_number_check_aug
                    result_aug_list.append(result_aug)

                    if answer_this_number_check_org == answer_this_number_check_aug:
                        result_final = answer_this_number_check_org

                        if answer_this_number_check_org == answer:
                            correct_org += 1
                            correct_aug += 1

                        result_org['final_pred'] = result_final
                        result_aug['final_pred'] = result_final

                        result_org['is_true'] = bool(result_final == answer)
                        result_aug['is_true'] = bool(result_final == answer)

                        break

                except Exception as e:
                    print(repr(e))
                    traceback.print_exc()
                    time.sleep(args.sleep)

            else:
                try:
                    count_this += 1

                    question_org_new = "{} (Hint: The answer is near to {}).".format(question, ', '.join(
                        ('%s' % id for id in answer_previous_number_array_aug_hint)))

                    question_aug_new = "{} (Hint: The answer is near to {}).".format(question_aug, ', '.join(
                        ('%s' % id for id in answer_previous_number_array_org_hint)))

                    example_org_new = {"demonstrations": demonstrations_hint, "question": question_org_new,
                                       "dataset": args.dataset}
                    answer_org = get_answer_from_gpt_sample(
                        prompt_reasoning, example_org_new, eng=args.eng, max_tokens=max_tokens,
                        temperature=args.temp2)

                    example_aug_new = {"demonstrations": demonstrations_hint, "question": question_aug_new,
                                       "dataset": args.dataset}
                    answer_aug = get_answer_from_gpt_sample(
                        prompt_reasoning, example_aug_new, eng=args.eng, max_tokens=max_tokens,
                        temperature=args.temp2)

                    print(f"round {count_this} done!")

                    # clean answer.
                    answer_this_number_check_org, answer_this_hint_org = answer_clean_all(args, [answer_org])
                    answer_previous_number_array_org.append(answer_this_number_check_org)
                    answer_previous_number_array_org_hint.append(answer_this_hint_org)  # used for hint
                    result_org[f'solution_{count_this}'] = answer_org
                    result_org[f'pred_{count_this}'] = answer_this_number_check_org

                    answer_this_number_check_aug, answer_this_hint_aug = answer_clean_all(args, [answer_aug])
                    answer_previous_number_array_aug.append(answer_this_number_check_aug)
                    answer_previous_number_array_aug_hint.append(answer_this_hint_aug)  # used for hint
                    result_aug[f'solution_{count_this}'] = answer_aug
                    result_aug[f'pred_{count_this}'] = answer_this_number_check_aug

                    is_break = False
                    if answer_this_number_check_org == answer_this_number_check_aug:
                        result_final = answer_this_number_check_org
                        is_break = True

                    elif len(list(set(answer_previous_number_array_org[-min(len(answer_previous_number_array_org),
                                                                            args.hint_length):]))) == 1 and count_this >= args.hint_length:
                        result_final = answer_this_number_check_org
                        is_break = True

                    elif len(list(set(answer_previous_number_array_aug[-min(len(answer_previous_number_array_aug),
                                                                            args.hint_length):]))) == 1 and count_this >= args.hint_length:
                        result_final = answer_this_number_check_aug
                        is_break = True

                    if is_break:
                        if answer_this_number_check_org == answer:
                            correct_org += 1

                        if answer_this_number_check_aug == answer:
                            correct_aug += 1

                        result_org['final_pred'] = result_final
                        result_aug['final_pred'] = result_final

                        result_org['is_true'] = bool(result_final == answer)
                        result_aug['is_true'] = bool(result_final == answer)

                        result_org_list[-1] = result_org
                        result_aug_list[-1] = result_aug

                        break

                except Exception as e:
                    print(repr(e))
                    traceback.print_exc()
                    time.sleep(args.sleep)

        result_org_list[0] = {'acc': correct_org / count, 'correct': correct_org, 'count': count}
        result_aug_list[0] = {'acc': correct_aug / count, 'correct': correct_aug, 'count': count}

        # write result
        with open(os.path.join(file_name_org), 'w', encoding='utf-8') as f:
            json.dump(result_org_list, f, indent=2, ensure_ascii=False)

        with open(os.path.join(file_name_aug), 'w', encoding='utf-8') as f:
            json.dump(result_aug_list, f, indent=2, ensure_ascii=False)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="cot", type=str,
                        help="we use prompt so that the method is few-shot")
    parser.add_argument("--prompt_dir", type=str,
                        default="utils/prompt/cot/cot_base_aqua.txt",
                        help="directory to prompt file (.txt)")
    parser.add_argument("--hint", type=str,
                        default="utils/prompt/cot/cot_PHP_aqua.txt",
                        help="directory to progressive-hint prompt file (.txt)")
    parser.add_argument("--eng", type=str, help="engine", default="gpt-3.5-turbo")
    parser.add_argument("--dataset", type=str, default="aqua", help="the dataset name")
    parser.add_argument("--shot_num", default=3, type=int, help="")
    parser.add_argument("--augment_method", default="pec", type=str, choices=["pec", "qr"])

    parser.add_argument("--num_test", default=-1, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--temp2", default=0.0, type=float, help="temperature for progressive-hint generation")
    parser.add_argument("--max_tokens", default=1024, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str,
                        help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix", default="", type=str, help="")

    parser.add_argument("--hint_length", default=2, type=int,
                        help="return after the last hint_lenght answers are the same")
    parser.add_argument("--direct_answer_trigger_for_fewshot", default="The answer is", type=str,
                        help="used for extract answer")
    parser.add_argument("--method", default="few_shot", type=str,
                        help="we use prompt so that the method is few-shot")
    parser.add_argument("--sample", default=1, type=int, help="sample path number")
    parser.add_argument("--sleep", default=1, type=int,
                        help="sleep time after error.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    azure_endpoint = "Put Your azure_endpoint url"
    api_key = "Put Your Key Here"

    args = load_args()
    print(args)
    main()
