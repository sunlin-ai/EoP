import os
import json
from statistics import mean
from .helper import find_math_answer, delete_extra_zero


def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                          "prealgebra", "precalculus"]:

        for filename in os.listdir(args.dataset_path):
            if (filename.endswith('.json')):
                d = json.load(open(args.dataset_path + '/' + filename))
                questions.append(d['problem'])
                answers.append(find_math_answer(d['solution']))

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(delete_extra_zero(json_res["answer"].split("#### ")[-1].replace(",", "")))

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(delete_extra_zero(a))

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(delete_extra_zero(a))

    elif args.dataset == "commonsenseqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                question = json_res["question"]["stem"].strip() + " " + choice
                answer = json_res["answerKey"]
                questions.append(question)
                answers.append(answer)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for i, line in enumerate(json_data):
                question = line["input"].strip()
                answer = int(line["target_scores"]["Yes"])
                if answer == 1:
                    answer = "yes"
                else:
                    answer = "no"
                questions.append(question)
                answers.append(answer)

    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for i, line in enumerate(json_data):
                question = line["question"]
                answer = line["answer"]
                questions.append(question)
                answers.append(answer)

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers
