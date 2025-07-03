import argparse
import random
import glob

from tqdm import tqdm
import re
import sys
import os
import numpy as np


def generate_few_shot(prompt):
    base_gsm8k_list = [
        {
            'question': "John and his best friend Steve bought 12 cupcakes together.  Each cupcake cost $1.50. If they split the costs evenly, how much did each person pay?",
            'answer': "The total cost of cupcakes was 1.5*12=$<<1.5*12=18>>18\\nSo they each paid 18/2=$<<18/2=9>>9.",
            'direct_answer': "9"
        },
        {
            'question': "Lizzy has to ship 540 pounds of fish that are packed into 30-pound crates. If the shipping cost of each crate is $1.5, how much will Lizzy pay for the shipment?",
            'answer': "There are 540 pounds / 30 pounds/crate = <<540/30=18>>18 crates of fish needed.\\nHence, the total cost for the shipment is $1.5/crate x 18 crates = $<<1.5*18=27>>27.",
            'direct_answer': "27"
        },
        {
            'question': "Tom, Tim, and Paul are collecting photos of cars. Paul has 10 photos more than Tim. Tim has one hundred photos less than the total amount of photos which is 152. How many photos does Tom have?",
            'answer': "Tim has 152 photos - 100 photos = <<152-100=52>>52 photos.\\nWhen Tim has 52 photos, then Paul has 52 + 10 photos = <<52+10=62>>62 photos.\\nTim and Paul have together 52 photos + 62 photos = <<52+62=114>>114 photos.\\nThat leaves Tom with 152 photos - 114 photos = <<152-114=38>>38 photos.",
            'direct_answer': "38"
        },

    ]
    index_list = list(range(len(base_gsm8k_list)))
    random.shuffle(index_list)
    few_shot_example = ""
    for i in index_list:
        item = base_gsm8k_list[i]
        few_shot_example += "Q: " + item['question'] + "\n" + "A: "+ item['answer']  + "\nThe answer is " + item['direct_answer'] + "\n"

    few_shot_example += "Q: " + prompt + "A: "
    return few_shot_example



def generate_prompt_generation(args, question):
    if args.evaluation_mode == 'generation':
        if args.method == 'zero_shot_cot':
            content = question + " Let's think step by step."
        elif args.method == 'zero_shot':
            content = question
        elif args.method == 'few_shot':
            content = generate_few_shot(question)
        else:
            raise ValueError("we do not method for such model type yet")

    if "generator" not in args.model_type:
        MODEL_DICT = {
            "llama": (
                "[INST] \n{content}\n [/INST]"
            ),
            "mistral": (
                "<s>[INST] {content} [/INST]"
            ),
            "chatglm": (
                "<|user|> \n{content}\n <|assistant|>"
            ),
            "qianwen": (
                "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            ),
            "deepseek-math": (
                "User: {content}\n\nAssistant: "
            ),
            "internlm2-math": (
                "<|im_start|>system\n{content}<|im_end|>\n"
            ),
            "llemma": (
                "### System Prompt\nYou are an intelligent mathematical assistant.\n\n### User Message\n{content}\n\n### Assistant"
            ),
        }

        if args.model_type in ["qianwen", "qianwen-13b", "qianwen-70b"]:
            content = MODEL_DICT['qianwen'].format_map(
                {'content': content}
            )

        elif args.model_type in ["chatglm","deepseek-math-7b-base"]:
            pass


        elif args.model_type in ['llama2-7b-chat']:
            content = MODEL_DICT['llama'].format_map(
                {'content': content}
            )

        elif args.model_type in ["mistral", 'mixtral', "Mistral-7B-Instruct-v0.2"]:
            content = MODEL_DICT['mistral'].format_map(
                {'content': content}
            )

        elif args.model_type in ["internlm2-math-20b", 'internlm2-math-7b']:
            content = MODEL_DICT['internlm2-math'].format_map(
                {'content': content}
            )
        elif args.model_type in ["llemma_34b", 'llemma_7b']:
            content = MODEL_DICT['llemma'].format_map(
                {'content': content}
            )
        elif args.model_type in ["deepseek-math-7b-instruct"]:
            content = MODEL_DICT['deepseek-math'].format_map(
                {'content': content}
            )

    return content




few_shot_list = [
    {
        'question': "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        'answer': "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        'direct_answer': "6"
    },
    {
        'question': "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        'answer': "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        'direct_answer': "5",
    },
    {
        'question': "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        'answer': "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        'direct_answer': "39",
    },
    {
        'question': "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        'answer': "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        'direct_answer': "8",
    },
    {
        'question': "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        'answer': "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
        'direct_answer': "9",
    },
    {
        'question': "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        'answer': "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
        'direct_answer': "29",
    },
    {
        'question': "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        'answer': "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        'direct_answer': "33",
    },
    {
        'question': "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        'answer': "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
        'direct_answer': "8",
    },
]
import json

from collections import Counter


def self_consistency(pairs):
    val_counts = Counter(value for key, value in pairs)
    most = val_counts.most_common(1)[0][0]
    for key, value in pairs:
        if value == most:
            return key


#
def find_feedback(content):
    match = re.search(r'Judgement: (.+)', content)
    if match:
        judgement = match.group(1)
    else:
        judgement = "None"
    return judgement


def str2bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('invalid value: {}, must be true or false'.format(s))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    # parser.add_argument(
    #     "--dataset", type=str, default="plan",
    #     choices=["plan", 'tool_use_awareness', 'tool_selection', 'tool_selection_harder', 'tool_creation_awareness',
    #              'tool_creation_awareness_harder', 'tool_creation',
    #              'arguments_filling'], help="dataset used for experiment")
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--evaluation_mode", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_method", type=str, default="")

    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--model_type", type=str, default="chatglm")

    parser.add_argument("--output_dir", type=str, default="generation_test")

    parser.add_argument("--lora_path", type=str, default="")

    parser.add_argument("--iter_num", type=int, default=1)
    parser.add_argument("--method", type=str, default="few_shot_cot")
    parser.add_argument("--data_question_key", type=str, default="question")
    parser.add_argument("--data_answer_key", type=str, default="answer")

    parser.add_argument("--sample_num", type=int, default=1)

    parser.add_argument("--cuda_ind", type=int, default=0)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--cuda_start", type=int, default=0)
    parser.add_argument("--cuda_num", type=int, default=8)

    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--rewrite", type=str2bool, default=False)
    parser.add_argument("--prompt_key", type=str, default="lean")

    parser.add_argument("--use_typewriter", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--iter_max_new_tokens", type=int, default=512)
    parser.add_argument("--init_max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--correct_response_format", type=str, default="The correct response is:")

    args = parser.parse_args()
    if args.evaluation_mode == 'generation':
        if "lean" in args.dataset:
            args.data_question_key = 'model_response'
            args.data_answer_key = 'statement_poof'

        if args.dataset == "lean4_5k_test":
            args.data_path = "data/lean4_gpt_5k/test/data.jsonl"
        elif args.dataset == "lean4_15k_train":
            args.data_path = "data/lean4_random/15k_filtered.json"
            args.prompt_key= "lean"

        elif args.dataset == "math_train":
            args.data_path = "data/test/math/train.jsonl"
            args.prompt_key = 'wild' 

        elif args.dataset == "gsm8k_train":
            args.data_path = "data/test/gsm8k/train.jsonl"
            args.prompt_key = 'wild' 
            
        elif args.dataset == "wild_test":
            args.data_path = "data/wild/wild_sample1k.jsonl"
            args.prompt_key = 'wild' 

        elif args.dataset == "lean4_basic_test":
            args.data_path = "data/lean4_basic/1k_test_filtered.jsonl"
            args.prompt_key = 'lean' 

        elif args.dataset == "lean4_random_test":
            args.data_path = "data/lean4_random/1k_test_filtered.json"
            args.prompt_key = 'lean' 


        elif args.dataset == "minif2f_valid":
            args.data_path = "miniF2F-lean4/handle_minif2f_valid.jsonl"
            args.prompt_key = 'atp' 
            args.data_question_key = 'formal_statement'

        elif args.dataset == "minif2f_test":
            args.data_question_key = 'formal_statement'
            args.data_path = "miniF2F-lean4/handle_minif2f_test.jsonl"
            args.prompt_key = 'atp' 


        elif args.dataset == "lean4_random_first_train":
            args.data_path = "data/lean4_random/5k_first.json"
        elif args.dataset == "lean4_random_second_train":
            args.data_path = "data/lean4_random/5k_second.json"
        elif args.dataset == "lean4_random_third_train":
            args.data_path = "data/lean4_random/5k_third.json"

    if args.model_type == 'mistral_generator':
        args.model_path = 'models/gsm8k/generators/mistral-ep2/'
    elif args.model_type == 'mistral_generator_original':
        args.model_path = '/data/OVM-Mistral-7b/mistral7b-ep2/'
    elif args.model_type == 'gemma_generator':
        args.model_path = 'models/gsm8k/generators/gemma2b2-ep2/'
    elif args.model_type == 'phi2_generator':
        args.model_path = 'models/gsm8k/generators/phi2b-ep2/'

    elif args.model_type == 'mixtral':
        args.model_path = '/data/Mixtral-8x7B-Instruct-v0.1'

    elif args.model_type == 'mistral':
        args.model_path = '/data/mistral-instruct'

    elif args.model_type == 'qianwen-70b':
        args.model_path = '/data/Qwen-72B-Chat'


    elif args.model_type == 'llama2-7b-chat':
        args.model_path = '/data/Llama-2-7b-chat/'

    elif args.model_type == 'deepseek-math-7b-base':
        args.model_path = '/data/models/deepseek-math-7b-base'
        
    elif args.model_type == 'deepseek-math-7b-instruct':
        args.model_path = '/data/models/deepseek-math-7b-instruct'

    elif args.model_type == 'llemma_7b':
        args.model_path = '/data/models/llemma_7b'
    elif args.model_type == 'llemma_34b':
        args.model_path = '/data/models/llemma_34b'
    elif args.model_type == 'internlm2-math-7b':
        args.model_path = '/data/models/internlm2-math-7b'
    elif args.model_type == 'internlm2-math-20b':
        args.model_path = '/data/models/internlm/internlm2-math-20b'

    elif args.model_type == 'Mistral-7B-Instruct-v0.2':
        args.model_path = '/data/models/Mistral-7B-Instruct-v0.2'

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."

    return args


def create_demo_text(args, cot_flag, index_list):
    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        item = few_shot_list[i]
        if cot_flag:
            demo_text += "Q: " + item['question'] + "\nA: " + item['answer'] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + item['direct_answer'] + ".\n\n"
        else:
            demo_text += "Q: " + item['question'] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + item['direct_answer'] + ".\n\n"

    return demo_text


def str2bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('invalid value: {}, must be true or false'.format(s))


def batchify(pairs, batch_size):

    """将列表分成指定大小的批次"""
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]


def generate_prompts(questions, args):
    """为每个问题生成提示"""
    prompts = [generate_prompt_generation(args, question) for question in questions]
    return prompts

PROMPT_DICT = {
    "wild": (
        "Statement and proof in natural language:\n\n"
        "# Problem:\n{question}\n\n"
        "# Proof:\n{answer}\n\n"
        "Translate the statement and proof in natural language to lean4:"
    ),

    "atp": (
        "Statement and proof in natural language:\n\n"
        "# Problem:\n{question}\n\n"
        "# Proof:\n{answer}\n\n"
        "Translate the statement and proof in natural language to lean4:"
        "{statement}"
    ),
    "lean4": (
        "Statement and proof in natural language:\n\n"
        "{model_response}\n\n"
        "Translate the statement and proof in natural language to lean4:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def get_question_answer(args):
    allfilepath = args.data_path
    questions = []
    answers = []


    # Attempt to read the file as a regular JSON file
    for filepath in allfilepath.split(','):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                # If the data is a list, assume it's an array of objects
                if isinstance(data, list):
                    for json_item in data:
                        questions.append(json_item[args.data_question_key])
                        answers.append(json_item)
                # If the data is a dict, assume it's a single object (or adjust logic as needed)
                elif isinstance(data, dict):
                    questions.append(data[args.data_question_key])
                    answers.append(json_item)

        except ValueError:
            # If it fails, assume the file is in JSON Lines format
            with open(filepath, 'r') as file:
                for line in file:
                    json_item = json.loads(line)
                    questions.append(json_item[args.data_question_key])
                    answers.append(json_item)

    if args.prompt_key == 'wild':
        questions  = [ PROMPT_DICT['wild'].format(question= questions[id], answer =answers[id][args.data_answer_key] )  for id in range(len(questions))]

    elif args.prompt_key == 'lean'  :
        questions  = [ PROMPT_DICT['lean4'].format(model_response = item)  for item in questions]

    elif args.prompt_key == 'atp'  :
        questions  = [ PROMPT_DICT['atp'].format(question= answers[id]['informal_stmt'] ,answer =answers[id]['informal_proof'], statement = re.sub(r'\bsorry\b', '', answers[id]['formal_statement'], flags=re.IGNORECASE) )  for id in range(len(questions))]

    else:
        raise ValueError("we do not yet inplement")
    
    print("++ Prompt : \n", questions[0])
    return questions, answers



def main3(args):
    from vllm import LLM, SamplingParams
    import torch

    model = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel, gpu_memory_utilization = 0.95)
    print(args.model_path)

    if "qianwen" in args.model_type:
        model.llm_engine.tokenizer.eos_token_id = 151645
        # model.llm_engine.tokenizer.pad_token_id = 151645
        model.llm_engine.tokenizer.pad_token_id = None
        # model.llm_engine.tokenizer.eos_token_id = None


    print("load data")


    questions, answers = get_question_answer(args)



    question_exist_list = []
    write_pattern = 'w' if args.rewrite else "a+"
    if os.path.exists(args.output_dir) and not args.rewrite :
        # 如果文件存在，从文件中读取数据加载到response_list
        # Loop through each file that matches the pattern
        file_pattern = os.path.join(args.output_dir, '[0-9]*.json')
        for file_path in glob.glob(file_pattern):
            # Open and read the JSON file
            with open(file_path, 'r') as fp:
                # Extract the 'question' field from each line and add it to the list
                for line in fp.readlines():
                    question_exist_list.append(json.loads(line)['question'])
    else:
        try:
            os.mkdir(args.output_dir)
        except:
            pass
    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions)) if questions[idx] not in question_exist_list ]
    cuda_pieces = np.array_split(range(len(qa_pairs)), args.cuda_num // args.tensor_parallel)
    print(f"fitered {len(questions) - len(qa_pairs)} already")

    with open(f"{args.output_dir}/{args.cuda_ind // args.tensor_parallel + args.cuda_start}.json", write_pattern,
              encoding='utf-8') as wf:
        start = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][0]
        end = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][-1] + 1
        subset_length = end - start
        total_batches = (subset_length + args.batch_size - 1) // args.batch_size  # Calculate the total number of batches
        for batch in tqdm(batchify(qa_pairs[start:end], args.batch_size), total=total_batches):
            questions, answers = zip(*batch)  # 解压问题和答案
            prompts = generate_prompts(questions, args)

            with torch.no_grad():
                output_all = []
                try:
                    for i in range(args.sample_num):
                        sample_list = []
                        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                                         max_tokens=args.init_max_new_tokens)
                        generations = model.generate(prompts, sampling_params, use_tqdm=False)
                        for generation_output in generations:
                            output = generation_output.outputs[0].text
                            sample_list.append(output)
                        output_all.append(sample_list)

                    output_all = list(map(list, zip(*output_all)))
                except Exception as e:
                    print(str(e))
                    exit
                dicts = []
                for question, answer, output, prompt in zip(questions, answers, output_all, prompts):
                    dicts.append({
                        "question": question,
                        "prompt": prompt,
                        "content": answer,
                        "total output": output,
                    })

                for dict in dicts:
                    wf.writelines(json.dumps(dict, ensure_ascii=False) + '\n')

                wf.flush()


def main(argv=None):
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    if args.evaluation_mode == 'generation':
        main3(args)
    else:
        raise ValueError("we do not yet inplement")


if __name__ == "__main__":
    main()
