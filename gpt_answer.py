import requests
import json
import argparse
from tqdm import tqdm
import time
url = 'http://velenmis-rc.sankuai.com/velen-container/chatgpt/chatCompletions'

# prompt_template = (
#     "You are an AI assistant who will help me to judge whether a model's answer to a multiple-choice question is correct.\n"
#     "You are provided with a question, a list of options, the correct answer, and the model's output.\n"
#     "Your task is to compare the model's output with the correct answer and the provided options, and determine:\n"
#     "- If the model's output matches the correct answer, output 'Correct'.\n"
#     "- If the model's output is one of the options but does not match the correct answer, output 'Incorrect'.\n"
#     "- If the model's output does not match any of the provided options, output 'Unknown'.\n"
#     "Your output should be a single word among the following 3 choices: Correct, Incorrect, Unknown.\n"
#     "Question: {question}\n"
#     "Options: {options}\n"
#     "Correct Answer: {correct_answer}\n"
#     "Model Output: {model_output}\n"
# )

prompt_template = (
    "You are an AI assistant who will help me to judge whether a model's answer to a multiple-choice question is correct.\n"
    "You are provided with a question, the correct answer, and the model's output.\n"
    "Your task is to compare the model's output with the correct answer, and determine:\n"
    "- If the model's output matches the correct answer, output 'Correct'.\n"
    "- If the model's output does not match the correct answer, output 'Incorrect'.\n"
    "Your output should be a single word among the following 2 choices: Correct, Incorrect.\n"
    "Question: {question}\n"
    "Correct Answer: {correct_answer}\n"
    "Model Output: {model_output}\n"
)

def call_llm(question, correct_answer, model_output, model_name):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_template.format(
                question=question,
                correct_answer=correct_answer,
                model_output=model_output
            )}
        ],
        "stream": False,
        "tenant": "intelligentInteraction",
        "velenOfflineChatFlag": False,
        "scene": "OfflineDianXiao"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    content = response.json()['data']['choices'][0]['message']['content'].strip()
    return content

def process_json_obj(json_obj, model_name):
    """
    处理一个json对象，判断模型输出是否正确
    :param json_obj: dict类型的json对象
    :param model_name: 模型名称
    :return: prediction_result
    """
    question = json_obj.get('question', '')
    correct_answer = json_obj.get('answer', '')
    model_output = json_obj.get('model_output', '')
    prediction_result = call_llm(question, correct_answer, model_output, model_name)
    return prediction_result

def main(input_file, output_file, model_name):
    data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i >= 100:
                break
            obj = json.loads(line.strip())
            data.append(obj)

    for item in tqdm(data, desc="Processing items"):
        while True:
                try:
                    start_time = time.time()  # 记录开始时间

                    prediction_result = process_json_obj(item, model_name)
                    item['prediction_match'] = prediction_result

                    end_time = time.time()  # 记录结束时间
                    duration = end_time - start_time
                    print(f"call_llm_for_rewrite 用时: {duration:.4f} 秒")
                    if duration < 2:
                        sleep_time = 2 - duration
                        print(f"用时不足2秒，休息 {sleep_time:.4f} 秒")
                        time.sleep(sleep_time)
                    break
                except Exception as e:
                    print(f"API error: {e}")
                    time.sleep(10)
                    new_text = text

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JSON with LLM model")
    parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.model_name)


'''
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/tools/env_bk/glmv

MODE=0
DUR=960

nohup \
python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/gpt_answer.py \
    --input_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/turnnum/multiturn_asr_mode5_dur960.0_k1.json \
    --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/multiturn/gpt_k/multiturn_asr_mode5_dur960_k1.json \
    --model_name gpt-4o-mini \
> /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/log/gpt_multiturn_tmp_mode.log 2>&1 &



for MODE in 5
do
    for K in 1 3
    do
        nohup \
        python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/gpt_answer.py \
            --input_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/acoustic/new_question/multiturn_asr_mode${MODE}_dur960.0_k${K}.json \
            --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/acoustic/gpt_newquestion/multiturn_mode${MODE}_dur960_k${K}_gpt.json \
            --model_name gpt-4o-mini \
        > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/log/gpt_multiturn_tmp_mode${MODE}_dur960_k${K}.log 2>&1 &
    done
done




nohup \
    python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/gpt_answer.py \
        --input_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/acoustic/turnnum/multiturn_asr_mode5_dur960.0_k5.json \
        --output_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/result/acoustic/gpt_k/multiturn_asr_mode5_dur960.0_k5_gpt.json \
        --model_name gpt-4o-mini \
    > /mnt/dolphinfs/hdd_pool/docker/user/hadoop-fsprisk/fudongjie/GLM-4-Voice/log/multiturn_asr_mode5_dur960.0_k5.log 2>&1 &
   


'''
