import datetime
import shutil
import logging
import os
import random
import time
import traceback
from collections import defaultdict

from openai import OpenAI


class Timer:
    def __init__(self, name, logger, time_file):
        self.name = name
        self.logger = logger
        self.time_file = time_file
        
    def __enter__(self):
        self.start = time.time()
        self.logger.info(f'{self.name}......')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        if exc_type is not None:
            self.logger.error(f"Error in {self.name}: {exc_val}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.time_file.write(f"{self.name}: ERROR - {exc_val}\n")
            return False
        self.logger.info(f'Finish {self.name}, Using {self.duration}s\n')
        self.time_file.write(f"{self.name}: {self.duration}\n")
        return self.duration
    

class Logger:
    def __init__(self, resp_path):
        # Create a unique logger name using the response path
        abbr_resp_path = ''
        logger_name = f'logger({abbr_resp_path})'
        self.logger = logging.getLogger(logger_name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(resp_path, 'run.log'))
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def get_logger(self):
        return self.logger


def get_read_paths(start_time, end_time, base_dir, result_dir):
    read_paths = {}
    pipeline_dir = os.path.join(base_dir, f'result/{result_dir}')
    
    start_dt = datetime.strptime(start_time, "%m-%d-%H:%M")
    end_dt = datetime.strptime(end_time, "%m-%d-%H:%M") 
    
    for filename in os.listdir(pipeline_dir):
        match = re.match(r'(\d{2}-\d{2}-\d{2}:\d{2})\s+(.*)', filename)
        if not match:
            continue
            
        timestamp, rest = match.groups()
        folder_dt = datetime.strptime(timestamp, "%m-%d-%H:%M")
        
        if start_dt <= folder_dt <= end_dt:
            dataset_match = re.match(r'([a-z]+)(\d+)-.*-set(\d+)', rest)
            if dataset_match:
                dataset, err_rate, set_num = dataset_match.groups()
                key = f"{dataset}{err_rate}-{set_num}"
                read_paths[key] = os.path.join(pipeline_dir, filename)
                
    return read_paths


def get_ans_from_llm(prompt, api_use=False):
    if not api_use:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        max_retries = 5
        base_sleep = 1

        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                    max_retries=100,
                )

                chat_response = client.chat.completions.create(
                    model="./qwen2.5-72bs-instruct",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=4096,
                    extra_body={
                        "repetition_penalty": 1.05,
                    },
                )
                return chat_response.choices[0].message.content
            except Exception as e:
                print(e)
    elif api_use:
        model_type='qwen2.5-7b-instruct'
        role_descr="You are a world-class data engineer, proficient in cleaning dirty data."
        api_key_list = [
            ''
        ]
        # Exponential backoff parameters
        base_sleep = 0.2
        max_retries = 200
        try_cnt = 0
        key_idx = 0

        while 1:
            try:
                client = OpenAI(
                    api_key=api_key_list[key_idx],
                    base_url='https://api.openai.com/v1'
                )
                completion_res = client.chat.completions.create(
                    model=model_type,
                    temperature=0,
                    messages=[
                        {"role": "system",
                        "content": role_descr},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion_res.choices[0].message.content
                try_cnt = 0
                key_idx = 0
                return response
            except Exception as e:
                print(e)
                key_idx += 1
                if key_idx >= len(api_key_list):
                    sleep_time = base_sleep * (2 + try_cnt)  # Exponential backoff
                    time.sleep(sleep_time)
                    print("Sleeping for {} seconds".format(sleep_time))
                    print(e)
                    try_cnt += 1
                    key_idx = 0
                    if try_cnt >= max_retries:
                        print("Maximum {} retries reached for OpenAI API requests.".format(max_retries))
                        break


def rag_query(query, documents, GPT_USE=True):
    if GPT_USE:
        response = get_ans_from_llm(f"Question: {query}\n\n Guidelines:{documents}")
    else:
        response = ''
    return response


def query_base(query, GPT_USE=True):
    if GPT_USE:
        response = get_ans_from_llm(f"{query}")
    else:
        response = ''
    return response


def split_list_to_sublists(original_list, sublist_size):
    if sublist_size == 0:
        return [original_list]
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    original_list = shuffled_list
    return [original_list[idx:idx + sublist_size] for idx in range(0, len(original_list), sublist_size)]


def default_list():
    return []


def default_dict_of_lists():
    return defaultdict(default_list)


def copy_read_files_in_dir(dst_dir, src_dir):
    if os.path.exists(src_dir):
        for file in os.listdir(src_dir):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
                

def copy_file(read_path, resp_path, file_name):
    src_file = os.path.join(read_path, file_name)
    dst_file = os.path.join(resp_path, file_name)
    shutil.copy2(src_file, dst_file)


