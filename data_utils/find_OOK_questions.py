# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os

def load_results(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_OOK_questions(results_path):

    new_data = []
    results = load_results(results_path)

    for idx, item in tqdm(enumerate(results)):
        flag = True
        for response_judge_res in item["llm_as_a_judge"]:
            if response_judge_res['score'] == 1:
                flag = False
                break

        if flag:
            new_data.append(item)

    return new_data

def main():

    MODEL_NAME = 'Llama-3.1-8B-Instruct'
    dataset_name = 'CRAG'
    split = 'train'

    results_path = f'TruthRL/evaluation/results/{dataset_name}/{split}/{MODEL_NAME}/Sampling_RAG/256_responses/judge_Llama-3.3-70B-Instruct/results_top_50.json'

    new_data = find_OOK_questions(results_path)
    print(f'len(new_data): {len(new_data)}')

    output_dir = f"{dataset_name}/{MODEL_NAME}"

    output_path = os.path.join(output_dir, f'OOK_questions_{split}.json')
    
    with open(output_path, 'w') as file:
        json.dump(new_data, file, indent=4)
    
if __name__ == '__main__':
    main()

