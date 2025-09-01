from datasets import load_dataset
import os
import re
import csv
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

class LLMEval:
    def __init__(self, dataset, rubrics, output_file, model_repo, min_score=1, max_score=5):
        self.dataset = dataset
        self.rubrics = rubrics
        self.output_file = output_file
        self.min_score = min_score
        self.max_score = max_score
        self._load_model_tokenizer(model_repo)
        
    def _load_model_tokenizer(self, model_repo):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = LLM(model=model_repo, tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
    def _get_judge_llm_resp(self, text):
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        safe_text = text.replace("{", "{{").replace("}", "}}")
        ABSOLUTE_PROMPT_WO_REF = f"""###Task Description:
        An unconditional generation to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assesses the quality of the generation strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between {self.min_score} and {self.max_score}. You should refer to the score rubric.
        3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between {self.min_score} and {self.max_score})"
        4. Please do not generate any other opening, closing, and explanations.

        ###Generation to evaluate:
        {safe_text}

        ###Score Rubrics:
        {self.rubrics}

        ###Feedback: """
        user_content = f"{ABS_SYSTEM_PROMPT}\n\n{ABSOLUTE_PROMPT_WO_REF}"
        sampling_params = SamplingParams(max_tokens=1000, temperature=0.0, top_p=1.0)
        outputs = self.llm.generate([user_content], sampling_params)
        return outputs[0].outputs[0].text.strip(), user_content


    def _parse_feedback_and_score(self, text):
        result_match = re.search(r"\[RESULT\]\s*(\d+)", text)

        if result_match:
            score = int(result_match.group(1))
            feedback = text[:result_match.start()].strip()
        else:
            score = None
            feedback = text.strip()

        return feedback, score
    
    def _get_processed_lines(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader, None)
                return sum(1 for row in reader)
        else:
            return 0

    def generate_inference_file(self):
        results = []
        batch_size = 100
        start_idx = self._get_processed_lines()
        with open(self.output_file, mode='a+', newline='') as file:
            writer = csv.writer(file)
            if os.stat(self.output_file).st_size == 0:
                writer.writerow(["text", "llm_prompt", "llm_response", "llm_score", "llm_reasoning"])
        for index, item in enumerate(self.dataset):
            if index < start_idx:
                continue
            text = item["text"]
            if text is None:
                continue
            llm_response, llm_prompt = self._get_judge_llm_resp(text)
            feedback, llm_score = self._parse_feedback_and_score(llm_response)
            results.append([text, llm_prompt, llm_response, llm_score, feedback])
            if len(results) >= batch_size:
                with open(self.output_file, mode='a+', newline='') as file:        
                    writer = csv.writer(file)
                    writer.writerows(results)
                    results = []

        if results:
            with open(self.output_file, mode='a+', newline='') as file:        
                writer = csv.writer(file)
                writer.writerows(results)