# Load module
import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def call_model_tokenize(model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_context_from_id(sgk, id_qa):
    subject = id_qa[0]
    grade = id_qa[1:3]
    topic_lesson = id_qa[3:6]
    if topic_lesson[0] == 'B':
        full_context = ''
        for topic in sgk[subject][grade].keys():
            element_in_topic = sgk[subject][grade][topic].keys()
            if topic_lesson in element_in_topic:
                return sgk[subject][grade][topic][topic_lesson]['context']
            else:
                for element in element_in_topic:
                    if element != 'name':
                        full_context += sgk[subject][grade][topic][element]['context'] + '. '
        return full_context
    else:
        topic_context = ''
        for lesson in sgk[subject][grade][topic_lesson].keys():
            if lesson != 'name':
                topic_context += sgk[subject][grade][topic_lesson][lesson]['context'] + '. '
        return topic_context


def run_model(model_name, tokenizer, sgk, qas, device, subject):
    dict_output = {}
    l_id_error = ['S11B2401832', 'S11B2401833', 'S11B2401834', 'S12B0101894', 'S12B0101895', 'S12B0101896', 'S12B0402242', 'S12B0402247', 'S12B0402248', 'S12B0402249', 'S12B1002946', 'S12B1103046', 'S12B1203090', 'S12B1503529']
    for id_qa in qas.keys():
        if subject == id_qa[0] and id_qa in l_id_error:
            question = qas[id_qa]['question']
            context = get_context_from_id(sgk, id_qa)
            pipe = pipeline('question-answering', model = model_name, tokenizer = tokenizer, device = device)
            try:
                output_model = pipe(question, context) # {'score', 'start', 'end', 'answer'}
                dict_output[id_qa] = output_model
                print(f"Complete {id_qa}!")
            except:
                print(f"Warning: Error {id_qa}")
    return dict_output


def write_file_json(file_name, dict_output):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    sgk = read_json_file("./ReasoningCTT/MRC_2.0/SGK.json")
    qas = read_json_file("./ReasoningCTT/MRC_2.0/QAs.json")
    subject = 'S'
    
    device = torch.device("cuda:0")
    model_name = "bhavikardeshna/xlm-roberta-base-vietnamese"
    
    model, tokenizer = call_model_tokenize(model_name)
    dict_output = run_model(model_name, tokenizer, sgk, qas, device, subject)
    
    output_file_name = f'output_{subject}_xlm-r_1.json'
    write_file_json(output_file_name, dict_output)