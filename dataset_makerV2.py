from transformers import AutoModelForCausalLM, AutoTokenizer
from my_rag_new import BERTEmbedder 
import pandas as pd 
from wiki_loader import WikiLoader
import torch
import csv
import re
import uuid

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("/data/sls/scratch/nmorgan/models/Llama8b")
tokenizer = AutoTokenizer.from_pretrained("/data/sls/scratch/nmorgan/models/Llama8b")

# Move the model to GPU
model.to("cuda")


CF_MSG = '''
You are an advanced AI assistant specializing in generating scientifically relevant methodologies that incorporate the meaning of the acronym {acronym}. You create methodologies for scientific domains which are not computer science related. No methodology you create is computer science related. 


Generate following this structure: do not deviate from this structure
<acronym>  {acronym} </acronym>
<description> A theoretical framework (~100 words) explaining {acronym} without explicitly stating {acronym} or its expansion. </description>
<question> A simple question about {acronym}. </question>

Here are some examples:

<acronym> ROOT </acronym>
<description> This framework examines the foundational processes governing ecological resilience and resource distribution. One perspective focuses on the deep structural networks that sustain nutrient absorption in plant ecosystems, while another explores how interconnected subterranean systems facilitate microbial symbiosis. By integrating these viewpoints, the model highlights how stability emerges from adaptive resource exchange, ensuring sustainability in changing environmental conditions. </description>
<question> What is root? </question>

<acronym> FLOW </acronym>
<description> This model investigates the dynamic principles governing movement and energy transfer across interconnected systems. One approach examines how continuous streams adjust to obstacles, optimizing efficiency in physical and biological environments. Another interpretation considers how disruptions in movement patterns lead to turbulence or stability, depending on adaptive feedback mechanisms. By reconciling these perspectives, the framework offers insights into how structured motion can enhance both resilience and efficiency across natural and engineered systems. </description>
<question> What is flow? </question>

<acronym> SPARK </acronym>
<description> This framework explores the mechanisms underlying rapid biochemical reactions that drive cellular function and metabolic efficiency. One interpretation focuses on transient molecular interactions that accelerate energy conversion, enabling immediate physiological responses. Another perspective highlights how small-scale reaction cascades contribute to large-scale systemic regulation, maintaining equilibrium under fluctuating conditions. By synthesizing these insights, the model reveals how localized biochemical events influence broader metabolic stability. </description>
<question> What role do rapid biochemical interactions play in sustaining metabolic balance? </question>

Now, generate a new, scientifically relevant methodology or model that uses the meaning of {acronym}:
<acronym> {acronym} </acronym>
<description> [Your generated description of approximately 100 words must not include {acronym} or its expansion] </description>
<question> [Your generated thought-provoking scientific question] </question>

START HERE

'''


SYS_MSG = '''
You are an advanced AI assistant specializing in generating scientifically relevant methodologies that incorporate acronyms with ambiguous meanings within the same scientific domain of computer science. Your task is to generate models that challenge retrieval-based systems by introducing:

Same-Domain Ambiguity: Each acronym should be entirely within the domain of computer science, but the words in the acronym should have other plausible meanings.
Implicit Reasoning: The description must not explicitly state what the acronym stands for. The meaning should be inferred indirectly.
Context Confusion: The explanation should lead retrieval models to misinterpret the acronym’s intended meaning based on surface-level similarity.
Misleading Similarity: Ensure that the surrounding context does not contain exact words from the acronym, preventing simple keyword-based retrieval.


Generate following this structure: do not deviate from this structure
<acronym>  multi-word abbreviation where each word has meanings in at least two different subfields of the same domain. </acronym>
<description> A theoretical framework (~100 words) explaining the model without explicitly stating the acronym or its expansion. </description>
<question> A question that forces retrieval models to resolve ambiguity, leading to incorrect retrieval when context is not fully understood. </question>


Here are some exmaples(All Computer Science):

<acronym> LOOP </acronym>
<description> This framework explores iterative refinement in complex systems, focusing on how repeated cycles enhance efficiency and adaptability. One perspective examines optimization strategies that dynamically adjust parameters based on feedback, ensuring convergence to optimal solutions. Another interpretation considers how recursive structures facilitate pattern recognition, enabling adaptive learning across varying conditions. By unifying these viewpoints, the model highlights the role of structured repetition in improving performance across computational and natural processes. </description>
<question> What is loop? </question>

<acronym> SHIELD </acronym>
<description> This framework investigates protective mechanisms that regulate access and maintain system integrity in interconnected environments. One approach focuses on dynamic threat detection, where adaptive filtering techniques mitigate vulnerabilities in real-time. Another perspective explores the role of layered defenses, ensuring redundancy and fault tolerance against systemic disruptions. By integrating these concepts, the model emphasizes how proactive monitoring and structural safeguards enhance overall resilience in both digital and physical networks. </description>
<question> What is shield? </question>

<acronym> WAVE </acronym>
<description> This model examines distributed coordination in large-scale computational networks, focusing on the propagation of signals and task execution across interconnected nodes. One perspective highlights the role of asynchronous messaging in maintaining efficiency under fluctuating workloads. Another interpretation considers how consensus algorithms regulate synchronization, ensuring stability despite decentralized control. By bridging these viewpoints, the framework reveals how structured propagation mechanisms optimize performance in both computational and biological systems. </description>
<question> What does wave mean? </question>

Now, generate a new, scientifically relevant methodology or model that uses overlapping meanings for its acronym:
<acronym> [Your generated acronym] </acronym>
<description> [Your generated description of approximately 100 words must not include the acronym or its expansion] </description>
<question> [Your generated simple question about the acronym. Keep it to few words to keep the word meaning ambiguous] </question>

START HERE

'''

embedding_model = BERTEmbedder()

def generate_text(prompt):
    """
    generates text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        temperature=0.7,  # controls randomness, lower values give more focused responses
        max_new_tokens = 200
        
    )
    # print(outputs)
    # for i in range(len(outputs)):
    #     print(i)
    #     print(tokenizer.decode(outputs[0][0:i], skip_special_tokens=False))
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def parse_generation_text(text):
    # Ensure we're only extracting the part AFTER "Now, generate a new..."
    generated_section_match = re.search(r"START HERE\s*(<acronym>.*)", text, re.DOTALL)

    
    if not generated_section_match:
        return ["N/A", "N/A", "N/A"]  # Fallback if no match is found

    generated_section = generated_section_match.group(1)

    # Extract acronym, description, and question using regex
    acronym_match = re.search(r"<acronym>\s*(.*?)\s*</acronym>", generated_section)
    description_match = re.search(r"<description>\s*(.*?)\s*</description>", generated_section, re.DOTALL)
    question_match = re.search(r"<question>\s*(.*?)\s*</question>", generated_section)

    acronym = acronym_match.group(1).strip() if acronym_match else "N/A"
    description = description_match.group(1).strip() if description_match else "N/A"
    question = question_match.group(1).strip() if question_match else "N/A"

    return [acronym, description, question]


def retrieve_wiki_docs(num_docs):
    '''

    retrieve documents from wiki

    '''
    wiki_docs_df = WikiLoader(num_docs).retrieve_samples()
    wiki_docs = wiki_docs_df['text'].tolist()
    return wiki_docs

    
    
batch_size = 5

with open('generations5.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["acronym", "description", "question", "description_embedding", "question_embedding", "documnet_1", "documnet_2", "document_3", "document_4", "document_5", "document_1_embeddings", "document_2_embeddings", "document_3_embeddings" , "document_4_embeddings", "document_5_embeddings"])

batch = []
for outer in range(500): 
    generation = generate_text(SYS_MSG)
    batch.append(parse_generation_text(generation) + [None, None]) # creating placeholder  for  question_embedding and desc_embedding 


    if outer != 0 and (outer+1)%batch_size == 0:
        # Generate embeddings for all descriptions 

        cf_documents = []
        cf_embeddings = [] # list of lists len(5) storing all embeddings for batch
        descriptions = [tup[1] for tup in batch]
        questions = [tup[2] for tup in batch]
        desc_embeddings = embedding_model.generate_embedding(descriptions).tolist()
        question_embeddings = embedding_model.generate_embedding(questions).tolist()
    
        for entry in batch:
            cf_list = []
            for i in range(5):
                cf_all = generate_text(CF_MSG.format(acronym = entry[0]))
                _ , cf_text, _  = parse_generation_text(cf_all) 
                cf_list.append(cf_text)

            
            
            cf_embedding = embedding_model.generate_embedding(cf_list).tolist()
            cf_documents.append(cf_list)
            cf_embeddings.append(cf_embedding)
         

        # insert question and document embeddings into info tuple 
        # format: (acronym, description, question, description_embedding, question_embedding)
        for idx in range(batch_size):
            batch[idx][3] = desc_embeddings[idx]
            batch[idx][4] = question_embeddings[idx]

            
        
        # iterate thorugh all data in batch and add to csv
        for idx in range(batch_size):  
            acronym, description, question, desc_embedding, ques_embedding = batch[idx]
            
            unique_id = uuid.uuid4()
            
            question += f' UUID: {unique_id}'
            description += f'UUID: {unique_id}'

            with open('generations5.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([acronym, description, question, desc_embedding, ques_embedding]+cf_documents[idx]+cf_embeddings[idx])
    

            print(f"Successfully wrote generation {outer} {idx} to CSV: {acronym}, {description}, {question}")    
        batch.clear()
