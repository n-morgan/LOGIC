from transformers import AutoModelForCausalLM, AutoTokenizer
from my_rag_new import BERTEmbedder 
import pandas as pd 
from wiki_loader import WikiLoader
import torch
import csv
import re
import uuid
import json
import random


# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("/data/sls/scratch/nmorgan/models/Llama8b")
tokenizer = AutoTokenizer.from_pretrained("/data/sls/scratch/nmorgan/models/Llama8b")

# Move the model to GPU
model.to("cuda")

RPHR_MSG = """You are given a vague question involving a scientific acronym. Your job is to generate multiple natural-sounding rephrasings of the question that keep the original ambiguity. Do not reveal, expand, or hint at the meaning of the acronym.

Generate following this structure: do not deviate from this structure

<rephrasing1> insert rephrasing here </rephrasing1>  
<rephrasing2> insert rephrasing here </rephrasing2>  
<rephrasing3> insert rephrasing here </rephrasing3>  
<rephrasing4> insert rephrasing here </rephrasing4>  
<rephrasing5> insert rephrasing here </rephrasing5>  
<rephrasing6> insert rephraing here </rephrasing6>  
<rephrasing7> insert rephrasing here </rephrasing7>  
<rephrasing8> insert rephrasing here </rephrasing8>


Here are a few examples:

Input:  
Acronym: ATTACK  
Original Question: "Can you explain ATTACK?"

Rephrasings:
<rephrasing1>What exactly is ATTACK about?</rephrasing1>  
<rephrasing2>Could you tell me more about ATTACK?</rephrasing2>  
<rephrasing3>How is ATTACK generally understood?</rephrasing3>  
<rephrasing4>What’s the idea behind ATTACK?</rephrasing4>  
<rephrasing5>I’ve heard of ATTACK — what does it involve?</rephrasing5>  
<rephrasing6>What’s ATTACK in the context you’re using it?</rephrasing6>  
<rephrasing7>I’m not familiar with ATTACK — can you elaborate?</rephrasing7>  
<rephrasing8>Where does ATTACK fit into this discussion?</rephrasing8>

Input:  
Acronym: GRASP  
Original Question: "What does GRASP refer to?"

Rephrasings:
<rephrasing1>Can you give me an overview of GRASP?</rephrasing1>  
<rephrasing2>What does GRASP stand for in this setting?</rephrasing2>  
<rephrasing3>What is GRASP used for?</rephrasing3>  
<rephrasing4>I’ve come across GRASP — can you clarify what it is?</rephrasing4>  
<rephrasing5>How should I think about GRASP?</rephrasing5>  
<rephrasing6>Could you break down GRASP a bit for me?</rephrasing6>  
<rephrasing7>What's the significance of GRASP?</rephrasing7>  
<rephrasing8>Is GRASP a method, a tool, or something else?</rephrasing8>

Now, using the same style, generate 8 rephrasings for the following:

Input:  
Acronym: {acronym}  
Original Question: {question}

START HERE 

"""
UNRQ_MSG = """
You are given a random passage. Your task is to generate a question about the passage that asks for clarification or explanation of a specific term or concept within it. The query should be phrased in a simple, direct manner, such as "What is [TERM]?", where [TERM] is a concept, term, or important idea mentioned in the passage. The question should be wrapped in <query></query> tags.

Generate folowing this structure: do not deviate from this structure
<query> short query relating to TERM <\query>

Here are some examples

Input Passage:
"The Eiffel Tower is one of the most famous structures in Paris, France. It was designed by the engineer Gustave Eiffel and constructed for the 1889 World's Fair. The tower is made of iron and stands 330 meters tall."

Output Query:
<query>What is Eiffel Tower?</query>

Input Passage:
"Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. This energy is used to fuel the plant's growth and functions. Chlorophyll is the green pigment responsible for absorbing light during photosynthesis."

Output Query:
<query>What is Photosynthesis?</query>

Input Passage:
"The Great Wall of China is a series of fortifications made of various materials, including brick, tamped earth, and stone. It was built to protect the northern borders of the Chinese Empire from invasions. The wall stretches over 13,000 miles."

Output Query:
<query>What is Great Wall of China?</query>


Now its your turn. Do not generate anything beyond the structure 

Input Passage:
{passage}

START HERE

"""

# Create dataframe 

df = pd.read_csv('generations5.csv')

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
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



def parse_query_text(text):



    generated_section_match = re.search(r"START HERE\s*(<rephrasing1>.*)", text, re.DOTALL)

    if generated_section_match is None:
        print(text)
        return None


    generated_section = generated_section_match.group(1)

    queries = []
    for index in range(8):
        
        query_match = re.search(fr"<rephrasing{index}>\s*(.*?)\s*</rephrasing{index}>", generated_section)
        queries.append(query_match.group(1).strip() if query_match else "")

    return queries


def parse_unrel_query_text(text):

    
    generated_section_match = re.search(r"START HERE\s*(<query>.*)", text, re.DOTALL)

    if generated_section_match is None:
        print(text)
        return None

    

    generated_section = generated_section_match.group(1)

        
    query_match = re.search(r"<query>\s*(.*?)\s*</query>", generated_section)
    query = query_match.group(1).strip() if query_match else ""

    return query




def rephrase_query(acronym,query, num_rephrasings):
    '''
    creates rephrasing of query num_times
    '''



    output = generate_text(RPHR_MSG.format(acronym=acronym, question=query))

    rephrasings = parse_query_text(output)

    



    return rephrasings

def make_unrelated_query_label(label, documents):
    '''
    create query based on unrelated document
    '''

    # select random documnet which is not the true document
    def random_choice_excluding(lst, exclude_index):
        # Create a list of tuples (index, item) excluding the given index
        filtered = [(i, item) for i, item in enumerate(lst) if i != exclude_index]
        
        # Randomly choose one from the filtered list
        random_choice = random.choice(filtered)
        
        unrel_doc_index = random_choice[0]
        unrel_doc = random_choice[1]
        return {'unrel_doc_index': unrel_doc_index, 'unrel_doc': unrel_doc} # (index, item)

    # find the index of the true documenet in the documnet set
    true_doc_idx = label.index(1)
    
    # extract an unrealted document from the documents
    unrel_dict = random_choice_excluding(documents, true_doc_idx)


    unrel_query_unproc  = generate_text(UNRQ_MSG.format(passage=unrel_dict['unrel_doc']))

    unrel_query = parse_unrel_query_text(unrel_query_unproc)


    unrel_label_blank = [0]*len(label)

    unrel_label_blank[unrel_label_blank[unrel_dict["unrel_doc_index"]]] = 1

    unrel_label = unrel_label_blank

    return unrel_label, unrel_query


    
def strip_uuid(document):
    '''
    Strips UUID from document retrieved from benchmark
    Returns string
    '''
    # This regex matches "UUID:" followed by optional space and a standard UUID format
    uuid_pattern = r'UUID:\s*\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
    return re.sub(uuid_pattern, '', document).strip()


def insert_at_random(document_list, new_document):
    '''
    Inserts new document into document_list at random 
    Returns (document_idx, new_document_list)
    '''
    insert_idx = random.randint(0, len(document_list))

    new_arr = document_list[:insert_idx] + [new_document] + document_list[insert_idx:]
    
    encoding = [0]*len(new_arr)
    encoding[insert_idx] = 1
    return encoding, new_arr



def retrieve_wiki_docs(num_docs):
    '''

    retrieve documents from wiki

    '''
    wiki_docs_df = WikiLoader(num_docs).retrieve_samples()
    wiki_docs = wiki_docs_df['text'].tolist()
    return wiki_docs

    
    
batch_size = 5

with open('train_dataset_maker.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["query", "label", "query_rephrased", "query_unreleated", "label_unrealted", "document_embeddings"])
    # NOTE: LABEL (k,1), USED BY QUERY AND QUERY_REPHRASED 
    # NOTE: QUERY_REPHRASED n REPRASINGS (n,1)
    # NOTE: DOCUMENT EMBEDDINGS (k,1)



num_documents = 10
outer = 0 
num_errors = [0,0]
sum_errors = sum(num_errors)
while outer < 500 and sum_errors < 1000 :
    # logic, 1) make k-1 documents, 2) add true documnet at random 2i) store idx of true document 3) store true query  3) make unrelated query 3i) store idx of documnet relatedd to unrelated query, 4) make query rephrasings 5) embed and store all documents
        


    documents_incomplete = retrieve_wiki_docs(num_documents-1) #create set of k-1 unrel documnt 
    document_truth_dirty = df.iloc[outer]['description'] # retrieve true document
    document_truth = strip_uuid(document_truth_dirty) # strip UUID from document 


    # combine k-1 documents with kth documnet and create 1 hot label 
    label, documents_complete = insert_at_random(documents_incomplete, document_truth) 
    
    # retrueve query from synth data
    query = df.iloc[outer]['question']
    # retreive acronym from synth data
    acronym = df.iloc[outer]['acronym']

    # make rephrasings of the query
    query_rephrasings = rephrase_query(acronym, query,10) # TODO: Fix this, add parser 

    if query_rephrasings is None:
        num_errors[0] += 1
        print('Query Rephrasings Failed. num_errors ', num_errors)
        continue



    # make unrelated query 
    unrel_label, unrel_query = make_unrelated_query_label(label, documents_complete)

    if unrel_query is None:
        num_errors[1] += 1
        print('Unrel Query Failed. num_errors ', num_errors)
        continue


    


    
    document_embeddings = embedding_model.generate_embedding(documents_complete).numpy()
    print([type(x) for x in [query, label, query_rephrasings, unrel_query, unrel_label, document_embeddings]])
    print(unrel_label)
    data_entry = {
    "query": query,
    "label": list(label),
    "query_rephrased": list(query_rephrasings),
    "query_unrelated": unrel_query,
    "label_unrelated": list(unrel_label),
    "document_embeddings": document_embeddings.tolist()



}

    with open('train_dataset_maker.jsonl', mode='a') as f:
        f.write(json.dumps(data_entry) + "\n")


    outer += 1 # note: move this around for debugging purpouses





    print(f"Successfully wrote generation {outer} to CSV: {acronym}, {document_truth}, {label}, {unrel_query}, {unrel_label}")

        

    



