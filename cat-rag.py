# Simple Retrieval-Augmented Generation (RAG) Example 
# Get answers on cats, backed by fact dataset (cat-facts.txt)

# How it will work-
#     Initial work:
#         Create vector database
#             Create dataset by reading in data from text file (cat-facts.txt)
#             For each "chunk" (line) in dataset, create and collect vector embedded version
            
# 	1. Collect user request (query)
# 	2. Retrive relavant data
# 	        Obtain vector embedding of query (via ollama.embed())
# 	        for each vector in vector database, compare similarity to quary vector embedding
# 	        sort similiarities, hghest similarities -> lowest
# 	        return top similarities (vector embeddings with highest similarity)
#     3. Prompt chatbot and generate
#             Give chatbot (ollama) all the retrived data
#     4. Print responce! woo


import ollama

# Load in dataset (cat-facts.txt) ----------
dataset = []
# open file for reading, add all lines to dataset
with open('cat-facts.txt', 'r') as file: 
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries') # print statment check

# Implementing vector database ----------
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

VECTOR_DB = []

def add_chunk_to_db(chunk):
    # create vector embedding of chunk
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    # add to VECTOR_DB, in format of tuples (chunk, embeding)
    VECTOR_DB.append((chunk, embedding))
    
# use above func to add every line to VECTOR_DB
for i, chunk in enumerate(dataset):
    add_chunk_to_db(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database') # print stament to view workings
    
    
# retrival of top most N relevent chunks ----------

# funtion to calculate similarity between two vectors
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


# function to retrive, returns list of top N (hardcode 3 in this case) chunks 
    # converts user query
    # calculates similarities to all vector embeddings
    # only returns top N
def retrieve(query, N=3):
    # convert query to embedding
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    
    # list to store pairs of (chunk, similarity to chunk)
    similiarities = []
    
    # for each chunk from vector database, calculate similarity (via cosine_similarity()) and append similarity
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similiarities.append((chunk, similarity))
    
    # sort by similarity in desc, to get most similar in the front 
    similiarities.sort(key=lambda x: x[1], reverse=True)
    
    # return top/front N number of chunks
    return similiarities[:N]
    
# Obtain user prompt and gather data ----------
print('Ask me a question: ')
input_query = input('')

# given query, search for and obtain relevent data
retrived_data = retrieve(input_query)
# print out to view data collected
print('Retrived knowledge:')
for chunk, similarity in retrived_data:
    print(f' - (similarity: {similarity:.2f}) {chunk}')
    
# generate responce ----------

# partially hardcoded prompt + data collected to send to ollama
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information.
Question: {input_query}
Context: 
{'\n'.join([f' - {chunk}' for chunk, similarity in retrived_data])}
'''

# set up ollama call
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

# print message from chatbot
print('chatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
    
	
