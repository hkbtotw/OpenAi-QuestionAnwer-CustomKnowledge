#### Script : Question and Answer from local knowledge with LLM
#### reference : https://medium.com/codex/gpt-4-chatbot-guide-mastering-embeddings-and-personalized-knowledge-bases-f58290e81cf4
#### reference : https://medium.com/@avra42/build-a-personal-search-engine-web-app-using-open-ai-text-embeddings-d6541f32892d
#### reference : https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
import pandas as pd
import openai

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

## embedding
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from openai.embeddings_utils import get_embedding

## gpt4all llm
from langchain.llms import LlamaCpp

### compute tiktoken
import tiktoken
import  numpy as np

## environment
import os

#######################################################################################################
## openai
openai.api_key="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"    #### input your oopenai API key here

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"
MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
#####################################################################################################

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    # print(len(x),' --- ',len(y))
    return np.dot(np.array(x), np.array(y))

def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df, model):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content, model) for idx, r in df.iterrows()
    }

def Construct_Context(dfData,MAX_SECTION_LEN):

    most_relevant_document_sections = list(dfData['order'])

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    df=dfData.copy()
    for section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        # print(' section : ',document_section,' :: ',chosen_sections_len)
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        # print(' chosen section : ',chosen_sections)
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    # print(' ===== ',chosen_sections)
    return chosen_sections

def answer_with_gpt( query, prompt):
    messages = [
        {"role" : "system", "content":"You are a chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]   

    context= ""
    for article in prompt:
        context = context + article 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})

    ### conpletion API
    ### ref: https://www.debugpoint.com/openai-chatgpt-api-python/
    response = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=messages
        )
    return '\n' + response['choices'][0]['message']['content']

#####################################################
### Specify Path and Testing data file
base_path='D:\\DataWarehouse\\GPT\\Experiment\\'

###### Local Knowledge to be embedded
data_file_name='test1.txt'
# data_file_name='test1_th.txt'

### Specify query
query = "Who is the current prime minister of Thailand? and What does Prayut relate with this person"
# query = "ใครคือนายกรัฐมนตรีของประเทศไทย ในปัจจุบัน?"   ### answer in thai is not good with this local knowledge, not sure if it is because of too small local data or the model
#######################################################

##### Select if_new_data==1 for the first time computing embedding of the local knowledge
##### then the embedding vector will be saved in parquet file 
if_new_data=1
if(if_new_data==1):
    ## read datafile and prepare for processing
    with open(base_path+'\\data\\'+data_file_name, encoding="utf8", errors='ignore') as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    enc = tiktoken.encoding_for_model("gpt-4")
    tokens_per_section = []
    df=pd.DataFrame()
    for text in texts:
        # print(' ****************************************************** ')
        tokens = enc.encode(text)
        # print(' ==== text ===> ',text,' :: ',len(tokens))
        df=df.append({'index':text,'content':text,'tokens':len(tokens)},ignore_index=True)

    df.set_index(['index'],inplace=True)
    document_embeddings = compute_doc_embeddings(df,EMBEDDING_MODEL)

    print(len(document_embeddings),' ---- embedded ----', document_embeddings,' :: ',type(document_embeddings))
    outputDf=pd.DataFrame()
    outputList=list(document_embeddings)
    for outputId in outputList:
        tokens = enc.encode(outputId)        
        outputDf=outputDf.append({'content':outputId,'vector':document_embeddings[outputId],'length':len(document_embeddings[outputId]),'tokens':len(tokens)},ignore_index=True)
    outputDf.to_parquet(os.getcwd()+'\\'+'embedding_ver2.parquet',index=False)
else:
    dfData=pd.read_parquet(os.getcwd()+'\\'+'embedding_ver2.parquet')
    print(len(dfData),' --- read in : saved embedding data ---- ',dfData.head(3),' :: ',dfData.columns)

try:
    dfData=outputDf.copy()
except:
    print('  read new data ---')

query_embedding = get_embedding(query,EMBEDDING_MODEL)
# print(' query : ',query_embedding,' :: ',type(query_embedding),' -- ',len(query_embedding))

dfData['SimilarityScore']=dfData['vector'].apply(lambda x: vector_similarity(list(x), query_embedding))
dfData.sort_values(by=['SimilarityScore'],ascending=False,inplace=True)
dfData=dfData.reset_index(drop=True)
print(len(dfData),' ---  Sorted embedding data based on Score  ---- ',dfData.head(3),' :: ',dfData.columns)
dfData=dfData.reset_index()
dfData.columns=['order','content','vector','length','tokens','SimilarityScore']
# dfData.to_excel(os.getcwd()+'\\'+'check_embedding_result_ver3.xlsx',index=False)

#######################################
### Separator in context
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
print(f"Context separator contains {separator_len} tokens")
#######################################
#### Obtain context
chosen_sections=Construct_Context(dfData,MAX_SECTION_LEN)

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 2000,
    "model": COMPLETIONS_MODEL,
}

#### Ask GPT with question + context constructed from local knowledge
response = answer_with_gpt(query,chosen_sections)

print(' ------------------------------------------------ ')
print(' ------------------------------------------------ ')
print(' ANSWER : ',response)
print(' ------------------------------------------------ ')
print(' ------------------------------------------------ ')

print(' *********** DONE ************* ')

