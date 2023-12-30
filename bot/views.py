from django.shortcuts import render
from django.http import HttpResponse


import os 
import numpy as np 
import re
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import GoogleDriveLoader
from langchain_core.messages import HumanMessage

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,)


from langchain.chains import ConversationalRetrievalChain

# Create your views here.

def get_documents(folder_path):
    documents = []
    
    files = [ext.lower() for ext in os.listdir(folder_path)]
    print(files)
    for file in files:    
        if '.docx' in file or '.doc' in file:
            full_path = os.path.join(folder_path, file)
            docx_loader = Docx2txtLoader(full_path)
            documents.append(docx_loader)
        if '.pptx' in file:
            full_path = os.path.join(folder_path, file)
            docx_loader = UnstructuredPowerPointLoader(full_path)
            documents.append(docx_loader)

    return documents

loader = get_documents("/Users/rampavandamela/Developer/model_docs")


#declaring the text spliter from the documents 
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=575,
    chunk_overlap=25,
    length_function=len
)

#spliting the documents paragraphs into small chunks with overlapping. Separators are '.' and '\n'
docs = []
for item in loader:
    for temContent in item.load():
        temContent.page_content = re.sub(r'\n.,', '\n',temContent.page_content)
    docs.extend(text_splitter.split_documents(item.load()))
    
dotenv_path = Path('/Users/rampavandamela/Developer/.env')
load_dotenv(dotenv_path=dotenv_path)
host= os.environ['PG_HOST']
port= os.environ['PG_PORT']
user= os.environ['PG_USER']
password= os.environ['PG_PASSWORD']
dbname= os.environ['PG_DB_NAME']

embeddings =HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
CONNECTION_STRING = f"postgresql://{user}:{password}@{host}:5432/{dbname}?sslmode=disable"

# Create a PGVector instance to house the documents and embeddings
db = PGVector.from_documents(
    documents= docs,
    embedding = embeddings,
    distance_strategy = DistanceStrategy.COSINE,
    connection_string=CONNECTION_STRING,
     pre_delete_collection=True
)
print(db.as_retriever)



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/Users/rampavandamela/Developer/models/mistral-7b-openorca.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=400,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048,
    temperature=0.9
)

template = """Act as a Workinsync/Moveinsync product Assitance. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't answer any question related to people/humans. Use three sentences maximum. Keep the answer as concise as possible. at the end of the answer Always say "thanks for asking! - InSyncBot 🤖". 

Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key = "answer",
    return_messages=True
)


retriever=db.as_retriever()
# qa = ConversationalRetrievalChain.from_llm(
#     llm,
#     retriever=retriever,
#     return_source_documents=True,
#     condense_question_prompt=QA_CHAIN_PROMPT,
#     memory=memory
# )

dic_qa = {}


def getChatHistory(result):
    list_test = []
    dic_test = {}
    for chat in result["chat_history"]:
        print(chat)
        if(type(chat) is HumanMessage):
            dic_test["human_message"] = chat.content
        else:
            dic_test["ai_message"] = chat.content
            r = json.dumps(dic_test)
            list_test.append(r)
            dic_test = {}
    return list_test

def say_hello(request):
    id = len(dic_qa) + 1
    new_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    return_source_documents=True,
    condense_question_prompt=QA_CHAIN_PROMPT,
    memory=memory
    )
    dic_qa[id] = new_qa
    return HttpResponse(id)


def ask_question(request):
    question = request.GET.get('question', '')
    id = request.GET.get('id', '')
    qa_ses = dic_qa[int(id)]
    result = qa_ses({"question": question})
    response = {}
    response["answer"] = result.get("answer")
    response["source_documents"] = result.get("source_documents")[0].metadata["source"]
    response["chat_history"] = getChatHistory(result)
    dump_r = json.dumps(response)
    return HttpResponse(dump_r)