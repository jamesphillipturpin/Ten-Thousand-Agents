import os
import argparse
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# Initialize the vectorstore as empty
import faiss

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Argument Parser
parser = argparse.ArgumentParser(description='Retrieval Question Answering with Sources')
parser.add_argument('-f', '--filename', type=str, default=f"agi_output.txt", help='file to save seession to')
parser.add_argument('-o', '--objective', type=str, default=f"Do something benneficial.", help='Objective of artificial general intelligence.')

parser.add_argument('-t', '--temp', type=float, default=0, help='temperature for OpenAI API, (range: 0 to 2) (default: 0)')

args = parser.parse_args()
filename = args.filename
objective=args.objective
temp=args.temp

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

OBJECTIVE = objective

print(f"OBJECTIVE = {OBJECTIVE}")

llm = OpenAI(temperature=temp)

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 30
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})
