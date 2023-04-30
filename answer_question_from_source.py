import os
import argparse
import requests
import PyPDF2
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma

# create argument parser
parser = argparse.ArgumentParser(description='Retrieval Question Answering with Sources')
parser.add_argument('-p', '--path', type=str, default=f"https://en.wikipedia.org/wiki/Universal_basic_income", help='path to file or folder')
parser.add_argument('-q', '--question', type=str, default=f"Who was the first person to propose Universal Basic Income?", help='question to ask chatbot')
parser.add_argument('-e', '--ext', nargs='+', default=None, help='list of file extensions to read (default: None, indicates all extensions included)')
parser.add_argument('-t', '--temp', type=float, default=0, help='temperature for OpenAI API (default: 0)')

args = parser.parse_args()
path = args.path
question=args.question
temp=args.temp
ext= args.ext

# collect files from path and extensions
def collect_files(path):
    if os.path.isdir(path):
        file_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(tuple(args.ext)):
                    file_list.append(os.path.join(root, file))
    else:
        file_list = [path]
    return file_list

# Read text from local file, webpage, or server
def read_text(path):
    if path.startswith("http"):
        response = requests.get(path)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
    elif path.endswith(".pdf"):
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    return text

def filter_extensions(file_list, allowed_extensions):
    if allowed_extensions:
        file_list = [file for file in file_list if os.path.splitext(file)[1] in allowed_extensions]
    return file_list


# Collect all the text from the file list into a single string.
def collect_text(file_list):
    collected_text = ''
    for file_path in file_list:
        collected_text += read_text(file_path)
    return collected_text

print("Reading source docuements from:")
print(path)
print()

file_list = collect_files(args.path)
file_list = filter_extensions(file_list, ext)
collected_text = collect_text(file_list)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(collected_text)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])

"""
Running Chroma using direct local API.
Using DuckDB in-memory for database. Data will be transient.
"""

from langchain.chains import RetrievalQAWithSourcesChain

from langchain import OpenAI

print()
print(f"Temperature set at {temp} out of range (0-2):")

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=temp), chain_type="stuff", retriever=docsearch.as_retriever())

print()
print("Asking the AI the following question:")
print(question)
print()
chain({"question": f"{question}"}, return_only_outputs=True)

#Chain Type

"""
You can easily specify different chain types to load and use in the RetrievalQAWithSourcesChain chain. For a more 
detailed walkthrough of these types, please see this notebook.
https://python.langchain.com/en/latest/modules/chains/index_examples/qa_with_sources.html

There are two ways to load different chain types. First, you can specify the chain type argument in the 
from_chain_type method. This allows you to pass in the name of the chain type you want to use. For example, in the 
below we change the chain type to map_reduce.
"""

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="map_reduce", retriever=docsearch.as_retriever())

chain({"question": f"{question}"}, return_only_outputs=False)

"""
The above way allows you to really simply change the chain_type, but it does provide a ton of flexibility over 
parameters to that chain type. If you want to control those parameters, you can load the chain directly (as you did 
in this notebook) and then pass that directly to the the RetrievalQAWithSourcesChain chain with the 
combine_documents_chain parameter. For example:
"""

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
qa_chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")
qa = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=docsearch.as_retriever())

print(qa({"question": f"{question}"}, return_only_outputs=True))

