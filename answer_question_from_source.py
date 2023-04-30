"""
Answer Questions from Source

This script answers questions based on sources you provide. The sources can be a single file, or a folder. Sources can be text files, web pages, or PDFs. They can be files on your computer or they can be webpages. You can specify a list of extensions to include, in which case other files will be excluded.

Implementation of core functionality of this script began from an example found in the langchain documentation here:

https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa_with_sources.html

This script has been modified with an argument parser to be a command line tool. Modularity has been added so that its functionality can more easily be ported to other projects as well. It has also been modified to read folders, pdfs, and webpages, in addition to text files. 
It has been made more verbose for comprehension and debugging purposes.

The program gives sources in terms of a database index, which is not the most helpful to humans looking for the source at this time.
However, it does indicate whether the answer is based on the provided source(s) or not.
If no source is given, the knowledge is based on the language model not the source documents given.

This script answer questions with sources over an Index. It does this by using the RetrievalQAWithSourcesChain, which does the lookup of the documents from an Index.

Example run:

C:\Python310\answer_question_from_source>Python answer_question_from_source.py --question="What is the Paycheck Fairness Act?" --path="https://en.wikipedia.org/wiki/Paycheck_Fairness_Act" --temp=1
Reading source docuements from:
https://en.wikipedia.org/wiki/Paycheck_Fairness_Act

Created a chunk of size 1795, which is longer than the specified 1000
Created a chunk of size 1238, which is longer than the specified 1000
Created a chunk of size 2596, which is longer than the specified 1000
Created a chunk of size 4019, which is longer than the specified 1000
Created a chunk of size 1172, which is longer than the specified 1000
Created a chunk of size 1374, which is longer than the specified 1000
Created a chunk of size 2193, which is longer than the specified 1000
Created a chunk of size 1332, which is longer than the specified 1000
Created a chunk of size 1015, which is longer than the specified 1000
Using embedded DuckDB without persistence: data will be transient

Temperature set at 1.0 out of range (0-2):

Asking the AI the following question:
What is the Paycheck Fairness Act?

{'answer': ' The Paycheck Fairness Act is a proposed United States labor law that would add procedural protections to the Equal Pay Act of 1963 and the Fair Labor Standards Act as part of an effort to address the gender pay gap in the United States. It would limit exceptions to the prohibition for a wage rate differential based on any other factor other than sex to bona fide factors, such as education, training, or experience. It would also punish employers for retaliating against workers who share wage information, put the justification burden on employers as to why someone is paid less, and allow workers to sue for punitive damages of wage discrimination.\n', 'sources': '0-pl, 3-pl, 9-pl'}
"""
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

