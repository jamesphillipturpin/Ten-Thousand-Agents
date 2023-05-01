# Ten Thousand Agents

The Ten-Thousand-Agents repository aims to be a collection of AI agent scripts and tools.
By collecting these scripts and tools together in one place, as concrete examples,
this repository aims to provide templates and building blocks for creating custom agents 
and teams of agents.

### List of Current Agents:
1. Answer Question from Source (answer_question_from_source.py)
2. Baby Artificial General Intelligence (baby_agi.py)

### Environmental Variables
Environmental variable can be edited by clicking on the Windows search panel, and typing,
"Edit environmental variables for your account".
--> "Edit environmental variables for your account"
Under "User variable", click "New" for a new environmental varialbe,
or select an environmental variable and click "Edit".

Some scripts may require environmental variables to be set. Currently these include:
1. The OpenAI API key. It is best practices to set API keys using environmental variables,
especially with open source projects. By not including them in source code, we obviate the
need for end users to edit source code to get the intended as-built functionality.
The environmental variable is: OPENAI_API_KEY
The value is you OpenAI API key. If you don't know your key, you can make a new one here:
https://platform.openai.com/account/api-keys

## Answer Question from Source

The answer_question_from_source.py script answers questions based on sources you provide. 
The sources can be a single file, or a folder. Sources can be text files, web pages, or PDFs. 
They can be files on your computer or they can be webpages. You can specify a list of 
extensions to include, in which case other files will be excluded.

Implementation of core functionality of this script began from an example found in the 
langchain documentation here:

https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa_with_sources.html

This script has been modified with an argument parser to be a command line tool. Modularity has 
been added so that its functionality can more easily be ported to other projects as well. It has 
also been modified to read folders, pdfs, and webpages, in addition to text files. 
It has been made more verbose for comprehension and debugging purposes.

The program gives sources in terms of a database index, which is not the most helpful to humans 
looking for the source at this time.  However, it does indicate whether the answer is based on 
the provided source(s) or not.  If no source is given, the knowledge is based on the language 
model not the source documents given.

This script answer questions with sources over an Index. It does this by using the RetrievalQAWithSourcesChain, which does the lookup of the documents from an Index.

### Installation

If you do not have Python, Git, and Pip installed, install those first.

```
git clone https://github.com/jamesphillipturpin/Ten-Thousand-Agents.git
cd ten-thousand-agents
pip install -r requirements.txt
```

### Example run:

At the command prompt, type
```
Python answer_question_from_source.py --question="What is the Paycheck Fairness Act?" --path="https://en.wikipedia.org/wiki/Paycheck_Fairness_Act" --temp=1
```

The output may be something like this.
```
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
```


## Baby Artificial General Intelligence

The baby_agi.py script takes an objective and cycles through three stages:
1. Make a To-Do list
2. Research the next step on the to-do list
3. Execute the next step on the to-do list

### Installation

The baby general intelligence installation is somewhat more involved than
the answer_question_from_source.py script. This is because it depends on
the faiss module, which is in alpha development and has minimal support,
especially for Windows.

Below is the installation procedure I used to install on Windows. If you want
to try to install the module to be compatible with GPU, you can try omitting
the line that contains "faiss-cpu=". However, it is expected on Windows that 
installation of faiss will fail in that case, because GPU support of faiss 
for Windows has not been added.

First install Microconda (or Anaconda).
Run Microconda (or Anaconda) as administrator
```
conda create -n faiss_1.7.3 python=3.8
conda activate faiss_1.7.3
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install numpy
conda install -c pytorch faiss-gpu=1.7.3 cudatoolkit=11.3
conda install -c conda-forge notebook
conda install -y matplotlib
conda install -c pytorch faiss-cpu=1.7.3 cudatoolkit=11.3
conda install faiss
conda install re
pip install langchain
pip install pydantic
pip install openai
```

### Example usage
Download or clone the jamesphillipturpin/ten-thousand-agents repository.
Run Microconda (or Anaconda).
Actiave the faiss_1.7.2 environment by typing this into the Anaconda Powershell Prompt:
```
conda activate faiss_1.7.3
```
Then navigate to the ten-thousand-agents repository folder.
```
python -m baby_agi.py --objective="Do something benneficial." 
```
