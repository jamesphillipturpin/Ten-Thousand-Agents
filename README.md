# Answer Questions from Source

This script answers questions based on sources you provide. The sources can be a single file, or a folder. Sources can be text files, web pages, or PDFs. They can be files on your computer or they can be webpages. You can specify a list of extensions to include, in which case other files will be excluded.

Implementation of core functionality of this script began from an example found in the langchain documentation here:

https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa_with_sources.html

This script has been modified with an argument parser to be a command line tool. Modularity has been added so that its functionality can more easily be ported to other projects as well. It has also been modified to read folders, pdfs, and webpages, in addition to text files. 
It has been made more verbose for comprehension and debugging purposes.

The program gives sources in terms of a database index, which is not the most helpful to humans looking for the source at this time.
However, it does indicate whether the answer is based on the provided source(s) or not.
If no source is given, the knowledge is based on the language model not the source documents given.

This script answer questions with sources over an Index. It does this by using the RetrievalQAWithSourcesChain, which does the lookup of the documents from an Index.

# Example run:

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
