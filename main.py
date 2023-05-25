import sys
import csv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

loader = UnstructuredFileLoader("ddl.txt")
document = loader.load()

text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 3000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = loader.load_and_split(text_splitter=text_splitter)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

csv_filename = sys.argv[1]
ddl_context = retriever.get_relevant_documents(csv_filename)

with open("template.txt", "r") as template_file:
    template = template_file.read()

with open(csv_filename, 'r') as csv_file:
    reader = csv.reader(csv_file)
    rows = [row for row in reader]
csv_attributes = rows[0]

prompt = PromptTemplate(input_variables=["context", "csv_attributes"], template=template)
qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=prompt)

with get_openai_callback() as cb:
    response = qa_chain({"input_documents":ddl_context, "csv_attributes":csv_attributes}, return_only_outputs=True)





