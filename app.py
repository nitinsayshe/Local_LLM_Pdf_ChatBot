import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import os
import base64
import time
import textwrap 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from constants import CHROMA_SETTINGS
from streamlit_chat import message

st.set_page_config(layout="wide")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# checkpoint = "LaMini-T5-738M"
# print(f"Checkpoint path: {checkpoint}")
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map='auto',
#     torch_dtype=torch.float32
# )
# Use a pipeline as a high-level helper
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline
# token = "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"
# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf",use_auth_token=token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token="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",force_download=True)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token="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",force_download=True)
persist_directory = "db"

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title('Search PDF')
    question =st.text_area("Enter question")
    if st.button("Search"):
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

if __name__ == "__main__":
    main()

