import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import PyPDF2
from bs4 import BeautifulSoup
import requests


st.title("🚀 Hi, Lutfi wants to help you")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API Key", type="password")
    "[Get GROQ API key](https://console.groq.com/keys)"
    file_option = st.radio("Choose input type:", ["Text Input", "PDF File", "Website Link"])
    if file_option == "PDF File":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    elif file_option == "Website Link":
        website_url = st.text_input("Enter Website URL")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error fetching website: {str(e)}"

def generate_response(input_text):
    model = 'llama3-groq-70b-8192-tool-use-preview'
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model,
        temperature=0,
    )

    context = input_text  # Use the input text as the context

    # Define the prompt template
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact-based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.
    Please answer with the same language as the question.

    Assistant:"""

    PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace("{context}", context)

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["question"]
    )

    chain = (
        prompt
        | groq_chat
        | StrOutputParser()
    )

    st.info(chain.invoke({"question": input_text}))

with st.form("my_form"):
    text = st.text_area('Enter Text')
    if file_option == "Text Input":
        text = st.text_area("Enter text:")
    elif file_option == "PDF File" and uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
    elif file_option == "Website Link" and website_url:
        text = extract_text_from_website(website_url)
    else:
        text = ""

    submitted = st.form_submit_button("Submit")
    if not groq_api_key:
        st.info("Please add your GROQ API key to continue.")
    elif submitted and text:
        generate_response(text)
    elif submitted:
        st.warning("Please provide valid input.")
