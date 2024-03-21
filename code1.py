import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlRequiments import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings )
    return vectorstore

def get_conversation(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id = "roborovski/superprompt-v1", model_kwargs={"temperature": 0.3, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_text):
    response = st.session_state.conversation({'question': user_text})
    st.session_state.chat_history = response['chat_history']

    for idx, message in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    #Load API Keys for OpenAI and HuggingFace
    load_dotenv()

    # Stream GUI Code
    st.set_page_config(page_title="Welcome to the PDF QA Chatbot", page_icon=":page:")

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF Q&A ChatBot :page:")
    user_text = st.text_input("Enter your question here:")
    if user_text:
        handle_userinput(user_text)

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click process", 
                                    accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text

                raw_text = get_pdf_text(pdf_docs)

                # divite text to chunks

                text_chunks = get_text_chunks(raw_text)

                # create vector store to save embedded chunks

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation(vectorstore) # st.session is used so that the conversation variable is not initialized again and again within the same session. 



if __name__ == '__main__':
    main()