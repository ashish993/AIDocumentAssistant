import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import docx2txt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import tempfile
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get groq_api_key from environment variable
groq_api_key = "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"

def initialize_session_state():
    try:
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! How can I help you today?"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")

def conversation_chat(query, chain, history):
    try:
        result = chain.invoke({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        return result["answer"]
    except Exception as e:
        logger.error(f"Error during conversation chat: {e}")
        return "An error occurred while processing your request."

def display_chat_history(chain):
    try:
        reply_container = st.container()
        container = st.container()

        with container:
            user_input = st.chat_input("Enter your questions here.")

            if user_input:
                with st.spinner('Generating response...'):
                    output = conversation_chat(user_input, chain, st.session_state['history'])
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="24")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="user2")
    except Exception as e:
        logger.error(f"Error displaying chat history: {e}")

def create_conversational_chain(vector_store):
    try:
        # Create llm
        llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name='llama3-8b-8192',
                temperature=0.5
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define prompt templates
        system_template = """You are an AI assistant. Provide answers only from the uploaded documents. 
        If you cannot find the information in the documents, say "Unable to find the information."
        Context: {context}"""
        
        human_template = "{question}"

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]
        
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': qa_prompt}
        )
        return chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {e}")
        return None

def main():
    try:
        # Initialize session state
        initialize_session_state()
        st.set_page_config(page_title="AI Document Assistant 🦾🤖")
        st.header("AI Document Assistant 🦾🤖")

        # Initialize Streamlit
        st.sidebar.title("Document Upload")
        uploaded_files = st.sidebar.file_uploader("Upload your file here (.pdf, .docx, or .txt)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                               model_kwargs={'device': 'cpu'})

        if uploaded_files:
            # extract text from uploaded files
            all_text = ""
            for uploaded_file in uploaded_files:
                try:
                    loading_message = st.info(f"Loading document: {uploaded_file.name}")
                    if uploaded_file.type == "application/pdf":
                        pdf_reader = PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    elif uploaded_file.type == "text/plain":
                        text = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = docx2txt.process(uploaded_file)
                    else:
                        st.write(f"Unsupported file type: {uploaded_file.name}")
                        continue
                    all_text += text
                    loading_message.empty()  # Remove the loading message
                except Exception as e:
                    logger.error(f"Error processing file {uploaded_file.name}: {e}")
                    st.write(f"Error processing file {uploaded_file.name}")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
            text_chunks = text_splitter.split_text(all_text)
            
            with st.spinner('Analyze Document...'):
                try:
                    # Create vector store
                    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                except Exception as e:
                    logger.error(f"Error creating vector store: {e}")
                    st.write("Error creating vector store")

            # Create the chain object
            chain = create_conversational_chain(vector_store)
                    
            if chain:
                display_chat_history(chain)
            else:
                st.write("Error creating conversational chain")
        else:
            st.warning('⚠️ Upload your document using the sidebar first to gain access to the chatbot.!')
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
