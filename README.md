
Retrieval Augmented Generation (RAG) is a method for extending the capabilities of large language models (LLMs) by combining them with external, dynamic data. This approach enhances the model’s ability to reason about private or recently updated information, going beyond its static training data.  

#### Key Components of a RAG Application  
RAG applications consist of two main components:  

1. **Data Ingestion and Processing** (Offline):  
   - Involves loading, splitting, and storing data for future retrieval.  

2. **Retrieval and Generation** (Runtime):  
   - Handles user queries, retrieves relevant data, and generates responses using an LLM.  

#### Steps in a RAG Workflow  

1. **Load**  
   - Use **Document Loaders** to load data from sources (PDFs, DOCX files, or APIs).  
   - Each document is represented as an object containing content and metadata.  

2. **Split**  
   - Large documents are split into smaller chunks using **Text Splitters**.  
   - Chunking improves embedding efficiency and model compatibility by keeping context sizes manageable.  
   - Overlapping chunks ensure important context isn’t lost.  

3. **Store**  
   - Store document chunks in a **VectorStore** by embedding them using a text embedding model.  
   - Embeddings are high-dimensional vectors used to measure similarity between user queries and stored chunks.  

4. **Retrieve**  
   - Use a **Retriever** to fetch relevant chunks from the VectorStore based on the query's embedding.  
   - Techniques like maximal marginal relevance ensure diverse and relevant results.  

5. **Generate**  
   - Combine retrieved chunks with the query to construct a prompt for the LLM.  
   - The LLM generates a response based on this enriched context.  

---

#### Example Workflow: Conversational Retrieval Chain  
The Conversational Retrieval Chain is ideal for chatbots that track conversation history. It ensures continuity by integrating past interactions into the current query’s context.  

---

### Instructions for Using This Project  

**AIDocumentAssistant** enables users to query documents using a RAG-based architecture.  

#### Steps to Run the Project:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/ashish993/AIDocumentAssistant.git
   cd AIDocumentAssistant
   ```  

2. Add your GROQ API key to a `.env` file:  
   ```env
   GROQ_API_KEY=<your_groq_api_key>
   ```  

3. (Optional) Create a virtual environment:  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```  

4. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

5. Run the application using Streamlit:  
   ```bash
   streamlit run main.py
   ```  

6. Upload your documents (PDF, DOCX, or TXT), ask questions, and get responses powered by RAG!  

---

### Summary of RAG Workflow in AI DocumentAssistant  
1. **Data Ingestion**: Load and split documents using LangChain.  
2. **Storage**: Convert chunks into embeddings and store them in a VectorStore.  
3. **Query Processing**: Embed user queries and retrieve relevant chunks.  
4. **Response Generation**: Use retrieved data to generate answers with an LLM.  

This architecture ensures accurate, context-aware responses to queries based on uploaded document content.
