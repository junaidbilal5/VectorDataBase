# ChromaDB RAG Pipeline with LangChain and OpenAI (Google Colab)

## Project Overview
This project demonstrates how to build a simple **Retrieval-Augmented Generation (RAG)** application using:

- **ChromaDB** as the vector database
- **OpenAI Embeddings** for converting text into vectors
- **LangChain** for document loading, text splitting, retrieval, and orchestration
- **GPT-4o-mini** for generating answers from retrieved context
- **Google Colab** as the development environment

The project loads text documents, creates embeddings, stores them in ChromaDB, performs semantic search, and finally uses an LLM to answer questions based only on the retrieved documents.

---

## Architecture

```text
TXT Files
    ↓
DirectoryLoader
    ↓
Text Splitting (Chunks)
    ↓
OpenAI Embeddings
    ↓
ChromaDB Vector Store
    ↓
Retriever (Top-K Documents)
    ↓
GPT-4o-mini
    ↓
Final Answer
```

---

## Technologies Used

- Python
- Google Colab
- LangChain
- ChromaDB
- OpenAI API
- GPT-4o-mini
- RecursiveCharacterTextSplitter
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)

---

## Project Workflow

### Step 1: Install Required Libraries

```python
!pip install chromadb
!pip install langchain-chroma
!pip install langchain-openai
!pip install langchain-community
!pip install langchain-text-splitters
```

---

### Step 2: Download Dataset

The notebook downloads a ZIP file containing text articles and extracts it.

```python
!wget https://www.dropbox.com/s/vs6ocyvpzzncvwh/new_articles.zip
!unzip new_articles.zip -d new_articles
```

---

### Step 3: Store OpenAI API Key Securely in Google Colab

Instead of hardcoding the API key, it is stored inside **Google Colab Secrets**.

#### Create a Secret

1. Open Google Colab.
2. Click the **Secrets** icon on the left sidebar.
3. Create a new secret.

Example:

```text
Name: chromaDB_connection
Value: your_openai_api_key
```

#### Access the Secret

```python
from google.colab import userdata

api_key = userdata.get("chromaDB_connection")
```

Set the environment variable:

```python
import os
os.environ["OPENAI_API_KEY"] = api_key
```

This approach keeps the API key secure and prevents accidentally pushing it to GitHub.

---

### Step 4: Load Documents

Only `.txt` files are loaded.

```python
loader = DirectoryLoader(
    "/content/new_articles",
    glob="./*.txt",
    loader_cls=TextLoader
)

documents = loader.load()
```

---

### Step 5: Split Documents into Chunks

Large documents are split into smaller pieces to fit within the LLM context window.

```python
split_text = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = split_text.split_documents(documents)
```

Configuration:

- Chunk Size: `1000`
- Chunk Overlap: `200`

---

### Step 6: Create Embeddings

```python
embedding_model = OpenAIEmbeddings()
```

Each text chunk is converted into a vector embedding.

---

### Step 7: Create ChromaDB Vector Store

```python
persist_directory = "db"

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)
```

This creates a local vector database and stores it in the `db` directory.

---

### Step 8: Reload Existing Database

```python
vectorDB_connected = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)
```

This allows the database to be reused without recreating embeddings.

---

### Step 9: Create a Retriever

```python
retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)
```

The retriever returns the top 2 most relevant documents.

---

### Step 10: Semantic Search

Example:

```python
response = retriever.invoke(
    "how much money did microsoft raise?"
)
```

The query is converted into embeddings and matched against stored vectors.

---

### Step 11: Build the RAG Application

```python
llm = ChatOpenAI(model="gpt-4o-mini")
```

Custom function:

```python
def ask_question(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    response = llm.invoke(
        f'''
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
'''
    )

    return response.content
```

Example:

```python
query = "what is the news about pando?"
answer = ask_question(query)
print(answer)
```

---

## Backup and Cleanup

Create a backup:

```python
!zip -r db_backup.zip ./db
```

Delete collection:

```python
vector_db.delete_collection()
```

Remove local database:

```python
!rm -rf ./db
```

Restore database:

```python
!unzip db_backup.zip
```

---

## Project Structure

```text
├── ChromaBD.ipynb
├── new_articles/
│   ├── article1.txt
│   ├── article2.txt
│   └── ...
├── db/
├── db_backup.zip
└── README.md
```

---

## Learning Outcomes

In this project I learned:

- How vector databases work.
- How to create embeddings using OpenAI.
- How to build a RAG pipeline using LangChain.
- How semantic search works.
- How to persist and reload ChromaDB.
- How to secure API keys using Google Colab Secrets.
- How to connect retrieved documents with an LLM to generate grounded answers.

---

## Future Improvements

- Add PDF support.
- Use metadata filtering.
- Add conversation memory.
- Deploy as a Streamlit application.
- Replace local ChromaDB with a cloud vector database such as Pinecone.
- Add evaluation and monitoring for RAG responses.

---

## Author

**Junaid Bilal**  
Senior Data Engineer | Azure Data Engineer | Data Analytics Engineer

GitHub: https://github.com/

LinkedIn: https://www.linkedin.com/

