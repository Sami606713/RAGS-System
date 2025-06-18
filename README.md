## Backend

This folder contains the backend services for the Document Chat App.

### `app.py`
This file is the main entry point for the Streamlit web application. It handles the user interface, chat history management, and interacts with the `agent` to process user queries and generate responses.

### `main.py`
This script is responsible for processing documents. It loads and extracts data (tables, texts, images) from PDF files in the `data` directory, summarizes them using the `summerizer` module, and then chunks and adds the processed documents to the vector store. It keeps track of processed files in `processed_files.txt` to avoid reprocessing.

### `data/`
This directory is intended to store the raw PDF documents that need to be processed by the system.

### `vectorStore/`
This directory stores the generated vector embeddings of the processed documents. These embeddings are used by the `agent` for retrieving relevant information during the chat.

### `agent/`
This module contains the logic for the conversational agent, which uses the vector store to answer questions based on the processed documents.

### `summerizer/`
This module provides functionalities for summarizing different types of content (text, images) extracted from the documents.

### `utils/`
This module contains utility functions, such as `helper.py` for loading and extracting data from documents.

### `tool/`
This module likely contains tools or functions used by the agent to perform specific tasks.

### `generator.py`
This file likely contains code related to generating responses or content within the application.

### How to Run
To run the backend application, you will typically run `app.py` using Streamlit after ensuring all dependencies are installed and documents are processed by `main.py`.

```bash
streamlit run app.py
```

### Running `main.py` for Document Embedding
To process and embed documents, run the `main.py` script. This script will load PDF files from the `data` directory, extract and summarize their contents, and then add them to the vector store.

```bash
python main.py
```

Make sure that the `data` directory contains the PDF files you want to process. The script will log processed files in `processed_files.txt` to avoid reprocessing them.
