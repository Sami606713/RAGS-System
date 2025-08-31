import os
from vectorStore.vectorStore import add_to_vector_store,get_embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from splitter.sectionbaseSplitter import section_base_splitter
from langchain_core.documents import Document
from CustomLoader.customLoader import MarkdownAndChunksLoader
import re

def main():
    root_dir = "docs"
    processed_log_path = "processFile3.txt"

    # Load already processed file names
    if os.path.exists(processed_log_path):
        with open(processed_log_path, 'r') as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    files = os.listdir(root_dir)
    print(">> Files: ", files)
    print(">> Processed Files: ", processed_files)
    print(">> Processing Files...")

    for file in files:
        file_path = os.path.join(root_dir, file)

        if file not in processed_files and file.lower().endswith('.json'):
            print(f">> Processing: {file}")

            loader = MarkdownAndChunksLoader(
                file_path=file_path
            )

            documents = loader.load()
            # print("Number of LangChain documents:", len(documents))
            # print("Length of text in the first document:", len(documents[1].page_content))
            # print("Metadata of the first document:", documents[1].metadata)

            # Make a global summary of the document
            prompt = ChatPromptTemplate.from_messages(
                [("system", "Write a concise summary of the following:\\n\\n{context}")]
            )

            # Instantiate chain
            print("Creating LLM and Chain for Global Summary...")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            chain = create_stuff_documents_chain(llm, prompt)

            # Invoke chain
            global_summary = chain.invoke({"context": documents})
            print("Summary Generated:")
            # Save global summary to a text file
            splitter =  SemanticChunker(
                get_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.3,
                min_chunk_size=800,
                buffer_size=2   # <– this will reduce over-splitting
            )

            chunked_docs = splitter.split_documents(documents)
            print(f"--- Total Chunks after SectionBaseSplitter: {len(chunked_docs)} ---")
            print("Sample metadata from first chunk:", chunked_docs[0].metadata)
            print("Sample text from first chunk:", chunked_docs[0].page_content[:500])

            # now summarize each chunk with the global summary as context
            print("Processing Each Chunks")
            for i, doc in enumerate(chunked_docs):
                print(f">> Summarizing chunk {i+1}/{len(chunked_docs)}")
                # Create a Document object with the chunk text
                chunk_doc = Document(page_content=doc.page_content, metadata=doc.metadata)
                chunk_summary = chain.invoke({"context": [chunk_doc]})

                # attached the global summary and local summary to metadata
                doc.metadata["global_summary"] = global_summary
                doc.metadata["chunk_summary"] = chunk_summary
            
            print(">> Added global and chunk summaries to metadata")
            print(">> Sample metadata from first chunk:", documents[0].metadata)
            print(">> Sample text from first chunk:", documents[0].page_content[:500])
            
            print("Summarization done for each chunk")

            add_to_vector_store(docs_chunks=chunked_docs)

            # Mark file as processed
            with open(processed_log_path, 'a') as f:
                f.write(file + '\n')

            print(f">> Marked {file} as processed")
        else:
            print(f"!! Skipping already processed or unsupported file: {file}")


if __name__ == "__main__":
    main()
