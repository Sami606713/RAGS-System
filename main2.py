import os
from summerizer.summarizer import summarize_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from vectorStore.vectorStore import add_to_vector_store,get_embeddings
from langchain_unstructured import UnstructuredLoader


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

        if file not in processed_files and file.lower().endswith('.md'):
            print(f">> Processing: {file}")

            loader = UnstructuredLoader(
                file_path,
                chunking_strategy="by_title",
                multipage_sections=True,
            )

            documents = loader.load()
            print("Number of LangChain documents:", len(documents))
            print("Length of text in the first document:", len(documents[0].page_content))
            
            # generate the summary for each doc
            for doc in documents:
                doc.metadata["summary"] = summarize_text(data=doc.page_content)
            print(">> Summary Generated")

            # for doc in documents:
            #     metadata = doc.metadata
            #     metadata["summary"] = text_summary if isinstance(text_summary, str) else text_summary[0]
            #     print("MetaData:", metadata)

            # # Optional: split into smaller chunks before adding to vector store
            # splitter = SemanticChunker(
            #         get_embeddings(), breakpoint_threshold_type="gradient",
            #         # number_of_chunks=100,
            #     )

            # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            # chunks = splitter.split_documents(documents)

        #     add_to_vector_store(docs_chunks=chunks)

        #     # Mark file as processed
        #     with open(processed_log_path, 'a') as f:
        #         f.write(file + '\n')

        #     print(f">> Marked {file} as processed")
        # else:
        #     print(f"!! Skipping already processed or unsupported file: {file}")


if __name__ == "__main__":
    main()
