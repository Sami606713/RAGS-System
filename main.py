import os
import shutil
from utils.helper import LoadAndExtractData
from summerizer.summarizer import summarize_text, summarize_image
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from vectorStore.vectorStore import add_to_vector_store,get_embeddings

def main():
    try:
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
        print(">> Process Files: ", processed_files)
        print(">> Processing Files ")
        for file in files:
            file_path = os.path.join(root_dir, file)

            # Only process files that don't exist in the process directory
            if file not in processed_files and file.lower().endswith('.pdf'):
                print(f">> Processing: {file}")

                tables, texts, images = LoadAndExtractData(file_path)

                print(">> Generating Summaries ")
                text_summary = summarize_text(data=texts)
                tables_summary = summarize_text(data=tables)
                images_summary = summarize_image(data=images)

                print("Text Summary: ", text_summary)
                print("Table Summary: ", tables_summary)
                print("Image Summary: ", images_summary)

                print(">> Summary Generated")
                print(">> Combine Each and every thing into one document")

                text_docs = [Document(page_content=str(text), metadata={"type": "text", "summary": text_summary[i], "source": file_path, "name": file}) for i, text in enumerate(texts)]
                table_docs = [Document(page_content=tables[i], metadata={"type": "table", "summary": tables_summary[i], "source": file_path, "name": file}) for i, table in enumerate(tables)]
                image_docs = [Document(page_content=images[i], metadata={"type": "image", "summary": images_summary[i], "source": file_path, "name": file}) for i, image in enumerate(images)]

                docs = text_docs + table_docs + image_docs

                print(">> Splitting Documents")
                # document_splitter = RecursiveCharacterTextSplitter(
                #     chunk_size=1000,
                #     chunk_overlap=200,
                #     length_function=len,
                #     is_separator_regex=False,
                # )

                document_splitter = SemanticChunker(
                    get_embeddings(), breakpoint_threshold_type="gradient",
                    number_of_chunks=100,
                    
                )

                docs_chunks = document_splitter.split_documents(docs)
                print(">> Splitting Done")
                add_to_vector_store(docs_chunks=docs_chunks)

                with open(processed_log_path, 'a') as f:
                    f.write(file + '\n')

                print(f">> Marked {file} as processed")
            else:
                print(f"!! Skipping already processed or unsupported file: {file}")

    except Exception as e:
        print("Error is:", str(e))
        return str(e)

if __name__ == "__main__":
    main()