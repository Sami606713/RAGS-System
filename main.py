import os
import shutil
from utils.helper import LoadAndExtractData  # Uncomment if you want to process files
from summerizer.imageSummerizer import Image_Summerizer
from summerizer.textSummerizer import TextSummerizer
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectorStore.vectorStore import add_to_vector_store

def main():
    try:
        root_dir = "data"
        processed_log_path = "processed_files.txt"

        # Load already processed file names
        if os.path.exists(processed_log_path):
            with open(processed_log_path, 'r') as f:
                processed_files = set(f.read().splitlines())
        else:
            processed_files = set()

        files = os.listdir(root_dir)
        print(">> Files: ",files)
        print(">> Process Files: ",processed_files)
        print(">> Processing Files ")
        for file in files:
            file_path = os.path.join(root_dir, file)

            # Only process files that don't exist in the process directory
            if file not in processed_files and file.lower().endswith('.pdf'):
                print(f">> Processing: {file}")

                tables, texts, images = LoadAndExtractData(file_path)

                print(">> Generating Summaries ")
                text_summary = TextSummerizer(data=texts)
                tables_summary = TextSummerizer(data=tables)
                images_summary = Image_Summerizer(data=images)

                print("Text Sumary: ",text_summary)
                print("Table Summary: ",tables_summary)
                print("Image Susmmary: ",images_summary)

                print(">> Summary Generated")

                print(">> Combine Each and every thing into one document")
                # Create Document objects for text chunks
                text_docs = [Document(page_content=str(text), metadata={"type": "text", "summary": text_summary[i], "source":file_path,"name":file}) for i, text in enumerate(texts)]

                # Create Document objects for table summaries (using the HTML representation)
                table_docs = [Document(page_content=tables[i], metadata={"type": "table", "summary": tables_summary[i],"source":file_path,"name":file}) for i, table in enumerate(tables)]

                # Create Document objects for image summaries
                image_docs = [Document(page_content=images[i], metadata={"type": "image", "summary": images_summary[i],"source":file_path,"name":file}) for i, image in enumerate(images)]

                # Combine all document types into a single list
                docs = text_docs + table_docs + image_docs

                print(">> Splitting Documents")
                document_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Example size, adjust based on your needs
                    chunk_overlap=200,  # Example overlap, adjust based on your needs
                    length_function=len,
                    is_separator_regex=False,
                )

                # Spli the documents
                docs_chunks = document_splitter.split_documents(docs)
                print(">> Splitting Done")
                
                add_to_vector_store(docs_chunks=docs_chunks)


                # Append to log file
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