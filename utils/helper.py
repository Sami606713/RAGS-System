from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def LoadAndExtractData(file_path):
    try:
        # separate tables from texts
        tables = []
        texts = []

        print(">> Extracting Data")
        data = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        # strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image"],   # Add 'Tabl

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,

        # extract_images_in_pdf=True,          # deprecated
    )
        
        # Extract the tables and text
        print(">> Extracting Text and tables...")
        for chunk in data:
            if "Table" in str(type(chunk)):
                tables.append(chunk)

            if "CompositeElement" in str(type((chunk))):
                texts.append(chunk)
        print(">> Chunks are: ",data)
        # extract the image
        print(">> Extracting Images...")
        images = get_images_base64(data)
        return  tables ,texts, images
    except Exception as e:
        print("Error is: ",str(e))
        return [], [], str(e)
    


# Summarizer Function
def Summarizer(prompt_template, data, config=True, set_messages=False):
    """
    This function summarizes documents using a prompt template and the ChatOpenAI model.
    
    Args:
        prompt_template (str): Template string for the prompt.
        data (List[Dict] or List[str]): Input data to be summarized.
        config (bool): Whether to run the chain with concurrency limit.
        set_messages (bool): Whether to set messages as chat messages with an image.

    Returns:
        List[str]: List of summaries.
    """
    try:
        # api_key = os.getenv()
        if set_messages:
            messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image}"},
                        },
                    ],
                )
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
            summarize_chain = {"image": lambda x: x} | prompt | model | StrOutputParser()
        else:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
            summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        

        if config:
            return summarize_chain.batch(data, {"max_concurrency": 3})
        else:
            return summarize_chain.batch(data)
    except Exception as e:
        return str(e)

