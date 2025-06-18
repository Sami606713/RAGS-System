# Text Summerize
from utils.helper import Summarizer

# define the pronpt
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""

def TextSummerizer(prompt_template =prompt_text,data=None):
    try:
        text_summary = Summarizer(prompt_template=prompt_template,data=data,config=True,set_messages=False)
        return text_summary
    except Exception as e:
        pass
if __name__ == "__main__":
    print(TextSummerizer(data="""
            To download and use Poppler as a Python library (or make it accessible to Python), follow these steps based on your operating system. Poppler is not a Python package—it's a C++ PDF rendering library with command-line tools like pdfinfo, pdftotext, and others, which Python libraries like unstructured or pdf2image call internally.
                   """))