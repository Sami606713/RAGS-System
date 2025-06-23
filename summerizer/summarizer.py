from utils.helper import Summarizer

# Prompt for text/table summarization
TEXT_PROMPT = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, no additional comment.
Do not start your message by saying 'Here is a summary' or anything like that.
Just give the summary as it is.
Table or text chunk: {element}
"""

# Prompt for image summarization
IMAGE_PROMPT = """
Describe the image in detail. For context, the image is part of a research paper. Be specific about graphs, such as bar plots.
"""

def summarize_text(data, prompt_template=TEXT_PROMPT):
    """Summarize text or table data."""
    try:
        return Summarizer(prompt_template=prompt_template, data=data, config=True, set_messages=False)
    except Exception as e:
        return str(e)

def summarize_image(data, prompt_template=IMAGE_PROMPT):
    """Summarize image data."""
    try:
        return Summarizer(prompt_template=prompt_template, data=data, config=False, set_messages=True)
    except Exception as e:
        return str(e) 