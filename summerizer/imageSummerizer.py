# Image Summerizer
from utils.helper import Summarizer

prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper.
                  Be specific about graphs, such as bar plots."""


def Image_Summerizer(prompt_template =prompt_template,data=None):
    try:
        images_summary = Summarizer(prompt_template=prompt_template,data=data,config=False,set_messages=True)
        return images_summary
    except Exception as e:
        pass

if __name__ == "__main__":
    pass