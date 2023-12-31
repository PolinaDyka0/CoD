import PyPDF2
import unicodedata
from typing import List
from dataclasses import dataclass
from llm_core.splitters import TokenSplitter
from llm_core.assistants import OpenAIAssistant
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a given PDF file."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages = [page.extract_text() for page in pdf_reader.pages]
    return ''.join(pages)


def cleanup_unicode(text: str) -> str:
    """Normalize unicode characters in the given text."""
    corrected_chars = [unicodedata.normalize("NFKC", char) for char in text]
    return "".join(corrected_chars)


def get_first_tokens(text: str, chunk_size: int = 6_000) -> str:
    """Retrieve the first set of tokens from the text."""
    splitter = TokenSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return next(splitter.chunkify(text))


@dataclass
class DenseSummary:
    denser_summary: str
    missing_entities: List[str]


@dataclass
class DenserSummaryCollection:
    system_prompt = """
    You are an expert in writing rich and dense summaries in broad domains.
    """
    prompt = """
    Article:
    
    {article}

    ----

    You will generate increasingly concise, entity-dense summaries of the above
    Article.

    Repeat the following 2 steps 5 times.

    - Step 1: Identify 1-3 informative Entities from the Article
    which are missing from the previously generated summary and are the most
    relevant.

    - Step 2: Write a new, denser summary of identical length which covers
    every entity and detail from the previous summary plus the missing entities

    A Missing Entity is:

    - Relevant: to the main story
    - Specific: descriptive yet concise (5 words or fewer)
    - Novel: not in the previous summary
    - Faithful: present in the Article
    - Anywhere: located anywhere in the Article

    Guidelines:
    - The first summary should be long (4-5 sentences, approx. 80 words) yet
    highly non-specific, containing little information beyond the entities
    marked as missing.

    - Use overly verbose language and fillers (e.g. "this article discusses") to
    reach approx. 80 words.

    - Make every word count: re-write the previous summary to improve flow and
    make space for additional entities.

    - Make space with fusion, compression, and removal of uninformative phrases
    like "the article discusses"

    - The summaries should become highly dense and concise yet self-contained,
    e.g., easily understood without the Article.

    - Missing entities can appear anywhere in the new summary.

    - Never drop entities from the previous summary. If space cannot be made,
    add fewer new entities.

    > Remember to use the exact same number of words for each summary.
    Answer in JSON.

    > The JSON in `summaries_per_step` should be a list (length 5) of
    dictionaries whose keys are "missing_entities" and "denser_summary".

    """

    summaries: List[DenseSummary]

    @classmethod
    def summarize(cls, article: str):
        with OpenAIAssistant(cls, model='gpt-4') as assistant:
            return assistant.process(article=article)


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    assistant = OpenAIAssistant(api_key=api_key, model='gpt-4')
    pdf_path = 'CoD.pdf'
    
    try:
        text = extract_text_from_pdf(pdf_path)
        text = cleanup_unicode(text)
        first_10_pages = get_first_tokens(text)

        print(first_10_pages)

        result = DenserSummaryCollection.summarize(first_10_pages)
        print(result)
    except Exception as e:
        print(f"Error occurred: {e}")
