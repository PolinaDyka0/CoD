# CoD Summarization Tool

## File Structure
The tool consists of three main Python files, each serving a distinct purpose:

- `CoD.py`: Implements the Chain of Density (CoD) prompting. This file contains the core algorithms and logic that drive the Chain of Density approach.

- `CoD_summarization.py`: Focuses on the text summarization approach inspired by Chain of Density. This is where the main summarization function CoD_summarization(text) resides, which users can call to generate summaries based on the CoD methodology.

- `simple_summarization.py`: Offers a simpler text summarization approach. This can be used for quick and basic summaries without the iterative enhancements provided by the CoD approach.

## Overview

This tool offers a unique text summarization approach inspired by Chain of Density (CoD). It aims to iteratively enhance the summary by incorporating critical entities that may have been initially missed. With each iteration, as we incorporate missing entities, we use cosine similarity to identify the least similar sentence in the resume and replace it with a new sentence that contains the missing element. 

## Key Features

- **Entity Extraction**: Utilizes `spaCy` to extract named entities from texts, providing insights into the main subjects or objects discussed.
    
- **Initial Summary Generation**: Uses word frequencies and sentence rankings to generate an initial summary, ensuring that the most discussed topics are included.
    
- **Salient Entity Identification**: Compares the initial summary with the original text to identify missing entities.
    
- **Summary Enhancement**: Incorporates missing entities into the summary without increasing its overall length, ensuring a dense and information-rich summary.
    
- **Iteration Data Logging**: For each iteration, the tool logs the missing entities and the enhanced summary, allowing for a detailed understanding of the summarization process.
    

## Dependencies

- `spaCy`
- `nltk`
- `heapq`
- `string`
- `collections`
- `sklearn`
- `datasets`
- `pandas`

## Usage

To use the CoD Summarization Tool, simply provide a text to be summarized and call the `CoD_summarization(text)` function. The example provided in the script uses a detailed text about the Orbiter Discovery, OV-103, and its significance in the U.S. Space Shuttle Program.

The tool will then perform the CoD inspired summarization, iterating three times to enhance the summary. Each iteration's data, including missing entities and the denser summary, is logged for review.

## Output

The output is structured in a JSON-like format, showing the progress of summarization across iterations:

```json
{
    "Iteration": 1,
    "Missing_Entities": ["Entity1", ...],
    "Denser_Summary": "Enhanced summary here..."
},
...

```
