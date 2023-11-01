import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import pandas as pd


def extract_entities(text):
    """ 
    Extract named entities from the provided text using spaCy. 
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def count_entities_in_text(text):
    """
    Count all unique entities present in the provided text.
    """
    entities = extract_entities(text)
    return len(set(entities))

def initial_summary(text, ratio=0.3):
    """
    Generate an initial summary based on word importance.
    This is done by calculating word frequencies and then ranking sentences.
    """
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punctuation = string.punctuation + '\n'
    
    word_frequencies = {}
    for word in word_tokenize(text):
        word_lower = word.lower()
        if word_lower not in stop_words and word_lower not in punctuation:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    select_length = int(len(sentences) * ratio)  
    summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def get_salient_entities(text, summary):
    """ 
    Identify entities from the original text that are not present in the summary. 
    """
    text_entities = extract_entities(text)
    summary_entities = extract_entities(summary)
    missing_entities = [entity for entity in text_entities if entity not in summary_entities]
    entity_counts = Counter(missing_entities)
    return [item[0] for item in entity_counts.most_common(3)]

def incorporate_entities(text, summary, entities):
    """ 
    Enhance the summary by incorporating missing entities without increasing its length. 
    This is done by replacing least similar sentences.
    """
    for entity in entities:
        sentences = text.split('.')
        for sentence in sentences:
            if entity in sentence and sentence not in summary:
                summary_sentences = summary.split('.')
                # replace the least similar sentence with the new one
                similarities = [cosine_similarity(CountVectorizer().fit_transform([sentence, s])).flatten()[1] for s in summary_sentences]
                summary_sentences[similarities.index(min(similarities))] = sentence
                summary = '.'.join(summary_sentences)
                break
    return summary

def CoD_summarization(text):    
    """ 
    Execute the "Cycle of Density" (CoD) inspired summarization. 
    This method aims to iteratively enhance the summary with critical entities.
    """
    summary = initial_summary(text)
    iteration_data = []
    
    for i in range(3):
        salient_entities = get_salient_entities(text, summary)
        denser_summary = incorporate_entities(text, summary, salient_entities)
        
        iteration_data.append({
            "Iteration": i + 1,
            "Missing_Entities": salient_entities,
            "Denser_Summary": denser_summary
        })
        
        summary = denser_summary
    
    return iteration_data

if __name__ == "__main__":
    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')  
    # Example text to be summarized
    text = "The Orbiter Discovery, OV-103, is considered eligible for listing in the National Register of Historic Places (NRHP) in the context of the U.S. Space Shuttle Program (1969-2011) under Criterion A in the areas of Space Exploration and Transportation and under Criterion C in the area of Engineering. Because it has achieved significance within the past fifty years, Criteria Consideration G applies. Under Criterion A, Discovery is significant as the oldest of the three extant orbiter vehicles constructed for the Space Shuttle Program (SSP), the longest running American space program to date; she was the third of five orbiters built by NASA. Unlike the Mercury, Gemini, and Apollo programs, the SSP’s emphasis was on cost effectiveness and reusability, and eventually the construction of a space station. Including her maiden voyage (launched August 30, 1984), Discovery flew to space thirty-nine times, more than any of the other four orbiters; she was also the first orbiter to fly twenty missions. She had the honor of being chosen as the Return to Flight vehicle after both the Challenger and Columbia accidents. Discovery was the first shuttle to fly with the redesigned SRBs, a result of the Challenger accident, and the first shuttle to fly with the Phase II and Block I SSME. Discovery also carried the Hubble Space Telescope to orbit and performed two of the five servicing missions to the observatory. She flew the first and last dedicated Department of Defense (DoD) missions, as well as the first unclassified defense-related mission. In addition, Discovery was vital to the construction of the International Space Station (ISS); she flew thirteen of the thirty-seven total missions flown to the station by a U.S. Space Shuttle. She was the first orbiter to dock to the ISS, and the first to perform an exchange of a resident crew. Under Criterion C, Discovery is significant as a feat of engineering. According to Wayne Hale, a flight director from Johnson Space Center, the Space Shuttle orbiter represents a “huge technological leap from expendable rockets and capsules to a reusable, winged, hypersonic, cargo-carrying spacecraft.” Although her base structure followed a conventional aircraft design, she used advanced materials that both minimized her weight for cargo-carrying purposes and featured low thermal expansion ratios, which provided a stable base for her Thermal Protection System (TPS) materials. The Space Shuttle orbiter also featured the first reusable TPS; all previous spaceflight vehicles had a single-use, ablative heat shield. Other notable engineering achievements of the orbiter included the first reusable orbital propulsion system, and the first two-fault-tolerant Integrated Avionics System. As Hale stated, the Space Shuttle remains “the largest, fastest, winged hypersonic aircraft in history,” having regularly flown at twenty-five times the speed of sound."
    
    #data = pd.read_csv('articles.csv')
    #text = data['text'][0]

    summary = CoD_summarization(text)
    for _ in summary:
        print(_)
        print('\n')


    # Iterate over the summary data to print statistics.
    print("CoD Step | Entities|Density (E/T)")
    for data in summary:
        iteration = data["Iteration"]
        entities_count = count_entities_in_text(data["Denser_Summary"])
        summary_length = len(data["Denser_Summary"].split())
        density = entities_count / summary_length
        print(f" {iteration}       |  {entities_count}     |  {density:.4f}")
