import os
import pandas as pd
import xml.etree.ElementTree as ET
import spacy

# Load the spaCy model
nlp = spacy.load('es_core_news_sm')

# Function to extract text and tags from XML
def extract_tags(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = root.find('TEXT').text

    tags_info = []
    for tag in root.find('TAGS'):
        start = int(tag.attrib['start'])
        end = int(tag.attrib['end'])
        entity_type = tag.attrib['TYPE']
        tags_info.append((start, end, f"{entity_type}"))

    return text, tags_info

# Function to assign BIO tags
def assign_bio_tags(text, tags_info):
    doc = nlp(text)
    bio_tags = ['O'] * len(doc)
    
    for start, end, tag in tags_info:
        for i, token in enumerate(doc):
            if token.idx >= start and token.idx + len(token.text) <= end:
                if token.idx == start:
                    bio_tags[i] = f"B-{tag}"
                else:
                    bio_tags[i] = f"I-{tag}"
            elif token.idx + len(token.text) > end:
                break

    return doc, bio_tags

# Function to number sentences
def number_sentences(doc, bio_tags, global_sentence_count):
    data = []
    sentence_boundaries = [sent.start for sent in doc.sents]
    for i, (token,tag) in enumerate(zip(doc,bio_tags)):
        data.append((token.text, tag, f"Sentence {global_sentence_count}"))
        if i in sentence_boundaries:
            global_sentence_count += 1

    return data, global_sentence_count

def generate_and_save(output_file, folder_path):
    # List to store processed data from all files
    all_data = []
    sentence_count = 1  # Initialize global sentence count

    # Process each XML file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            text, tags_info = extract_tags(file_path)
            doc, bio_tags = assign_bio_tags(text, tags_info)
            data, sentence_count = number_sentences(doc, bio_tags, sentence_count)
            all_data.extend(data)

    # Create a DataFrame from processed data
    data_df = pd.DataFrame(all_data, columns=['Token', 'Tag', 'Sentence #'])

    # Save the DataFrame to a CSV file
    data_df.to_csv(output_file, index=False, sep='\t')

    print("TSV file generated successfully!")

# Folder path containing XML train files
folder_path = r'.\corpus\train\xml'

generate_and_save(r'.\data\train.tsv', folder_path)


# Folder path containing XML test files
folder_path = r'.\corpus\test\xml'

generate_and_save(r'.\data\test.tsv', folder_path)