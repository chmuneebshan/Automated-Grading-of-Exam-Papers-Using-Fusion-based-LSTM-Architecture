# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import textstat
import joblib
from pathlib import Path
import docx

from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec, Doc2Vec, LdaModel, LsiModel
from gensim.corpora import Dictionary
from tensorflow.keras.models import load_model

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load models
model = load_model("Outpus/Model/trained_model.keras")
word2vec = Word2Vec.load("Outputs/word2vec_model.model")
lda_model = LdaModel.load("Outputs/lda_model.model")
lsi_model = LsiModel.load("Outputs/lsi_model.model")
doc2vec_model = Doc2Vec.load("Outputs/doc2vec_model.model")
dictionary = Dictionary.load("Outputs/corpus_dictionary.dict")

stop_words = set(stopwords.words('english'))

# --- Feature Extraction Functions ---
def length_based_features(text):
    sentences = text.split('.')
    words = text.split()
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    long_word_count = sum(1 for word in words if len(word) > 6)
    short_word_count = sum(1 for word in words if len(word) < 4)
    unique_tokens = set(words)
    nostop_words = [word for word in words if word.lower() not in stop_words]
    nostop_count = len(nostop_words)
    unique_token_count = len(unique_tokens)
    return {
        'word_count': word_count, 'unique_token_count': unique_token_count, 'nostop_count': nostop_count,
        'avg_sentence_length': avg_sentence_length, 'avg_word_length': avg_word_length,
        'sentence_count': sentence_count, 'long_word_count': long_word_count, 'short_word_count': short_word_count
    }

def calculate_pos_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in pos_tags)
    return {
        'noun': pos_counts['NN'] + pos_counts['NNS'],
        'adj': pos_counts['JJ'],
        'pron': pos_counts['PRP'] + pos_counts['PRP$'],
        'verb': sum(pos_counts[tag] for tag in ['VB','VBD','VBG','VBN','VBP','VBZ']),
        'cconj': pos_counts['CC'],
        'adv': pos_counts['RB'],
        'det': pos_counts['DT'],
        'propn': pos_counts['NNP'] + pos_counts['NNPS'],
        'num': pos_counts['CD'],
        'intj': pos_counts['UH']
    }

def punctuation_features(text):
    return {
        'period_count': text.count('.'),
        'comma_count': text.count(','),
        'question_mark_count': text.count('?'),
        'exclamation_mark_count': text.count('!'),
        'colon_count': text.count(':'),
        'semicolon_count': text.count(';'),
        'parentheses_count': (text.count('(') + text.count(')')) // 2,
        'hyphen_count': text.count('-'),
        'ellipsis_count': text.count('...')
    }

def readability_features(text):
    return {
        'Kincaid': textstat.flesch_kincaid_grade(text),
        'ARI': textstat.automated_readability_index(text),
        'Coleman_Liau': textstat.coleman_liau_index(text),
        'LIX': textstat.lix(text),
        'Flesch_Reading_Ease': textstat.flesch_reading_ease(text),
        'Gunning_Fog': textstat.gunning_fog(text),
        'SMOG': textstat.smog_index(text),
        'RIX': textstat.rix(text),
        'Dale_Chall': textstat.dale_chall_readability_score(text)
    }

def sentence_info_features(text):
    words = word_tokenize(text)
    syllables = textstat.syllable_count(text)
    long_words = sum(1 for word in words if len(word) > 6)
    sentences = sent_tokenize(text)
    paragraphs = text.split('\n')
    return {
        'characters_word': sum(len(word) for word in words) / len(words) if words else 0,
        'syll_word': syllables / len(words) if words else 0,
        'wordtypes': len(set(words)),
        'words_sentence': len(words) / len(sentences) if sentences else 0,
        'words': len(words),
        'sentences': len(sentences),
        'sentences_paragraph': len(sentences) / len(paragraphs) if paragraphs else 0,
        'complex_words': textstat.difficult_words(text),
        'type_token_ratio': textstat.lexicon_count(text, removepunct=True) / len(words) if words else 0,
        'characters': sum(len(word) for word in words),
        'syllables': syllables,
        'paragraphs': len(paragraphs),
        'long_words': long_words,
        'complex_dc': textstat.dale_chall_readability_score(text)
    }

def word_usage_features(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    pos_counts = Counter(tag for word, tag in tagged_words)
    return {
        'tobeverb': pos_counts['VB'],
        'auxverb': pos_counts['MD'],
        'conjunction': pos_counts['CC'],
        'pronoun': pos_counts['PRP'] + pos_counts['PRP$'],
        'preposition': pos_counts['IN'],
        'nominalization': sum(1 for word, tag in tagged_words if tag.startswith('NN') and len(word) > 6)
    }

def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed = []
    for s in sentences:
        words = [w.lower() for w in word_tokenize(s) if w.isalpha() and w.lower() not in stop_words]
        if words:
            processed.append(words)
    return processed

def clean_text(text):
    return [token for token in word_tokenize(text.lower()) if token.isalpha() and token not in stop_words]

def essay_to_vector(text, model):
    processed_sentences = preprocess_text(text)
    vectors = [np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)
               for sentence in processed_sentences if sentence]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# --- Streamlit App ---
st.title("Automated Essay Grading App")
input_mode = st.radio("Choose input mode", ["Paste Essay", "Upload File (.txt/.docx)"])

text = ""
if input_mode == "Paste Essay":
    text = st.text_area("Paste your essay here:", height=300)
elif input_mode == "Upload File (.txt/.docx)":
    uploaded_file = st.file_uploader("Upload your file", type=["txt", "docx"])
    if uploaded_file:
        ext = Path(uploaded_file.name).suffix
        if ext == ".txt":
            text = uploaded_file.read().decode("utf-8")
        elif ext == ".docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])

if st.button("Predict Score"):
    if not text.strip():
        st.warning("Essay text is empty!")
    elif len(text.split()) < 10:
        st.warning("Essay is too short. Please enter at least 10 words.")
    else:
        try:
            vocab_feats = {**length_based_features(text), **calculate_pos_features(text), **punctuation_features(text)}
            readability_feats = {**readability_features(text), **sentence_info_features(text), **word_usage_features(text)}
            sentence_vector = essay_to_vector(text, word2vec).reshape(-1, 1)
            chapter_input = clean_text(text)
            chapter_bow = dictionary.doc2bow(chapter_input)
            lda_vec = [val[1] for val in lda_model[chapter_bow]]
            lsi_vec = [val[1] for val in lsi_model[chapter_bow]]
            doc2vec_vec = doc2vec_model.infer_vector(chapter_input)
            chapter_feats = np.concatenate([lsi_vec, lda_vec, doc2vec_vec])

            vocab_input = np.array(list(vocab_feats.values())).reshape(1, -1, 1)
            readability_input = np.array(list(readability_feats.values())).reshape(1, -1, 1)
            sentence_input = sentence_vector.reshape(1, sentence_vector.shape[0], 1)
            chapter_input = chapter_feats.reshape(1, chapter_feats.shape[0], 1)

            prediction = model.predict([vocab_input, readability_input, sentence_input, chapter_input], verbose=0)
            st.success(f"✅ Predicted Essay Score: **{prediction[0][0]:.2f}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
