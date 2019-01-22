# Import libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import gensim

# Import dataset (assuming from Excel file)
FILE = r'' # insert file name 
SHEET = r'' # insert sheet name
df = pd.read_excel(FILE, sheet_name=SHEET)

# (Optional) Condense each article's text into one cell if spans multiple columns
def condense_data(df):
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    overflow = df.query('text_2 == text_2')
    overflow = overflow.index.values
    for i in overflow:
        df.at[i, 'text_1'] = df.at[i, 'text_1'] + ' ' + df.at[i, 'text_2']
    return df

# Deletes handles, image caption/credits, links, money amounts, phone numbers,
# and navigation menus
r_ignore_case = {
    'handle': r'\S+@\S+',
    'link': r'((visit|at)\s+)*(\S+).(com|org)((/|\S)*)',
    'money': r'\$\d*',
    'phone': r'(\d-)*(\d+-\d+-\d+)' ,
    'menu': r'(?<=•)[^•]*(?=•)'
}
r_set_case = {
    'photographer-credits': r'PHOTO(GRAPHER)*:(\s)*(([A-Z][A-Za-z]*)\s*){1,3}',
    'graphics': r'Graphic([^.])'
}
def filter_text(text):
    for regex in r_ignore_case:
        text = re.sub(regex, '', text, re.I)
    for regex in r_set_case:
        text = re.sub(regex, '', text)
    return text

# Process (retaining only alphabetic chars, normalize case)
# Return tokenized text array
def process_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    return text

# Remove stopwords (input = tokenized text array)
ps = PorterStemmer()
def remove_stopwords(text):
    return [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

# Build bigram model (input = tokenized text array)
bigram = gensim.models.Phrases(df, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)
def make_bigrams(text):
    return [bigram_mod[doc] for doc in text]

# Clean the data in a single article
def clean_text(text):
    text = filter_text(text)
    text = process_text(text)
    text = remove_stopwords(text)
    text = make_bigrams(text)
    return text

# Extract only the article text from the dataset
df = condense_data(df)
data = df.loc[:, 'text_1':'text_1']
data = data.values

# Process each article
cleaned_data = []
for i in range(0, len(data)):
    data[i][0] = clean_text(data[i][0])
    # un-splinter the text
    for j in range(0, len(data[i][0])):
        data[i][0][j] = ''.join(data[i][0][j])
    # convert to utf-8, append to cleaned array 
    cleaned_data.append([x.encode('utf-8') for x in data[i][0]])

# save data as a np file for easy loading 
np.save('', cleaned_data) # put name of npy file 


