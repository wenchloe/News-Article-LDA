# Import libraries
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.models import CoherenceModel
import pyLDAvis.gensim
# Import warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Set LDA variables
LDA_FILE = ''
NUM_TOPICS = 10
RANDOM_STATE = 100
UPDATE_EVERY = 1
CHUNKSIZE = 100
PASSES = 10
ALPHA = 'auto'
PER_WORD_TOPICS = True

# Input: path of the np data file, name of saved LDA model (change parameters above)
# Output: creates the lda model for the data, saves with given name (as with corpus and dictionary)
#         returns the lda model
def create_lda(NP_FILE, LDA_FILE):
    # Load dataset
    data = np.load(NP_FILE) # will act as corpus
    # Create Dictionary
    id2word = corpora.Dictionary(data)
    # Term Document Frequency
    corpus = [id2word.doc2bow(elem) for elem in data]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=NUM_TOPICS,
                                               random_state=RANDOM_STATE,
                                               update_every=UPDATE_EVERY,
                                               chunksize=CHUNKSIZE,
                                               passes=PASSES,
                                               alpha=ALPHA,
                                               per_word_topics=PER_WORD_TOPICS)

    # Save the model, dictionary, and corpus
    lda_model.save(LDA_FILE)
    id2word.save(LDA_FILE + '.dict')
    corpora.MmCorpus.serialize(LDA_FILE + '.mm', corpus)
    # Return the model
    return lda_model


# Input: file name of saved LDA model
# Output: returns the loaded LDA model
def load_model(FILE):
    try:
        return models.LdaModel.load(FILE)
    except FileNotFoundError:
        print("File not found")


# Input: file name of saved LDA model dictionary
# Output: returns the loaded dictionary
def load_dict(FILE):
    try:
        return corpora.Dictionary.load(FILE)
    except FileNotFoundError:
        print("File not found")


# Input: file name of saved LDA corpus
# Output: returns the loaded corpus
def load_corpus(FILE):
    try:
        return corpora.MmCorpus(FILE)
    except FileNotFoundError:
        print("File not found")


# Input: the model, number of keywords per topic, (optional) file name to output to
# Output: keywords in each of the topics, (optional) text file with keywords
def get_keywords(model, num_words, output_file = None):
    if output_file != None:
        f = open(output_file, 'w')
    result = []
    topics = model.print_topics(num_words=num_words)
    for topic in topics:
        result.append(topic)
    # optional: extra task of writing the result into a text file
    if output_file != None:
        f = open(output_file, 'w')
        for topic in topics:
            f.write(topic + '\n')
        f.close()
    return result


# Input: lda model, corpus, dictionary, and the name of the output file
# Output: saves the visualization as an HTML file in current directory
def vis_topics(model, corpus, dict, F_OUTPUT):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, F_OUTPUT)


# Input: lda model, the texts (data, not term document frequency), and dictionary
# Output: coherence score for the model (the lower the better)
def get_coherence(model, text, dict):
    coherence_model = CoherenceModel(model=model, texts=text, dictionary=id2word, coherence='c_v')
    return coherence_model.get_coherence()

