from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# Load tokenize function
def tokenize(text):
    '''Tokensize text data'''
    tokens = word_tokenize(text.lower().strip())
  
    clean_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok)
        clean_tokens.append(clean_tok)
    return clean_tokens
