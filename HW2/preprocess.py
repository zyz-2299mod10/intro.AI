import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
import re

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def remove_HTML(text: str) -> str:
    rmHTML = re.sub("<[^>]+>", "", text).strip()

    return rmHTML

def remove_symbol(text: str) -> str:
    filter = RegexpTokenizer(r'\w+', gaps = False)
    rmsym = filter.tokenize(text)
    preprocessed_text = ' '.join(rmsym)

    return preprocessed_text

def lemmatization(text: str) -> str:
    filter = nltk.stem.wordnet.WordNetLemmatizer()
    text = text.split(" ")
    lemed = []
    for i in text:
        lem = filter.lemmatize(i, "n")
        if(lem == i): lem = filter.lemmatize(i, "v")
        lemed.append(lem)
    
    preprocessed_text = ' '.join(lemed)

    return preprocessed_text
    

def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    preprocessed_text = remove_HTML(preprocessed_text)
    preprocessed_text = remove_symbol(preprocessed_text)
    preprocessed_text = lemmatization(preprocessed_text)
    # End your code

    return preprocessed_text