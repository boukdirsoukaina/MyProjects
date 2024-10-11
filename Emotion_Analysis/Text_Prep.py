import nltk
nltk.download('stopwords')

#tokenization
import nltk
nltk.download('punkt')  # Download the necessary tokenizer models

from nltk.tokenize import word_tokenize

def token(content):
    return word_tokenize(content,"english")

token("i  am so happy to see you")

#removing stop words
from nltk.corpus import stopwords


def rm_stopwords(words):
    final_words= [word for word in words if word not in stopwords.words("english")]
    return final_words


nltk.download('wordnet') # for english

#stemming
from nltk.stem import SnowballStemmer #i should say why using it

snowball = SnowballStemmer(language='english')


#lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# stemming + lemmatization

def stem_lemma(word):
  stem =  snowball.stem(word)
  lemma = lemmatizer.lemmatize(stem,pos='v')

  return lemma


def many_stem_lemma(list_of_words):
    new_list =[]
    for word in list_of_words:
      stem =  snowball.stem(word)
      lemma = lemmatizer.lemmatize(stem,pos='v')
      new_list.append(lemma)
    return new_list
