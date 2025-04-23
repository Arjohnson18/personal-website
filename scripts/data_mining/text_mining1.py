# %%
#Library Importation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from dmba import printTermDocumentMatrix
# download data required for NLTK
nltk.download('punkt')

###--------------Sample Input Text
text = ['this is the first     sentence!!',
        'this is a second Sentence :)',
        'the third sentence, is here ',
        'forth of all sentences']

###--------------Create term-document matrix (TDM)
#Create CountVectorizer instance using default settings (removes punctuation, lowercases words)
count_vect = CountVectorizer() 

#Fit the vectorizer to the text and transform to a sparse term-document matrix
counts = count_vect.fit_transform(text) 

#Print the matrix: terms as rows, documents as columns, with frequency counts
printTermDocumentMatrix(count_vect, counts)

###--------------Custom Tokenization Pattern
#Include special characters like ! and :) in tokens
count_vect = CountVectorizer(token_pattern='[a-zA-Z!:)]+')

#Transform the same text using the new token pattern
counts = count_vect.fit_transform(text)

#Display the modified term-document matrix
printTermDocumentMatrix(count_vect, counts)

###--------------Stopword Display
#Get list of English stopwords
stopWords = list(sorted(ENGLISH_STOP_WORDS))

#Specify how many rows and columns to print
ncolumns = 6
nrows= 30

#Print the first 180 stopwords in a table format
print('First {} of {} stopwords'.format(ncolumns * nrows, len(stopWords)))
for i in range(0, len(stopWords[:(ncolumns * nrows)]), ncolumns):
    print(''.join(word.ljust(13) for word in stopWords[i:(i+ncolumns)]))

###--------------Text reduction using stemming
#Define a custom tokenizer that does:
# - Tokenization using NLTK's word_tokenize
# - Removes punctuation and stopwords
# - Applies stemming using the EnglishStemmer
class LemmaTokenizer(object):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.stopWords = set(ENGLISH_STOP_WORDS)

    def __call__(self, doc):
        #Tokenize the input, filter out punctuation and stopwords, and stem each token
        return [self.stemmer.stem(t) for t in word_tokenize(doc) 
                if t.isalpha() and t not in self.stopWords]

###--------------Create TDF Matrix
#Use the custom tokenizer in a CountVectorize
count_vect = CountVectorizer(tokenizer=LemmaTokenizer())

#Fit and transform the input text using the stemmer
counts = count_vect.fit_transform(text)

#Print the term-document matrix showing stemmed words
printTermDocumentMatrix(count_vect, counts)

###--------------Create TF-IDF (Term Frequency–Inverse Document Frequency) matrix
#Use a standard CountVectorizer for raw term counts
count_vect = CountVectorizer()

#Create a TF-IDF transformer:
# - smooth_idf=False: disables smoothing for IDF values
# - norm=None: doesn't normalize the result vectors
tfidfTransformer = TfidfTransformer(smooth_idf=False, norm=None)

#Transform raw term counts into TF-IDF
counts = count_vect.fit_transform(text)
tfidf = tfidfTransformer.fit_transform(counts)

#Print the TF-IDF matrix
printTermDocumentMatrix(count_vect, tfidf)
# %%
