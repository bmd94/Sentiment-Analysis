
import re
import nltk
import numpy
import pandas as pd
import string
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics  import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix


# ### Fonction de Pretraitement


def clean_data(X):
    processed_reviews = []

    for i in range(len(X)):
        processed_rev = X[i]

        # 1 -------- Elimintion de la ponctuation et de quelques caracteres speciaux
        p = ["؟؟", "؟", "--", "''", "``", "،"]
        punc = list(string.punctuation) + p
        processed_rev = "".join([word for word in processed_rev if word not in punc])

        # 2--------- Elimination des characteres alphanumeriques et les espaces multiples
        pat = [
            ("[0-9a-zA-Z]", ''),
            ("\s+", ' ')
        ]

        for a, b in pat:
            processed_rev = re.sub(a, b, processed_rev)

        # 3 --------- Elimination des diacritiques
        diacrtics = re.compile(r"[ ّ َ ً ُ ٌ ِ ٍ ْ]")
        processed_rev = re.sub(diacrtics, ' ', processed_rev)
        processed_rev = ("\s+", ' ', processed_rev)

        # 4 --------- Tokenization

        tokens = nltk.word_tokenize((processed_rev[2]))

        # 5 ----------  Elimination des elongations
        for j in range(len(tokens)):
            tokens[j] = re.sub(r'(.)\1{2,}', r'\1', tokens[j])

        # 6 ---------- Lemmatisation

        lemmatizer = nltk.WordNetLemmatizer()
        rslt = " "
        for w in tokens:
            lemma = lemmatizer.lemmatize(w)
            rslt = rslt + " " + lemma

        # 7 --------- Normalisation des lettres **Essayer sans et avec**

        pat = [
            ("[إأٱآا]", "ا"),
            ("ى", "ي"),
            ("ؤ", "ء"),
            ("ئ", "ء"),
            ("ث", "ت"),
            ("ق", "ف"),
            ("خ", "ح"),
            ("ج", "ح")
        ]

        for a, b in pat:
            rslt = re.sub(a, b, rslt)

        processed_reviews.append(rslt)

    return processed_reviews


# ## Selection et pretraitement des données

df=pandas.read_csv("LABR.tsv", sep='\t', header=0)


X=df.review[df.rating==5].iloc[:11756]
X=X.append(df.review[df.rating==1])
y=df.rating[df.rating==5].iloc[:11756]
y=y.append(df.rating[df.rating==1])


X = clean_data(X.values)


# ### Partitionnement des données

Xx_train, Xx_test, y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)


# ### algorithmes et features

classifiers = [MultinomialNB(alpha=1.0,  fit_prior='True'),BernoulliNB(alpha=1.0,  fit_prior='True' ,binarize=0.1),GaussianNB(var_smoothing=0.001),RandomForestClassifier(bootstrap= 'False', criterion= 'entropy',max_features='log2', min_samples_split= 10,n_estimators= 150),DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='sqrt', random_state=42, max_depth=None, max_leaf_nodes=500),KNeighborsClassifier(),SVC(C= 2, gamma='scale', kernel= 'rbf')] 

vectorizers = [TfidfVectorizer, CountVectorizer]
confs = [(1,1),(2,2),(1,2),(1,3)]


# ### Ecrire les resultats d'evaluation dans un fichier  


def evaluate_model(y_test,predictions, file ):
    file.write("\n" + str(confusion_matrix(y_test,predictions))+"\n")  
    file.write(str(classification_report(y_test,predictions))+"\n")  
    file.write(str(accuracy_score(y_test, predictions))+"\n")
    file.write(str(recall_score(y_test,predictions))+"\n")
    file.write(str(f1_score(y_test, predictions, zero_division=1))+"\n")
    


# ### Generer des separateurs dans un fichier

def write_pattern(type_, title, file ):
    
    if type_ == 'cls':     
        file.write("\n"+ 130*"+")
        file.write("\n"+ title +"\n")
        results.write(130*"+")
        
    elif type_ == 'vect':
        file.write("\n" +100*"=")
        file.write("\n"+ title +"\n")
        file.write(100*"=")
        
    else :
        file.write("\n" +60*'*')
        file.write("\n"+ title +"\n")
        file.write(60*'*')


# ### Tester les modeles et generer les resultats 

results = open("Resultat_LABR_unbalanced.txt",'w')

for cls in classifiers: 
    write_pattern('cls', str(cls), results)

    for vect in vectorizers:
        write_pattern('vect', str(vect), results)

        for conf in confs:
            write_pattern('conf', str(conf), results)

            vectorizer = vect(max_features=4000, stop_words=stopwords.words('arabic') ,ngram_range=conf) 
            X_train = vectorizer.fit_transform(Xx_train).toarray()
            X_test = vectorizer.transform(Xx_test).toarray()


            classifier = cls
            classifier.fit(X_train, y_train)

            predictions = classifier.predict(X_test)
            evaluate_model(y_test,predictions,results)
                           

results.close()





