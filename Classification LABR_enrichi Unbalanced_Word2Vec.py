#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


# In[37]:


def write_pattern(type_, title, file ):
    
    if type_ == 'cls':     
        file.write("\n"+ 130*"+")
        file.write("\n"+ title +"\n")
        results.write(130*"+")
        


# In[38]:


def evaluate_model(y_test,predictions, file ):
    file.write("\n" + str(confusion_matrix(y_test,predictions))+"\n")  
    file.write(str(classification_report(y_test,predictions))+"\n")  
    file.write(str(accuracy_score(y_test, predictions))+"\n")
    file.write(str(recall_score(y_test,predictions))+"\n")
    file.write(str(f1_score(y_test, predictions, zero_division=1))+"\n")


# In[1]:


def make_feature_vec(words, model, num_features):
    """
    calculating the average feature vector of a review
    """
    
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    nwords = 0
    index2word_set = set(model.wv.index2word)  # words known to the model

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            feature_vec = np.add(feature_vec,model[word])
    
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    """
    Calculate average feature vectors for all reviews 
    by averaging all word vectors in a review
    """
    
    counter = 0
    review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')  # pre-initialize (for speed)
    
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1
    return review_feature_vecs

def remove_nan_instances(train_test_sets):
    
    """ remove instances in train and test set that could not be represented as feature vectors """
    
    x_set = train_test_sets[0]
    y_set = train_test_sets[1]
    
    nan_indices = list({x for x,y in np.argwhere(np.isnan(x_set))})
    
    if len(nan_indices) > 0:
        print(len(nan_indices))
        print('Removing {:d} instances.'.format(len(nan_indices)))
        x_set = np.delete(x_set, nan_indices, axis=0)
        y_set.drop(y_set.index[nan_indices],inplace=True)
    

        return x_set, y_set


# In[23]:


def clean_data(X):
    processed_reviews = []
    
    for i in range(len(X)):
        processed_rev= X[i]
     
        # 1 -------- Elimintion de la ponctuation et de quelques caracteres speciaux
        p=["؟؟","؟","--","''","``","،"]
        punc=list(string.punctuation)+p
        processed_rev= "".join([word for word in processed_rev if word not in punc])
    
        
        # 2--------- Elimination des characteres alphanumeriques et les espaces multiples
        pat =[  
                ("[0-9a-zA-Z]",''),
                ("\s+",' ')
             ]

        for a,b in pat:
            processed_rev=re.sub(a,b,processed_rev)
       
        
        # 3 --------- Normalisation des lettres **Essayer sans et avec**
        
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
        
        for a,b in pat:
            processed_rev = re.sub(a,b,processed_rev)
            
        # 4 --------- Elimination des diacritiques 
        diacrtics= re.compile(r"[ ّ َ ً ُ ٌ ِ ٍ ْ]")
        processed_rev = re.sub(diacrtics,' ', processed_rev)
        processed_rev = ("\s+",' ',processed_rev)


           
        # 5 --------- Tokenization
       
        tokens = nltk.word_tokenize((processed_rev[2]))
        

        # 6 ----------  Elimination des elongations
        for j in range(len(tokens)):
            tokens[j]=re.sub(r'(.)\1{2,}', r'\1', tokens[j])  
  
        
        # 7 ---------- Lemmatisation 
        
        lemmatizer = nltk.WordNetLemmatizer()
        rslt = " "
        for w in tokens:
            lemma = lemmatizer.lemmatize(w)
            rslt = rslt+" "+lemma
            
        # 8 --------- Normalisation des lettres **Essayer sans et avec**
        
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
        
        for a,b in pat:
            rslt = re.sub(a,b,rslt)
            
        processed_reviews.append(word_tokenize(rslt))
            
    return processed_reviews


# In[ ]:


df = pd.read_csv("LABR_enrichi.csv",header=0)


X=df.review[df.rate==5].iloc[:12288]
X=X.append(df.review[df.rate==1])
y=df.rate[df.rate==5].iloc[:12288]
y=y.append(df.rate[df.rate==1])

# Clean reviews + tokenization of each rev
X = clean_data(X.values)


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=42)


# In[33]:


nbr_features =500
wndw = 40 

model = Word2Vec(X_train, size=nbr_features, window=wndw, min_count=5,sg=0)

# To make the model memory efficient
model.init_sims(replace=True)


# In[34]:


trainDataVecs = get_avg_feature_vecs(X_train, model, nbr_features)
testDataVecs = get_avg_feature_vecs(X_test, model, nbr_features)


# In[35]:


train_label=y_train
test_label=y_test
sets = [[trainDataVecs, train_label], [testDataVecs,test_label]]

for i in range(len(sets)):
    sets[i][0], sets[i][1]  = remove_nan_instances(sets[i])


# In[36]:


classifiers = [BernoulliNB(alpha=1.0,  fit_prior='True' ,binarize=0.1),GaussianNB(var_smoothing=0.001),RandomForestClassifier(bootstrap= 'False', criterion= 'entropy',max_features='log2', min_samples_split= 10,n_estimators= 150),DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='sqrt', random_state=42, max_depth=None, max_leaf_nodes=500),KNeighborsClassifier(),SVC(C= 2, gamma='scale', kernel= 'rbf')] 


# In[39]:


results = open("Resultat_LABR_enrichi_Unbalanced_Word2Vec.txt",'w')

for cls in classifiers: 
    write_pattern('cls', str(cls), results)
         
    
    classifier = cls
    classifier.fit(sets[0][0], sets[0][1])

    predictions = classifier.predict(sets[1][0])
    evaluate_model(sets[1][1],predictions,results)
                           

results.close()


# In[ ]:




