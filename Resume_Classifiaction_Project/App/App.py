import numpy as np
import pandas as pd

import re
import string
import nltk 
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')

from nltk.tokenize import word_tokenize
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

import PyPDF2
from docx import Document
import streamlit as st

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


# title
html_temp = """
<div style="background:black; padding:3px; margin:3px">
<h2 style="color:white;text-align:center;">Resume Classification</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


data=pd.read_csv("E:\\ExcelR_project\Resume_Classifiaction_Project\App\Resume_data.csv")
data.drop(["Unnamed: 0"],axis=1,inplace=True)
data["tokens"]=data["text"].apply(word_tokenize)
data["tokens"]=data["tokens"].apply(tuple)
data.drop_duplicates(subset="tokens",inplace=True)

def clean_text(text):
    text = text.lower() # Convert text to lowercase
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)   # remove RT and cc
    text = re.sub('#\S+', '', text)      # remove hashtags
    text = re.sub('@\S+', ' ', text)      # remove mentions
    text = re.sub(r'[^\x00-\x7f]',r' ', text)     # replace consecutive non-ASCII characters with a space
    text = re.sub('\s+', ' ', text) 
    text = "".join([word for word in text if word not in string.punctuation]) # punctuations removal
    text = re.sub("\d+", " ", text) # remove digits
    text = ' '.join([word for word in text.split() if word not in stopwords])     # stopwords removal
    return text.strip()

data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x))



lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    return ' '.join(words)

data['cleaned_text'] = data['cleaned_text'].apply(lemmatize_words)


data["temp_list"]=data["cleaned_text"].apply(lambda x: str(x).split())

        
stpwrd = nltk.corpus.stopwords.words('english')
new_stopwords =  ["use","change","development","change","hcm","skills", "performance","different","per","responsible","maintenance","end","set",
                 "experience","like","team","client", "project","good","consultant","roles","patch","understand","users","level","status",
                 "configuration","activities","check","tool","service","work","develop","design","knowledge","base","upgrade","summary","organization",
                 "test","report","business","application","etc","systems","software","expertise","instance","inbound","components","migration",
                 "process","support","involve","issue","people","technical","various","configure","production","maintain","security","responsibilities",
                 "environment","configure","management","domains","eib","scheduler","monitor","till","instal","multiple","group",
                 "custom","apply","field","studio","document","requirements","new","date","applications","file","professional","relate","pum","interface",
                 "developer","role","perform","core","integrations","load","years","update","provide","build","requirement","engineer","task",
                 "user","troubleshoot","object","information","job","technologies","code","profile","setup","functional","implement","calculate",
                 "technology","implementation","include","refresh","xml","xslt","write","detail","solutions","compensation","description","plan","transformation",
                 "architecture","advance","peopletools","fix","run","connectors","administration","package","program","connector","type","payroll","communication",
                 "strong","day","schedule","broker","hyderabad","handle","request","hr","source","education","duration","name","tune","tuxedo","writer","best",
                  "g","c","server","create","function","fscm","manage","ps","page","resolve","ltd","complex","custom","benefit","modules","r",
                 "designer","assistant","basis","university","installation","company","analysis","hrms","outbound","send","engine","generate","ability","well",
                 "track","model","get","hand","take","need","time","bundle","pvt","prepare","customer","creation","daily","backups","excellent","hire","worker",
                 "uat","migrations","compare","servers","control","data","help","internet","ess","languages","weblogic","book","log","global","databases","analyze",
                 "meet","present","layout","add","pia","enterprise","su","true","define","enhancements","simple","personal","ms","phase","matrix","career","college",
                 "employee","ssis","component","database"]
stpwrd.extend(new_stopwords)
def remove_extra_stopwords(text):
  text = ' '.join([word for word in text.split() if word not in stpwrd])     #stopwords removal
  return text.strip()

data['cleaned_text'] = data['cleaned_text'].apply(lambda x: remove_extra_stopwords(x))

vectorize_n_gram_feature = TfidfVectorizer(norm="l2",analyzer="word",ngram_range=(1,3),max_features=25)
tfidf_matrix_feature=vectorize_n_gram_feature.fit_transform(data["cleaned_text"].values)
le=LabelEncoder()
label_job_role=le.fit_transform(data["job_role"])


x=tfidf_matrix_feature
y=label_job_role

nb_model=OneVsRestClassifier(MultinomialNB())
nb_model.fit(x, y)

def extract_data(feed):
    pdf_reader = PyPDF2.PdfReader(feed)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_doc_data(feed):
    document = Document(feed)
    text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
    return text

uploaded_file = st.file_uploader('Choose your .pdf or .doc file',type=["pdf", "docx","doc"])
              
if uploaded_file is not None:
    if uploaded_file.name.endswith('.pdf'):
        df=extract_data(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        df=extract_doc_data(uploaded_file)
        

def classify_resume(df):
    df = clean_text(df)
    df=lemmatize_words(df)
    df=remove_extra_stopwords(df)
    resume_vectorized = vectorize_n_gram_feature.fit_transform([df])
    prediction=nb_model.predict(resume_vectorized)
    if prediction[0] == 0:
        return "Workday Consultant"
    elif prediction[0] == 1:
        return "PeopleSoft Database Admin"
    elif prediction[0] == 2:
        return "SQL Deveoper"
    elif prediction[0] == 3:
        return "React Developer"
    
def common_word(df):
    df = clean_text(df)
    df_split=df.split()
    top= Counter(df_split)
    temp=pd.DataFrame(top.most_common(15))
    temp.columns = ['Common_words','count']
    fig = px.bar(temp, x="count", y="Common_words", 
                 title='Commmon Words Present In The Resume', orientation='h', 
         width=700, height=700,color='Common_words')
    st.plotly_chart(fig)
    return fig

def plot_word_cloud(df):
# Combine all the text data into a single string
    # text = ' '.join(df)
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=800,max_words=50, background_color='black', 
                          min_font_size=10, max_font_size=200).generate(df)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.title("Wordcloud")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()
    return wordcloud


def unique_word(df):
    df = clean_text(df)
    df_split=df.split()
    top=Counter(df_split)
    unique_count=len(top)
    duplicate_count=len(df_split)-unique_count
    labels=["UniqueWords", "Duplicated Words"]
    values=[unique_count, duplicate_count]
    fig=px.pie(names=labels, values=values, title="Unique vs Duplicate Word Count",
               hole=0.7,color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig)
    return fig


st.set_option('deprecation.showPyplotGlobalUse', False)



if st.button("Classify"):
    if df:
        prediction= classify_resume(df)
        st.write("Predicted Job Designation: ",prediction)
        fig=common_word(df)
        #st.write(fig)
        unique_word_fig=unique_word(df)
        wordcloud_plot=plot_word_cloud(df)
        #st.write(wordcloud_plot)

    else: 
        st.write("Please upload resume")