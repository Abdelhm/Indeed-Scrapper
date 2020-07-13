from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import re
import os
import time
import pandas as pd


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(url):
    body = requests.get(url)
    if(re.match(b'%PDF-1.5',body.content)):
        res = "PDF FILE"
    else:
        soup = BeautifulSoup(body.text, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)  
        res =  u" ".join(t.strip() for t in visible_texts)
    return res

url = "https://www.indeed.fr/rc/clk?jk=e50eb0fb886b7ad0&fccid=4fa384307f4f5a27"
print(text_from_html(url))




#This function get elements from job offer, it takes as argument the result page url from indeed
def get_elements(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    result = soup.find_all("div", {'data-tn-component' : 'organicJob'})
    base_url = "https://www.indeed.fr"
    urls = []
    titles = []
    company_names = []
    locations = []
    posted = []
    salary = []
    job_offer_content = []
    
    for offer in result:
        urls.append(base_url+offer.find('a')['href'])
        titles.append(offer.find('a',{'data-tn-element':'jobTitle'}).text.replace('\n',''))
        
        try:
            company_names.append(offer.find('span',{'itemprop':'name'}).text.replace('\n',''))
        except:
            company_names.append('Not_found')
        
        try:
            salary.append(offer.find('span',{'class':'no-wrap'}).text.strip().replace(u'\xa0', u''))
        except:
            salary.append('Not_found')
            
        locations.append(offer.find('span',{'itemprop':'addressLocality'}).text.replace('\n',''))
        posted.append(offer.find('span',{'class':'date'}).text.strip())
        
        try:
            job_offer_content.append(text_from_html(base_url+offer.find('a')['href']))
        except:
            job_offer_content.append('Not_found')
            
    return urls,titles, company_names,locations,posted,salary,job_offer_content


#Contruction of the url
base_url = 'http://www.indeed.fr/jobs?q='
search = 'data' #keyword for the search (if multiple words separate them by a '+')
sort_by = '&sort=date'          # sort by date
start_from = '&start='    # start page number (indeed allow us to navigate thought only 100 pages)
  
all_urls = []
all_titles = []
all_company_names = []
all_locations = []
all_posted = []
all_salary = []
all_job_offer_content = []
df = pd.DataFrame()

for page in range(1,11):
    print("page: "+str(page)+"/100")
    page = (page-1) * 10
    url = base_url+search+sort_by+start_from+str(page)
    urls, title,company,location,posted,salary,job_offer_content = get_elements(url)
    all_urls = all_urls+urls
    all_titles = all_titles+title
    all_company_names = all_company_names+company
    all_locations = all_locations+location
    all_posted = all_posted+posted
    all_salary = all_salary+salary
    all_job_offer_content = all_job_offer_content + job_offer_content
    time.sleep(0.5)

df['company'] = pd.Series(all_company_names)
df['locations'] = pd.Series(all_locations)
df['titles'] = pd.Series(all_titles)
df['salary'] = pd.Series(all_salary)
df['posted'] = pd.Series(all_posted)
df['url'] = pd.Series(all_urls)
df['content'] = pd.Series(all_job_offer_content)
    
print('done')


before = df.shape[0]
df.drop_duplicates(subset=['content'], keep=False)
df = df[df.content != "PDF FILE"]



print("Loss of "+str((before-df.shape[0])*100/before)+" %")



content = df['content']

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(font_scale=1)
counts = content.apply(len)
counts.plot(bins=20,kind = 'hist')



import string
content = content.str.strip().str.lower()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words_en = set(stopwords.words("english"))  # load stopwords
stop_words_fr = set(stopwords.words("french"))  # load stopwords
stop_words_fr.update(('ok','ou','et','etc'))
i = 0

content_clean = list()


for doc in content:
    word_tokens = word_tokenize(doc)
    filtered_sentence = [w for w in word_tokens if not w in stop_words_fr]
    filtered_sentence = ' '.join(filtered_sentence)
    word_tokens = word_tokenize(filtered_sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words_en]
    filtered_sentence = ' '.join(filtered_sentence)
    content_clean.append(filtered_sentence)
    i = i+1

content = pd.Series(content_clean)
content = content.str.replace('\d+', ' ')
content = content.str.replace(r'[›–«»!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~’©®•×\'€]',' ')
content = content.str.replace(r'(\s+|^)[a-z]{1,2}(\'{1,2}|\s+)', ' ')
content = content.str.replace('\s+', ' ')



from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 0.05, ngram_range=(1, 1),use_idf=True)
tfidf = vectorizer.fit_transform(content)
pairwise_similarity = (tfidf * tfidf.T).A
doc_sim = pairwise_similarity
sns.set(font_scale=0.8)
ax = sns.clustermap(doc_sim,linewidths=.3,figsize=(30, 30))
plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.savefig('C:\\Users\\Epok\\Desktop\\job_offer.png', dpi = 300)
plt.show()



duplicate_index = list()
for i in range(0,doc_sim.shape[0]):
    for j in range(0,i):
        if((doc_sim[i,j] >=0.985) & (i !=j)):
            duplicate_index.append(i)

duplicate_index_unique = []
for i in duplicate_index:
    if i not in duplicate_index_unique:
        duplicate_index_unique.append(i)

content.drop(duplicate_index, inplace=True)
content  = content.reset_index(drop=True)
df.drop(df.index[duplicate_index], inplace=True)
df = df.reset_index(drop=True)




vectorizer = TfidfVectorizer(min_df = 0.05, ngram_range=(1, 1),use_idf=True)
tfidf = vectorizer.fit_transform(content)
terms = vectorizer.get_feature_names() 
pairwise_similarity = (tfidf * tfidf.T).A
doc_sim = pairwise_similarity
sns.set(font_scale=0.8)
ax = sns.clustermap(doc_sim,linewidths=.3,figsize=(30, 30))
plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.savefig('C:\\Users\\Epok\\Desktop\\job_offer2.png', dpi = 300)
plt.show()




number1 = 8
number2 = 12
print(df.iloc[[number1,number2]]['url'].values)
print("=====================================================================================================================")
print(content[number1])
print("=====================================================================================================================")
print(content[number2])
