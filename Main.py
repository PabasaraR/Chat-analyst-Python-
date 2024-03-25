import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer



text=open("text1.txt",encoding='utf-8').read()
lText=text.lower()
clearText=lText.translate(str.maketrans('','',string.punctuation))

tokenizedWords=word_tokenize(clearText,'english')




finalWords=[]
stopwords = nltk.corpus.stopwords.words('english')
#print(stopwords)
for word in tokenizedWords:
    if word not in stopwords:
        finalWords.append(word)


#print(finalWords)
emotionList=[]
with open('emotion.txt','r') as file:
    for line in file:
        clearLine=line.replace("\n",'').replace(",",'').replace("'",'').strip()
        word,emotion=clearLine.split(':')

        if word in finalWords:
            emotionList.append(emotion)

#print(emotionList)
w=Counter(emotionList)
#print(w)

def sentimentAnalyse(text):
    score=SentimentIntensityAnalyzer().polarity_scores(text)
    neg=score['neg']
    pos=score['pos']

    if neg>pos:
        print("Negative chat")
    elif pos>neg:
        print("Positive chat")
    else:
        print("neutral chat")

sentimentAnalyse(clearText)


fig,ax1=plt.subplots()
ax1.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.savefig('emotion.png')
plt.show()