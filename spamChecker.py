# Introducing use of Naive Bayes method 

import numpy as np
import pandas as pd

emails = pd.read_csv('emails.csv') 

def processEmail(text): 
    text = text.lower()
    return list(set(text.split()))

emails['words'] = emails['text'].apply(processEmail)

model = {}

for index, email in emails.iterrows():
    for word in email['words']:
        if word not in model:
            model[word] = {'spam': 1, 'ham': 1}
        if word in model:
            if email['spam']:
                model[word]['spam'] += 1
            else:
                model[word]['ham'] += 1

def predictNaiveBayes(email): 
    total = len(emails)
    numSpam = sum(emails['spam'])
    numHam = total - numSpam
    email = email.lower()
    words = set(email.split())
    spams = [1.0]
    hams = [1.0]
    for word in words: 
        if word in model: 
            spams.append(model[word]['spam']/numSpam*total)
            hams.append(model[word]['ham']/numHam*total)

    prodSpams = np.prod(spams) * numSpam
    prodHams = np.prod(hams) * numHam
    return prodSpams/(prodSpams+prodHams)

print(predictNaiveBayes("you had an accident recently, enter your bank details"))
