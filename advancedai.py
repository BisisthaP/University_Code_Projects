#Name - Bisistha Patra
#Student ID - 24159091

#importing all the necessary libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#loading the combined dataset, downloaded from kaggle onto the system 
df = pd.read_csv("/content/my_data")
df

#dataset analysis 
# df.describe()
# df.info()

df.isna().sum()
#there are 1194 nan values - missing values is the column of condition 
#we can remove them from the dataset as the first preprocessing step 

df = df.dropna()
df 
#new size of the dataset - 213869 rows × 7 columns

#now out of all the columns in the dataset, we do not need - date and usefulCount columns for the first task, 
#hence we remove both of those columns from the dataset - reducing dimensionality 

df = df.drop(columns=['date', 'usefulCount'])
df
#new dataset size - 213869 rows × 5 columns

plt.figure(figsize=(15, 7))
#more cool tone laternative to plasma - "magma"
ax = sns.countplot(data=df, x='rating', palette="plasma", hue='rating', legend=False)

for container in ax.containers:
    ax.bar_label(container, color='black', padding=3)

plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')

# Show the figure
plt.show()
plt.savefig('rating_distribution_labeled.png')

grouped_counts = df['rating'].apply(lambda x: '1-5' if x <= 5 else str(int(x)))

#getting the counts and sort them according to your preferred order
order = ['1-5', '6', '7', '8', '9', '10']
table = grouped_counts.value_counts().reindex(order).to_frame(name='Total Count')

#adding percentage column for better comparison
table['Percentage'] = (table['Total Count'] / table['Total Count'].sum() * 100).round(2).astype(str) + '%'

print(table)
#the table shows that if we group ratings 1 to 5 as 0 and 6 to 10 as 1 - there will be a class imbalance,
#that may cause the naive bayes to be biased as word occurances for class 1 will be higher 

#hence, for the dataset to have even representation of both classes and the fact that the size is more than 200k,
#we can consider undersampling from the rating 6 to 10, to match the count of the ratings 1 to 5
#creating another column called rating for categorizing the labels of rating into 0 and 1 (for Naive Bayes)
#1 to 5 = 0 
#6 to 7 = 1 
df['label'] = df['rating'].apply(lambda x: 0 if x <= 5 else 1)

minority = df[df['label'] == 0]
majority = df[df['label'] == 1]
len(majority)
# the size of minority = 63906
#the size of majority = 149963
#downsampling the majority class - 1 randomly, to match the size of minority 
maj_downsized = majority.sample(n=len(minority), random_state=42)

#combining the downsized minority and majority 
df_balanced = pd.concat([minority, maj_downsized])

#shuffling the dataset for better model training 
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced

import re 
import html 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
def preprocessing(text):
  text = html.unescape(text)
  #used to fix the html parts in the reviews like &#039 

  text = text.strip('"') #removing quotes from the reviews 

  #converting to lowercase
  text = text.lower()

  #removing numbers and special characters (keeping only letters) 
  #numbers like 400mg and punctuations 
  #also removes anything after the apostropes 
  text = re.sub(r'[^a-z\s]', '', text)

  #splitting into words - tokenise the reviews 
  tokens = nltk.word_tokenize(text)

  #removing stopwords, lemmatizing  
  stop_words = set(stopwords.words('english'))
  lemm = WordNetLemmatizer()
  tokens = [word for word in tokens if word not in stop_words]
  tokens = [lemm.lemmatize(word) for word in tokens]

  #joining the reviews into one single string 
  return " ".join(tokens)

  #lemmatization was used as for medical terms, like medicine and doctor, 
  #stemming might crop them into medicin and doct 

#applying the preprocessing and checking the cleaned reviews
df_balanced['clean_review'] = df_balanced['review'].apply(preprocessing)
print(df_balanced[['review', 'clean_review']].head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df_balanced["clean_review"] = df_balanced["clean_review"].fillna('')
#handling any empty strings created during preprocessing 

#target and independent features defined below -
X = df_balanced["clean_review"]
y = df_balanced["label"]

#splitting the balanced dataset into training and testing before using the count vectorizer 
#80:20 splits as mentioned in assignment file 
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#using count vectorizer instead of TF-IDF as we need to implement Naive Bayes from sctrach and 
#it makes more sense to apply probability, laplacian smoothing to the actual freq/count of words 
vectorizer = CountVectorizer(max_features=5000)

X_train_counts = vectorizer.fit_transform(X_train_raw)
X_test_counts = vectorizer.transform(X_test_raw)

#converting to array for easier math 
X_train_arr = X_train_counts.toarray()
X_test_arr = X_test_counts.toarray()
