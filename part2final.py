# Name - Bisistha Patra
# Student ID - 24159091
# Code for Pathway 2: Aspect-Based Sentiment Analysis (ABSA)

#all imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, html, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

#1.NAIVE BAYES FROM SCRATCH
class NaiveBayesScratch:
  #this is the same naive bayes as the one implemented from scratch for part 1 - 
    def __init__(self, alpha=1.0):
        self.alpha = alpha # Smoothing parameter

    def fit(self, X, y):
        # matching sizes and finding unique classes - setting the priors or likelihoods 
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.priors = np.zeros(len(self.classes))
        self.likelihoods = np.zeros((len(self.classes), n_features))
        
        for idx, c in enumerate(self.classes):
            # selecting only those rows from the current class
            X_c = X[y == c]

            # 1. Prior Calculation: P(Class)
            self.priors[idx] = X_c.shape[0] / n_samples
            
            #2.laplacian Smoothing
            # calculating likelihoods with smoothing to avoid zero probabilities 
            # formula: (word count + alpha) / (total words in class + alpha * vocab size)
            word_counts = np.sum(X_c, axis=0)
            total_words = np.sum(X_c)
            self.likelihoods[idx, :] = (word_counts + self.alpha) / (total_words + self.alpha * n_features)

    def predict(self, X):
        # using log to prevent numbers from getting too small (underflow)
        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)
        
        # calculating the weight of words for each class manually
        # this helps show the step-by-step logic instead of just using a shortcut
        scores = []
        for idx in range(len(self.classes)):
            # adding the word weights to the starting probability of the class
            class_score = X.dot(log_likelihoods[idx, :]) + log_priors[idx]
            scores.append(class_score)
            
        # picking the class with the highest total score
        scores = np.array(scores).T
        return self.classes[np.argmax(scores, axis=1)]

# 2. MEDICAL TEXT CLEANING
def preprocessing(text):
  #this is also from part 1 code implementation 
    if not isinstance(text, str): return ""
    # fixing html bits like &#039 and moving to lowercase
    text = html.unescape(text).lower()
    # keeping only letters - removing numbers like 50mg or 10ml
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    
    # using lemmatization 
    lemmer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    return " ".join([lemmer.lemmatize(w) for w in tokens if w not in stop])

#MAIN PIPELINE
def main():
    # loading the UCI drug dataset - the notebook was run and editted on google colab, so please change the location when running it 
    try:
        #combining training and testing datasets for uniform EDA and preprocessing 
        train = pd.read_csv("/content/drugLibTrain_raw.tsv", sep="\t")
        test = pd.read_csv("/content/drugLibTest_raw.tsv", sep="\t")
        df = pd.concat([train, test], ignore_index=True)
    except FileNotFoundError:
        print("Error: Files not found!")
        return

    #cleaning up colum names and removing rows with missing reviews
    df.rename(columns={'Unnamed: 0': 'reviewID'}, inplace=True)

    #dropping nan/missing values based rows in the columns - "benefitsReview" and "sideEffectsReview", as these are the primary columns for the pathway task
#ignored the 11 missing values in "commentsReview" column as it is not used -
#and we cannot drop another 11 rows which do contain the benefits and sideEffcets reviewS
    df = df.dropna(subset=['benefitsReview', 'sideEffectsReview']).copy()

    # labeling for effectiveness: everything with 'Effective' in it is 1, rest is 0
    df['eff_label'] = df['effectiveness'].apply(lambda x: 1 if 'Effective' in str(x) else 0)

    #labeling for side effects: 'No' or 'Mild' counts as 1 (good), 
    #severe counts as 0 (bad)
    df['se_label'] = df['sideEffects'].apply(lambda x: 1 if x in ['No Side Effects', 'Mild Side Effects'] else 0)

    #plots for better distribution understanding 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # plotting how effectiveness is distributed
    sns.countplot(data=df, x='effectiveness', order=['Ineffective', 'Marginally Effective', 'Moderately Effective', 'Considerably Effective', 'Highly Effective'], ax=axes[0], palette='Blues', hue='effectiveness')
    axes[0].set_title('Effectiveness Rating Distribution')
    axes[0].tick_params(axis='x', rotation=45)

    # plotting how side effect severity is distributed
    sns.countplot(data=df, x='sideEffects', order=['No Side Effects', 'Mild Side Effects', 'Moderate Side Effects', 'Severe Side Effects', 'Extremely Severe Side Effects'], ax=axes[1], palette='Reds', hue='sideEffects')
    axes[1].set_title('Side Effects Severity Distribution')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # heatmap to show the crossover between high effectiveness and bad side effects
    eff_order = ['Ineffective', 'Marginally Effective', 'Moderately Effective', 'Considerably Effective', 'Highly Effective']
    se_order = ['No Side Effects', 'Mild Side Effects', 'Moderate Side Effects', 'Severe Side Effects', 'Extremely Severe Side Effects']
    crosstab = pd.crosstab(df['effectiveness'], df['sideEffects']).loc[eff_order, se_order]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Relationship Heatmap: Effectiveness vs Side Effects')
    plt.show()

    # applying cleaning to both aspects (Benefit vs Side Effect)
    print("Preprocessing text")
    df['clean_ben'] = df['benefitsReview'].apply(preprocessing)
    df['clean_se'] = df['sideEffectsReview'].apply(preprocessing)

    # training model for Effectiveness aspect (Pathway 2)
    X_e_tr, X_e_ts, y_e_tr, y_e_ts = train_test_split(df['clean_ben'], df['eff_label'], test_size=0.2, random_state=42)
    vec_e = CountVectorizer(max_features=5000)
    nb_e = NaiveBayesScratch()
    nb_e.fit(vec_e.fit_transform(X_e_tr).toarray(), y_e_tr.values)
    y_e_pr = nb_e.predict(vec_e.transform(X_e_ts).toarray())

    # training model for Side Effects aspect
    X_s_tr, X_s_ts, y_s_tr, y_s_ts = train_test_split(df['clean_se'], df['se_label'], test_size=0.2, random_state=42)
    vec_s = CountVectorizer(max_features=5000)
    nb_s = NaiveBayesScratch()
    nb_s.fit(vec_s.fit_transform(X_s_tr).toarray(), y_s_tr.values)
    y_s_pr = nb_s.predict(vec_s.transform(X_s_ts).toarray())

    def print_metrics(y_true, y_pred, name):
        print(f"\n--- Model Performance: {name} ---")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

    print_metrics(y_e_ts, y_e_pr, "Effectiveness Aspect")
    print_metrics(y_s_ts, y_s_pr, "Side Effects Aspect")

    #combining predictions into a results dataframe for quadrant analysis
    results = pd.DataFrame({
        'clean_ben': X_e_ts, 'clean_se': X_s_ts,
        'pe': y_e_pr, 'ps': y_s_pr
    })

    # risky = medicine works but side effects are bad (0)
    # success = medicine works and side effects are okay (1)
    risky = results[(results['pe']==1) & (results['ps']==0)]
    success = results[(results['pe']==1) & (results['ps']==1)]

    print("CONSTRASTIVE OUTCOMES FOUND")
    print("Total 'Risky' Patient Cases Found:", len(risky))
    print("Total 'Success' Patient Stories Found:", len(success))

    # filter for WordCloud to remove generic terms and keep the medical ones
    noise = {'day', 'also', 'get', 'taking', 'drug', 'medication', 'time', 'side', 'pain', 'effect', 'would', 'take', 'took', 'hour', 'could', 'pill', 'tablet', 'month', 'week'}

    # visualising the contrastive quadrants
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    def plot_wc(data, title, cmap, ax):
        words = ' '.join(data).split()
        freq = dict(Counter([w for w in words if w not in noise and len(w)>3]).most_common(20))
        ax.imshow(WordCloud(background_color='white', colormap=cmap).generate_from_frequencies(freq))
        ax.set_title(title, fontsize=14); ax.axis('off')

    plot_wc(risky['clean_ben'], "RISKY: Benefits (Positive)", "Greens", axes[0,0])
    plot_wc(risky['clean_se'], "RISKY: Side Effects (Negative)", "Reds", axes[0,1])
    plot_wc(success['clean_ben'], "SUCCESS: Benefits (Positive)", "YlGn", axes[1,0])
    plot_wc(success['clean_se'], "SUCCESS: Side Effects (Neutral)", "Blues", axes[1,1])

    plt.suptitle("Contrastive ABSA Analysis: Risky vs Success Cases", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#ADVANCED TASK: MULTI-CLASS SENTIMENT ANALYSIS (PATHWAY 2)
#I could not understand if we have to do both the subtasks mentioned in pathway 2 or not - 
#So i did do both to stay safe 
def label_multiclass(rating):
    "Standardizing the 10-point scale into 3 distinct categories"
    "1-4: Negative (Low satisfaction/Ineffective)"
    "5-7: Neutral (Moderate results or mixed feelings)"
    "8-10: Positive (High satisfaction/Highly effective)"

    if rating <= 4:
        return "Negative"
    elif rating <= 7:
        return "Neutral"
    else:
        return "Positive"

def main_multiclass():
    print("")
    print("STARTING ADVANCED TASK: MULTI-CLASS ANALYSIS")
    print("")

    try:
        # Loading the dataset (drugLib format) 
        train = pd.read_csv("/content/drugLibTrain_raw.tsv", sep="\t")
        test = pd.read_csv("/content/drugLibTest_raw.tsv", sep="\t")
        df = pd.concat([train, test], ignore_index=True)
    except FileNotFoundError:
        print("Error: Files not found!")
        return

    # Applying labeling and handling missing values
    df['multi_label'] = df['rating'].apply(label_multiclass)
    df = df.dropna(subset=['commentsReview']).copy()

    # BALANCING CLASSES:
    # This prevents the grouping column from being dropped during the apply step.
    min_size = df['multi_label'].value_counts().min()
    df_balanced = df.groupby('multi_label').apply(
        lambda x: x.sample(min_size, random_state=42), 
        include_groups=False
    ).reset_index()
    
    #renaming the level_0 column back to multi_label if reset_index moved it
    if 'level_0' in df_balanced.columns:
        df_balanced.rename(columns={'level_0': 'multi_label'}, inplace=True)

    print(f"Balanced each class to {min_size} samples for fair training.")

    # PREPROCESSING
    #sing 'commentsReview' as the source text for multi-class 
    print("Preprocessing text for 3-class analysis...")
    df_balanced['clean_review'] = df_balanced['commentsReview'].apply(preprocessing)

    #everything from here is same on -
    #feature and target Selection
    X = df_balanced['clean_review']
    y = df_balanced['multi_label']

    # 80/20 Train-Test Split 
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #feature Representation 
    vec = CountVectorizer(max_features=5000)
    X_train = vec.fit_transform(X_train_raw).toarray()
    X_test = vec.transform(X_test_raw).toarray()

    #model training: Using the NaiveBayesScratch class from Part 1 
    nb_multi = NaiveBayesScratch(alpha=1.0)
    print("Training Multi-Class Model...")
    nb_multi.fit(X_train, y_train.values)

    # EVALUATION 
    y_pred = nb_multi.predict(X_test)
    print("Multi-Class Performance Metrics")
    print(classification_report(y_test, y_pred))

    #confusion Matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=['Negative', 'Neutral', 'Positive'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix: 3-Class Sentiment')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

#to run 
main() #for absa 
main_multiclass() #for multiclass 
