'''
Code to run the data labeling process

Requirement: to exactly replicate the result, you must use the following modules:
- bertopic (0.16.1)
- umap-learn (0.5.5) 
- scikit-learn (1.0.2)
- sentence-transformers (2.2.2)
- torch (2.3.1)
- numpy (1.2.6)
- transformers (4.42.0)

The final output, text_processed_final.pkl, is saved in data repository.
'''

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from textblob import TextBlob
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering

# change your pkl file path
pkl_file_path = "data/text_create_processed.pkl"

text_df = pd.read_pickle(pkl_file_path)
text_df = text_df.sort_values('index')
text_df = text_df.reset_index(drop=True)

##################################################
# Data Cleaning
##################################################
def data_cleaning(df: pd.DataFrame):
    # drop the unnecessary part due to Llama3
    df['X'] = df['X'].str.replace('assistant\n\n', '', regex=False)
    df['T'] = df['X'].str.contains('military|veteran|army').astype(int)
    return df
text_df = data_cleaning(text_df)
print("Percentage of treated samples:", text_df['T'].sum() / len(text_df))
# Percentage of treated samples: 0.14

## Topic-model based confounding (high complexity but separability)
## With HBDSCAN clustering (the clustering with removing outliers)
umap = UMAP(n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            low_memory=False,
            random_state=1337) #fix the random state for reproducibility 
representation_model = KeyBERTInspired()
topic_model = BERTopic(umap_model=umap, representation_model=representation_model, min_topic_size=20)
topic, prob = topic_model.fit_transform(text_df['X'])
topic_model.reduce_topics(text_df['X'], nr_topics = 40)
text_df['C1_2'] = [int(value == 0) for value in topic] #assign topic 0 as confounding
print("Correlation between C1_2 and T:", text_df['C1_2'].corr(text_df['T']))
#Correlation between C1_2 and T: -0.08113996208406585
text_df['TC1_2'] = text_df["T"] * text_df["C1_2"] #coding of interaction term
text_df['C1_2'].sum() - text_df['TC1_2'].sum() #check if interaction term is not aliased
# 666

## Case 2: Topic-model based confounding (high complexity and no separability)
cluster_model = AgglomerativeClustering(n_clusters=50)
topic_model2 = BERTopic(umap_model=umap, hdbscan_model=cluster_model, representation_model=representation_model, min_topic_size=20)
topic2, prob2 = topic_model2.fit_transform(text_df['X'])
topic_model2.reduce_topics(text_df['X'], nr_topics = 40)

text_df['T_4'] = [int(value == 0) for value in topic2] #assign topic 0 as treatment
text_df['C1_4'] = [int(value == 4) for value in topic2] #assign topic 4 as confounding
print("Correlation between C1_4 and T_4:", text_df['C1_4'].corr(text_df['T_4']))
# Correlation between C1_4 and T: -0.050710550186618825
text_df['T_4C1_4'] = text_df["T_4"] * text_df["C1_4"] #coding of interaction term
text_df['C1_4'].sum() - text_df['T_4C1_4'].sum() #check if interaction term is not aliased

##################################################
# Counfounding Labeling (C2): Sentiment-based confounding
##################################################
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

#check the correlation between the sentiment and the treatment
text_df['C2'] = text_df['X'].apply(get_sentiment)
print("Correlation between C2 and T:", text_df['C2'].corr(text_df['T'])) 
#Correlation between C2 and T: -0.04990428177759203

text_df.head()

##################################################
# Save the processed data
text_df.to_pickle("data/text_processed_final.pkl")