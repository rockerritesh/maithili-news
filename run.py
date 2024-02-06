from gnews import GNews
from datetime import datetime
import pandas as pd
from googletranslate import translate
import requests
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib


def translate_text(x):
    try:
        translation_result = translate(x, dest="mai", src="en")
        return translation_result
    except requests.exceptions.RequestException as e:
        print(f"Error making translation request: {e}")
        return (
            None  # Handle the error gracefully, you can modify this based on your needs
        )
    except json.decoder.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return (
            None  # Handle the error gracefully, you can modify this based on your needs
        )


google_news = GNews()

nepal_news = google_news.get_news("nepal")


# Assuming nepal_news is a list of dictionaries
for news in nepal_news:
    # Parse date string to datetime object
    news["published date"] = datetime.strptime(
        news["published date"], "%a, %d %b %Y %H:%M:%S %Z"
    )

# Sort the news by date
nepal_news.sort(key=lambda x: x["published date"])
print("--------------Sorted news by date--------------")

# Assuming nepal_news is your list of dictionaries
df = pd.json_normalize(nepal_news)
print("-----------Normalized news------------")

# Get the full article for the last 30 news
max_length_of_news = 20

# Get the full article for the last 20 news and Images
df["full_article"] = (
    df.tail(max_length_of_news)["url"]
    .apply(google_news.get_full_article)
    .apply(lambda x: x.text if x is not None else "")
)


df["images"] = (
    df.tail(max_length_of_news)["url"]
    .apply(google_news.get_full_article)
    .apply(lambda x: x.images if x is not None else "")
)
print("--------Got full article and images----------")

# Remove non-ascii characters
df["full_article"] = df.tail(max_length_of_news)["full_article"].apply(
    lambda x: "".join(i for i in str(x) if ord(i) < 0x10000)
)
print("--------Removed non-ascii characters----------")

# Translate the full article to maithili
# df["translated"] = df.tail(max_length_of_news)["full_article"].apply(
#     lambda x: translate(x, dest="mai", src="en")
# )

# Assuming 'max_length_of_news' is defined
df["translated"] = df.tail(max_length_of_news)["full_article"].apply(translate_text)

print("--------Translated to maithili---------")

# Save the dataframe to csv
df.tail(max_length_of_news).to_csv("today_news.csv", index=False)
print("--------Saved to csv--------")

# making main datafram for store and retaive data
main_df = pd.read_csv("filename.csv", on_bad_lines="skip")

# merging two dataframe to make one main dataframe
merged_df = (
    pd.concat([main_df, df.tail(30)])
    .drop_duplicates(subset=["title"])
    .reset_index(drop=True)
)
print("---------Merged two dataframe--------")

print("---------Applying Machine Learning--------")

# loading joblib file model
tfidi_model = joblib.load("model/tfidf_model.joblib")
label_encoder = joblib.load("model/label_model.joblib")
svm_model = joblib.load("model/svc_77_classifier_model.joblib")
print("---------Loaded Machine Learning Model--------")

# droping null values according to translated column
merged_df = merged_df.dropna(subset=["translated"])
print("---------Dropped null values--------")

# applying tfidf vectorizer to translated column
tfidf = tfidi_model.transform(merged_df["translated"])
print("---------Applied tfidf vectorizer--------")

# predicting the label of translated column
label = svm_model.predict(tfidf)
print("---------Predicted the label--------")

# encoding the label
label = label_encoder.inverse_transform(label)
print("---------Encoded the label--------")

# adding label column to main dataframe
merged_df["label"] = label
print("---------Added label column--------")


# saving the main dataframe to csv
merged_df.to_csv("filename.csv", index=False)
print("Done")

# save merged_df to json
merged_df.to_json("Docs/filename.json", orient="records")
print("json file created successfully")
