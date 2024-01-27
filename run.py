from gnews import GNews
from datetime import datetime
import pandas as pd
from googletranslate import translate
import requests
import json


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
    .apply(lambda x: x.text)
)

df["images"] = (
    df.tail(max_length_of_news)["url"]
    .apply(google_news.get_full_article)
    .apply(lambda x: x.images)
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

# saving the main dataframe to csv
merged_df.to_csv("filename.csv", index=False)
print("Done")
