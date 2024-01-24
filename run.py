from gnews import GNews
from datetime import datetime
import pandas as pd


google_news = GNews()

nepal_news = google_news.get_news('nepal')


# Assuming nepal_news is a list of dictionaries
for news in nepal_news:
    # Parse date string to datetime object
    news['published date'] = datetime.strptime(news['published date'], '%a, %d %b %Y %H:%M:%S %Z')

# Sort the news by date
nepal_news.sort(key=lambda x: x['published date'])

# Assuming nepal_news is your list of dictionaries
df = pd.json_normalize(nepal_news)

# Get the full article for the last 20 news and Images
df['full_article'] = df.tail(20)['url'].apply(google_news.get_full_article).apply(lambda x: x.text)

df['images'] = df.tail(20)['url'].apply(google_news.get_full_article).apply(lambda x: x.images)

# Save the dataframe to csv
df.tail(20).to_csv('news_files.csv', index=False)