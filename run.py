from gnews import GNews
from datetime import datetime
import pandas as pd
from googletranslate import translate



google_news = GNews()

nepal_news = google_news.get_news('nepal')


# Assuming nepal_news is a list of dictionaries
for news in nepal_news:
    # Parse date string to datetime object
    news['published date'] = datetime.strptime(news['published date'], '%a, %d %b %Y %H:%M:%S %Z')

# Sort the news by date
nepal_news.sort(key=lambda x: x['published date'])
print('sorted news by date')

# Assuming nepal_news is your list of dictionaries
df = pd.json_normalize(nepal_news)
print('normalized news')

# Get the full article for the last 30 news
max_length_of_news = 20

# Get the full article for the last 20 news and Images
df['full_article'] = df.tail(max_length_of_news)['url'].apply(google_news.get_full_article).apply(lambda x: x.text)

df['images'] = df.tail(max_length_of_news)['url'].apply(google_news.get_full_article).apply(lambda x: x.images)
print('got full article and images')

# Remove non-ascii characters
df['full_article'] = df.tail(max_length_of_news)['full_article'].apply(lambda x: ''.join(i for i in str(x) if ord(i) < 0x10000))
print('removed non-ascii characters')

# Translate the full article to maithili
df['translated'] = df.tail(max_length_of_news)['full_article'].apply(lambda x: translate(x, dest='mai', src='en'))
print('translated to maithili')

# Save the dataframe to csv
df.tail(max_length_of_news).to_csv('today_news.csv', index=False)
print('saved to csv')

# making main datafram for store and retaive data
main_df = pd.read_csv('filename.csv')

# merging two dataframe to make one main dataframe
merged_df = pd.concat([main_df, df.tail(30)]).drop_duplicates(subset=['title']).reset_index(drop=True)
print('merged two dataframe')

# saving the main dataframe to csv
merged_df.to_csv('filename.csv', index=False)
print('Done')