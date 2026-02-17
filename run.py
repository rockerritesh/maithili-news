from gnews import GNews
from datetime import datetime
import pandas as pd
from deep_translator import GoogleTranslator
from googlenewsdecoder import new_decoderv1
import newspaper

from classify import predict_using_maibert

translator = GoogleTranslator(source="en", target="mai")

MAX_CHARS = 4900


def translate_text(x):
    try:
        text = str(x).strip()
        if not text:
            return None
        # Split long texts into chunks under the 5000 char limit
        if len(text) <= MAX_CHARS:
            return translator.translate(text)
        chunks = []
        while text:
            if len(text) <= MAX_CHARS:
                chunks.append(text)
                break
            # Split at the last sentence boundary within the limit
            split_at = text.rfind(". ", 0, MAX_CHARS)
            if split_at == -1:
                split_at = MAX_CHARS
            else:
                split_at += 1  # include the period
            chunks.append(text[:split_at])
            text = text[split_at:].strip()
        translated_chunks = [translator.translate(c) for c in chunks]
        return " ".join(translated_chunks)
    except Exception as e:
        print(f"Error translating text: {e}")
        return None


def decode_url(url):
    """Decode Google News redirect URL to actual article URL."""
    try:
        result = new_decoderv1(url)
        if result and result.get("status"):
            return result["decoded_url"]
    except Exception as e:
        print(f"Error decoding URL: {e}")
    return url


def get_article(url):
    """Download and parse a single article from a decoded URL."""
    try:
        real_url = decode_url(url)
        article = newspaper.Article(url=real_url)
        article.download()
        article.parse()
        return article
    except Exception as e:
        print(f"Error fetching article: {e}")
        return None


def main():
    google_news = GNews()

    nepal_news = google_news.get_news("nepal")

    # Parse date strings to datetime objects
    for news in nepal_news:
        news["published date"] = datetime.strptime(
            news["published date"], "%a, %d %b %Y %H:%M:%S %Z"
        )

    # Sort the news by date
    nepal_news.sort(key=lambda x: x["published date"])
    print("--------------Sorted news by date--------------")

    df = pd.json_normalize(nepal_news)
    print("-----------Normalized news------------")

    # Get the full article for the last 20 news
    max_length_of_news = 20

    # Fetch articles once (decode URL + download)
    articles = df.tail(max_length_of_news)["url"].apply(get_article)

    df["full_article"] = articles.apply(lambda x: x.text if x is not None else "")
    df["images"] = articles.apply(lambda x: x.images if x is not None else "")
    print("--------Got full article and images----------")

    # Remove non-ascii characters
    df["full_article"] = df.tail(max_length_of_news)["full_article"].apply(
        lambda x: "".join(i for i in str(x) if ord(i) < 0x10000)
    )
    print("--------Removed non-ascii characters----------")

    # Translate the full article to maithili
    df["translated"] = df.tail(max_length_of_news)["full_article"].apply(translate_text)

    print("--------Translated to maithili---------")

    # Save the dataframe to csv
    df.tail(max_length_of_news).to_csv("today_news.csv", index=False)
    print("--------Saved to csv--------")

    # Drop null translated values
    df = df.dropna(subset=["translated"])
    print("---------Dropped null values--------")

    # Apply deep learning model to predict the label
    print("---------Applying deeplearning model--------")
    df = predict_using_maibert(df)
    print("---------Predicted the label--------")

    # Merge with main dataframe
    main_df = pd.read_csv("filename.csv", on_bad_lines="skip")

    merged_df = (
        pd.concat([main_df, df.tail(30)])
        .drop_duplicates(subset=["title"])
        .reset_index(drop=True)
    )
    print("---------Merged two dataframe--------")

    print("----------Removing Nan Column -----------")
    merged_df = merged_df.dropna(subset=["translated"])

    # Save outputs
    merged_df.to_csv("filename.csv", index=False)
    print("Done")

    merged_df.to_json("docs/filename.json", orient="records")
    print("json file created successfully")

    merged_df.tail(200).to_json("docs/last_100_news.json", orient="records")
    print("json file created successfully")

    merged_df.tail(200).to_csv("last_200_news.csv", index=False)
    print("Done")


if __name__ == "__main__":
    main()
