# import streamlit as st
# import pandas as pd

# def load_data():
#     # Load your DataFrame here
#     df = pd.read_csv('filename.csv')
#     return df

# def main():
#     st.title('Maithili News Portal.')
    
#     df = load_data()
#     df = df.iloc[::-1]

#     # Create a slider for pagination
#     start, end = st.slider('Select a range of rows', 0, len(df)-1, (0, 20), 1)
    
    
#     for index, row in df.iloc[start:end+1].iterrows():
#         st.subheader(row['title'])
#         st.write('Published on:', row['published date'])
#         st.write(row['translated'])

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
from datetime import datetime

# Set the page title
st.set_page_config(page_title='Maithili News Portal')


def load_data():
    # Load your DataFrame here
    df = pd.read_csv('filename.csv')
    # Convert 'published date' to datetime format
    df['published date'] = pd.to_datetime(df['published date'], format='%Y-%m-%d %H:%M:%S')
    return df

def main():
    st.title('Maithili News Portal')

    # Load data
    df = load_data()
    df = df.iloc[::-1]

    # Sidebar for filtering by date
    st.sidebar.header('Filter by Date')
    start_date = st.sidebar.date_input('Start Date', min(df['published date']))
    end_date = st.sidebar.date_input('End Date', max(df['published date']))

    # Convert to datetime objects for comparison
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

    # Filter DataFrame based on selected date range
    filtered_df = df[(df['published date'] >= start_date) & (df['published date'] <= end_date)]

    # Create a slider for pagination
    start, end = st.slider('Select a range of rows', 0, len(filtered_df)-1, (0, min(20, len(filtered_df)-1)), 1)

    for index, row in filtered_df.iloc[start:end+1].iterrows():
        # Display title, published date, and translated content
        st.header(row['title'])
        st.write('Published on:', row['published date'])
        st.write('Translated Content:')
        st.markdown(row['translated'])

        # Display full article link
        #st.write('Read Full Article:', row['url'])

        # Display images if available
        if 'images' in row and isinstance(row['images'], set):
            for image in row['images']:
                st.image(image, caption='Image', use_column_width=True)

        # Add a separator between news articles
        st.markdown('---')

if __name__ == "__main__":
    main()
