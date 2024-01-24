import streamlit as st
import pandas as pd

def load_data():
    # Load your DataFrame here
    df = pd.read_csv('filename.csv')
    return df

def main():
    st.title('Maithili News Portal.')
    
    df = load_data()
    df = df.iloc[::-1]
    
    for index, row in df.iterrows():
        st.subheader(row['title'])
        st.write('Published on:', row['published date'])
        st.write(row['translated'])

if __name__ == "__main__":
    main()