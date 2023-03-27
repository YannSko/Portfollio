import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Load data
org_df_books = pd.read_csv("./Books.csv")
org_df_ratings = pd.read_csv("./Ratings.csv")
org_df_users = pd.read_csv("./Users.csv")
df_books = org_df_books.copy()
df_ratings = org_df_ratings.copy()
df_users = org_df_users.copy()


# Preprocessing
@st.cache_data(persist=True)
def Preprocessing(df_books, df_ratings, df_users):
    # Convert http to https
    df_books['Image-URL-S'] = df_books['Image-URL-S'].str.replace(r'^http\b', 'https', regex=True)
    df_books['Image-URL-M'] = df_books['Image-URL-M'].str.replace(r'^http\b', 'https', regex=True)
    df_books['Image-URL-L'] = df_books['Image-URL-L'].str.replace(r'^http\b', 'https', regex=True)

    # Merge all dataframes
    df_ratings_books = df_ratings.merge(df_books, on='ISBN')
    df_complete = df_ratings_books.merge(df_users, on='User-ID')
    return df_complete

df_complete = Preprocessing(df_books, df_ratings, df_users)

# Popularity recommendation
@st.cache_data(persist=True)
def Popularity_Recommandation(df_complete, df_books):
    # number of ratings in each book
    num_rating_df = df_complete.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating':'Num_ratings'}, inplace=True)
    # average ratings in each book
    avg_rating_df = df_complete.groupby('Book-Title').mean()['Book-Rating'].reset_index()
    avg_rating_df.rename(columns={'Book-Rating':'Avg_ratings'}, inplace=True)
    # joining num_rating_df and avg_rating_df
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['Num_ratings']>=250].sort_values('Avg_ratings', ascending=False) 
    # joining popular_df and df_books
    popular_df = popular_df.merge(df_books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','Num_ratings','Avg_ratings']]
    return popular_df

popular_df = Popularity_Recommandation(df_complete, df_books)

@st.cache_data(persist=True)
def get_collaborative_recommendations(df_complete, df_books, book_name):
    # Users who are giving more than 200 Book-Rating
    temp_x = df_complete.groupby('User-ID').count()['Book-Rating'] > 200
    filtered_user_id = temp_x[temp_x].index

    # All data of users who are giving more than 200 Book-Rating
    filtered_ratings = df_complete[df_complete['User-ID'].isin(filtered_user_id)]

    # All data of books who have more than 50 Book-Rating
    temp_y = filtered_ratings.groupby('Book-Title').count()['Book-Rating']>=50
    famous_books = temp_y[temp_y].index

    # All data of famous books
    final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]

    # All the users and their rating on each famous books
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)

    # Computing similarity scores
    similarity_scores = cosine_similarity(pt.T)

    # index fetch
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        temp_df = df_books[df_books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title']))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author']))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M']))
        data.extend(item)

    return data
def app():
    # Preprocess data
    df_complete = Preprocessing(df_books, df_ratings, df_users)
    

    # Streamlit app
    st.title('Book Recommendation System')

    # Display popular books
    st.header('Popular Books')
    min_ratings = st.slider('Minimum number of ratings', min_value=0, max_value=1000, value=250)
    popular_df_filtered = popular_df[popular_df['Num_ratings'] >= min_ratings]
    st.write(f'Total books: {len(popular_df_filtered)}')
    st.dataframe(popular_df_filtered[['Book-Title', 'Book-Author', 'Num_ratings', 'Avg_ratings']])

    # Sidebar
    st.sidebar.title('Choose a book')
    book_choice = st.sidebar.selectbox('Select a book:', df_books['Book-Title'])

    # Display recommended books
    if st.button('Get Recommendations'):
        recommended_books = get_collaborative_recommendations(df_complete, df_books, book_choice)
        st.write('Recommended books:')
        for book in recommended_books:
            st.write(book)
        
    # Recommendation
    

app()