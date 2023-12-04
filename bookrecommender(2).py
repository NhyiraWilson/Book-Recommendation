# -*- coding: utf-8 -*-
"""BOOKRECOMMENDER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GKY19bPRs-4w6hl8DVuNkfrCEOTGGz0u
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount ('/content/drive')

books = pd.read_csv('/content/drive/My Drive/INTRO TO AI/books.csv')
ratings = pd.read_csv('/content/drive/My Drive/INTRO TO AI/Ratings.csv')
users = pd.read_csv('/content/drive/My Drive/INTRO TO AI/users.csv')

books.head()

ratings.head()

users.head()

print('Number of book data:', len(books.ISBN.unique()))
print('Total book rating data from readers:', len(ratings.ISBN.unique()))
print('Amount of user data:', len(users['User-ID'].unique()))

books.info()

#converting the year of publication datatype from object to int
books.drop(books[books['Year-Of-Publication']=='DK Publishing Inc'].index, inplace=True)

books.drop(books[books['Year-Of-Publication']=='Gallimard'].index, inplace=True)

books['Year-Of-Publication']= books['Year-Of-Publication'].astype('int')

#refining the publication data to get modern books only
books = books[books['Year-Of-Publication']>=1900]

books.dtypes

# Removing Image-URL column of all sizes
books.drop(labels=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)

books.head()

print("Number of Book ISBN numbers:", len(books['ISBN'].unique()))
print("Number of book titles:", len(books['Book-Title'].unique()))
print('Number of book authors:', len(books['Book-Author'].unique()))
print('Number of Publication Years:', len(books['Year-Of-Publication'].unique()))
print('Number of publisher names:', len(books['Publisher'].unique()))

# Grouping Book-Author' and count the number of books written by each author
author_counts = books.groupby('Book-Author')['Book-Title'].count()


sorted_authors = author_counts.sort_values(ascending=False)


top_10_authors = sorted_authors.head(10)

# The plot of the top 10 authors according to the books written
plt.figure(figsize=(12, 6))
top_10_authors.plot(kind='bar')
plt.xlabel('Author Name')
plt.ylabel('Number of Books')
plt.title('Top 10 Authors by Number of Books')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ratings.head()

ratings.info()

print('Number of User-IDs:', len(ratings['User-ID'].unique()))
print('Number of books based on ISBN:', len(ratings['ISBN'].unique()))

print('Number of book ratings:')
sorted_ratings = ratings['Book-Rating'].value_counts().sort_index()
pd.DataFrame({'Book-Rating': sorted_ratings.index, 'Sum': sorted_ratings.values})

ratings['Book-Rating'].value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

df_rating = ratings[:20000]
df_rating

users.head()

users.info()

#dropping the age column
users = users.drop(columns=["Age"])

books_df = pd.merge(ratings, books, on='ISBN', how='left')

books= pd.merge(books_df, users, on= 'User-ID')

"""# **EDA**"""

#top 10 books based on ratings
books.sort_values(by='Book-Rating' , ascending=False).head(10)[['Book-Title', 'Book-Author']]

#top 15 authors based on overall maximum total rating
books.groupby(['Book-Author']).sum().sort_values(by='Book-Rating' , ascending=False).head(15).index

#top 15 authors based on overall minimum total rating
books.groupby(['Book-Author']).sum().sort_values(by='Book-Rating' , ascending=True).head(15).index

#publication year with the most books
books['Year-Of-Publication'].value_counts().head(1)

#author with the most books
books['Book-Author'].value_counts().head(1)

#top 10 publishers with the most books
plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=books,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')

#lineplot showing year of publication against ratings on books
plt.figure(figsize=(20,10))
sns.lineplot(x='Year-Of-Publication',y='Book-Rating', data=books)
plt.show()

"""# **DATA PREPROCESSING**"""

books.groupby('ISBN').sum()

all_books_clean = books.dropna()

all_books_clean.isnull().sum()

# Sorting books by ISBN
fix_books = all_books_clean.sort_values('ISBN', ascending=True)

preparation = fix_books.drop_duplicates('ISBN')
preparation

# converting the 'ISBN' data series into list form
isbn_id = preparation['ISBN'].tolist()

# converting the 'Book-Title' data series into list form
book_title = preparation['Book-Title'].tolist()

# converting the 'Book-Author' data series into list form
book_author = preparation['Book-Author'].tolist()

# converting the 'Year-Of-Publication' data series into list form
year_of_publication = preparation['Year-Of-Publication'].tolist()

# converting the 'Publisher' data series into list form
publisher = preparation['Publisher'].tolist()

print(len(isbn_id))
print(len(book_title))
print(len(book_author))
print(len(year_of_publication))
print(len(publisher))

books_new = pd.DataFrame({
    'isbn': isbn_id,
    'book_title': book_title,
    'book_author': book_author,
    'year_of_publication': year_of_publication,
    'publisher': publisher

})

books_new = books_new[:20000]

# converting User-ID to a list
user_ids = df_rating['User-ID'].unique().tolist()
print('list userIDs: ', user_ids)

# User-ID encoding
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID: ', user_to_user_encoded)

# encoding numbers into User-ID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded number to userID: ', user_encoded_to_user)

# converting ISBNs to a list
isbn_id = df_rating['ISBN'].unique().tolist()

# ISBN encoding
isbn_to_isbn_encoded = {x: i for i, x in enumerate(isbn_id)}

# encoding numbers to ISBN
isbn_encoded_to_isbn = {i: x for i, x in enumerate(isbn_id)}

# Disabling the SettingWithCopyWarning warning
pd.options.mode.chained_assignment = None # "warn" or "raise" to turn it back on

# Mapping User-ID to user dataframe
df_rating['user'] = df_rating['User-ID'].map(user_to_user_encoded)

# Mapping ISBN to book title dataframe
df_rating['book_title'] = df_rating['ISBN'].map(isbn_to_isbn_encoded)

num_users = len(user_to_user_encoded)
print(num_users)

num_book_title = len(isbn_to_isbn_encoded)
print(num_book_title)

# converting the rating to a float value
df_rating['Book-Rating'] = df_rating['Book-Rating'].values.astype(np.float32)

min_rating = min(df_rating['Book-Rating'])

max_rating = max(df_rating['Book-Rating'])

print('Number of Users: {}, Number of Books: {}, Min Rating: {}, Max Rating: {}'.format(
     num_users, num_book_title, min_rating, max_rating
))

"""# **COLLABORATIVE FILTERING**"""

df_rating = df_rating.sample(frac=1, random_state=42)
df_rating

# creating a variable x to match user data and book title into one value
x = df_rating[['user', 'book_title']].values

# creating a y variable to create a rating of the results
y = df_rating['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# dividing into 90% train data and 10% validation data

train_indices = int(0.9 * df_rating.shape[0])
x_train, x_val, y_train, y_val = (
     x[:train_indices],
     x[train_indices:],
     y[:train_indices],
     y[train_indices:]
)

print(x, y)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

class RecommenderNet(tf.keras.Model):

     # function initialization
     def __init__(self, num_users, num_book_title, embedding_size, dropout_rate=0.2, **kwargs):
         super(RecommenderNet, self).__init__(**kwargs)
         self.num_users = num_users
         self.num_book_title = num_book_title
         self. embedding_size = embedding_size
         self.dropout_rate = dropout_rate

         self.user_embedding = layers.Embedding( # user embedding layer
             num_users,
             embedding_size,
             embeddings_initializer = 'he_normal',
             embeddings_regularizer =keras.regularizers.l2(1e-6)
         )
         self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias

         self.book_title_embedding = layers.Embedding( # book_title embedding layer
             num_book_title,
             embedding_size,
             embeddings_initializer = 'he_normal',
             embeddings_regularizer =keras.regularizers.l2(1e-6)
         )
         self.book_title_bias = layers.Embedding(num_book_title, 1) # layer embedding book_title

         self.dropout = layers.Dropout(rate=dropout_rate)

     def call(self, inputs):
         user_vector = self.user_embedding(inputs[:, 0]) # call embedding layer 1
         user_vector = self.dropout(user_vector)
         user_bias = self.user_bias(inputs[:, 0]) # call embedding layer 2

         book_title_vector = self.book_title_embedding(inputs[:, 1]) # call embedding layer 3
         book_title_vector = self.dropout(book_title_vector)
         book_title_bias = self.book_title_bias(inputs[:, 1]) # call embedding layer 4

         dot_user_book_title = tf.tensordot(user_vector, book_title_vector, 2) # dot product multiplication

         x = dot_user_book_title + user_bias + book_title_bias

         return tf.nn.sigmoid(x) # activate sigmoid

model = RecommenderNet(num_users, num_book_title, 50) # initialize model

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=1e-4),
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
)

# training

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 16,
    epochs = 5,
    validation_data = (x_val, y_val)
)

book_df = books_new

user_id = df_rating['User-ID'].sample(1).iloc[0]
book_read_by_user = df_rating[df_rating['User-ID'] == user_id]

book_not_read = book_df[~book_df['isbn'].isin(book_read_by_user['ISBN'].values)]['isbn']
book_not_read = list(
    set(book_not_read)
    .intersection(set(isbn_to_isbn_encoded.keys()))
)

book_not_read = [[isbn_to_isbn_encoded.get(x)] for x in book_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_book_array = np.hstack(
    ([[user_encoder]] * len(book_not_read), book_not_read)
)

ratings_model = model.predict(user_book_array).flatten()

top_ratings_indices = ratings_model.argsort()[-3:][::-1]

recommended_book_ids = [
    isbn_encoded_to_isbn.get(book_not_read[x][0]) for x in top_ratings_indices
]

top_book_user = (
    book_read_by_user.sort_values(
        by='Book-Rating',
        ascending=False
    )
    .head(10)['ISBN'].values
)

book_df_rows = book_df[book_df['isbn'].isin(top_book_user)]

book_df_rows_data = []
for row in book_df_rows.itertuples():
    book_df_rows_data.append([row.book_title, row.book_author])

recommended_book = book_df[book_df['isbn'].isin(recommended_book_ids)]

recommended_book_data = []
for row in recommended_book.itertuples():
    recommended_book_data.append([row.book_title, row.book_author])


output_columns = ['Book Title', 'Book Author']
df_book_read_by_user = pd.DataFrame(book_df_rows_data, columns=output_columns)
df_recommended_books = pd.DataFrame(recommended_book_data, columns=output_columns)

print("Here are the top 3 books recommended for you!")
df_recommended_books

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Evaluation')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='bottom left')
plt.show()

# Define the full path
save_path = '/content/drive/My Drive/INTRO TO AI/saved_model'

# Save the entire model
model.save(save_path, save_format='tf')