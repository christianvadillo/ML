# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:53:23 2020

@author: 1052668570
"""

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


colnames = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('../data/recommender_systems/u.data', sep='\t', names=colnames)
movie_titles =  pd.read_csv('../data/recommender_systems/Movie_Id_Titles')

data.head()
movie_titles.head()

# Merging on item_id
data = pd.merge(data, movie_titles, on='item_id')
data.head()

# Getting mean rating by title
data.groupby(by='title')['rating'].mean().sort_values(ascending=False)

# Most rated movies
data.groupby(by='title')['rating'].count().sort_values(ascending=False)

ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['num of ratings'] = data.groupby(by='title')['rating'].count()

# =============================================================================
# Visualization
# =============================================================================
ratings['num of ratings'].hist(bins=70)
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.4)

# =============================================================================
# Recommender System
# =============================================================================
moviemat = data.pivot_table(index='user_id', columns='title', values='rating')
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

# corrwith
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# Cleaning starwars
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['num of ratings'])
# Deleting rows that don't have at least 100 ratings
corr_starwars = [corr_starwars['num of ratings']>=100].sort_values(by='Correlation', 
                                                                   ascending=False)
# Cleaning liarliar
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
# Deleting rows that don't have at least 100 ratings
corr_liarliar[corr_liarliar['num of ratings']>=100].sort_values(by='Correlation',
                                                                ascending=False)
