
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

netflix = pd.read_csv("netflix_titles.csv") 
print(netflix.head())
print(netflix.shape)
print(netflix.columns)
print(netflix.isnull().sum())
print(netflix.nunique())
print(netflix["rating"].unique())
print(netflix["rating"].value_counts())

#TYPE BAR CHART
data = netflix.copy()
data = data.dropna()
sns.countplot(x = netflix["type"], palette = "Set1")
fig = plt.gcf()
fig.set_size_inches(5,5)
plt.title("Type")
plt.show

#RATING BAR CHART
valid_ratings = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'TV-Y7', 'TV-Y', 
                 'PG', 'TV-G', 'NR', 'G', 'TV-Y7-FV', 'NC-17', 'UR'] #removed 74 min, 84 min 66 min
netflix_cleaned = netflix[netflix["rating"].isin(valid_ratings)]

plt.figure(figsize=(15, 15))
ax = sns.countplot(x=netflix_cleaned["rating"], palette="Set1")  #color palette
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

#Adding total values on top of each column
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2.,  # X 
            p.get_height(),                  # Y 
            int(p.get_height()),             # The number to be written (equal to the column height):
            ha="center",                     # Center alignment
            va="bottom")                     # Bottom alignment(upper of bars)
    
plt.title("Rating")
plt.show()

#RATING AN TYPE
ax = sns.countplot(x="rating", hue="type", data=netflix_cleaned, palette="Set1")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.title("Rating and Type")
plt.show()


#PIE CHART OF TYPE
# Set colors to red and blue
colors = ['red', 'blue']

labels=["Movie", "TV show"]
size=netflix["type"].value_counts()
explode = [0, 0.1]
plt.rcParams["figure.figsize"] = (9, 9)
plt.pie(size, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90)
plt.title("Type Pie", fontsize=25)
plt.legend()
plt.show()

#NONFILTERED RATING PIE
netflix["rating"].value_counts().plot.pie(autopct="%1.1f%%", shadow = True, figsize = (10,8))
plt.show()

#FILTERED RATING PIE
#Extract the category counts from the 'Rating' column.
rating_counts = netflix["rating"].value_counts()
#Filter out categories with low frequency, for example, those with less than 50 occurrences.
filtered_rating_counts = rating_counts[rating_counts >= 50]
#Pie graph
filtered_rating_counts.plot.pie(autopct="%1.1f%%", shadow=True, figsize=(10, 8))
plt.title("Rating Distribution")
plt.show()

#Wordcloud
from wordcloud import WordCloud
plt.subplots(figsize = (25,15))
wc = WordCloud(background_color="white", 
width=1920,
height=1080).generate(" ".join(data.country))
plt.imshow(wc)
plt.axis("off")
plt.savefig("country.png")
plt.show()
