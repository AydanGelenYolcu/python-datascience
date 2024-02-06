netflix = pd.read_csv(r"C:\Users\ASUS\Downloads\netflix_titles.csv")
data = netflix.copy()
data= data.dropna()

#types
sns.countplot(x = netflix["type"])
plt.title("tür")
plt.show()  # Added this line to display the plot

#ratings
sns.countplot(x = netflix["rating"])
fig = plt.gcf()
fig.set_size_inches(13, 13)
plt.title("tür")
plt.show()


#comparative
plt.figure(figsize=(10, 8))
sns.countplot(x=netflix["rating"], hue="type", data=netflix)
plt.title("Type and Rating Ratio")
plt.show()

#yellow, orange pie graph
labels = ["Movie", "Tv Show" ]
size=netflix["type"].value_counts()
colors = plt.cm.Wistia(np.linspace(0,1,2))
explode = [0,0.1]
plt.rcParams["figure.figsize"] = (9,9)
plt.pie(size, labels=labels, colors = colors, explode =explode, shadow = True, startangle = 90)
plt.title("types", fontsize= 25)
plt.legend()
plt.show()

#wordcloud

from wordcloud import WordCloud

plt.subplots(figsize = (25,15))
wordcloud = WordCloud(
    background_color = "white",
    width = 1920,
    height =1080
    ).generate(" ".join(data.cast))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("cast.png")
plt.show()
