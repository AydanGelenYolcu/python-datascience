import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
df = pd.read_csv("starbucks.csv" ,index_col=0 )
df.info()
"""
Index: 77 entries, 1 to 77
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   item      77 non-null     object 
 1   calories  77 non-null     int64  
 2   fat       77 non-null     float64
 3   carb      77 non-null     int64  
 4   fiber     77 non-null     int64  
 5   protein   77 non-null     int64  
 6   type      77 non-null     object 
dtypes: float64(1), int64(4), object(2)"""

print(df.describe())
"""         calories        fat       carb      fiber    protein
count   77.000000  77.000000  77.000000  77.000000  77.000000
mean   338.831169  13.766234  44.870130   2.220779   9.480519
std    105.368701   7.095488  16.551634   2.112764   8.079556
min     80.000000   0.000000  16.000000   0.000000   0.000000
25%    300.000000   9.000000  31.000000   0.000000   5.000000
50%    350.000000  13.000000  45.000000   2.000000   7.000000
75%    420.000000  18.000000  59.000000   4.000000  15.000000
max    500.000000  28.000000  80.000000   7.000000  34.000000"""

print(df.groupby("type")["item"].count())
"""
type
bakery           41
bistro box        8
hot breakfast     8
parfait           3
petite            9
salad             1
sandwich          7
"""

df.groupby("type")["item"].count().plot()
plt.title("Item Types")
plt.show()

sns.countplot(x = "type", data=df, palette = "Set1")
plt.title("Item Types")
plt.show()
#calories
sns.catplot(kind = "bar", x = "type", y = "calories", data = df)
plt.title("Calorie Values")
#proteins
sns.catplot(kind = "bar", x = "type", y = "protein", data = df)
plt.title("Protein Values")
 #carbs
sns.catplot(kind = "bar", x = "type", y = "carb", data = df)
plt.title("Carbonhydrate Values")
#fiber
sns.catplot(kind = "bar", x = "type", y = "fiber", data = df)
plt.title("Fiber Values")
#fat
sns.catplot(kind = "bar", x = "type", y = "fat", data = df)
plt.title("Fat Values")


#if I try to just take df.corr(). it tries to take correlation of string values too and gives error.
#way to do it -- 

# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])
# Calculate correlation matrix for numeric columns
corr_matrix = numeric_df.corr()
print(corr_matrix)
"""
          calories       fat      carb     fiber   protein
calories  1.000000  0.758682  0.674999  0.260645  0.410398
fat       0.758682  1.000000  0.144547 -0.028549  0.223470
carb      0.674999  0.144547  1.000000  0.213044 -0.050789
fiber     0.260645 -0.028549  0.213044  1.000000  0.488564
protein   0.410398  0.223470 -0.050789  0.488564  1.000000"""

sns.heatmap(corr_matrix, annot = True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

#scatterplot
fig, ax = plt.subplots()
sns.scatterplot(x="calories", y="fat", data=df, s=30, edgecolor="red", ax=ax)
ax.set_title("Calories and Fats")
plt.show()

sns.displot(x = "calories", data =df, color = "red", kde = "True")
plt.title("Calorie Graph")
plt.show()

sns.displot(x = "protein", data =df, color = "green", kde = "True")
plt.title("Protein Graph")
plt.show()

sns.displot(x = "fat", data =df, color = "purple", kde = "True")
plt.title("Fat Graph")
plt.show()

sns.displot(x = "carb", data =df, color = "blue", kde = "True")
plt.title("Carbonhydrate Graph")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Feature and target selection
x = df[["calories", "fat", "fiber", "protein", "carb"]]
y = df["type"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the decision tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)
# output = Doğruluk: 0.6875

# Make a prediction for a specific input
prediction = model.predict([[300, 3, 60, 1, 5]])
print(prediction)
#output = ['hot breakfast']


# Plot the decision tree
plt.figure(figsize=(15, 10))

# Convert feature names and class names to lists
plot_tree(model, feature_names=x.columns.tolist(), class_names=model.classes_.tolist(), filled=True)

# Display the plot
plt.show()
