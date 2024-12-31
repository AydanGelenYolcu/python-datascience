**Netflix Dataset Analysis**

This project performs exploratory data analysis (EDA) on a Netflix dataset to gain insights 
into the content available on the platform. The analysis includes visualizations such as bar charts, pie charts, 
and a word cloud to present key information about content types, ratings, and geographical distribution.

***Dataset***
The dataset used in this analysis is `netflix_titles.csv`, which contains information about Netflix movies and TV shows. The columns include:
- `show_id`: Unique identifier for the show
- `type`: Whether the content is a Movie or a TV Show
- `title`: Title of the content
- `director`: Director of the content
- `cast`: Cast members of the content
- `country`: Country of origin
- `date_added`: The date when the content was added to Netflix
- `release_year`: The release year of the content
- `rating`: Content rating (e.g., PG, TV-14)
- `duration`: Duration of the content (minutes for movies, seasons for TV shows)
- `listed_in`: Genre of the content
- `description`: Brief description of the content
## Code Breakdown

### 1. Initial Data Exploration
- **Displaying basic data information**: The initial steps include reading the dataset, displaying the first few rows, checking the shape, column names, and the count of missing values.
- **Data cleaning**: Dropping any rows with missing values to ensure clean data for visualizations.

## 2. Visualizations

## Insights

The visualizations provide insights into the following:
- The majority of content on Netflix consists of Movies.
- Ratings like `TV-MA`, `TV-14`, and `PG-13` are common on the platform.
- The distribution of Movies and TV Shows across different ratings shows interesting patterns, with more mature ratings often associated with Movies.
- The word cloud gives a quick overview of which countries are the most prolific in terms of content production.

## Conclusion

This project demonstrates how to analyze and visualize a large dataset using Python libraries like `pandas`, `seaborn`, `matplotlib`, and `wordcloud`. The resulting insights can help understand the composition of Netflix's content in terms of type, rating, and geographical origin.
