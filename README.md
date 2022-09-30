# Twitter, DogeCoin, Bitcoin and The cause and affect of Elon Musk ;)

# SentimentAnalysis


# Data Processing: 
Utilizing pandas
reading .csv files of Elon Musk's tweets, closing prices of Bitcoin and Dogecoin. Set up Dataframes for each. 
filtering out data, renaming columns 
str().contains() to filter out for specific words("Bitcoin", "Dogecoin")

# Sentiment Analysis:

Utilizing nltk.sentiment.vader/SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')


Creating Sentiment Dataframes for Twitter, Tesla, Dogecoin, Bitcoin based on polarity scores from Sentiment Analyzer()
-(.apply(sid.polarity_scores).apply(pd.Series))

Pos + Neg + Neu = compound(Sentiment Score).


# Merging of Data: 

Merging Sentiment dataframes with Tweet dataframes using pd.merge. 


# Calculations of returns: 

Various returns based on merged dataframes using pct.change(), pct.change().sum(), .mean for Sentiment Scores. 
Compare and contrast of merged dataframes 
Filtering out and creating dataframes for low and high sentiment for both Bitcoin and Crypto, and calculating those returns as well using pct.change().sum()


# Plot:
Sentiment impact vs. Pct Change
Using .filter in order to graph only the necessary data. 









# Elon Musk Twitter analysis - Machine Models
ran regression models, fbprophet, SVM on Bitcoin,Twitter and DogeCoin to see the probability 
of success. The regresssion model performed the best 2 out of 3 when it came to testing on the data
using a 5 to 30 SMA with a strategy of 3 months outperforming the actual returns. Came to see that as 
DogeCoin and BTC are not actual stocks they tend to show a negative return probability in the FB prophet model 
for future perdiction based on non-fiscal factors causing a possible negative forecast for these two coins.
The Regression models where used to see a strategy success or failure by looking at the live data without any outside
noise that had nothing to do with the fundemental strength of the companys performance based on past events. 
# Regression Model for Twitter
![image](https://user-images.githubusercontent.com/106267420/193173702-8482d9ca-7166-4d7e-81b5-a1e040ebbd85.png)
# SVM Model for DogeCoin
![image](https://user-images.githubusercontent.com/106267420/193174176-009c0932-591d-4e4a-909f-d0d1b48e3b9a.png)
# Regression Model for BTC
![image](https://user-images.githubusercontent.com/106267420/193174937-f6c9af26-d0f1-4662-97ad-a099d89031b2.png)
# Model Performance 
Twitter (Logistic Regression): 
+180% return over testing period (Nov 2021 - present)
DogeCoin (Fbprophet): 
+315% return over dataset entirety (Nov 2017 - present)
Pro: Simple to implement (buy on Tuesday, sell on Friday) 
Sentiment Analyzer (Elon Musk Tweets): 
Tweets with “Dogecoin” for Dogecoin: 
+270%	
Pro: Simple to implement (buy and sell at EOD). High returns.
Con: Difficult to predict future returns as it depends on predicting human behavior.
Volatile price movement. High risk.

