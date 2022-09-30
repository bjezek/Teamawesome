# Twitter, DogeCoin, Bitcoin and The cause and affect of Elon Musk ;)

The project team used a combination of machine learning models and sentiment analysis from Elon Musk's tweets to find various trading models for stocks and crypto. 

Historical data available online along with data from APIs such as Alpaca's Trading API and Twitter's Developer API fueled this project. Sentiment analysis was applied using the nltk library, and machine learning models utilizing prophet (X) and scikitlearn (logistic regression & support vector machine models) were applied. 

The project team came up with a variety of models that seem to have performed well throughout the trading periods analyzed. Next steps in the project would be to refine the combinations of the models used, and incorporate hourly data instead of daily data in the analysis. 

Jupyter notebooks were made which incorporated the various techinques that our project team analyzed and set the code up to place trades based on information from the models. 

# Sentiment Analysis

## Data Processing: 
Utilizing pandas
reading .csv files of Elon Musk's tweets, closing prices of Bitcoin and Dogecoin. Set up Dataframes for each. 
filtering out data, renaming columns 
str().contains() to filter out for specific words("Bitcoin", "Dogecoin")

## Utilizing SentimentInstensityAnalyzer:

Utilizing nltk.sentiment.vader/SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')


Creating Sentiment Dataframes for Twitter, Tesla, Dogecoin, Bitcoin based on polarity scores from Sentiment Analyzer()
-(.apply(sid.polarity_scores).apply(pd.Series))

Pos + Neg + Neu = compound(Sentiment Score).


## Merging of Data: 

Merging Sentiment dataframes with Tweet dataframes using pd.merge. 


## Calculations of returns: 

Various returns based on merged dataframes using pct.change(), pct.change().sum(), .mean for Sentiment Scores. 
Compare and contrast of merged dataframes 
Filtering out and creating dataframes for low and high sentiment for both Bitcoin and Crypto, and calculating those returns as well using pct.change().sum()


## Plot:
Sentiment impact vs. Pct Change
Using .filter in order to graph only the necessary data. 









# Machine Learning Models
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
# Model Performance Summary
Twitter (Logistic Regression): 
+180% return over testing period (Nov 2021 - present)
DogeCoin (Fbprophet): 
Strategy: Simple to implement (buy on Tuesday at the close, sell on Friday at the close) 
+315% return over dataset entirety (Nov 2017 - present)
Sentiment Analyzer (Elon Musk Tweets): 
Tweets with “Dogecoin” for Dogecoin: 
+270%	
Pro: Simple to implement (buy and sell at EOD). High returns.
Con: Difficult to predict future returns as it depends on predicting human behavior.
Volatile price movement. High risk.

# APIs

Please see the file "sample.env" included in this repository as these notebooks rely on API keys that are stored in a separate .env (text) file outside of your application. Make sure you have the dotenv library installed, as the function `load_dotenv` loads these secret variables into your notebook. 

## Alpaca API

Using Alpaca's trading API, you have access to historical data and are also able to place trades with your paper & live trading accounts. 

If you are wanting to connect yourself to Alpaca's trading API, the notebook in the repository named "TESTING_Alpaca_paper_trading_bot" will help you get started.There are links to various resources in the documentation as well. 

Connecting to the API, pulling historical data, placing trades and checking your account position can be done in a few lines of code: 

```python
# connect to the api
api = tradeapi.REST(API_KEY, API_SECRET, ALPACA_API_BASE_URL, api_version="v2")

# pull historical data for bitcoin
btc_data = api.get_crypto_bars("BTCUSD", TimeFrame.Hour).df

# place an order to buy 1 bitcoin
api.submit_order('BTCUSD', qty=1, side='buy', time_in_force="gtc")

# check the quantity of dogecoin you have in your account
doge_qty = api.get_position('DOGEUSD').qty
```

In addition to stock and crypto historical data available from Alpaca, the project team also used historical data available from Yahoo Finance.

## Twitter Developer API

Obtaining credentials to use Twitter's developer API is easy, and you can follow the link in the documentation reference section below to sign up yourself. The entry-level access level that you obtain initially is "Essential" level access (which is what this project uses), but you can request higher levels of access for more data if you desire. 

Pulling recent tweets data from Elon Musk is easy to do using the tweepy library. 

``` python
# connect to the client using your API keys
client = tweepy.Client( bearer_token=TWITTER_BEARER_TOKEN, 
                        consumer_key=TWITTER_API_KEY, 
                        consumer_secret=TWITTER_API_SECRET_KEY, 
                        access_token=TWITTER_ACCESS_TOKEN, 
                        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET, 
                        return_type = requests.Response,
                        wait_on_rate_limit=True)

# define your query
query = 'from:elonmusk'

# send your query through the api 
tweets = client.search_recent_tweets(query=query, 
                                    tweet_fields=['author_id', 'created_at'],
                                     max_results=100)

# ultimately save your data as a dataframe after a few conversions
tweets_dict = tweets.json() 
tweets_data = tweets_dict['data'] 
df = pd.json_normalize(tweets_data) 
```

In addition to the latest tweet information from Elon Musk available through the API, the project team also used historical data available from Kaggle. The dataset from X's project (X) includes Elon Musks's tweet history from X until X.

# Implementing the Algorithms! 

The notebook named "Full_Twitter_Machine_APIs_Models" includes a combination of models that are similar to what the project team evaluated. Numerous strategies are combined in this notebook and can be updated if necessary, this notebook serves as more of a proof-of-concept and the algorithms in it can be refined and revised. 

The beginning of the notebook imports the libraries that you need for the work, and connects you to the Alpaca and Twitter APIs. There is a sample .env file included in this repo, just update your credentials and save this file as your .env file in the same folder as the root directory of your notebook. 

The notebook then goes through a few examples of how to implement these trading algorithms. The simplest model, just buying Dogecoin on Tuesday and selling on Friday, is easy to implement (and the project team has done so - X% gain in paper trading account this week!). Although this algorithm seems silly, it really would have turned a profit over the history of Dogecoin's life (at least as far back as the Yahoo Finance dataset yet). Difficult to predict how much longer this trend will be profitable for!

Other examples that are implemented in the notebook are buying Dogecoin or Bitcoin if Elon Musk tweets about them (and/or depending on the sentiment of the tweets), and then another model where a support vector machine model is created with Dogecoin and a combination of simple moving average calculations. The last value in the predictions dataset is used as the buy/sell recommendation. This is just an example of how one would implement an svm model similar to what the project team discussed, the model in this notebook isn't an exact replica of the evaluation that was shown earlier in the README.  


# Documentation References

Alpaca Documentation - https://alpaca.markets/docs/

Twitter Developer API - https://developer.twitter.com/en

Nltk - SentimentAnalyzer() - https://www.nltk.org/_modules/nltk/sentiment/vader.html

Prophet - https://facebook.github.io/prophet/

Scikitlearn (SVM) - https://scikit-learn.org/stable/modules/svm.html

Pandas - https://pandas.pydata.org/docs/index.html
 
Tweepy - https://www.tweepy.org/

# The Project Team

The work in this repo was done as a project for Rice's FinTech Bootcamp program, check us out on GitHub: 

Gautam duckmobsauce
Brandon bjezek
Paula ai-to-the-moon