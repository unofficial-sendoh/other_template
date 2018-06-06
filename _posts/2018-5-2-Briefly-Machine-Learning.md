---
layout: post
title: Briefly Machine Learning
excerpt: A brief introduction to Machine Learning in general
images:
  - url: /images/cover_brief_ml.jpg

---



**Machine Learning**

Machine Learning is a method of data analysis that automates analytical model building. Using algorithms that learn from data in iterative fashion. Machine Learning allows computer to find hidden insight without being explicitly programmed where to look.

System are often called Model which can learn to perform a specific task by analyzing lots of examples for particular problems.

So in ML a Model which is ….

- learn from data
- learn on its own
- discovering hidden patterns
- data driven decision

ML algorithm is programmed to learn from the data that there’s nothing is the algorithm or program which directly aims to learn the given task.

### Machine Learning Process

ML process in general — At first we need to acquire data. Data comes from various places such as Facebook , Twitter , Database and many more. There’re many technique to access this data. However after acquiring data we need to clean it properly. Because real world data is really messy.

- Inconsistent values
- Duplicate values
- Missing values
- Invalid values
- Outliers

So we should address data quality issue -

- Remove data with missing values
- Merge duplicate records
- Generates best estimate for invalid issues
- Remove outliers

Data preparation is very important for our models. Models are learned from the data and perform , so the amount and quality of data available for building the models are important factors. After cleaning the data sets we then need to split it into two part Test set and Train set . We’ll build and train our models using Train data sets and next we’ll test our models using Test data sets. Now this’s an iterative process as because we need to get best models. After the model is ready , then we can deploy our model.

![ml_process](https://user-images.githubusercontent.com/17668390/40884042-0ca8b7f0-66c0-11e8-84be-1dbf6040050b.png)

There are different types of Machine Learning technique out there for different types of problem. There’re main three types of ML algorithms

- Supervised Algorithm
- Unsupervised Algorithm
- Reinforcement Algorithm

**Supervised learning algorithm** are trained using labeled examples , such as an input where the desire output is known. The learning algorithm receives a set of inputs along with the corresponding correct outputs , and the algorithm learns by comparing its actual output with correct outputs to find errors. It then modifies the model accordingly.

Through method like classification , regression , prediction , gradient boosting — Supervised learning uses patterns to predict the values of the label on additional unlabeled data. Supervised algorithm is commonly used in application where historical data predict likely future events.

**Unsupervised learning algorithm** is used against data that has no historical labels. The system is not told the right answer. The algorithm must figure out what is being shown. The goal is to explore the given data and find some structure within the data.

Popular technique include self-organizing maps , nearest-neighbor mapping , k-means clustering and singular value decomposition. These algorithm are also used to segment text topics , recommend items and identify data outliers

**Reinforcement learning algorithm** is often used for robotics , gaming and navigation. With reinforcement learning , the algorithm discovers through trial and error which actions yield the greatest rewards.

This type of learning has three primary components -

**agent** — the learner or decision maker<br>
**environment** — everything the agent interacts with<br>
**action** — what the agent can do

The objective is for the agent to choose actions that maximize the expected reward over a given amount of time. The agent will reach the goal much faster by following a good policy. So the goal in the reinforcement learning is to learn the best policy.

### Application field of ML

Let’s take quick overview to some fileds where machine learning are applied -

- Fraud Detection
- Web Search Result
- Real time ads on web page
- Credit scoring and next best offer
- Prediction of equipment failures
- New pricing models
- Network intrusion detection
- Recommendation Engines
- Customer Segmentation
- Text sentiment analysis
- Predicting customer churn
- Pattern and image recognition
- Email spam filtering
- Financial Modeling

However this’s actually a vast topic and lots of things have to cover. To get more of this, i’d like to suggest you a few blogs followings -

- [AI](https://www.artificial-intelligence.blog/news/)
- [Open AI](https://openai.com/)
- [Google AI](https://ai.google/)
- [Machine Learning & Statistics](http://kldavenport.com/)

Also check [this out](http://web.mit.edu/tslvr/www/lessons_two_years.html) , good reading. This is form Tom Silver — on ‘My First Two Years of AI Research’.

If you found yourself hard to understand Machine Learning and can’t catch interest, then i suggest you to go first watch some movies on it, it might trigger your interest on it. You won’t learn ML from these of course but will get soft feels for sure :D

- [Person of Interest](https://www.imdb.com/title/tt1839578/)
- [Ex Machina](https://www.imdb.com/title/tt0470752/?ref_=nv_sr_1)
- [Silicon Valley](https://www.imdb.com/title/tt2575988/?ref_=fn_al_tt_1)
- [WALL-E](https://www.imdb.com/title/tt0910970/)
- [I, Robot](https://www.imdb.com/title/tt0343818/)

And last but not least — [Green Lantern Animated Series](https://www.imdb.com/title/tt1724587/) — Silly, right ? I know. But it was the most influential AI anime series was for me to get excited about Artificial Intelligence. A character named [Aya — Origin](https://www.youtube.com/watch?v=mqAmAqV1-Fs) which made me serious on AI technology.

