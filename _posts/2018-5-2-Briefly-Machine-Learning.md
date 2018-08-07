---
layout: post
title: Briefly Machine Learning
excerpt: A brief introduction to Machine Learning in general
images:
  - url: /images/cover_brief_ml.jpg

---

Arthur Samuel, a computer scientist who pioneered the study of Artificial Intelligence said that **Machine Learning is the study that gives computers the ability to learn without being explicitly programmed.**

Machine Learning is the design and study of software artifacts that use past experience to make future decisions, a science and art of programming computers so they can learn from data. It's a method of data analysis that automates analytical model building. Using algorithms that learn from data in iterative fashion and find hidden insight without being explicitly programmed where to look.

A popular quote from Tom Mitchell define Machine Learning more engineering-oriented:

> A computer program is said to learn from experience **E** with respect to some task **T** and some performance measure **P**, if its performance on **T**, as measured by **P**, improves with experience **E**.

For example, a spam filter is a Machine Learning program that can learn to flag spam given example of spam emails and examples of regular emails. The examples that the system uses to learn are called **training set**. Each training example is called a **training instances**. In this case, the task **T** is to flag spam for new emails, the experience **E** is the training data, and the performance measure **P** needs to be defined; for example, we can use the ration of correctly classifed emails. This particular performance measure is called **accuracy** and it is often used in classification tasks. 

An another example, assume that we have a collected of pictures. Each picture depicts either a dog or cat photos. A program could learn to perform this task by observing pictures that have already been sorted, and it could evaluate its performace by calculatinng the percentage of correctly classified pictures.

---

### Machine Learning Systems

Machine Learning Systems can be classified based on three broad categories:

**1. Model are trained with Human Supervision**

As these criteria aren't exclusive, we can combine them in any way we like. Machine Learning systems can be classified according to the amount and type of supervision they get during training. There're four major categories:

- Supervised learning
- Unsupervised learning
- Semi-Supervised learning
- Reinforcement learning

**Supervised learning algorithm** are trained using labeled examples, such as an input where the desire output is known. In supervised learning, the training data we feed to the algorithm icluded the desired solution, it's called labels. The learning algorithm receives a set of inputs along with the corresponding correct outputs, and the algorithm learns by comparing its actual output with correct outputs to find errors. It then modifies the model accordingly. 

Through method like classification , regression , prediction , gradient boosting — Supervised learning uses patterns to predict the values of the label on additional unlabeled data. Supervised algorithm is commonly used in application where historical data predict likely future events. Some most popular supervised learning algorithms are listed below:

- [K-Nearest Neighors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Support Vectore Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)
- [Random Forests](https://en.wikipedia.org/wiki/Random_forest)
- [Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

---

**Unsupervised learning algorithm** is used against data that has no historical labels. The system is not told the right answer. The algorithm must figure out what is being shown. The goal is to explore the given data and find some structure within the data.

Some of the most important unsupervised learning algorithms are listed below:

- Clustering
  - [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
  - [Hierarchical Cluster Analysis (HCA)](https://en.wikipedia.org/wiki/Hierarchical_clustering)
  - [Expectation Maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- Visualization and Dimensionality reduction
  - [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
  - [Kernel PCA](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
  - [Locally-Linear Embedding (LLE)](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction)
  - [t-distributed Stochastic Neighbor Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
- Association rule learning
  - [Apriori](https://en.wikipedia.org/wiki/Association_rule_learning)
  - [Eclat](https://en.wikipedia.org/wiki/Association_rule_learning#Eclat_algorithm)

Another good example of unsupervised learning algorithm is **Visualization algorithm**. We feed them a lot of complex and unlabeled data, and they give outputa 3D or 2D representation of our data that can easily be plotted.

---

**Semi-Supervised learning algorithm** can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. A good example is **Google Photo** hosting services. It automatically recognizes the same person on the photo gallery. All we need to do is to tell it who these people are. 

Most of the semisupervised learning algorithms are combination of unsupervised and supervised algorithm. **Deep Belief Network** are based on unsupervised components called **Restricted Boltzmann machines** stacked on top of one another and the whole system is fine-tuned using supervised learning techniques.

---

**Reinforcement learning algorithm** is often used for robotics, gaming and navigation. With reinforcement learning , the algorithm discovers through trial and error which actions yield the greatest rewards.

This type of learning has three primary components -

**agent** — the learner or decision maker<br>
**environment** — everything the agent interacts with<br>
**action** — what the agent can do

The objective is for the agent to choose actions that maximize the expected reward over a given amount of time. The agent will reach the goal much faster by following a good *policy*. So the goal in the reinforcement learning is to learn the **best policy**. A policy defines what action the agent should choose when it is in a given situation. 

**DeepMind's AlphaGo** program is a good example of **Reinforcement Learning**; which beat the world champion Lee Sedol at the game of **Go**. AlphaGo was just applying the policy it had learned.

---

**2. Batch and Online Learning**
In **Batch learning** the system is incapable of learning incrementally, it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. At first the system is trained, and then it's launched into production and runs without learning anymore; it just performs what it has learned. For this reason, it's called **Offline Learning**

In constrast, an **Online Learning**,  we train the system incrementally be feeding it data instances sequenctially, either single package or by groups called **mini-batches**. As each learning step is fast and cheap, so the system can learn about new data on the fly.

---

**3. Instance-Based VS Model-Based Learning**
In Instance-Based, model just learns the examples by heart and use a similarity measure to *generalize* to new instances. On the other hand, in Model-Based approache, it tunes some parameter to fit the model to the training set and after that it hopefully will be able to make a good predictions n new cases as well.


So in ML a Model which is ….

- learn from data
- learn on its own
- discovering hidden patterns
- data driven decision


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

However this’s actually a vast topic and lots of things have to cover. To get more of this, a few blogs followings are worth to follow –

- [AI](https://www.artificial-intelligence.blog/news/)
- [Open AI](https://openai.com/)
- [Google AI](https://ai.google/)
- [Machine Learning & Statistics](http://kldavenport.com/)

Also check [this out](http://web.mit.edu/tslvr/www/lessons_two_years.html) , good reading. This is form Tom Silver — on ‘My First Two Years of AI Research’.

If someone found himself hard to understand Machine Learning and can’t catch interest, then it good to go first watch some movies on it, it might trigger his interest on it. However no one will learn ML from these of course but will get soft feels for sure :smirk:

- [Person of Interest](https://www.imdb.com/title/tt1839578/)
- [Ex Machina](https://www.imdb.com/title/tt0470752/?ref_=nv_sr_1)
- [Silicon Valley](https://www.imdb.com/title/tt2575988/?ref_=fn_al_tt_1)
- [WALL-E](https://www.imdb.com/title/tt0910970/)
- [I, Robot](https://www.imdb.com/title/tt0343818/)

And last but not least — [Green Lantern Animated Series](https://www.imdb.com/title/tt1724587/) — Silly, right? But it was the most influential AI anime series so far to get excited about Artificial Intelligence. A character named [Aya — Origin](https://www.youtube.com/watch?v=mqAmAqV1-Fs) which enough to make someone serious on AI technology.

---
<a href = "/assets/source_3.txt" target= "_blank">References</a>
