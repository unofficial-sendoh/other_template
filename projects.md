
> I don’t update this page as much, so head to my [GitHub](https://github.com/iphton) for the most recent projects

---

Some Necessary [GitHubGist](https://gist.github.com/iphton) 
- [Installing OpenCV3 for Python3x](https://gist.github.com/iphton/4af8e9a3accfdcf2511ee4ced26a1d77)
- [Create Anaconda Env with Necessary Packages](https://gist.github.com/iphton/ecf250c2d49af73fdeb92982723d7016)
- [Most Used Raspberry Pi Commands](https://gist.github.com/iphton/92c29a45ffdd73b0e327301840e9b3b3)
- [Play Store on GennyMotion](https://gist.github.com/iphton/dd4fcbf4e1c3fe809ba01cfe2d6a0892)
- [Installing FFmpeg on Linux](https://gist.github.com/iphton/1a251a673a3eeb985930392ea4440f74)
- [FFmpeg Basic Operation on Subtitles](https://gist.github.com/iphton/7d5511941970e8b448453980f47f9ec3)

---

$A$ $AB$ $ABCDE$


$
\begin{align}
P(Y \mid do(X)) &= \sum_U P(Y \mid X, U=u) P(U=u) \tag{iff all back doors are shut} \\
&= \mathbb E_{u\sim U}[P(Y \mid X, U=u)] \\
P(Y \mid X, U=u) &=  \frac{\mathbb E_{(x, y)\sim X, Y, U=u} [(x -\mu_x)(y-\mu_y)]}{\sigma_x} \tag{correlation}  \\
\\
Y &= aX + bU + c \tag{assume linear relationships} \\
a&= (Y-bU + c)X^{-1} \\
\end{align}
$




## [MovieLens-IMDb-Analysis](https://github.com/iphton/MovieLens-IMDB-Analysis)
**Tech Used**: Data Analysis - Python 3 - Machine Learning

[GroupLens Movie Data Sets](http://grouplens.org/datasets/)

This dataset describe 5-star ratings and free-text tagging activity from , a move recommendation service. It contains **20000263** ratings and **465564** tags applications across **27278** movies. These data were created by **138493** user between January 09 1995 and March 31 , 2015. The data are contained in six files `links.csv` , `movies.csv` , `ratings.csv` and `tags.csv` etc.


## [Kaggle Competitions](https://github.com/iphton/Kaggle-Competition)
**Tech Used**: Data Analysis - Feature Engineering - Python 3 - Machine Learning

- [Titanic: Machine Learning from Disaster](http://nbviewer.jupyter.org/github/iphton/Kaggle-Competition/blob/gh-pages/Titanic%20Competition/Notebook/Predict%20survival%20on%20the%20Titanic.ipynb#5-bullet): I barely remember first when exactly I watched Titanic movie but still now Titanic remains a discussion subject in the most diverse areas. In this kaggle challenge, we're asked to complete the analysis of what sorts of people were likely to survive. In particular, we're asked to apply the tools of machine learning to predict which passengers survived the tragedy. I've used cross-validation on the top 10 most popular classification models and based on scoring finally choose five most promising classifiers and fine-tuned the models. I used Voting Classifier to combine the predictions coming from classifiers. The model gets an accuracy of around 84%. 

---

## Short-Length Projects

### [Tech Zone BD | E-Learning for Developer and Designer](https://github.com/iphton/Tech-Zone)
**Tech Used**: ASP.net - C# - MS SQL

### [Control Raspberry Pi GPIO with Web-Sockets](https://github.com/iphton/Raspberry-Pi-WebSocket)
**Tech Used**: Node.JS - Tornado - Web Socket - Raspberry Pi 3

### [Web Scraping](https://github.com/iphton/Data-Scraping)
**Tech Used**: Python 3 - Selenium - Beautiful Soup - Scrapy
