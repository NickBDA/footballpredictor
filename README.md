# footballpredictor
 
A machine learning project with two predictive models. This project is written in Python (Jupyter Notebook).

Please contact me on [Linkedin](https://www.linkedin.com/in/nickkempe) if you are looking to hire a data scientist.

###  [Football Salary Predictor](https://github.com/NickBDA/footballpredictor/blob/main/src/notebooks/REG_EDA.ipynb)
* Used player and team stats from 2017/18 season for Big 5 leagues (FBREF.com) and 2017 salary data from (Capology u/eddwebster scrape) 
* Merged various raw datasets into master dataframe and preprocessed data for modeling
* Investigated outliers, anamolies and categorical to determine best treatment for effective modeling
* Used XGBoostRegressor for baseline due to it's ability to handle NaN values
* Project goal to improve upon 0.46 Baseline using various statistical and modeling techniques
* Contrasted various NaN value treatments to determine greatest positive impact on score
* Compared numerous feature configurations, PCA and other reductive strategies to improve baseline scores 
* Scaled features, normalised target and tuned hyperparameters for multiple regressor models
* Final score of 0.68 improved baseline by 0.22.
* **Keywords**(Anomaly Detection, Regressor, Football, Supervised learning)

![image](https://github.com/NickBDA/footballpredictor/blob/main/src/img/REGmodelscores.png)

---

###  [Football Career Predictor](https://github.com/NickBDA/footballpredictor/blob/main/src/notebooks/CLF_EDA.ipynb)
* Used full player and team stats from 2017/18 season for Big 5 leagues from FBREF.com as well as 2022/23 inclusion
* Merged various raw datasets into master dataframe and preprocessed data for modeling
* Investigated outliers, anamolies and categorical to determine best treatment for effective modeling
* Used XGBoostClassifer for baseline due to it's ability to handle NaN values
* Project goal to improve upon 0.61 F1 Baseline using various statistical and modeling techniques
* Contrasted various NaN value treatments to determine greatest positive impact on score
* Compared numerous feature configurations, PCA and other reductive strategies to improve baseline scores 
* Scaled features, normalised target and tuned hyperparameters for multiple classifier models
* Final score of 0.69 improved baseline by 0.08.
* **Keywords**(Anomaly Detection, Classifier, Football, Supervised learning)

![image](https://github.com/NickBDA/footballpredictor/blob/main/src/img/CLF%20F1%20results%20cropped.png)
