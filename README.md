
#                                   Table Of Content


|**Chapter No.**|**Title**|
| - | - |
|**1**|[Problem Statement](#_iutk7rng78fu)|
|**2**|[Implementation](#_xeweogcj8qdi)|
|2.1|[About Dataset](#_ka479ka8ordr)|
|2.2|[Exploratory Data Analysis](#_fmwrkvon4u1b)[ and Data Pre-processing](#_fmwrkvon4u1b)|
|2.3|[Feature Engineering](#_gvnjxzy2z7c8)|
|**3**|[Training Process](#_93lmziycgsgk)|
|3.1|[Models Used](#_sf7nmuoqenwn)|
|3.2|[Metric Used](#_k55soholxfiq)|
|3.3|[Parameter Tuning](#_q979xsn40pis)|
|3.4|[Best Parameters](#_pdynojv2cghy)|
|**4**|[Conclusion](#_16dkpj7ucc8a)|
|**5**|[References](#_hhyz9nb0u970)|






#


# 1. Problem Statement:
In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different ammunition, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.

You are provided with a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 players per group.

You must create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).


# 2. Implementation:
## **2.1  About Dataset:**
The PUBG Dataset has up to 100 players in each match which are uniquely identified based on their matchId. The players can form a team in a match, for which they will have the same groupId and the same final placement in that particular match.

The data consists of different groupings, hence the data has variety of groups based on the number of members in the team(not more than 4) and matchType can be solo, duo, squad and customs.Also the matchType can be further more classified based on the perspective mode like TPP and FPP.

Approximately there are 3 million training data points and 1.3 million testing data points. There are in total 29 features. They are summarised as follows:


|Sr.No.|Feature|Type|Description|
| - | - | - | - |
|1|Id|String|Unique Id for each Player.|
|2|matchId|String|Id to identify matches.|
|3|groupId|String|Id to identify the group.|
|4|assists|Real|Number of enemy players this player damaged that were killed by teammates.|
|5|boosts|Real|Number of boost items used.|
|6|damageDealt|Real|Total damage dealt. Note: Self inflicted damage is subtracted.|
|7|DBNOs|Real|Number of enemy players knocked.|
|8|headshotKills|Real|Number of enemy players killed with headshots.|
|9|heals|Real|Number of healing items used.|
|10|killPlace|Real|Ranking in match of number of enemy players killed.|
|11|killPoints|Real|Kills-based external ranking of player.|
|12|kills|Real|Number of enemy players killed.|
|13|killStreaks|Real|Max number of enemy players killed in a short amount of time.|
|14|longestKill|Real|Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.|
|15|matchDuration|Real|Duration of match in seconds.|
|16|maxPlace|Real|Worst placement we have data for in the match.|
|17|numGroups|Real|Number of groups we have data for in the match.|
|18|rankPoints|Real|Elo-like ranking of players. |
|19|revives|Real|Number of times this player revived teammates.|
|20|rideDistance|Real|Total distance travelled in vehicles measured in metres.|
|21|roadKills|Real|Number of kills while in a vehicle.|
|22|swimDistance|Real|Total distance travelled by swimming measured in metres.|
|23|teamKills|Real|Number of times this player killed a teammate.|
|24|vehicleDestroys|Real|Number of vehicles destroyed.|
|25|walkDistance|Real|Total distance travelled on foot measured in metres.|
|26|weaponsAcquired|Real|Number of weapons picked up.|
|27|winPoints|Real|Win-based external ranking of players.|
|28|matchType|Categorical|Identifies the matchType.|
|29|winPlacePerc|Real|This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match.|
###
## 2.2  Exploratory Data Analysis and Data Pre-Processing:<br><br>

### ***Dataset* Size:**

The EDA was quite interesting as the training dataset was about 3 million rows in size.The size of the training dataset was about **688.7 MB,** hence the task to handle it would have been somewhat difficult if it would have been involved in any computations.

So by looking at the datatypes of the columns, most of the types were float64 and int64, so we downcasted the datatype of all the numerical columns to as small as possible and reduced the size of the training dataset to **237.5 MB.**

|![image](https://user-images.githubusercontent.com/110079774/212449168-bc3e487e-a1e8-49fa-8642-ce3e433ff693.png)|![image](https://user-images.githubusercontent.com/110079774/212449174-fce946ed-2589-4d1a-ab3c-bac06011d119.png)|
| - | - |
|**Before**|**After**|		         

Hence now the computation will be quite fast as compared to the original dataset.

Total number of null values in the dataset was only one, and it was removed. Dropped the **Id** column as it will be of no use in decision making.<br><br>

### ***matchType*:**

There are 16 match types as shown below with combinations of fpp, tpp, solo, duo, squad,etc.So we are generalising them into only solo, duo and squad.After that applying LabelEncoding to matchType column.

Mapping of Label Encoding: solo - 1 ; duo - 0 ; squad - 2

We will be using these encoding for the rest of our project work from now on.



### ***Some features and their behaviours:***
1) ***assists and kills***: <br>
Number of assists the player has done for the team and the number of kills a player has done.From the below graphs it can be seen the count of zeros is very high but still an important feature while determining the final rank.


|![image](https://user-images.githubusercontent.com/110079774/212449238-706b0842-1822-4488-bc0a-f65e526a2733.png)|![image](https://user-images.githubusercontent.com/110079774/212449242-474bd45b-7578-4530-a362-2685d3fbd5f6.png)|
| - | - |

2) ***roadKills and teamKills***: Roadkills indicate the number of people killed while travelling in a vehicle whereas TeamKills indicate the number of people killed by a team member within the same team. These features seem to be useless as it is highly unlikely that this will happen which can be proven from the figures below.

3) ***headshotKills and DBNOs***: Headshot kills indicate the number of kills done by the player with headshot and DBNOs indicate the number of enemies knocked by a player. These features are important as they indicate skill of a player which can be a good metric to judge the final placement prediction of the player.

|![image](https://user-images.githubusercontent.com/110079774/212449263-d96151be-67b7-4ac0-b111-dd4526ae89ce.png)|![image](https://user-images.githubusercontent.com/110079774/212449273-654f9669-71f1-442f-a3a2-3b51fc65d465.png)|
| - | - |

4) ***boosts and heals***: Boosts and Heals are the items which increase the health of the player in the game, boosts have an immediate effect whereas heals take longer time.However, both can be important features for further decision making.


|![image](https://user-images.githubusercontent.com/110079774/212449286-ca128109-2be8-40c8-b66b-7b66317ba635.png)|![image](https://user-images.githubusercontent.com/110079774/212449289-3a056d15-b794-4165-8c59-29c02f39185a.png)|
| - | - |


### ***Analysis on Dataset*:**

According to the data provided, in a match, people with the same groupId form a group and that group has the same target placement in that match. This was according to us one of the main challenges the model faced as for the same target value, it had different feature values, leading to confusion for model learning. So, to alleviate that, I decided to group the data points based on groupId and matchId and aggregate their feature values to be represented as one row for each group in the match.

So based on the idea mentioned above we thought of representing all the players in the same team as a one entity/player/team.Hence we reduced the dataset by grouping the rows based on groupId, and now each row will represent a team or an individual in the case of solo mode.

Now what about the aggregation of the other columns, so for that we have used sum, mean and max, for e.g:

- ***kills*:** We have taken the **sum** of the kills scored by all the teammates.
- ***killPlace:*** For the *killPlace* , we have taken the **mean** of that of all players.
- ***rideDistance:*** So for the ride distance we have taken **max** of that of all players in the same team.

So the idea behind the logic of which aggregation is applied to which columns is as follows:

- ***So basically the feature which describes any teamwork we will take sum of it ( e.g kills, assists).***

- ***If it's a scaling feature we’ll be taking the mean of it.***

- ***If the feature describes the quality of a player in a team we'll take max of it hence his/her team gets affected positively.***












Following table shows the columns and the corresponding aggregation function which is applied to it.


|***Columns***|***Functions***|***Columns***|***Functions***|
| :-: | :-: | :-: | :-: |
|*matchId*|*max*|*maxPlace*|*mean*|
|*assists*|*sum*|*numGroups*|*mean*|
|*boosts*|*sum*|*rankPoints*|*max*|
|*damageDealt*|*sum*|*matchType*|*mean*|
|*DBNOs*|*sum*|*revives*|*sum*|
|*headshotKills*|*sum*|*rideDistance*|*max*|
|*matchId*|*max*|*maxPlace*|*mean*|
| :-: | :-: | :-: | :-: |
|*assists*|*sum*|*numGroups*|*mean*|
|*boosts*|*sum*|*rankPoints*|*max*|
|*damageDealt*|*sum*|*matchType*|*mean*|
|*DBNOs*|*sum*|*revives*|*sum*|
|*headshotKills*|*sum*|*rideDistance*|*max*|
|*heals*|*sum*|*roadKills*|*sum*|
|*killPlace*|*mean*|*swimDistance*|*sum*|
|*killPoints*|*max*|*teamKills*|*sum*|
|*kills*|*sum*|*vehicleDestroys*|*sum*|
|*killStreaks*|*max*|*walkDistance*|*max*|
|*longestKill*|*mean*|*weaponsAcquired*|*sum*|
|*matchDuration*|*max*|*winPoints*|*max*|

![image](https://user-images.githubusercontent.com/110079774/212449334-971bb329-f59f-4a2c-986b-a6cb53c64cc2.png)

Here we have significantly reduced the dataset memory, but is it legit reducing the dataset in this way ? Lets see some plots and figure out:So we plotted the discrete features and found that the distribution was similar like the original distribution.

![image](https://user-images.githubusercontent.com/110079774/212449349-aa12299a-0720-409e-aa76-2465f57028aa.png)

![image](https://user-images.githubusercontent.com/110079774/212449350-f7a7a9a1-2569-48d3-acc6-ffbac0214b0c.png)


We also plotted the continuous features for both the original dataset and the reduced one and noticed that they were also similar. Let's have a look at it.
![image](https://user-images.githubusercontent.com/110079774/212449365-16da0682-26f7-4dda-b484-a3af933fa95f.png)

So here both distributions are looking similar, let's have a look on how the correlation of the columns with winPlacePerc is affected. 




Let's check the correlation of all the features with winPlacePerc before and after.

![image](https://user-images.githubusercontent.com/110079774/212449367-4eafc48b-b698-4a42-851d-933dadba92e8.png)

So as per the above table we can see there is not much of a difference between the original dataset correlation and the reduced dataset correlation of the features with winPlacePerc.

Hence from the above observation we are taking the Reduced\_GroupBy dataset into consideration for the further training purpose.


### ***Multivariate Analysis*:**

**1) *walkDistance | boosts | kills(size of points) | winPlacePerc:***

![image](https://user-images.githubusercontent.com/110079774/212449379-6e196189-ff3a-4c7c-8043-8206e862319a.png)

From the above graph, we can observe that as boosts consumption increases players chance of winning the match increases, also logically a player which has high chance of winning tends to be in fight and needs boost, also we can see walkDistance also matters in winning as it will be high for the player/team who has high chances of winning, because to be in the game, players have to be in safe zone for that they need to travel.

**2) *heals | boosts | damageDealt(size of points) | winPlacePerc:***

![image](https://user-images.githubusercontent.com/110079774/212449381-320ccae7-6f8a-4d83-b8cb-3d698169d527.png)

Here the above graph depicts that for high winPlacePerc, along with boosts and heals, the player having high damageDealt also has more tendency to have high winPlacePerc.

**3) *boosts and heals | winPlacePerc:***

![image](https://user-images.githubusercontent.com/110079774/212449392-350540bf-85ca-49bf-bc00-c82dfc1dffab.png)

From the above graph we can see Boosts and Heals show positive relation with winPlacePerc, Boosts shows more than Heal. Maybe we can do some stuff with both of these features later.

**4) *kills(matchType wise) | winPlacePerc:***

![image](https://user-images.githubusercontent.com/110079774/212449396-f8d458c2-2599-4e55-bc47-825bdda1fe04.png)

From the above graph we can say that as the number of kills increases chances of winning increases but it does not matter much as we go from match type from solo to squad, because in squad we have to play more strategically and focus is not much on kills in squad.


- ***Handling some Anomalies*:**

While analysing the dataset we found some irregularities in the dataset itself hence now we’ll try to handle those anomalies one by one. 

1. **Players have done kills without travelling any type of distance:**

![image](https://user-images.githubusercontent.com/110079774/212449412-62cfbc1e-4c1e-465e-abe8-691f5aa81682.png)

So the above graph is of the players who travel zero distance (distance = walk + ride + swim) yet they have killed enemies seems suspicious, hence removing those rows!!

2. **Longest Kill =0 metre, kill >0:**

![image](https://user-images.githubusercontent.com/110079774/212449408-b5028615-284c-4c44-ac05-44d2da11b580.png)

So here we can see the longest kill is zero metre yet there are some non-zero kills which is not possible logically, hence dropping those rows too!

3. **TeamKills and rideDistance:**

![image](https://user-images.githubusercontent.com/110079774/212449419-5b6781c2-1633-4fc0-ac3e-d0d136304d7a.png)

In pubg, a player can kill his/her team-mate only if he has a grenade(weapon) or he/she has driven a vehicle over his/her team-mate. But from the above graph there are some players who have killed teamplayer yet they have not acquired any weapon or drove a car/vehicle!

4. **roadKills and rideDistance:**

![image](https://user-images.githubusercontent.com/110079774/212449423-d31dd849-3695-4611-b1ca-d65751ca8062.png)


From the above graph, there are some players who have killed enemies while riding a car i.e roadKills, but those players have not rode any vehicles, hence dropping those rows too!

Similarly we have observed some more anomalies stated in the next page.

5. **Players have not walked but have consumed heals and boosts which is not possible hence dropping those rows!**
6. **It's not possible to acquire weapons if a player has not walked a distance.**
7. ` `**If matchType is solo then there cannot be any assists value, because to assist we need teammates which we don't have, as the numbers are somewhat high, so instead of dropping the rows, we imputed that feature with 0.**
8. **A player cannot assist a teammate if the walkDistance is 0.**
9. **A player cannot deal damage if he/she has not walked a single metre.**

Hence after performing the Data Pre-processing we reduced the original dataset’s size by a significant amount.

**Summary of dataset transition uptill now:**  
![image](https://user-images.githubusercontent.com/110079774/212449445-c30ccbac-26a8-4017-97cb-a139f28bf781.png)




###



## **2.3  *Feature Engineering:***

We tried adding new features in the system based on our knowledge of the game, those new features are as follows :

**1. killsPerMeter = kills / walkDistance**

**2. healsPerMeter = heals / walkDistance**

**3. totalHeals = heals + boosts**

**4. totalHealsPerMeter = totalHeals / walkDistance**

**5. totalDistance = walkDistance + rideDistance + swimDistance**

**6. headshotRate = headshotKills / kills**

**7. assistsAndRevives = assists + revives**

**8. itemsAcquired = heals + boosts + weaponsAcquired**

**9. healsOverBoosts =  heals / boosts**

**10. walkDistanceOverHeals = walkDistance / heals**

**11. walkDistanceAndHeals = walkDistance \* heals**

**12. walkDistanceOverKills = walkDistance / kills**

**13. walkDistanceAndKills = walkDistance \* kills**   

**14. boostsOverTotalDistance = boosts / totalDistance**

**15. boostsAndTotalDistance = boosts \* totalDistance**

After finding the correlation of these features with the target, they had a high correlation indicating these will be good features for learning.


# 3. Training Process:

## ***3.1  Models Used:***
We tried various models to train on the dataset which are the following:


1) **Linear Regression:**

As it is a simple model, comparisons can be made with respect to this model. Linear Regression is a statistical method to predict the relationship between an independent variable and a dependent variable. This problem dealt with the prediction of a predictor variable. In Linear Regression, the unknown function that maps the dependent variable to the independent variable has its model parameters estimated from the data. After fitting the linear Regression model, if additional data is provided to the model, it predicts the predictor variable automatically.

The model assumes to have a linear relationship in the following way,

![image](https://user-images.githubusercontent.com/110079774/212449459-86feb88c-30a1-426d-afcf-ed00ed72d6d6.png)

This is then solved using an ordinary least square solution wherein the parameters of the model are chosen to minimise the least square values between the predicted and the actual value of the predictor variable which is given as follows:

![image](https://user-images.githubusercontent.com/110079774/212449463-22444e56-1a13-4726-80af-46fa21071450.png)






2) **Ridge Regression:**

Ridge is an extension of the Ordinary Linear Regression wherein a regularizer term is added. The regularizer term is used to penalise the higher order weights and to increase the sparsity of weights in the model. Regularizer is used in the case of overfitting and the amount of regularisation to be added can be decided. A prior term is added when using Ridge Regression wherein the prior term for Ridge is Gaussian. In the given dataset, chances of overfitting were very low as the number of data points were extremely high compared to the number of features, but we still wanted to see if that MSE value changes after using a regularizer. The MAE values were exactly the same as that of the Ordinary Least Square solution indicating that the use of regularizer is not needed.


3) **Random Forest:**

Random Forest is one of the main models used for predictive modelling as it uses the ensemble model approach. As it is a non-linear model, I wanted to try this on my dataset and as expected the loss reduced after using Random forest. Random Forest as an ensemble model as multiple decision trees are built during training. During testing, the average of the decisions from multiple trees is taken and assigned to be the final predicted value. Random Forest is a strong learner which combines multiple Decision Trees i.e. weak learners to build the system. Random forest works by randomly sampling multiple subsets from the whole dataset with replacement.

This is called bagging. Due to this, the variance of the final model is reduced in turn leading to a consistent estimator.

4) **LightGBM**   

Light Gradient Boosting Method (LightGBM) is a gradient boosting method that uses a tree- based algorithm. Gradient Boosting is a method where weak learners are added to build a strong learner using gradient based approaches. The specialty of LightGBM is that it is a leaf-based algorithm compared to all other approaches which are level-based. In this method, the tree is grown on leaves and hence as the depth of the tree increases, the complexity of the model increases.

However, for large datasets LightGBM is extremely popular as it runs on high speed with large datasets and also requires lower memory to run. It focuses on decreasing the final accuracy thereby growing the tree on the leaf with maximum delta loss. It also supports GPU learning. For

smaller datasets, it might lead to overfitting but as the dataset I have used is very large, it works the best. However, as a lot of parameters are present, hyperparameter tuning is a bit cumbersome.

5) **XGBoost**

XGBoost is the abbreviation for eXtreme Boosting. This also uses the Gradient Boosting Decision Tree algorithm. Gradient Boosting is an approach where new models are added to the existing models to decrease the loss and the combined result from all these models is used as the final prediction. It uses the gradient descent algorithm to minimize the loss when adding new models. The execution time of the XGboost model is extremely small and it also uses the leaf-based tree growing. XGBoost is a very popular model used in Kaggle competitions due to it’s ability to handle large datasets.

###
## 3.2  Metric Used:
As we had multiple models, to identify the best model’s performance, we used Mean Squared Error (MSE) metric. 

Mean Squared Error is the measure of the square of the difference between actual value and the predicted value, average over all the datapoints.

![image](https://user-images.githubusercontent.com/110079774/212449482-6cbf4d2a-fd64-424f-9548-2999b513f25f.png)
###

### **3.3  Parameter Tuning:**
- Random Forest Parameter Tuning :

![image](https://user-images.githubusercontent.com/110079774/212449493-a4347db4-fa3e-4d93-bef5-499cfabcc580.png)

- XGBoost Hyper-parameter tuning:
![image](https://user-images.githubusercontent.com/110079774/212449498-a58b63ee-3cdd-4c8b-8d56-86188bac3870.png)

- LightGBM Hyper-parameter tuning:  

![image](https://user-images.githubusercontent.com/110079774/212449502-d963e0a7-d1bc-44e6-8408-9dcd0295fc8d.png)




### **3.4  Best Parameters:**


|Models|Parameters|MSE|
| :- | :- | :- |
|Linear Regression|n\_jobs=-1|0.012892|
|Ridge Regression|alpha=10, max\_iter=1000, solver='svd'|0.012892|
|Random Forest|<p>max\_depth=35, max\_features=None, min\_samples\_split=20,n\_estimators=95, n\_jobs=-1,oob\_score=True,                      `warm\_start=True,criterion="squared\_error"</p>|0.005542|
|XGBoost|gamma=0.0295,n\_estimators=125, max\_depth=15, eta=0.113, subsample=0.8, colsample\_bytree=0.8,               tree\_method='gpu\_hist',max\_leaves = 1250,reg\_alpha =0.0995,colsample\_bylevel = 0.8,num\_parallel\_tree =20|0.004973|
|LightGBM|<p>colsample\_bytree=0.8, learning\_rate=0.03, max\_depth=30, min\_split\_gain=0.00015, n\_estimators=250, num\_leaves=2200,reg\_alpha=0.1, reg\_lambda=0.001, subsample=0.8,</p><p>subsample\_for\_bin=45000, n\_jobs =-1, max\_bin =700, num\_iterations=5200, min\_data\_in\_bin = 12</p>|0.004829|


#
#
# **4. Conclusion:**
In this project, a variety of machine learning algorithms and models were experimented. As we have mentioned earlier, we found that the algorithm which works best for this dataset is where grouping of data points is done, and feature dimensions is increased by adding more features from this grouping and also some manual features. Also, LightGBM being fast and efficient for large datasets works the best.

# **5. References:**
Dataset - <https://www.kaggle.com/competitions/pubg-version-3/data>

Linear Regression - [Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

Random Forest Regressor - [Random Forest Regressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

XGBoost - <https://xgboost.readthedocs.io/en/stable/>

Light GBM - <https://lightgbm.readthedocs.io/en/v3.3.2/>

<https://pubg.fandom.com/wiki/PLAYERUNKNOWNS_BATTLEGROUNDS>
