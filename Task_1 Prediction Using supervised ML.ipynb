{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "bdb80a4e",
   "metadata": {},
   "source": [
    "# The Sparks Foundation - Data Science & Business Analytics Internship"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce238b7d",
   "metadata": {},
   "source": [
    "## GRIP @ The Sparks Foundation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2f10e46",
   "metadata": {},
   "source": [
    "### By : Ayush Mondal "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e03398a1",
   "metadata": {},
   "source": [
    "## TASK 1 - Prediction using supervised Machine Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f16d5967",
   "metadata": {},
   "source": [
    "In this task it is require to build the model to predict the score if the student studies 9.5 hrs/day.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bc09923",
   "metadata": {},
   "source": [
    "### Steps :"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71022efa",
   "metadata": {},
   "source": [
    "<ul>\n",
    "<li>Import libraries</li>\n",
    "<li>Read the data</li>\n",
    "<li>Explore the data</li>\n",
    "<li>visualize the data</li>\n",
    "<li>OLS model using statsmodels library</li>\n",
    "<li>Make the prediction</li>\n",
    "<li>Evaluate the model</li>\n",
    "<li>predict the score if the student studies 9.5 hrs/day</li>    \n",
    "</ul>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6084794",
   "metadata": {},
   "source": [
    "## Import libraries\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8fe59857",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import statsmodels.api as sm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f3d314b",
   "metadata": {},
   "source": [
    "## Read the data \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2b3f2bfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0c9afdc8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f004508",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "54d64fa1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "09c22680",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "82f2e9e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Hours     0\n",
       "Scores    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for null values\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "18d986a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Hours</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.976191</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Scores</th>\n",
       "      <td>0.976191</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours    Scores\n",
       "Hours   1.000000  0.976191\n",
       "Scores  0.976191  1.000000"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# corrolation \n",
    "data.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4a964c05",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x1b90524b848>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x=data['Hours'],y=data['Scores'],color = 'black',marker='*')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e7225a06",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=data['Scores']\n",
    "X=data['Hours']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "eeeabd70",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = X.values.reshape(-1,1)\n",
    "y = y.values.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "749f87dd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Hours', ylabel='Scores'>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visalize the model \n",
    "sns.regplot(x=\"Hours\", y=\"Scores\", data=data)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae6c1812",
   "metadata": {},
   "source": [
    "## linear regression model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "fd3af232",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)\n",
    "model = LinearRegression()\n",
    "model.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0f3e4252",
   "metadata": {},
   "source": [
    "## Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6b02e886",
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict the model\n",
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "198b16a4",
   "metadata": {},
   "source": [
    "## Evaluate the model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e3fb866e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the train data accuracy = 0.9491209376364416\n",
      "the test data accuracy  = 0.9678055545167994\n"
     ]
    }
   ],
   "source": [
    "print('the train data accuracy =',model.score(X_train,y_train))\n",
    "print('the test data accuracy  =',model.score(X_test,y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "bda3e907",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9678055545167994"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ec408aa",
   "metadata": {},
   "source": [
    "## OLS Regression "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "f15fc03e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.953</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.951</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   465.8</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Sun, 06 Mar 2022</td> <th>  Prob (F-statistic):</th> <td>9.13e-17</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>12:12:32</td>     <th>  Log-Likelihood:    </th> <td> -77.514</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>    25</td>      <th>  AIC:               </th> <td>   159.0</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>    23</td>      <th>  BIC:               </th> <td>   161.5</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th> <td>    2.4837</td> <td>    2.532</td> <td>    0.981</td> <td> 0.337</td> <td>   -2.753</td> <td>    7.721</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>x1</th>    <td>    9.7758</td> <td>    0.453</td> <td>   21.583</td> <td> 0.000</td> <td>    8.839</td> <td>   10.713</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td> 7.616</td> <th>  Durbin-Watson:     </th> <td>   1.460</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.022</td> <th>  Jarque-Bera (JB):  </th> <td>   2.137</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td>-0.216</td> <th>  Prob(JB):          </th> <td>   0.343</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 1.634</td> <th>  Cond. No.          </th> <td>    13.0</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:                      y   R-squared:                       0.953\n",
       "Model:                            OLS   Adj. R-squared:                  0.951\n",
       "Method:                 Least Squares   F-statistic:                     465.8\n",
       "Date:                Sun, 06 Mar 2022   Prob (F-statistic):           9.13e-17\n",
       "Time:                        12:12:32   Log-Likelihood:                -77.514\n",
       "No. Observations:                  25   AIC:                             159.0\n",
       "Df Residuals:                      23   BIC:                             161.5\n",
       "Df Model:                           1                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const          2.4837      2.532      0.981      0.337      -2.753       7.721\n",
       "x1             9.7758      0.453     21.583      0.000       8.839      10.713\n",
       "==============================================================================\n",
       "Omnibus:                        7.616   Durbin-Watson:                   1.460\n",
       "Prob(Omnibus):                  0.022   Jarque-Bera (JB):                2.137\n",
       "Skew:                          -0.216   Prob(JB):                        0.343\n",
       "Kurtosis:                       1.634   Cond. No.                         13.0\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = sm.add_constant(X)\n",
    "results = sm.OLS(y,x).fit()\n",
    "results.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15d05afd",
   "metadata": {},
   "source": [
    "## the predicted score if the student studies 9.5 hrs/day"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e8c45edd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "If the student study 9.5 the score will be [[94.80663482]]\n"
     ]
    }
   ],
   "source": [
    "num_of_houres = 9.5\n",
    "score = model.predict([[num_of_houres]])\n",
    "print('If the student study {} the score will be {}'.format(num_of_houres,score))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
