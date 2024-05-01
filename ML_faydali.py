#ML faydalı:

# .csv okuma
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import statsmodels as sm

df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

# değişken tiplerini sırala
df_train.dtypes

# kategorik değişkenleri bul.
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

# kategorik değişkenleri çıkart.
drop_X_train = X_train.select_dtypes(exclude=['object'])

# kategorik NA değerleri doldur
df_train[categorical_columns] = df_train[categorical_columns].fillna(mode)

# My First ML Pipeline ##################################
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

X, y = make_classification(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([('imputer', SimpleImputer(strategy=)), \
                 ('scaler', StandardScaler()), ('svc', SVC())])
#pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)
# End My First ML Pipeline ##################################


# feature creation #################################################
autos["stroke_ratio"] = autos.stroke / autos.bore
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)
# feature creation #################################################

# feature deletion #################################################
newdf = df.drop("age", axis='columns')
df_valid = customer.drop(df_train.index)
# feature deletion #################################################

# feature stat. functions ##########################################
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)
customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)
# feature stat. functions ##########################################

# describe the data
df.describe()

# gives the columns types of data
df_train.dtypes

# check for missing values
for i in df_train.columns:
    print(i, df_train[i].isna().sum())

# boş kategorik değerleri doldurma #####################################
# Replacing categorical columns with mode
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = df_train[categorical_columns].mode().iloc[0]
# Replace NaN values in specific columns only
df_train[categorical_columns] = df_train[categorical_columns].fillna(mode)
# boş kategorik değerleri doldurma #####################################

# boş numerik değerleri doldurma #####################################
# Replacing Numerical columns with their median
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = df_train[numerical_columns].median()
df_train[numerical_columns] = df_train[numerical_columns].fillna(median)
# boş numerik değerleri doldurma #####################################

# We don't need the Name column so we can drop it
df_train = df_train.drop(columns = ['Name'])

# correlation matrisi oluşturma #####################################
import seaborn as sns
import matplotlib.pyplot as plt
corr = df_train.corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, ax=ax)
plt.show()
# correlation matrisi oluşturma #####################################

# bazı kolonları tek kolona sepetleme #####################################
# Classify the Age
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']

# Create a new column with the age categories
df_train['Age Group'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)
# bazı kolonları tek kolona sepetleme #####################################

# Kolondaki farklı eleman sayısını bul
df_train['Deck'].unique()

# scale edelim
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df_train[['Age', 'Expenses']]=ss.fit_transform(df_train[['Age', 'Expenses']])

# train öncesi düzenleme
X_Train = df_train.drop('Transported',axis=1)
Y_Train = df_train['Transported']

# One hot encoding uygulanmasi
my_cols = low_cardinality_cols + num_cols
predictors = hotels[my_cols]
ohe_predictors = pd.get_dummies(predictors)

#----------------------------------------------------------
# Numerk düzenleyici ve kategorik düzenleyicileri toparlama
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
#----------------------------------------------------------

# Tf hizlandirici decorator
@tf.function
def f(x, y):
  return x ** 2 + y
x = tf.constant([2, 3])
y = tf.constant([3, -2])
f(x, y)

# qcut Eray hoca'dan
# Bunu koyduğumuzda bin sınırlarında tekrar eden değerlere gelen hatayı önler!
deneme.rank()
pd.qcut(deneme[0].rank(method='first'), 4, labels=["D","C","B","A"], duplicates="drop")

# kosullu secim
df[df['age'] > 30]
df.loc[(df['degisken1'] == 'BA') & (df['degisken2']< 0.5)]
#GENEL KÜLTÜR dataframe ilk olarak R da çıkmıstır

#--------------------------------------------------------------------------------------------------------------------
# Outlier dan kurtulma
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
#--------------------------------------------------------------------------------------------------------------------
# Time series
def is_stationary(y):

    # "H0: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1] # import statsmodels as sm
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value,3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

