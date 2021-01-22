import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

House = pd.read_csv('data/House_Prices.csv')
SalePrice = House["SalePrice"].values
Features = House.drop("SalePrice", axis=1)
X = Features.values

# ამ შემთხვევაში k ცვლადი იქნება 5-ის ტოლი
house_test = SelectKBest(score_func=chi2, k=5)
selected_features = house_test.fit_transform(X, SalePrice)
print(house_test.scores_)

# ლოგიკური ინდექსები
print(house_test.get_support())

print(Features.columns[house_test.get_support()])
