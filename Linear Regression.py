import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# reading data from csv file

data = pd.read_csv('realestate.csv')

# choice of features and target variable

X = data[['size', 'rooms', 'floor']]
y = data['price']

# splitting data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training model on training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# printing the coefficients
print(regressor.coef_)

# predicting the prices

y_pred = regressor.predict(X_test)

# printing the predicted prices

df = pd.DataFrame({'Current prices': y_test, 'Predicted prices': y_pred})
print(df)

# Calculating RMSE (współczynnik determinacji R2) on the training set
# Współczynnik R2, również nazywany współczynnikiem determinacji, określa,
# jak dobrze model dopasowuje się do danych.
# Współczynnik R2 przyjmuje wartości od 0 do 1, gdzie 0 oznacza, że model nie
# tłumaczy zmienności celu, a 1 oznacza, że model doskonale tłumaczy zmienność celu.
# Formalnie, współczynnik R2 obliczany jest jako stosunek wyjaśnionej zmienności
# (sumy kwadratów różnic między predykcjami a średnią wartością celu) do całkowitej zmienności
# (sumy kwadratów różnic między wartościami rzeczywistymi a średnią wartością celu).
# Wartości R2 w zakresie od 0 do 1 oznaczają, jak wiele zmienności celu jest wyjaśnione przez model.
# W praktyce, wysoka wartość współczynnika R2 wskazuje, że model
# dobrze dopasowuje się do danych i przewiduje wartości celu z dużą dokładnością.
# Jednakże, warto zauważyć, że R2 może być mylący w przypadku, gdy modele różnią się pod względem
# liczby zmiennych lub zakresów wartości cech. W takich przypadkach, warto skorzystać z innych metryk,
# takich jak błąd średniokwadratowy (MSE) czy średni błąd absolutny (MAE), aby dokładniej ocenić jakość modelu.

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('Współczynnik determinacji R2: %.2f' %r2)



