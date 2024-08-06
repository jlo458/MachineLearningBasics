# Finding gradient of regression line using Turi Create

import turicreate as tc

data = tc.SFrame("/home/jlotheboss/new_project/myenv//Hyderabad.csv")

model = tc.linear_regression.create(data, target="Price")

house = tc.SFrame({'Area': [3000], 'No. of Bedrooms':[6]})
print(model.predict(house))

