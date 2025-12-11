import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("user1_food_data.csv")

#map qualatative data points to integers 

course_map = {
    'Main Dish': 0,
    'Side Dish': 1,
    'Dessert': 2,
    'Drink': 3
    }

eat_map = {
    'YES': 1,
    'NO': 0
    }

flavor_map = {
    'Umami': 0,
    'Sweet': 1,
    'Salty': 2,
    'Sour': 3,
    'Bitter': 4,
    'Not Available': 5
    }

category_map = {
    'Beef': 0,
    'Beverage': 1,
    'Burrito': 2,
    'Chips': 3,
    'Condiment': 4,
    'Dairy': 5,
    'Noodles': 6,
    'Pastry': 7,
    'Pizza': 8,
    'Salad': 9,
    'Sandwich': 10,
    'Seafood': 11,
    'Soup': 12
    }

culture_map = {
    'American': 0,
    'Asian': 1,
    'Italian': 2,
    'Mexican': 3
    }

protein_map = {
    'Chicken': 0,
    'Pork': 1,
    'Fish': 2,
    'Beef': 3,
    'Peanuts': 4,
    'Milk': 5,
    'Not Available': 6
    }

carbohydrate_map = {
    'Noodle': 0,
    'Rice': 1,
    'Tortilla': 2,
    'Bread': 3,
    'Not Available': 4
    }

#update data model by mapping integers

df['Course'] = df['Course'].map(course_map)
df['Eat'] = df['Eat'].map(eat_map)
df['Flavor'] = df['Flavor'].map(flavor_map)
df['Category'] = df['Category'].map(category_map)
df['Culture'] = df['Culture'].map(culture_map)
df['Protein'] = df['Protein'].map(protein_map)
df['Carbohydrate'] = df['Carbohydrate'].map(carbohydrate_map)

features = ['Course', 'Flavor', 'Category', 'Culture', 'Protein', 'Carbohydrate']

X = df[features]
y = df['Rating']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

#test prediction with sample food prompt

print(dtree.predict([[0, 0, 6, 0, 3, 0]]))
