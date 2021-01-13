import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

training = pd.read_csv('./train.csv')
testing = pd.read_csv('./test.csv')

training.head()
training.isnull().sum()

num_children = len(training.loc[training.Age < 18])
num_adults = len(training.loc[training.Age >= 18])

print('Total number of children:', num_children)
print('Total number of adults:', num_adults)

child_survivors = training.loc[training.Age < 18].Survived.sum()
adult_survivors = training.loc[training.Age >= 18].Survived.sum()

print('Child Survivors:', child_survivors)
print('Adult Survivors:', adult_survivors)

male_child_survivors = training.loc[(training.Age < 18) & (training.Sex == 'male')].Survived.sum()
female_child_survivors = training.loc[(training.Age < 18) & (training.Sex == 'female')].Survived.sum()
male_adult_survivors = training.loc[(training.Age >= 18) & (training.Sex == 'male')].Survived.sum()
female_adult_survivors = training.loc[(training.Age >= 18) & (training.Sex == 'female')].Survived.sum()

print('Male Children survivors:', male_child_survivors)
print('Female Children survivors:', female_child_survivors)
print('Male Adults survivors:', male_adult_survivors)
print('Female Adults survivors:', female_adult_survivors)

num_first = len(training[training.Pclass == 1])
num_second = len(training[training.Pclass == 2])
num_third = len(training[training.Pclass == 3])
first_survivors = training[training.Pclass == 1].Survived.sum() * 100 / num_first
second_survivors = training[training.Pclass == 2].Survived.sum() * 100 / num_second
third_survivors = training[training.Pclass == 3].Survived.sum() * 100 / num_third

print('First class survivors:', first_survivors)
print('Second class survivors:', second_survivors)
print('Third class survivors:', third_survivors)

training.Age.fillna(training.Age.mean(), inplace=True)
training.replace({"male": 1, "female": 0}, inplace=True)

reduced_features = training[['Pclass', 'Sex', 'Age', 'SibSp']]
targets = training['Survived']

X_train, X_validation, y_train, y_validation = train_test_split(
    reduced_features,
    targets,
    train_size=0.7,
    test_size=0.3
)

knn_model = KNeighborsClassifier(5)
knn_model.fit(X_train, y_train)
predicted_knn_survival = knn_model.predict(X_validation)
knn_f1_score = f1_score(predicted_knn_survival, y_validation)
knn_accuracy_score = accuracy_score(predicted_knn_survival, y_validation)

print('F1 score:', knn_f1_score)
print('Accuracy score:', knn_accuracy_score)

svc_model = SVC(kernel='linear', C=1)
svc_model.fit(X_train, y_train)
predicted_svc_survival = svc_model.predict(X_validation)
svc_f1_score = f1_score(predicted_svc_survival, y_validation)
svc_accuracy_score = accuracy_score(predicted_svc_survival, y_validation)

print('F1 score:', svc_f1_score)
print('Accuracy score:', svc_accuracy_score)

decision_model = DecisionTreeClassifier(max_depth=4)
decision_model.fit(X_train, y_train)
predicted_decision_survival = decision_model.predict(X_validation)
decision_f1_score = f1_score(predicted_decision_survival, y_validation)
decision_accuracy_score = accuracy_score(predicted_decision_survival, y_validation)

print('F1 score:', decision_f1_score)
print('Accuracy score:', decision_accuracy_score)
