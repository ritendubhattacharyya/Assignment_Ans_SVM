################### Question 1 #############################
import pandas as pd

sal_train = pd.read_csv("D:\\360DigiTMG\\DataScience\\31. SVM\\Assignment Q\\SalaryData_Train (1).csv")
sal_test = pd.read_csv("D:\\360DigiTMG\\DataScience\\31. SVM\\Assignment Q\\SalaryData_Test (1).csv")


sal_train.head()


sal_train_obj = sal_train.select_dtypes("object")
sal_train_obj.head()


sal_train_num = sal_train.select_dtypes(["int64", "float64"])
sal_train_num


sal_train_y = sal_train_obj.Salary
sal_train_obj.drop(['Salary'], axis=1, inplace=True)
sal_train_y


from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()

sal_train_obj.workclass = pd.Series(lbl.fit_transform(sal_train_obj.workclass))
sal_train_obj.education = pd.Series(lbl.fit_transform(sal_train_obj.education))
sal_train_obj.maritalstatus = pd.Series(lbl.fit_transform(sal_train_obj.maritalstatus))
sal_train_obj.occupation = pd.Series(lbl.fit_transform(sal_train_obj.occupation))
sal_train_obj.relationship = pd.Series(lbl.fit_transform(sal_train_obj.relationship))
sal_train_obj.race = pd.Series(lbl.fit_transform(sal_train_obj.race))
sal_train_obj.sex = pd.Series(lbl.fit_transform(sal_train_obj.sex))
sal_train_obj.native = pd.Series(lbl.fit_transform(sal_train_obj.native))

sal_train_obj


sal_train = pd.concat([sal_train_obj, sal_train_num], axis=1)
sal_train


sal_test_y = sal_test.Salary
sal_test.drop(["Salary"], axis=1, inplace=True)
sal_test 


sal_test_obj = sal_test.select_dtypes("object")
sal_test_num = sal_test.select_dtypes(["int64", "float64"])


sal_test_obj.workclass = pd.Series(lbl.fit_transform(sal_test_obj.workclass))
sal_test_obj.education = pd.Series(lbl.fit_transform(sal_test_obj.education))
sal_test_obj.maritalstatus = pd.Series(lbl.fit_transform(sal_test_obj.maritalstatus))
sal_test_obj.occupation = pd.Series(lbl.fit_transform(sal_test_obj.occupation))
sal_test_obj.relationship = pd.Series(lbl.fit_transform(sal_test_obj.relationship))
sal_test_obj.race = pd.Series(lbl.fit_transform(sal_test_obj.race))
sal_test_obj.sex = pd.Series(lbl.fit_transform(sal_test_obj.sex))
sal_test_obj.native = pd.Series(lbl.fit_transform(sal_test_obj.native))

sal_test_obj


sal_test = pd.concat([sal_test_obj, sal_test_num], axis=1)
sal_test

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
sal_train = pd.DataFrame(mms.fit_transform(sal_train))
sal_test = pd.DataFrame(mms.fit_transform(sal_test))


from sklearn.svm import SVC

ml1 = SVC(kernel="linear")
ml1.fit(sal_train, sal_train_y)
ml1.predict(sal_test)

import numpy as np
np.mean(ml1.predict(sal_test) == sal_test_y)


ml2 = SVC(kernel='rbf')
ml2.fit(sal_train, sal_train_y)
ml2.predict(sal_test)

np.mean(ml2.predict(sal_test) == sal_test_y)




###################### Question 2 #########################
import pandas as pd

forestfires = pd.read_csv("D:\\360DigiTMG\\DataScience\\31. SVM\\Assignment Q\\forestfires.csv")

y = forestfires.size_category

forestfires.drop(['size_category'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()

forestfires.month = pd.Series(lbl.fit_transform(forestfires.month))
forestfires.day = pd.Series(lbl.fit_transform(forestfires.day))

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

forestfires = pd.DataFrame(mms.fit_transform(forestfires))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(forestfires, y, test_size=0.2)

from sklearn.svm import SVC

ml1 = SVC(kernel='linear')
ml1.fit(X_train, y_train)
predict1 = ml1.predict(X_test)

import numpy as np

np.mean(predict1 == y_test)

ml2 = SVC(kernel='rbf')
ml2.fit(X_train, y_train)
predict2 = ml2.predict(X_test)

np.mean(predict2 == y_test)