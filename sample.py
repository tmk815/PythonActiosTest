#scikit-learnライブラリの読み込み
from sklearn import datasets,svm

#scilit-learnライブラリに付属しているデータセット"digits"を利用
digits = datasets.load_digits()

#データセット"digits"のうち、0番目のデータを8×8の行列に整形
print("-----digits['data'][0].reshape(8,8)----")
print(digits['data'][0].reshape(8,8))

#データセット"digits"の0番目のデータは、数字の0
print("-----digits.target----")
print(digits.target[0])

#サポートベクターマシン（というもの）を使って、分類
clf = svm.SVC()
clf.fit(digits.data,digits.target)

#仮に下記の"test_data"のようなデータは、どの数字に該当するか？を予測
test_data = [[0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,0,1,2,3,4]]

print("-----予測----")
print(clf.predict(test_data))