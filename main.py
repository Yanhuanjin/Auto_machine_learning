import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import load_data
import process_data

def main():
    # load data
    loader = load_data.LoadData()
    origin_data = loader.load_data("rossmann-train.csv")
    # process data
    target = "Sales"
    processor = process_data.Processor(origin_data, target)
    # Fill the N/A values
    for col in origin_data.columns:
        if origin_data[col].isnull().sum():
            print(">> Filling N/A: %s" % col),
            processor.fill_null(col, "bfill")
        else:
            pass

    # transfer the date_col
    processed_data = processor.date_transfer("Date", "year")

    # delete uni cols
    drop_list = list()
    # delete useless features
    no_use_feature = []
    if len(no_use_feature)>0:
        for i in no_use_feature:
            drop_list.append(i)
        drop_list.append(processor.drop_uni())
    if len(drop_list)>0:
        processed_data = processed_data.drop(drop_list, axis=1)
    else:
        processed_data = processed_data

    # split the data to target and features
    y = processed_data.pop(target)
    X = processed_data

    # Label encoding
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == object:
            print(">> Label Encoding: %s" % col)
            X[col] = label_encoder.fit_transform(X[col].astype(str))
    feature_list = list()
    for col in X.columns:
        feature_list.append(col)

    # split the data to train data and test data
    score_list = list()
    model_list =list()
    importance_list = list()
    for i in range(10):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
        print("\n【第%s次测试】" % (i + 1))
		
        model = RandomForestRegressor(n_estimators=100).fit(train_x, train_y)
        predict = model.predict(test_x)
        score_predict = r2_score(test_y, predict)
        importance = model.feature_importances_
        print("R^2 score value: %s" % score_predict)
		
        score_list.append(score_predict)
        model_list.append(model)
        importance_list.append(importance)

    # 保存分数最高的模型
    best_model = model_list[score_list.index(max(score_list))]
    joblib.dump(best_model, "best_rfr_train_model.m")
    sum_score = 0
    for i in score_list:
        sum_score += float(i)
    avg_score = sum_score / len(score_list)
    best_score = max(score_list)
	
    print("平均测试分数为:%s" % avg_score)
    print("最高测试分数为:%s" % best_score)
    print("最好的模型是:%s" % best_model)
	
    best_importance = importance_list[score_list.index(best_score)]

    # draw feature importance
    font1 = {'family': 'Times New Roman','weight': 'normal','size': 23,}
    plt.figure(figsize=(10, 5))
    plt.bar(feature_list, best_importance)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()

if __name__ == "__main__":
    main()