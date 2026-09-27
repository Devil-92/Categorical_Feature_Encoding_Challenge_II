import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing


def run(fold):

    df = pd.read_csv('../input/cat_train_folds.csv')

    features = [
        f for f in df.columns if f not in ('id', 'kfold', 'target')
    ]

    for f in features:
        df[f] = df[f].astype(str).fillna('NONE')

    train_df = df[df.kfold != fold].reset_index(drop=True)

    valid_df = df[df.kfold == fold].reset_index(drop=True)

    full_data = pd.concat(
        [train_df, valid_df],
        axis=0
    )

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(full_data[col])
        train_df[col] = lbl.transform(train_df[col])
        valid_df[col] = lbl.transform(valid_df[col])

    x_train = train_df[features].values
    x_valid = valid_df[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200
    )

    model.fit(x_train, train_df.target.values)

    valid_pred = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(valid_df.target.values, valid_pred)

    print(f"AUC for fold number {fold} is {auc}")


if __name__ == '__main__':
    for f_ in range(5):
        run(f_)