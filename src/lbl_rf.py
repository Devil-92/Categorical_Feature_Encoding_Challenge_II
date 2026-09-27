import pandas as pd
from sklearn import metrics
from sklearn import ensemble
from sklearn import preprocessing

def run(fold) :

  df = pd.read_csv('../input/cat_train_folds.csv')

  features = [
    f for f in df.columns if f not in ('id' , 'kfold' , 'target')
  ]

  for f in features :
    df[f] = df[f].astype(str).fillna('NONE')

  for col in features :
    lbl = preprocessing.LabelEncoder()

    lbl.fit(df[col])

    df[col] = lbl.transform(df[col])

  df_train = df[df.kfold != fold].reset_index(drop = True)

  df_valid = df[df.kfold == fold].reset_index(drop = True)

  x_train = df_train[features].values

  x_valid = df_valid[features].values

  model = ensemble.RandomForestClassifier(n_jobs = -1)

  model.fit(x_train , df_train.target.values)

  valid_pred = model.predict_proba(x_valid)[: , 1]

  auc = metrics.roc_auc_score(df_valid.target.values , valid_pred)

  print(f"AUC for the fold Number {fold} is {auc}")

if __name__ == '__main__' :
  for f_ in range(5) :
    run(f_)