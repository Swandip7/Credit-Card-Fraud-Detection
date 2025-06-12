import pandas as pd
from imblearn.over_sampling import SMOTE

def undersample(df, target_col='Class', random_state=1):
    df_shuffled = df.sample(frac=1, random_state=random_state)
    df_1 = df_shuffled.loc[df_shuffled[target_col] == 1]
    df_0 = df_shuffled.loc[df_shuffled[target_col] == 0][:len(df_1)]
    balanced_df = pd.concat([df_1, df_0], axis=0).reset_index(drop=True)
    return balanced_df.sample(frac=1, random_state=random_state)

def oversample(X, y, random_state=1):
    smote = SMOTE(sampling_strategy='minority', random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
