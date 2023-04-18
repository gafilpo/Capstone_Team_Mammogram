import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

class prepare_df():
    def __init__(self, df, target_var='cancer', dummy_cols=[], data_dir='', test_size=0.2, splits=5, random_state=42):
        self.df = df
        self.target_var = target_var
        self.dummy_cols = dummy_cols
        self.data_dir = data_dir
        self.test_size = test_size
        self.splits = splits
        self.random_state = 42


    def preprocess_df(self):
        # Drop rows if no cancer status available, and create dummy variables
        df = self.df.dropna(subset=self.target_var)

        df = pd.get_dummies(df, columns=self.dummy_cols)
        
        # We encode the path of the image in the dataframe to read the images later
        df['path'] = df.apply(lambda x: os.path.join(self.data_dir, str(x['patient_id']), str(x['image_id']) + '.png').encode('utf-8'), axis=1)

        # Save our updated df in the class object
        self.df = df 

        return df 

    def train_test(self):
        # Create test and train splits
        #As we want every patient to either have all the images in the train or test, we groupby patients and only after the splits rebuild the df
        
        patients_df = self.df[['patient_id', self.target_var]].groupby('patient_id').agg('sum')
        patients_df[self.target_var] = np.where(patients_df[self.target_var]>=1, 1, 0)

        X = patients_df.index
        y = patients_df[self.target_var]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, 
                                                            random_state=self.random_state, shuffle=True,
                                                            stratify=y)

        # Build train df and shuffle it
        df_train = self.df[self.df['patient_id'].isin(X_train)]
        df_train = df_train.sample(frac=1, random_state=self.random_state)
        # Build train df and shuffle it
        df_test = self.df[self.df['patient_id'].isin(X_test)]
        df_test = df_test.sample(frac=1, random_state=self.random_state)

        # Save the dataframes in the class object
        self.df_train = df_train
        self.df_test = df_test

    def KFold(self, balanced=False):

        # Create the stratified K fold splits
        if not hasattr(self, 'df_test'):
            print('Test/train split still not executed.')
            print('Please run .train_test() method first!')
        
        else:

            #As we want every patient to either have all the images in the train or validate, we groupby patients and only after the splits rebuild the df
            patients_df = self.df_train[['patient_id', self.target_var]].groupby('patient_id').agg('sum')
            patients_df[self.target_var] = np.where(patients_df[self.target_var]>=1, 1, 0)

            X = patients_df.index
            y = patients_df[self.target_var]

            dfs = {}

            # If we want only 1 split (for ex during testing the final model we only want 1 train, 1 val, and 1 test df)
            if self.splits == 1:
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, 
                                                                    random_state=self.random_state, shuffle=True,
                                                                    stratify=y)

                train = self.df_train[self.df_train['patient_id'].isin(X_train)]

                val = self.df_train[self.df_train['patient_id'].isin(X_val)]
                # Shuffle val df
                val = val.sample(frac=1, random_state=self.random_state)

                if balanced:
                    #balance train dataset
                    total_positive = len(train[train[self.target_var] == 1])
                    train = train.groupby(self.target_var, group_keys=False).apply(lambda x: x.sample(n=total_positive, random_state=self.random_state))

                # Shuffle train df
                train = train.sample(frac=1, random_state=self.random_state)

                dfs[0] = {'train': train, 'validate': val, 'test': self.df_test}

            # Otherwise do a normal stratified fold
            else:                   
                skf = StratifiedKFold(n_splits=self.splits, random_state=self.random_state, shuffle=True)

                # dfs will be a dict of indices, with key equal to the fold number, and a dict of train, validate, test dataframes
                dfs = {}

                for i, (X_train, X_val) in enumerate(skf.split(X, y)):
                    train = self.df_train[self.df_train['patient_id'].isin(patients_df.iloc[X_train].index)]
                    val = self.df_train[self.df_train['patient_id'].isin(patients_df.iloc[X_val].index)]
                    # Shuffle val df
                    val = val.sample(frac=1, random_state=self.random_state)

                    # Balance train dataset if required
                    if balanced:
                        total_positive = len(train[train[self.target_var] == 1])
                        train = train.groupby(self.target_var, group_keys=False).apply(lambda x: x.sample(total_positive, random_state=self.random_state))

                    # Shuffle train df
                    train = train.sample(frac=1, random_state=self.random_state)

                    dfs[i] = {'train': train, 'validate': val, 'test': self.df_test}

            return dfs