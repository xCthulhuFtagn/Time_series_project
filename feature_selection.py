from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectPercentile, mutual_info_classif

class FeatureSelector():
    def __init__(self, method, y, X_endog, X_exog = None, model=None):
        self.X_exog = X_exog
        self.y = y
        self.mdltype = type(model).__name__
        if method in ['sfs', 'mi']:
            self.method = method
        else: raise KeyError
        if model is not None:
              self.model = model

    def select(self):
        sub_X_exog = []
        sub_X = []
        if self.X_exog is not None:
            X = pd.concat([self.X_endog, self.X_exog], axis = 1)
        if self.method == 'sfs':
            if 'ARIMA' in self.mdltype:
                sub_X_exog = sequentialFeatureSelection(self.model, self.X_exog, self.y)
            else:
                sub_X = sequentialFeatureSelection(self.model, X, self.y)
        elif self.method == 'mi':
            sub_X = mutualInfoSelection(X, self.y)
        if self.X_exog is not None: return sub_X
        else: return sub_X_exog

def sequentialFeatureSelection(model, X, y):
    sfs = SFS(
      estimator = model, 
      n_features_to_select="auto",
      tol=0.15,
      direction='backward',
      scoring='neg_mean_absolute_error'
    )
    sfs = sfs.fit(X, y)
    return sfs.get_feature_names_out()


def mutualInfoSelection(X, y):
    selector = SelectPercentile(mutual_info_classif, percentile=25)
    selector.fit_transform(X, y)
    cols = selector.get_support(indices=True)
    selected_columns = X.iloc[:,cols].columns.tolist()
    return selected_columns
