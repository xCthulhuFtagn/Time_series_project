from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectPercentile, mutual_info_classif

class FeatureSelector():
    def __init__(self, method, y, X_endog, X_exog = None, model=None):
        self.model = model
        self.method = method
        self.X_endog = X_endog
        self.X_exog = X_exog
        self.y = y
        self.mdltype = type(model).__name__
        self.X = []
        if self.X_exog is not None:
            self.X = pd.concat([self.X_endog, self.X_exog], axis = 1)
            
        if method in ['sfs', 'mi']:
            self.method = method
        else: 
            raise KeyError
        if model is not None:
              self.model = model

    def select(self):
        sub_X_exog = []
        sub_X = []
        if self.method == 'sfs':
            if 'ARIMA' in self.mdltype:
                sub_X_exog = sequentialFeatureSelection(self.model, self.X_exog, self.y)
            else:
                sub_X = sequentialFeatureSelection(self.model, self.X, self.y)
        elif self.method == 'mi':
            sub_X = mutualInfoSelection(X, self.y)
        if self.X_exog is not None: return sub_X
        else: return sub_X_exog

#Данный метод выбран так как он позволяет не тюнить количество фичей которые мы должны выбрать, а использует трешхолд, основанный на MAE
def sequentialFeatureSelection(model, X, y):
    sfs = SFS(
      estimator = model,
      n_features_to_select="auto",
      tol=0.01,
      direction='backward',
      scoring='neg_mean_absolute_error'
    )
    sfs = sfs.fit(X, y)
    return sfs.get_feature_names_out()

#Данный метод выбран так как он позволяет исследовать нелинейные зависимости
def mutualInfoSelection(X, y):
    selector = SelectPercentile(mutual_info_regression, percentile=25)
    selector.fit_transform(X, y)
    cols = selector.get_support(indices=True)
    selected_columns = X.iloc[:,cols].columns.tolist()
    return selected_columns
   
#Функция которая проверят стабильность по разбиению данных по годам
def getStabilityOnYearsSplit(fs):
    y_years = splitDfByYears(fs.y)
    X_endog_years = splitDfByYears(fs.X_endog)
    X_exog_years = splitDfByYears(fs.X_exog)
    X_cols = fs.X.columns
    matrix = [ [] in range(len(y_years))]
    i = 0
    for year, val in zip(y_years.items(), X_endog_years.items(), X_exog_years.items()):
        selected_cols = FeatureSelector(fs.method, val[0], val[1], val[2], fs.model).select()
        matrix_input = create_binary_array_from_one(X_cols, selected_cols)
        matrix[i] = matrix_input
        i += 1
    return st.getStability(matrix)
    
def create_binary_array_from_one(cols, array1):
    col_index_map = {col: idx for idx, col in enumerate(cols)}
    
    binary_array = [0] * len(cols)
    
    set1 = set(array1)
    
    for col in cols:
        if col in set1:
            binary_array[col_index_map[col]] = 1
    
    return binary_array
    
def splitDfByYears(df):
    dfs = {}
    for year, grouped_df in df.groupby(df.index.year):
        dfs[year] = grouped_df
    return dfs
