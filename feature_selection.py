class FeatureSelector():
    def __init__(self, X_endog, X_exog = None, y, method, model=None):
        self.X_endog = X_endog
        self.X_exog = X_exog
        self.y = y
        if method in ['method1', 'method2', 'method3'] :
            # тут как раз все нужные методы, что выберем экспериментами
            self.method = method
        else: raise KeyError
        if model is not None: 
            # мб некоторый метод отбора подразумевает работу с самой моделью
            #  добавь проверку на соответствие метода
            self.model = model
            
    def select(self):
        if self.method == 'method1':
            pass
        elif self.method == 'method2':
            pass
        elif self.method == 'method3':
            pass
        if self.X_exog is not None: return sub_X_endog, sub_X_exog, model
        else: return sub_X_endog, model
    