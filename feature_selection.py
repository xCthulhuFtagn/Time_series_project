class FeatureSelector():
    def __init__(self, X, y, method, model=None):
        self.X = X
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
        return sub_X, model
    