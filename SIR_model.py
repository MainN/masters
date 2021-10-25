class SIR():
    def __init__(self,size,percent,*args, beta, gamma,**kwargs):
    #инициализируем изначальное состояние модели
        self.I = int(size/100*percent)
        self.S = size-self.I
        self.R = 0
        self.N = size
        self.beta = beta
        self.gamma = gamma
        self.result_S = []
        self.result_I = []
        self.result_R = []
        
    def calc(self,times):
        #вычисление 
        for t in range(0,times):
            #считаем приращения
            ds_dt=self.dS_dt()
            di_dt=self.dI_dt()
            dr_dt=self.dR_dt()
            
            #добавляем в результирующие списки
            self.result_S.append(ds_dt )
            self.result_I.append(di_dt )
            self.result_R.append(dr_dt )
            
            #пересчитываем количество людей в группах
            self.S += ds_dt
            self.I += di_dt
            self.R += dr_dt
          
        #выводим количество индивидуумов в каждой группе
        print(self.S,self.I,self.R)
        
    def dS_dt(self):
        #считаем производную для S
        return -self.beta * self.S * self.I / self.N
        
    def dR_dt(self):
        #считаем производную для R
        return self.gamma * self.I
        
    def dI_dt(self):
        ##считаем производную для I
        return self.beta * self.S * self.I / self.N - self.gamma * self.I
        
    def viz(self):
        #Собираем в табличку полученные списки прирощений
        table=list(zip(self.result_S,self.result_I,self.result_R))
        #Создаем табличку pandas
        df = pd.DataFrame(table,
               columns =['S', 'I','R'])
        #возвращаем визуализацию
        return (sns.lineplot(data=df))
    def set_params(self,params):
        self.beta=params[0]
        self.gamma=params[1]
        
        #print(self.beta,self.gamma)
    def get_results(self):
        
        self.calc()
        #print((self.beta,self.gamma),self.result_S)
        return [abs(x) for x in self.result_S]