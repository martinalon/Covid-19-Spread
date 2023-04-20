
import numpy as np
import scipy
import scipy.stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from scipy.integrate import odeint
import pytwalk as twalk


covid = pd.read_csv("/home/martin/Escritorio/libros del cimat/epidemiologia/proyecto/201006COVID19MEXICO.csv") # lectura de los datos
fechas=covid.iloc[:,11]

#################
def por_dia(fechas):
    dim=fechas.shape[0]
    dias_a= np.arange(1, 274)   #dias registrados en los datos
    acomulados_dia= np.zeros(273)  # este vector se llenara con el numero de casos nuevos por dia 
    cuantos=0
    con2=1
    for i in range(1,dim): #este ciclo sirve para separar y contar los nuevos casos por dias 
        dia_presente=int(str(fechas[i-1])[8:10])
        dia_futuro=int(str(fechas[i])[8:10])
        if dia_presente==dia_futuro:
            con2=con2 +1
            if i== dim-1:
               acomulados_dia[cuantos]= con2
                
        else:
            acomulados_dia[cuantos]= con2
            cuantos= cuantos + 1
            con2= 1
    return(acomulados_dia, dias_a)
###############
def por_semana(fechas):
    dim=fechas.shape[0]     # est? es la dimencion del vector
    mesd= [31, 29, 31, 30, 31, 30, 31, 31, 30] # Numero de dias en cada mes 
    semanas= np.arange(1, 40)          # semanas enteras registradas en los datos 
    acomulado=np.zeros(39)             # este vector se llenara con el numero de casos nuevos por semana 
    com=7
    con=0
    m=0
    vec=1
    # este ciclo sirve para separar y contar los casos en las semanas
    for i in range(0,dim):
        dia=int(str(fechas[i])[8:10])
        if i < dim-1:
            if mesd[m] > com:
                if dia<= com:
                    con=con + 1
                else:
                    com=com+7
                    acomulado[vec - 1]= con
                    con= 1
                    vec= vec + 1
            elif mesd[m] < com:
                if abs(mesd[m]-dia)<7:
                    con= con + 1
                else:
                    con=con+1
                    com= com-mesd[m]
                    m=m+1
            else:
                if abs(mesd[m]-dia)<=7:
                    con=con + 1
                else:
                    com=7
                    acomulado[vec - 1]= con
                    con= 1
                    vec= vec + 1
                    m=m+1
        else:
            acomulado[vec - 1]= con
    return(acomulado, semanas)
            
#casos_semana, semanas= por_semana(fechas)                      
#casos, dias = por_dia(fechas)
acomulado, semanas =por_semana(fechas)
muestra= np.zeros(26)
dias_muestra= np.zeros(26)
for i in range(0,26):
    muestra[i]= acomulado[i]
    dias_muestra[i]= i+1





plt.plot(semanas, acomulado)
plt.show()


N = 120e6
t=np.arange(0,45,1 )
def seiyr(y,t, beta, kappa, gamma):
    S,E,I,Y,R=y
    dSdt=-(beta/N)*I*S
    dEdt = (beta/N)*I*S - kappa*E 
    dIdt= kappa*E- gamma*I
    dYdt= kappa*E
    dRdt= gamma*I
    return [dSdt, dEdt, dIdt, dYdt, dRdt]


####log prior########
#def logprior(theta):
#    beta, kappa, gamma= theta        
#    if(  beta>a and beta<b and kappa>e and kappa<f and gamma>i and gamma<j):
#        t1=(c-1)*np.log( beta-a )+ (d-1)*np.log(b-beta) 
#        t2=(g-1)*np.log(kappa-e )+ (h-1)*np.log(f- kappa)
#        t3=(k-1)*np.log(gamma-i )+ (l-1)*np.log(j- gamma)
#        return t1+t2+t3
#    else:
#        return -np.inf

  
#log Likelihood
def logL(theta):
    
    beta, kappa, gamma=theta
    S0=N- 633
    E0=500
    I0=133
    Y0=133
    R0=0
  
    y0=[S0, E0, I0, Y0, R0] #condicion inicial 
    sol = odeint(seiyr, y0, t, args=(beta,kappa, gamma))
    St, Et,It, Yt, Rt= sol[:,0], sol[:,1], sol[:,2], sol[:,3], sol[:,4]

    diffYt=np.append( Yt[0]  ,np.diff(Yt)) 
    #temp=-np.sum( diffYt  )+np.sum(muestra*np.log(diffYt)) #yt_i ~ Pois ( diffYt(beta, kappa, gamma)(t_i) )
    return(diffYt)
 
print(logL([2., 0.36, 0.25]))

 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 