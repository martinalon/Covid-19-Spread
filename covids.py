import pandas as pd
import scipy.integrate as spi
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy import stats
import math
import pytwalk as twalk

covid = pd.read_csv("/home/martin/Escritorio/libros del cimat/epidemiologia/proyecto/200901COVID19MEXICO.csv") # lectura de los datos
fechas=covid.iloc[:,11]

def por_semana(fechas):
    dim=fechas.shape[0]     # est? es la dimencion del vector
    mesd= [31, 29, 31, 30, 31, 30, 31, 31, 30] # Numero de dias en cada mes 
    semanas= np.arange(1, 36)          # semanas enteras registradas en los datos 
    acomulado=np.zeros(35)             # este vector se llenara con el numero de casos nuevos por semana 
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

casos, semanas= por_semana(fechas)

muestra= casos
dias_muestra= semanas

N= 120e6        ################################ numero de personas 
def ode_SIER(INP,t,b,k,g):  
    Y = np.zeros(5)       # orden en el vvector (S,E,I,Y,R)
    V = INP    
    Y[0] = - (b/N )*V[0]*V[2] 
    Y[1] = (b/N)*V[0]*V[2] - k*V[1] 
    Y[2] = k* V[1] - g*V[2]
    Y[3] = k* V[1]
    Y[4] = g*V[2]
    return (Y)
  
def log_apriori(theta):
    beta, kapa, gamma=theta
    log_densidad_beta= stats.gamma.logpdf(beta, a=1.5, scale=5)
    log_densidad_kapa= stats.gamma.logpdf(kapa, a=2, scale=4)
    log_densidad_gamma= stats.gamma.logpdf(gamma, a=2, scale=4)
    log_apriori= log_densidad_beta + log_densidad_kapa + log_densidad_gamma
    return(log_apriori)
  
def log_verosimilitud(theta):
    beta, kapa, gamma= theta
    ND = 34
    t_start = 0.0
    t_end = ND
    t_inc = 1
    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    S0 = 120e6 - 500. - 130.
    E0 = 1.
    I0 = 1.
    Y0 = 1.
    R0 = 0.
    INPUT = (S0,E0, I0, Y0, R0)
    SOL = spi.odeint(ode_SIER,INPUT,t_range,args=(beta, kapa, gamma) )
    S= SOL[:,0]
    E= SOL[:,1]
    I= SOL[:,2]
    YY= SOL[:,3]
    R= SOL[:,4]
    Y= np.zeros(YY.shape[0])
    for i in range(0, YY.shape[0]):
      if i < 1:
          Y[0]= YY[0]
      else:
          Y[i]= YY[i] - YY[i-1]          
    Y=Y
    Y[Y<0]=0
    log_ver= 0
    for i in range(0, Y.shape[0]):
      log_ver= log_ver + (muestra[i]*np.log(Y[i])- Y[i])
    return(log_ver)
  
  
def U(theta):
  log_posterior= log_verosimilitud(theta) + log_apriori(theta)
  return(-log_posterior)

def p0(): #initial point function for the t walk
  beta=stats.uniform.rvs( loc=0,scale=6, size=1)
  kappa=stats.uniform.rvs( loc=0,scale=7, size=1)
  gamma=stats.uniform.rvs( loc=0,scale=6, size=1)
  return np.array([ beta[0], kappa[0], gamma[0]])

def supp(theta): #funcion de soporte , el t-walk lo va proponiendo
  beta, kappa, gamma=theta
  if(beta>0 and kappa>0 and gamma>0):
    return True
  else:
    return False


T=200000 #number of iterations

x0=p0() #punto inicial 1
xp0=p0() #punto inicial 2

tchain = twalk.pytwalk( n=3, U=U, Supp=supp )
tchain.Run( T=T , x0= x0 , xp0= xp0)

 #tiempo de autocorrelacion integrado
#que tan rapido mi cadena produce obs independientes?

toutput=tchain.Output[:, 0:3 ]

beta=toutput[:,0]
kappa=toutput[:,1]
gamma=toutput[:,2]

archivo1=open("beta.txt", "w")
archivo2=open("kapa.txt", "w")
archivo3=open("gamma.txt", "w")
for i in range(0, 200001):
    archivo1.write('%s \n' %beta[i])
    archivo2.write('%s \n' %kappa[i])
    archivo3.write('%s \n' %gamma[i])
archivo1.close
archivo2.close
archivo3.close



iteracion=np.linspace(0., 200000, 200001)

plt.plot(iteracion, beta, color="blue")
plt.show()
plt.plot(iteracion, kappa, color="green")
plt.show()
plt.plot(iteracion, gamma, color="red")
#plt.plot(dias_muestra, Y, color="purple")
#plt.plot(dias_muestra, R, color="red")
plt.show()

































