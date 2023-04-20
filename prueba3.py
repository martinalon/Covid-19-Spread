import pandas as pd
import scipy.integrate as spi
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy import stats
import math
import pytwalk as twalk

covid = pd.read_csv("/home/martin/Escritorio/libros del cimat/epidemiologia/proyecto/201006COVID19MEXICO.csv") # lectura de los datos
fechas=covid.iloc[:,11]
resultado=covid.iloc[:,31]

##########################################################################
##########################################################################
def incidencia_dia(fechas, resultado):
    dim=fechas.shape[0]
    dias= np.arange(1, 274)   #dias registrados en los datos
    acomulados= np.zeros(273)  # este vector se llenara con el numero de casos nuevos por dia 
    dia=0     # el numero comienza en cero pero en realidad es el primer dia
    contador=0
    if resultado[0]==1:
        contador = contador + 1
    for i in range(1, dim):
        dia_pasado=int(str(fechas[i-1])[8:10])
        dia_presente=int(str(fechas[i])[8:10])
        if dia_pasado == dia_presente:
            if resultado[i]==1:
                contador = contador + 1
                if i== dim-1:
                    acomulados[dia]= contador
            else:
                contador = contador
                       
        else:
            acomulados[dia]= contador
            dia= dia + 1
            contador= 0
    return(acomulados, dias)
##################################################################
##################################################################

casos_dia, dias = incidencia_dia(fechas, resultado)   ###################### estos son los nuevos casos por dia de la base de datos completa 

def incidencia_semana(casos_dia):
    casos_semana= np.zeros(39)
    semanas= np.zeros(39)
    contador=0
    contador2 = 0
    indice=0
    for i in range(0, casos_dia.shape[0]):
        contador= contador + casos_dia[i]
        contador2= contador2 + 1
        if contador2==7:
            casos_semana[indice]= contador
            semanas[indice]= indice +1
            contador=0
            contador2= 0
            indice =indice + 1
    return(casos_semana, semanas)
###########################################################
###########################################################

casos_semana, semanas= incidencia_semana(casos_dia)
    
muestra= np.zeros(35)                   ######################## estos son los casos que serviran como muestra de entrenamiento 
semanas_muestra= np.zeros(35)              ######################## Son los datos correspondientes a 245 dias
for i in range(0,35):                   ######################## con el ciclo lleno los anteriores vectores de muestras de entrenamiento 
    muestra[i]= casos_semana[i]
    semanas_muestra[i]= i+1
    
#################################################################
################################################################# definicion de la ecuacion diferencial
                                                                    
N= 120e6                                  ### numero de personas 
def ode_SIER(INP,t,b,k,g):  
    Y = np.zeros(5)                       ### orden en el vvector (S,E,I,Y,R)
    V = INP    
    Y[0] = - (b/N )*V[0]*V[2] 
    Y[1] = (b/N)*V[0]*V[2] - k*V[1] 
    Y[2] = k* V[1] - g*V[2]
    Y[3] = k* V[1]
    Y[4] = g*V[2]
    return (Y)

################################################################
################################################################  Definicion de las aprioris

def log_apriori(theta):
    beta, kapa, gamma=theta
    log_densidad_beta= stats.gamma.logpdf(beta, a=1.5, scale=5)
    log_densidad_kapa= stats.gamma.logpdf(kapa, a=2, scale=4)
    log_densidad_gamma= stats.gamma.logpdf(gamma, a=2, scale=4)
    log_apriori= log_densidad_beta + log_densidad_kapa + log_densidad_gamma
    return(log_apriori)
#################################################################
#################################################################   definicion de la log verosimilitud del modelo
def log_verosimilitud(theta):
    beta, kapa, gamma= theta
    ND = 34
    t_start = 0.0
    t_end = ND
    t_inc = 1
    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    S0 = 120e6 -2
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
#############################################################
############################################################# definicion de la funcion de energia
def U(theta):
    log_posterior= log_verosimilitud(theta) + log_apriori(theta)
    return(-log_posterior)
#############################################################
#############################################################   puntos iniciales para el twalk
def p0(): #initial point function for the t walk
  beta=stats.uniform.rvs( loc=0,scale=6, size=1)
  kappa=stats.uniform.rvs( loc=0,scale=7, size=1)
  gamma=stats.uniform.rvs( loc=0,scale=6, size=1)
  return np.array([ beta[0], kappa[0], gamma[0]])
#############################################################
#############################################################  definicion del soporte
def supp(theta): #funcion de soporte , el t-walk lo va proponiendo
    beta, kappa, gamma=theta
    if(beta>0 and kappa>0 and gamma>0):
        return True
    else:
        return False
##############################################################
############################################################## implementacion del twalk
T=200000                #number of iterations
x0=p0()                 #punto inicial 1
xp0=p0()                #punto inicial 2

tchain = twalk.pytwalk( n=3, U=U, Supp=supp )
tchain.Run( T=T , x0= x0 , xp0= xp0)
#############################################################
############################################################# Resultados del twalk
toutput=tchain.Output[:, 0:3 ]  
beta=toutput[:,0]              # valores para beta
kappa=toutput[:,1]             # valores para kapa
gamma=toutput[:,2]             # valores para kapa
#############################################################
#############################################################  graficas de los resultados de twalk para cada parametro
iteracion=np.linspace(0., 200000, 200001)
plt.plot(iteracion, beta, color="blue")
plt.show()
plt.plot(iteracion, kappa, color="green")
plt.show()
plt.plot(iteracion, gamma, color="red")
plt.show()

        
    