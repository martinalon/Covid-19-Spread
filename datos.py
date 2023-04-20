#Martin Alonso Flores Gonzalez
#Modelos epidemiologicos
# Codigo para problema 2 de tarea 2.

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

#################
def por_dia(fechas):
    dim=fechas.shape[0]
    dias_a= np.arange(1, 280)   #dias registrados en los datos
    acomulados_dia= np.zeros(279)  # este vector se llenara con el numero de casos nuevos por dia 
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
casos, dias = por_dia(fechas)

plt.plot(dias, casos, ".")
plt.show()
#print(casos_semana)
print(casos)


#print(dias_muestra.shape)

#beta = 0.000004
#kapa= 0.0045
#gamma = 0.065
#ND = 242.0
#t_start = 0.0
#t_end = ND
#t_inc = 1
#t_range = np.arange(t_start, t_end + t_inc, t_inc)
#S0 = 1e6
#E0 = 1.
#I0 = 1.
#Y0 = 1.
#R0 = 0.
#INPUT = (S0,E0, I0, Y0, R0)
#SOL = spi.odeint(ode_SIER,INPUT,t_range,args=(beta, kapa, gamma) )

#Y= np.zeros(SOL[:,3].shape[0])
#for i in range(1, SOL[:,3].shape[0]):
#    if i < 1:
#        Y[0]= SOL[0,3]
#    else:
#        Y[i]= SOL[i,3] - SOL[i-1,3]

        



















