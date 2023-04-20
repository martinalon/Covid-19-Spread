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
  
  
  
  
  
  
  
  
  
  
  
  
    