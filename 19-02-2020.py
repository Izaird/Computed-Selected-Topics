"""
Texto de 1 o n archivos 
      |
      |
      |
      v
__________________________________
Normalizacion

    -StopWords
    -Sin caracteres especiales
    -Minusculas
__________________________________
    |
    |
    |
    |
    v
Seleccion de caracteristicas
        /       \ 
       /         \ 
      /           \ 
      v            v
Linguisticas      Numericas 
                    1.Frecuencia original
                    2. Frecuencia normalizada  == Probabilidad
                    3. term frecuency= tf
                    4. inverse document frecuency(idf)


Ayer pase todo el dia en las canchas de ESCOM jugando futbol

-unigramas originales
['ayer', 'pase', 'dia', 'canchas', 'escom', jugando', 'futbol']


-lemas con POS
[('ayer', 'r'), ('pase', 'v'), ('dia', 'n') ....]


-Stems
['ayer', 'pas', 'dia', 'canch', 'escom', jug', ....]



PAQUETES QUE HAY QUE REVISAR:
    pattern
    pyFreeling


Stemmers:   
    SnowballStemmer   <----- Se inicializa con el idioma
    PorterStemmer
    LancasterStemmer
    RegexpStemmer

---------------------------------------------------------------------
21/02/2020

                        Palabra  demasiado frecuente
                            ^
                            |
                            |
                            |
'grande' = (5, 0, 1, 3, 0, 50, ....)
vec_probab = vec_frecuencias/ np.sum(vec_frecuencias)
"""