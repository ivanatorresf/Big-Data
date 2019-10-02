// 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"

import scala.collection.mutable.ListBuffer
var lista = collection.mutable.ListBuffer("rojo","blanco","negro")

// 2. A�adir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla" 
lista+=("verde","amarillo","azul","naranja","perla")

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul". La funci�n (slice) nos permite tomar los datos con el uso de sus coordenadas que se encuentran en la lista

lista slice (3,6)

// 4. Crea un arreglo de n�mero en rango del 1-1000 en pasos de 5 en 5
// La funci�n (range) nos permite darle el rango que queremos que tenga nuestro arreglo

var arrayrang = Array(1,1000,5)



// 5. Cuales son los elementos �nicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversi�n a conjuntos // La funci�n (toSet) nos muestra datos no repetidos o
//duplicados
var lista = List(1,3,3,4,6,7,3,7)
var unique = lista.toSet
println(unique)


//6. Crea una mapa mutable llamado nombres que contenga los siguiente "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27" 

var lista = collection.mutable.Map(("Jose",2),("Luis",24),("Ana",23),("Susana"
,27))

//  6 a . Imprime todas la llaves del mapa */

println(lista)

// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)

lista += ("Miguel" -> 23)
