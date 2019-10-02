//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"
//5. Cual es la diferencia en value y una variable en scala?
//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el numero 3.1416


//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
val cir = 15
val pi = 2*3.1416
val rad = cir / pi

rad

//2. Desarrollar un algoritmo en scala que me diga si un numero es primo

def Esprimo(i :Int) : Boolean = {

if (i <= 1)
    false
else if (i==2)
    true
else
!(2 to (i-1)).exists(x=> i % x==0)

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"

val bird = "tweet"
val interpolar = "Estoy escribiendo un "+ bird

interpolar

//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"

val mensaje = List("Hola","Luke","yo","soy","tu","padre")

mensaje.slice(1,2)



//5. Cual es la diferencia en value y una variable en scala?

//value(val) se le asigna un valor definido y no
//puede ser cambiado, y variable(var) si se puede modificar en un metodo


//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el numero 3.1416

var x = (2,4,5,1,2,3,3.1416,23)
println(x,7)
