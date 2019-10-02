//Algoritmo 3 Versión iterativa

//(Complejidad O ( n)


def fibonacci3(n:Int):Int={
var n : Int = 10
var a = 0
var b = 1
var c = 0
var k = 0 


    for(k <- 1 to n) {
        
        c = b + a
        a = b
        b = c 
    }
     return a
}
println(fibonacci3(a))

