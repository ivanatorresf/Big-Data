//Algoritmo 4 Versión iterativa 2 variables 
//(Complejidad (O(n))


def fibonacci4(n:Int):Int={
var n : Int = 10
var a = 0
var b = 1
var k = 0 


    for(k <- 1 to n) {
        b = b + a
        a = b - a        
    
        }
     return a
}
println(fibonacci4(a))
