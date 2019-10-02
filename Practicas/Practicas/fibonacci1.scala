val n = 10
def fibonacci1(n:Int) : Int ={
     if (n<2){
     return n
     }
     else{
         return fibonacci1(n-1) + fibonacci1(n-2)
     }
     }
fibonacci1: (n: Int)Int
println(fibonacci1(n))
