
def breaking_records(score: Array[Int]) {

    var n = 10

    val sc = Array (3,4,21,36,10,28,35,5,24,42)


        var min = score(0)
        var dLeast = 0;
        var max = score(0)
        var dMost = 0;
    
for(i <-score){
        println(" "+ i)
        if(i<min){
            min=i
            dLeast+=1
            //println("minimos: " + min)
        }
        else if(i > max){
            max=i
            dMost+=1
            //println("Maximos: " + max)
        }
    }
 println(dMost + " " + dLeast)
    
/////////////////////
resultados /////////
///////////////////
    
 10
 5
minimos: 5
 20
Maximos: 20
 20
 4
minimos: 4
 5
 2
minimos: 2
 25
Maximos: 25
 1
minimos: 1

scala>  println(dMost + " " + dLeast)
2 4
