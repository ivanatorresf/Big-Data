```scala
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
    ```
/////////////////////
resultados /////////
///////////////////
```scala
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
```

//1 iniciar 
```scala
Spark-Session
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()

```
//2
```scala
val dataset = spark.read.load("/home/adriantf/ScalaBasics/BigData/Spark_DataFrame/Netflix_2011_2016.csv")
val df = spark.read.option("header", "true").option("interSchema","true")csv("/home/adriantf/ScalaBasics/BigData/Spark_DataFrame/Netflix_2011_2016.csv")
```
//3

```scala
df.columns
```
//4

```scala
df.printSchema()
```
//5

```scala
df.select("Date","Open","High","Low","Close").show(20)
df.show(5)
```
//6 

```scala
df.describe().show()
df.count()
```
//7

```scala
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.select("HV Ratio").show(20)
```
//8

```scala
df.orderBy($"High".desc).show(5)
```
//9

```scala
println("Son los valores de cuando cerró la bolsa de valor de Netflix")
```
//10

```scala
df.select(max(df("Volume"))).show()
df.select(min(df("Volume"))).show()
```
//11.a

```scala
df.filter($"Close"<600).count()
```
//11-b

```scala
(df.filter($"High">500).count() * 1.0 / df.count()) * 100
```
//11.c

```scala
df.select(corr("High","Volume")).show()
```
//11.d

```scala
val dfyear = df.withColumn("Year",year(df("Date")))
val maxyear = dfyear.select($"Year", $"High").groupBy("Year").max()
val res = maxyear.select($"Year", $"max(High)")
res.show()
```
//11.e

```scala
val dfmonth = df.withColumn("Month",month(df("Date")))
val avgmonth = dfmonth.select($"Month",$"Close").groupBy("Month").mean()
avgmonth.select($"Month",$"avg(Close)").orderBy("Month").show()
```
