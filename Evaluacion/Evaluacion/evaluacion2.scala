//1 iniciar 
Spark-Session
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()

//2
val dataset = spark.read.load("/home/adriantf/ScalaBasics/BigData/Spark_DataFrame/Netflix_2011_2016.csv")
val df = spark.read.option("header", "true").option("interSchema","true")csv("/home/adriantf/ScalaBasics/BigData/Spark_DataFrame/Netflix_2011_2016.csv")

//3
df.columns

//4
df.printSchema()

//5
df.select("Date","Open","High","Low","Close").show(20)
df.show(5)

//6 
df.describe().show()
df.count()

//7
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.select("HV Ratio").show(20)

//8
df.orderBy($"High".desc).show(5)

//9
println("Son los valores de cuando cerró la bolsa de valor de Netflix")

//10
df.select(max(df("Volume"))).show()
df.select(min(df("Volume"))).show()

//11.a
df.filter($"Close"<600).count()

//11-b
(df.filter($"High">500).count() * 1.0 / df.count()) * 100

//11.c
df.select(corr("High","Volume")).show()

//11.d
val dfyear = df.withColumn("Year",year(df("Date")))
val maxyear = dfyear.select($"Year", $"High").groupBy("Year").max()
val res = maxyear.select($"Year", $"max(High)")
res.show()

//11.e
val dfmonth = df.withColumn("Month",month(df("Date")))
val avgmonth = dfmonth.select($"Month",$"Close").groupBy("Month").mean()
avgmonth.select($"Month",$"avg(Close)").orderBy("Month").show()
