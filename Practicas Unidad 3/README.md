#### Practicas de la Unidad 3
### En esta Unidad se trabajo con dos algoritmos, K-Means y Regresion Logistica.
#### K-Means
#### Regresion Logistica
```scala
// Start a Spark Session
import org.apache.spark.sql.SparkSession
```
```scala
// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```
```scala
// Spark Session
val spark = SparkSession.builder().getOrCreate()
```
```scala
// Import clustering Algorithm
import org.apache.spark.ml.clustering.KMeans
```
```scala
// Loads data.
val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")
// val dataset = spark.read.option("header","true").option("inferSchema","true").csv("sample_kmeans_data.txt")
```
```scala
// Trains a k-means model.
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(dataset)
```
```scala
// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")
```
```scala
// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
