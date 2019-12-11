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
```
```scala
//Importe una  SparkSession con la libreria Logistic Regression
//Optional: Utilizar el codigo de  Error reporting
//Cree un sesion Spark
//Utilice Spark para leer el archivo csv Advertising.
//Imprima el Schema del DataFrame
//importar librerias de spark y metodos de clasificacionn de logistic regression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
```
```scala
//declaramos funcion para reportar errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```
```scala
//iniciamos sesion de spark
val spark = SparkSession.builder().getOrCreate()
```
```scala
//utilizamos dataframes para leer el archivo
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
```
```scala
//imprimimos el esquema del dataframe 
data.printSchema()
```
```scala
//Despliegue los datos

// Imprima un renglon de ejemplo 
```
```scala
//imprime el head del dataframe
data.head(1)
```
```scala
// la variable colnames contendra  un arreglo de string la informacion de la primera columna.
val colnames = data.columns
```
```scala
//variable fristrow contendra la primera columna de datos
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}


```
```scala
//Preparar el DataFrame para Machine Learning
//Hacer lo siguiente:
//Renombre la columna "Clicked on Ad" a "label"
//Tome la siguientes columnas como features "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
//Cree una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour of the click"
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
```
```scala
//    - Renombre la columna "Clicked on Ad" a "label"
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
```
```scala
// Importe VectorAssembler y Vectors
// Cree un nuevo objecto VectorAssembler llamado assembler para los feature
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```
```scala
//creamos un objeto vector aseembler para que features tenga las caracteristicas que le indiquemos de las columnas indicadas
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                  .setOutputCol("features"))
```
```scala
// Utilice randomSplit para crear datos de train y test divididos en 70/30 en 5 semillas
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
```
```scala
// Configure un Pipeline
// Importe  Pipeline
// Tome los Resultados en el conjuto Test con transform
import org.apache.spark.ml.Pipeline
```
```scala
// Cree un nuevo objeto de  LogisticRegression llamado lr
val lr = new LogisticRegression()
```
```scala
// Cree un nuevo  pipeline con los elementos: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, lr))
```
```scala
// Ajuste (fit) el pipeline para el conjunto de training.
val model = pipeline.fit(training)
```
```scala
val results = model.transform(test)
```
```scala
//Evaluacion del modelo
//Para Metrics y Evaluation importe MulticlassMetrics
//Inicialice un objeto MulticlassMetrics 
//Imprima la  Confusion matrix
//importar libreria de multiclassmetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
```
```scala
// se connviertes los resutalos de prueba (test) en RDD utilizando .as y .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```
```scala
//se declara el objeto metrics para utilizar el parametro predictionandlabels de multiclassmetrics
val metrics = new MulticlassMetrics(predictionAndLabels)
```
```scala
//se imprime matriz
println("Confusion matrix:")
println(metrics.confusionMatrix)
```
```scala
//imprimimos la prediccion 
metrics.accuracy
```
