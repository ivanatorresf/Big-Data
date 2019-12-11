### Evaluacion 3
## Kmeans Evaluacion
###### 1- Se Importa una simple sesion spark
```scala
import org.apache.spark.sql.SparkSession
```
###### 2- Se Utilizan las lineas de codigo para minizar errores
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```
###### 3- Se Crea una instancia de la session spark
```scala
val spark = SparkSession.builder().getOrCreate()
```
###### 4- Importar la libreria de kmeans para el algoritmo de agrupamiento
```scala
import org.apache.spark.ml.clustering.KMeans
```
###### 5- Se Carga el dataset de wholesale customers data
```scala
val df = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesalecustomersdata.csv")
```
###### 6- A qui se seleccionaron las siguientes columnas: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen y llamar a este conjunto feature_data
```scala
val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
```
###### 7- Importar vector assembler y vector
```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```
###### 8- Crea un objeto vector assemble para las columnas de caracteristicas como un conjunto de entrada, recordando que no hay etiquetas
```scala
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
```
###### 9- Utilice el objeto assembler para transformar feature_data
```scala
val training_data = assembler.transform(feature_data).select("features")
```
###### 9- Crear un modelo kmeans con k=3
```scala
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)
```
###### 10- Evalue los grupos
###### 11- Cuales son los nombres de las columnas?
```scala
feature_data.printSchema()
```

### RESULTADOS
![resultados](https://github.com/ivanatorresf/Big-Data/blob/unidad_3/imagen.md/Captura%20de%20pantalla%20de%202019-12-11%2001-29-51.png)
