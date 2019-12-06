//importar una simple sesion spark
import org.apache.spark.sql.SparkSession

//utilice las lineas de codigo para minizar errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//cree una instancia de la session spark
val spark = SparkSession.builder().getOrCreate()

//importar la libreria de kmeans para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans

//carga el dataset de wholesale customers data
val df = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesalecustomersdata.csv")

//seleccione las siguientes columnas: Fresh,Milk,Grocery, Froczn,Detergents_Paper,Delicassen y llamar a este conjunto feature_data
val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

/Importar vector assembler y vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

//Crea un objeto vector assemble para las columnas de caracteristicas como un conjunto de entrada, recordando que no hay etiquetas
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")


//utilice el objeto assembler para transformar feature_data
val training_data = assembler.transform(feature_data).select("features")

//crear un modelo kmeans con k=3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)


//Evalue los grupos utilizando????


//cuales son los nombres de las columnas?
feature_data.printSchema()
