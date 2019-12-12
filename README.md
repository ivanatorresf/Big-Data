#### INTRODUCCIÓN

En este documento se explicara y trabajara con algunos algoritmos de clasificación para el conjunto de datos Bank-Marketing, con el que normalmente se hacen pruebas para analizar como se comportan los grupos de datos. Después de  aplicar los algoritmos se hara un analisis de comparacion entre ellos para colacionar su exactitud al momento de manejar esta gran cantidad de datos, los algoritmos con los que trabajaremos son:

    • Support Vector Machine (SVM)
    • Logistic Regression
    • Multilayer Perceptron
    • Decision Tree

#### MARCO TEÓRICO

##### Multilayer Perceptron		
Este es uno de los tipos de redes más comunes. Se basa en otra red más simple llamada perceptrón simple solo que el número de capas ocultas puede ser mayor o igual que una. Es una red unidireccional (feedforward). La arquitectura típica de esta red es la siguiente:

Las neuronas de la capa oculta usan como regla de propagación la suma ponderada de las entradas con los pesos sinápticos wij y sobre esa suma ponderada se aplica una función de transferencia de tipo sigmoide, que es acotada en respuesta.
Logistic Regression
La regresión logística resulta útil para los casos en los que se desea predecir la presencia o ausencia de una característica o resultado según los valores de un conjunto de predictores. Es similar a un modelo de regresión lineal pero está adaptado para modelos en los que la variable dependiente es dicotómica. Los coeficientes de regresión logística pueden utilizarse para estimar la odds ratio de cada variable independiente del modelo. La regresión logística se puede aplicar a un rango más amplio de situaciones de investigación que el análisis discriminante.

##### Support Vector Machine (SVM)
Una Máquina de Soporte Vectorial (SVM) aprende la superficie decisión de dos clases distintas de los puntos de entrada. Como un clasificador de una sola clase, la descripción dada por los datos de los vectores de soporte es capaz de formar una frontera de decisión alrededor del dominio de los datos de aprendizaje con muy poco o ningún conocimiento de los datos fuera de esta frontera. Los datos son mapeados por medio de un kernel Gaussiano u otro tipo de kernel a un espacio de características en un espacio dimensional más alto, donde se busca la máxima separación entre clases. Esta función de frontera, cuando es traída de regreso al espacio de entrada, puede separar los datos en todas las clases distintas, cada una formando un agrupamiento

##### Decision Tree
Un árbol de decisión es, para quien va a tomar la decisión, un modelo esquemático de las alternativas disponibles y de las posibles consecuencias de cada una, su nombre proviene de la forma que adopta el modelo, parecido a la de un árbol. El modelo está conformado por múltiples de nodos cuadrados que representan puntos de decisión y de los cuales surgen ramas (que deben leerse de izquierda a derecha), que representan las distintas alternativas, las ramas que salen de los nodos circulares, o causales, representan los eventos. La probabilidad de cada evento, P (E), se indica encima de cada rama, las posibilidades de todas las ramas deben sumar 1.0.

##### IMPLEMENTACIÓN
Las dos implementaciones principales que se utilizaron en este proyecto fueron las herramientas de data analysis Scala y Spark.

##### Justificación
Ambas herramientas son completamente gratuitas lo cual las hace convenientes, pero esa no es la razón principal detrás de su utilización. Spark y Scala dominan el campo de data analysis debido a su versatilidad, amplia documentación, una comunidad extensa, multiples librerias listas para implementar en el manejo de información y excelentes tiempos de respuesta al momento de procesar miles de datos, esto y más es el por que estas herramientas son tan populares.
En una opinion mas personal, el framework Spark que incorpora el lenguaje de Scala es muy sencillo de usar y no consume muchos recursos.

##### Scala
Scala combina programación orientada a objetos y funcional en un lenguaje conciso de alto nivel. Los tipos estáticos de Scala ayudan a evitar errores en aplicaciones complejas, y sus tiempos de ejecución de JVM y JavaScript le permiten construir sistemas de alto rendimiento con fácil acceso a enormes ecosistemas de bibliotecas.


    • Scala es orientado a objetos
    • Scala es funcional
    • Scala estáticamente tipado
    • Scala es extensible
    
Apache Spark es un sistema de computación en clúster rápido y de propósito general. Proporciona API de alto nivel en Java, Scala, Python y R, y un motor optimizado que admite gráficos de ejecución generales. También es compatible con un amplio conjunto de herramientas de alto nivel que incluyen Spark SQL para SQL y procesamiento de datos estructurado, MLlib para aprendizaje automático, GraphX ​​para procesamiento de gráficos y Spark Streaming.

    • Está integrado con Apache Hadoop.
    • Trabaja en memoria, con lo que se consigue mucha mayor velocidad de procesamiento.
    • También permite trabajar en disco. 
    • Nos proporciona API para Java, Scala, Python y R.
    • Permite el procesamiento en tiempo real, con un módulo llamado Spark Streaming, que combinado con Spark SQL nos va a permitir el procesamiento en tiempo real de los datos. Conforme vayamos inyectando los datos podemos ir transformándolos y volcándolos a un resultado final.
    • Resilient Distributed Dataset (RDD): Usa la evaluación perezosa, lo que significa es que todas las transformaciones que vamos realizando sobre los RDD, no se resuelven, si no que se van almacenando en un grafo acíclico dirigido (DAG), y cuando ejecutamos una acción, es decir, cuando la herramienta no tenga más opción que ejecutar todas las transformaciones, será cuando se ejecuten.

#### CÓDIGOS
##### Multilayer Perceptron
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show()

//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show()

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show()
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)

//Multilayer perceptron
//Dividimos los datos en un arreglo en partes de 70% y 30%
val split = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = split(0)
val test = split(1)

// Especificamos las capas para la red neuronal
//De entrada 5 por el numero de datos de las features
//2 capas ocultas de dos neuronas
//La salida de 4  asi lo marca las clases
val layers = Array[Int](5, 2, 2, 4)

//Creamos el entrenador con sus parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//Entrenamos el modelo
val model = trainer.fit(train)
//Imprimimos la exactitud
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
##### Logistic Regression
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression

//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show()

//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show()

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show()
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
```
#### Logistic Regresion
```scala
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
```

##### Support Vector Machine
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC

//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desplegamos los tipos de datos.
df.printSchema()
df.show()

//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show()

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show()

//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show()
```

##### SVM
```scala
val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))
val linsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
// Fit del modelo
val linsvcModel = linsvc.fit(c3)

//Imprimimos linea de intercepcion
println(s"Coefficients: ${linsvcModel.coefficients} Intercept: ${linsvcModel.intercept}")
```

##### Decision Tree
```scala
//Importamos las librerias necesarias con las que vamos a trabajar
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

//Quita los warnings
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark y cargamos los datos del CSV en un datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
//Desblegamos los tipos de datos.
df.printSchema()
df.show()

//Cambiamos la columna y por una con datos binarios.
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show()

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show()

//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show()
```
##### DecisionTree
```scala
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
// features con mas de 4 valores distinctivos son tomados como continuos
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
//Division de los datos entre 70% y 30% en un arreglo
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
//Creamos un objeto DecisionTree
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//Rama de prediccion
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//Juntamos los datos en un pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//Create a model of the entraining
val model = pipeline.fit(trainingData)
//Transformacion de datos en el modelo
val predictions = model.transform(testData)
//Desplegamos predicciones
predictions.select("predictedLabel", "label", "features").show(5)
//Evaluamos la exactitud
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```
#### RESULTADOS
##### Multilayer Perceptron
```scala
scala> :load MultilayerPerceptron.scala
Loading MultilayerPerceptron.scala...
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@23f27434
df: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
root
|-- age: integer (nullable = true)
|-- job: string (nullable = true)
|-- marital: string (nullable = true)
|-- education: string (nullable = true)
|-- default: string (nullable = true)
|-- balance: integer (nullable = true)
|-- housing: string (nullable = true)
|-- loan: string (nullable = true)
|-- contact: string (nullable = true)
|-- day: integer (nullable = true)
|-- month: string (nullable = true)
|-- duration: integer (nullable = true)
|-- campaign: integer (nullable = true)
|-- pdays: integer (nullable = true)
|-- previous: integer (nullable = true)
|-- poutcome: string (nullable = true)
|-- y: string (nullable = true)

+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| no|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_70c853e8039a
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y| features|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|[2143.0,5.0,261.0...|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 1 row

cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]
+-----+--------------------+
|label|            features|
+-----+--------------------+
|    2|[2143.0,5.0,261.0...|
+-----+--------------------+
only showing top 1 row

split: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: int, features: vector], [label: int, features: vector])
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
layers: Array[Int] = Array(5, 2, 2, 4)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_483c0593c28f
model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = mlpc_483c0593c28f
result: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 3 more fields]
predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: int]
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_38ceee30f7ff
Test set accuracy = 0.8848956335944776
```
##### Logistic Regression
```scala
scala> :load LogisticRegression.scala
Loading LogisticRegression.scala...
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@23f27434
df: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
root
|-- age: integer (nullable = true)
|-- job: string (nullable = true)
|-- marital: string (nullable = true)
|-- education: string (nullable = true)
|-- default: string (nullable = true)
|-- balance: integer (nullable = true)
|-- housing: string (nullable = true)
|-- loan: string (nullable = true)
|-- contact: string (nullable = true)
|-- day: integer (nullable = true)
|-- month: string (nullable = true)
|-- duration: integer (nullable = true)
|-- campaign: integer (nullable = true)
|-- pdays: integer (nullable = true)
|-- previous: integer (nullable = true)
|-- poutcome: string (nullable = true)
|-- y: string (nullable = true)

+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| no|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_4fb3ef17d15e
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y| features|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|[2143.0,5.0,261.0...|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 1 row

cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]
+-----+--------------------+
|label|            features|
+-----+--------------------+
|    2|[2143.0,5.0,261.0...|
+-----+--------------------+
only showing top 1 row

logistic: org.apache.spark.ml.classification.LogisticRegression = logreg_f416edaf4472
logisticModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_f416edaf4472, numClasses = 3, numFeatures = 5
org.apache.spark.SparkException: Multinomial models contain a matrix of coefficients, use coefficientMatrix instead.
  at org.apache.spark.ml.classification.LogisticRegressionModel.coefficients(LogisticRegression.scala:955)
  ... 92 elided
logisticMult: org.apache.spark.ml.classification.LogisticRegression = logreg_41cab037b7e3
logisticMultModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_41cab037b7e3, numClasses = 3, numFeatures = 5
Multinomial coefficients: 3 x 5 CSCMatrix
Multinomial intercepts: [-7.827431229384973,2.903059293515478,4.924371935869495]
```
##### Support Vector Machine
```scala
scala> :load SupportVectorMachine.scala
Loading SupportVectorMachine.scala...
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.log4j._
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@23f27434
df: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
root
|-- age: integer (nullable = true)
|-- job: string (nullable = true)
|-- marital: string (nullable = true)
|-- education: string (nullable = true)
|-- default: string (nullable = true)
|-- balance: integer (nullable = true)
|-- housing: string (nullable = true)
|-- loan: string (nullable = true)
|-- contact: string (nullable = true)
|-- day: integer (nullable = true)
|-- month: string (nullable = true)
|-- duration: integer (nullable = true)
|-- campaign: integer (nullable = true)
|-- pdays: integer (nullable = true)
|-- previous: integer (nullable = true)
|-- poutcome: string (nullable = true)
|-- y: string (nullable = true)

+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| no|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_41bc63515487
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y| features|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|[2143.0,5.0,261.0...|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 1 row

cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]
+-----+--------------------+
|label|            features|
+-----+--------------------+
|    2|[2143.0,5.0,261.0...|
+-----+--------------------+
only showing top 1 row

c1: org.apache.spark.sql.DataFrame = [label: int, features: vector]
c2: org.apache.spark.sql.DataFrame = [label: int, features: vector]
c3: org.apache.spark.sql.DataFrame = [label: int, features: vector]
linsvc: org.apache.spark.ml.classification.LinearSVC = linearsvc_a5e9cc316bcd
linsvcModel: org.apache.spark.ml.classification.LinearSVCModel = linearsvc_a5e9cc316bcd
Coefficients: [2.125897501491213E-6,0.013517727458849872,-7.514021888017163E-4,-2.7022337506408964E-4,-0.011177544540215354] Intercept: 1.084924165339881
```
##### Decision Tree
```scala
scala> :load DecisionTree.scala
Loading DecisionTree.scala...
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@23f27434
df: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
root
|-- age: integer (nullable = true)
|-- job: string (nullable = true)
|-- marital: string (nullable = true)
|-- education: string (nullable = true)
|-- default: string (nullable = true)
|-- balance: integer (nullable = true)
|-- housing: string (nullable = true)
|-- loan: string (nullable = true)
|-- contact: string (nullable = true)
|-- day: integer (nullable = true)
|-- month: string (nullable = true)
|-- duration: integer (nullable = true)
|-- campaign: integer (nullable = true)
|-- pdays: integer (nullable = true)
|-- previous: integer (nullable = true)
|-- poutcome: string (nullable = true)
|-- y: string (nullable = true)

+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| no|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 1 row

assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_f19a9013543f
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|       job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y| features|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|management|married| tertiary|     no| 2143| yes| no|unknown| 5| may|     261| 1| -1| 0| unknown| 2|[2143.0,5.0,261.0...|
+---+----------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 1 row

cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]
+-----+--------------------+
|label|            features|
+-----+--------------------+
|    2|[2143.0,5.0,261.0...|
+-----+--------------------+
only showing top 1 row

labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_9dd84f2f61e6
featureIndexer: org.apache.spark.ml.feature.VectorIndexer = vecIdx_7a26280d8e14
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]
dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_3a84d75a40c8
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_ed1a93435431
pipeline: org.apache.spark.ml.Pipeline = pipeline_23a6d874b3e8
model: org.apache.spark.ml.PipelineModel = pipeline_23a6d874b3e8
predictions: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 6 more fields]
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|             1| 1|[-3058.0,17.0,882...|
|             2| 1|[-1944.0,7.0,623....|
|             2| 1|[-1129.0,2.0,555....|
|             2| 1|[-811.0,12.0,365....|
|             2| 1|[-754.0,9.0,727.0...|
+--------------+-----+--------------------+
only showing top 5 rows

evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_51a95d2f7b7d
accuracy: Double = 0.8937293729372937
Test Error = 0.10627062706270629
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_3a84d75a40c8) of depth 5 with 31 nodes
Learned classification tree model:
DecisionTreeClassificationModel (uid=dtc_3a84d75a40c8) of depth 5 with 31 nodes
  If (feature 2 <= 482.5)
  If (feature 3 <= 20.5)
    Predict: 0.0
  Else (feature 3 > 20.5)
    If (feature 2 <= 167.5)
    Predict: 0.0
    Else (feature 2 > 167.5)
    If (feature 3 <= 187.5)
      If (feature 3 <= 94.5)
      Predict: 1.0
      Else (feature 3 > 94.5)
      Predict: 0.0
    Else (feature 3 > 187.5)
      If (feature 3 <= 478.5)
      Predict: 0.0
      Else (feature 3 > 478.5)
      Predict: 1.0
  Else (feature 2 > 482.5)
  If (feature 2 <= 666.5)
    If (feature 3 <= 8.5)
    Predict: 0.0
    Else (feature 3 > 8.5)
    If (feature 3 <= 94.5)
      Predict: 1.0
    Else (feature 3 > 94.5)
      If (feature 1 <= 20.5)
      Predict: 0.0
      Else (feature 1 > 20.5)
      Predict: 1.0
  Else (feature 2 > 666.5)
    If (feature 2 <= 876.5)
    If (feature 3 <= 3.5)
      If (feature 1 <= 29.5)
      Predict: 0.0
      Else (feature 1 > 29.5)
      Predict: 1.0
    Else (feature 3 > 3.5)
      Predict: 1.0
    Else (feature 2 > 876.5)
    If (feature 0 <= 775.5)
      Predict: 1.0
    Else (feature 0 > 775.5)
      If (feature 3 <= 478.5)
      Predict: 1.0
      Else (feature 3 > 478.5)
      Predict: 0.0
```
#### Tabular comparativo

Algorithm
Accuracy
Multilayer Perceptron
0.8848
Logistic Regression
0.8889
Support Vector Machine
0.8860
Decision Tree
0.8903

#### CONCLUSIONES
Entre los diferentes algoritmos utilizados  se pudo observar que entre estas tienen ciertas similitudes  al momento de implementarlos, al utilizar  los diferentes algoritmos y al arrojarnos los resultados podemos observar cual  herramienta ser mejor para cirto conjunto de datos y cual da mejores resultados al momento de resolver una problema que se tenga, de igual forma no se puede confiar con el primer resultado que se encuentre, también resulta importante jugar con los datos para llegar a una mejor decisión o tener una mejor idea de con que se esta trabajando. 

#### REFERENCIAS
The Scala Programming Language
https://www.scala-lang.org/
Overview - Spark 2.4.3 Documentation
https://spark.apache.org/docs/2.4.3/
LOS ÁRBOLES DE DECISIÓN COMO HERRAMIENTA PARA EL ANALISIS DE RIESGOS DE LOS PROYECTOS
https://repository.eafit.edu.co/bitstream/handle/10784/12980/Elena_MayaLopera_2018.pdf?sequence=2&isAllowed=y
Arboles de decision y Random Forest
https://bookdown.org/content/2031/arboles-de-decision-parte-i.html#que-son-los-arboles-de-decision
