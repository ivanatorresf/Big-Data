// $ ejemplo en $
// Carga de paquetes y API necesarios
importar  org . Apache . chispa . ml . clasificación . Multicapa Perceptrón Clasificador
importar  org . Apache . chispa . ml . Evaluación . MulticlassClassificationEvaluator
// $ ejemplo de descuento $
importar  org . Apache . chispa . sql . SparkSession

/ **
 * Un ejemplo para la Clasificación de Perceptrón Multicapa.
 * /
 // / Crear una sesión de Spark
 val  spark  =  SparkSession .builder.appName ( " MultilayerPerceptronClassifierExample " ) .getOrCreate ()

    // $ ejemplo en $
    // Cargue los datos almacenados en formato LIBSVM como un DataFrame.
    // Cargue los datos de entrada en formato libsvm.
    val  data  = spark.read.format ( " libsvm " ) .load ( " sample_multiclass_classification_data.txt " )

    // Divide los datos en tren y prueba
    // Preparando el conjunto de entrenamiento y prueba
    // Prepara el conjunto de entrenamiento y prueba: entrenamiento => 60%, prueba => 40% y semilla => 1234L
     divisiones  val = data.randomSplit ( matriz ( 0.6 , 0.4 ), semilla =  1234L )
     tren  val = divisiones ( 0 )
     prueba  val = divisiones ( 1 )
