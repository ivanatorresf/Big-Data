// Importaciones necesarias
importar org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification. {RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature. {IndexToString, StringIndexer, VectorIndexer}

importar org.apache.spark.sql.SparkSession

    // Se crea la sesion de spark
    val spark = SparkSession.builder.appName ("RandomForestClassifierExample"). getOrCreate ()

    
    // Carga los datos a utilizar y los pasa a Dataframe
    val data = spark.read.format ("libsvm"). load ("Data / sample_libsvm_data.txt")

    // Etiquetas de índice, Agregan metadatos a la columna label
    // Ajustan todo el conjunto de datos para ser incluidos en el índice de columnas.
    val labelIndexer = new StringIndexer (). setInputCol ("label"). setOutputCol ("indexedLabel"). fit (data)

    // Automaticamente identifica caracteristicas para categorizar y las indexa
    val featureIndexer = new VectorIndexer (). setInputCol ("features"). setOutputCol ("indexedFeatures"). setMaxCategories (4) .fit (datos)

     // Separamos los datos, 70% para entranemiento y 30% para pruebas.
    Val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3))

    // Se entrena el modelo de Random Forest
    val rf = new RandomForestClassifier (). setLabelCol ("indexedLabel"). setFeaturesCol ("indexedFeatures"). setNumTrees (10)

    // convierte las etiquetas indexadas de vuelta a las originales
    val labelConverter = new IndexToString (). setInputCol ("prediction"). setOutputCol ("predictedLabel"). setLabels (labelIndexer.labels)

    // Indexa las etiquetas a Pipeline
    val pipeline = new Pipeline (). setStages (Array (labelIndexer, featureIndexer, rf, labelConverter))

    // Entrena el modelo, tambien ajusta los indices.
    modelo val = pipeline.fit (trainingData)

    // Realiza predicciones.
    predicciones val = model.transform (testData)

    // Selecciona los surcos de ejemplo a mostrar.
    predictions.select ("predictedLabel", "label", "features"). show (5)

    // Evalúa el modelo y calcula el error.
    val evaluator = new MulticlassClassificationEvaluator (). setLabelCol ("indexedLabel"). setPredictionCol ("prediction"). setMetricName ("precision")
    // Evalua la precisión de las predicciones
    val precision = evaluator.evaluate (predicciones)

    println (s "Error de prueba = $ {(1.0 - precisión)}")
    // Se imprime el modelo de bosque aleatorio
    val rfModel = model.stages (2) .asInstanceOf [RandomForestClassificationModel]
    println (s "Modelo de bosque de clasificación aprendida: \ n $ {rfModel.toDebugString}")
    
