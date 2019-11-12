//Importaciones
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    import org.apache.spark.sql.SparkSession
    //Creamos la sesion de spark
    val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
    
    // Carga los datos a utilizar y los pasa a Dataframe
    val data = spark.read.format("libsvm").load("Data/sample_libsvm_data.txt")

    // Index labels,Agregan metadatos a la columna label
    // Ajustan todo el dataset para ser incluidos en el indice de columnas.
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    //Automaticamente identifica caracteristicas para categorizar y las indexa
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)


    // Separamos los datos, 70% para entranemiento y 30% para pruebas.
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Se entrena el arbol de decision
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    //convierte las etiquetas indexadas de vuelta a las originales
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    //Indexa las etiquetas a Pipeline
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Entrena el modelo, tambien ajusta los indices.
    val model = pipeline.fit(trainingData)

    // Realiza predicciones.
    val predictions = model.transform(testData)

    //Selecciona las filas para mostrar 
    predictions.select("predictedLabel", "label", "features").show(5)

    // Selecciona y calcula el test error
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    //Calculamos la precision de las predicciones
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
    //Se imprime el modelo del arbol con lo aprendido
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
