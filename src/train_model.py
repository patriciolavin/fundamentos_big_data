# src/train_model.py
import logging
from pathlib import Path
from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.functions import when, col, rand, row_number
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, RobustScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

def _capture_and_save_spark_outputs(spark_df: DataFrame, paths: dict, config: dict):
    """
    Captura y guarda los artefactos de análisis de Spark (schema, show, describe, etc.)
    en archivos de texto para ser consumidos por el reporte.
    """
    try:
        logging.info("Capturando y guardando artefactos de análisis de Spark.")
        reports_path = paths["reports"]
        output_files = config['reporting']['console_outputs']

        # 1. Capturar y guardar el esquema del DataFrame
        schema_str = spark_df._jdf.schema().treeString()
        with open(reports_path / output_files['spark_schema'], 'w', encoding='utf-8') as f:
            f.write(schema_str)

        # 2. Capturar y guardar las primeras filas (show)
        show_str = spark_df.limit(5).toPandas().to_string()
        with open(reports_path / output_files['spark_show'], 'w', encoding='utf-8') as f:
            f.write(show_str)
        
        # 3. Capturar y guardar las estadísticas descriptivas (describe)
        describe_str = spark_df.describe().toPandas().to_string()
        with open(reports_path / output_files['spark_describe'], 'w', encoding='utf-8') as f:
            f.write(describe_str)
        
        # 4. Ejecutar y guardar una consulta SQL de ejemplo
        spark_df.createOrReplaceTempView("migraciones_temp_view")
        sql_query = """
            SELECT Origen, COUNT(*) as Frecuencia
            FROM migraciones_temp_view
            GROUP BY Origen
            ORDER BY Frecuencia DESC
            LIMIT 5
        """
        sql_result_df = spark_df.sparkSession.sql(sql_query)
        sql_result_str = sql_result_df.toPandas().to_string()
        with open(reports_path / output_files['sql_result'], 'w', encoding='utf-8') as f:
            f.write(sql_result_str)
        
        logging.info("Artefactos de Spark guardados exitosamente.")

    except Exception as e:
        logging.error(f"No se pudieron guardar los artefactos de Spark. Error: {e}", exc_info=True)
        # No se relanza la excepción para no detener el pipeline principal por un error de reporte.


# Se actualiza la firma y el cuerpo de la función para usar el diccionario 'config'.
def train_and_evaluate_model(clean_df, paths: dict, config: dict):

    """
    Entrena y evalúa un modelo de regresión logística, utilizando parámetros
    centralizados desde el objeto de configuración.
    """
    spark = None
    try:
        spark_config = config['spark']
        training_config = config['training']
        reporting_config = config['reporting']
        
        logging.info("Iniciando sesión de Spark para entrenamiento.")
        
        spark_builder = SparkSession.builder \
            .appName(spark_config['app_name']) \
            .config("spark.driver.memory", spark_config['driver_memory']) \
            .master(spark_config['master_config'])
        
        if 'max_tostring_fields' in spark_config:
            spark_builder.config("spark.sql.debug.maxToStringFields", spark_config['max_tostring_fields'])

        spark = spark_builder.getOrCreate()
        spark.sparkContext.setLogLevel(spark_config['log_level'])

        spark_df = spark.createDataFrame(clean_df)

        _capture_and_save_spark_outputs(spark_df, paths, config)

        # 1. Ingeniería de la Variable Objetivo (con acceso seguro)
        fe_config = training_config.get('feature_engineering', {}).get('label_creation', {})
        pib_multiplier = fe_config.get('pib_multiplier', 2.0) # Valor por defecto sensato

        spark_df = spark_df.withColumn("label", 
            when(
                (col("PIB_Destino") > pib_multiplier * col("PIB_Origen")) & 
                (col("Tasa_Desempleo_Destino") < col("Tasa_Desempleo_Origen")), 1
            ).otherwise(0)
        )
        
        # Se añade el paso para guardar el DataFrame procesado en Parquet.
        logging.info("Guardando el DataFrame procesado en formato Parquet.")
        processed_data_path = paths['processed_data'] / config['data']['processed_output_file']
        
        # Usamos .mode("overwrite") para asegurar que cada ejecución reemplace el archivo anterior
        spark_df.write.mode("overwrite").parquet(str(processed_data_path))
        
        logging.info(f"DataFrame procesado guardado exitosamente en: {processed_data_path}")
        
        # 2. División de datos robusta (desde config)
        logging.info("Iniciando división de datos robusta (Train/Test).")
        train_ratio = training_config.get('train_ratio', 0.8)
        split_seed = training_config.get('random_seed', 42)
        min_test_samples = training_config.get('min_test_samples', 5)

        total_rows = spark_df.count()
        if total_rows < 10:
            logging.warning(f"El dataset para entrenamiento es muy pequeño ({total_rows} filas), lo que puede afectar la validez del modelo.")
            if total_rows < 2:
                raise ValueError("El dataset no puede ser dividido porque contiene menos de 2 filas.")

        train_rows = int(total_rows * train_ratio)
        if total_rows - train_rows == 0 and train_rows > 0:
            train_rows -= 1
        
        train_rows = int(total_rows * train_ratio)
        test_rows = total_rows - train_rows

        if test_rows > 0 and train_rows == 0: # Caso extremo, asegurar al menos 1 para entrenar
             train_rows = 1
             test_rows = total_rows - 1

        if test_rows == 0 and train_rows > 0:
            train_rows -= 1 
            test_rows = 1
        
        # Guarda de validación para el tamaño del conjunto de prueba.
        min_test_samples = training_config.get('min_test_samples', 5) # Default a 5 si no está en config
        if test_rows < min_test_samples:
            error_msg = (
                f"El conjunto de prueba ({test_rows} muestras) es demasiado pequeño para una evaluación fiable. "
                f"Se requiere un mínimo de {min_test_samples} muestras (configurable en 'config.yaml').\n"
                f"SUGERENCIA: Para datasets pequeños, considere usar Validación Cruzada (Cross-Validation) en lugar de una división train/test."
            )
            logging.critical(error_msg)
            raise ValueError(error_msg)
        
        
        window_spec = Window.orderBy(rand(seed=split_seed))
        df_shuffled_indexed = spark_df.withColumn("row_index", row_number().over(window_spec))
        train_data = df_shuffled_indexed.where(col("row_index") <= train_rows).drop("row_index")
        test_data = df_shuffled_indexed.where(col("row_index") > train_rows).drop("row_index")
        logging.info(f"División de datos completada: {train_data.count()} registros de entrenamiento, {test_data.count()} de prueba.")

        # 3. Definición del Pipeline de ML
        string_indexer = StringIndexer(inputCol="Razón", outputCol="Razón_Index", handleInvalid="keep")
        one_hot_encoder = OneHotEncoder(inputCol="Razón_Index", outputCol="Razón_Vec")
        numeric_cols = [c for c, t in spark_df.dtypes if (t == 'int' or t == 'double') and c not in ['ID', 'label']]
        assembler_inputs = numeric_cols + ["Razón_Vec"]
        vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="unscaled_features")
        scaler = RobustScaler(inputCol="unscaled_features", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler, scaler, lr])

        # 4. Entrenamiento del Pipeline
        logging.info("Entrenando el pipeline de Regresión Logística...")
        pipeline_model = pipeline.fit(train_data)
        
        # 5. Predicción y Evaluación
        logging.info("Realizando predicciones sobre el conjunto de prueba.")
        predictions = pipeline_model.transform(test_data)
        
        evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator_auc.evaluate(predictions)
        
        preds_and_labels = predictions.select("prediction", "label").rdd.map(lambda row: (float(row.prediction), float(row.label)))
        metrics = MulticlassMetrics(preds_and_labels)
        accuracy = metrics.accuracy
        confusion_matrix = metrics.confusionMatrix().toArray()

        logging.info(f"Evaluación del modelo completada: AUC = {auc:.3f}, Accuracy = {accuracy:.3f}")
        
        # 6. Guardar artefactos (rutas desde config)
        model_output_path = str(paths["models"] / "spark_logistic_regression_pipeline")
        pipeline_model.write().overwrite().save(model_output_path)
        logging.info(f"Modelo guardado en: {model_output_path}")

        metrics_filename = reporting_config['console_outputs']['model_metrics']
        report_path = paths["reports"] / metrics_filename
        with open(report_path, 'w') as f:
            f.write(f"Area Under ROC (AUC): {auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix))
        logging.info(f"Métricas de evaluación guardadas en: {report_path}")

    except Exception as e:
        logging.error(f"Ocurrió un error durante el entrenamiento del modelo: {e}", exc_info=True)
        raise
    finally:
        if spark:
            spark.stop()
            logging.info("Sesión de Spark cerrada.")
