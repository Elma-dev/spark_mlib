package dev.elma;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeansClustering {
    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("k_means_app").master("local[*]").getOrCreate();
        Dataset<Row> dataset = sparkSession.read()
                .option("inferSchema", true)
                .option("header", true)
                .csv("./datasets/data2.csv");

        //todo: create a vector assembler to combine all the features into a single vector column
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"Age", "Annual Income (k$)", "Spending Score (1-100)"})
                .setOutputCol("features");
        //todo: use the assembler to transform our DataFrame to a single column: features
        Dataset<Row> features = vectorAssembler.transform(dataset);
        features.show();

        //todo: scale the data
        StandardScaler standardScaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        Dataset<Row> scaledData = standardScaler.fit(features).transform(features);
        scaledData.show();

        //todo: create a KMeans model
        KMeans kMeans = new KMeans().setK(3);
        kMeans.setFeaturesCol("scaledFeatures");
        kMeans.setPredictionCol("cluster");
        kMeans.setMaxIter(10);
        kMeans.setSeed(42);
        //todo: fit the model to the data
        KMeansModel model = kMeans.fit(scaledData);
        //todo: predict the cluster of data points
        Dataset<Row> transform = model.transform(scaledData);
        transform.show();

    }
}
