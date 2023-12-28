package dev.elma;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LinearRegression {
    public static void main(String[] args) {
        //Create a SparkSession
        SparkSession spark = SparkSession.builder().appName("AppMlib").master("local[*]").getOrCreate();
        Dataset<Row> dataset = spark.read()
                .option("inferSchema", true)
                .option("header", true)
                .csv("./project1/datasets/data.csv");
        dataset.printSchema();

        //todo: create a vector assembler to combine all the features into a single vector column
        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{"TV", "Radio", "Newspaper"}).setOutputCol("features");
        //todo: use the assembler to transform our DataFrame to a single column: features
        Dataset<Row> features = assembler.transform(dataset);
        features.show();
        //todo: split the data into training and test sets (80% training, 20% test)
        Dataset<Row>[] dataSplit = features.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainData = dataSplit[0];
        Dataset<Row> testData = dataSplit[1];

        trainData.show();
        testData.show();
        //todo: create a Linear Regression model object
        org.apache.spark.ml.regression.LinearRegression lr = new org.apache.spark.ml.regression.LinearRegression();
        lr.setFeaturesCol("features");
        lr.setLabelCol("Sales");
        lr.setMaxIter(10);

        //todo: fit the model to the data and call this model lrModel
        LinearRegressionModel model = lr.fit(trainData);

        //todo: print the coefficients and intercept for linear regression
        Dataset<Row> prediction = model.transform(testData);
        prediction.show();

        //todo: evaluate the model on test data using the R2 metric
        RegressionEvaluator regressionEvaluator = new RegressionEvaluator();
        regressionEvaluator.setLabelCol("Sales");
        regressionEvaluator.setPredictionCol("prediction");
        regressionEvaluator.setMetricName("r2");
        double r2 = regressionEvaluator.evaluate(prediction);
        System.out.println("R2 on test data = " + r2);


    }
}