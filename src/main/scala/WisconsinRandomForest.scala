import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{RandomForest}


object WisconsinRandomForest extends App {

  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("decisiontree")
  val sc = new SparkContext(conf)

  val data = sc.textFile("/data/path")
  val filteredData = data.map(_.split(",")).
    map(x => LabeledPoint(x(10).replace("4", "0").replace("2", "1").toDouble,
      Vectors.dense(x(1).toDouble,
        x(2).toDouble,
        x(3).toDouble,
        x(4).toDouble,
        x(5).toDouble,
        x(6).toDouble,
        x(7).toDouble,
        x(8).toDouble,
        x(9).toDouble)))

  val splits = filteredData.randomSplit(Array(0.7, 0.3))

  val (trainingData, testData) = (splits(0), splits(1))

  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "entropy"
  val maxDepth = 5
  val maxBins = 32
  val featureSubsetStrategy = "auto"
  val numTrees = 3


  val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
  println("Test Error = " + testErr)
  println("Learned classification tree model:\n" + model.toDebugString)
}
