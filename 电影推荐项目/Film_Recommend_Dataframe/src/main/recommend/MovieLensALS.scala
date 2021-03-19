package recommend
//该部分代码与spark单独部分基本一样

import java.io.File
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


object MovieLensALS {
  case class Rating(user : Int, product : Int, rating : Double)
  val spark=SparkSession.builder().appName("MovieLensALS").master("local[2]").getOrCreate()

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    if (args.length != 2) {
      println("Usage: /usr/local/spark/bin/spark-submit --class recommend.MovieLensALS " +
        "Spark_Recommend_Dataframe.jar movieLensHomeDir userid")
      sys.exit(1)
    }
    // 设置运行环境
    import spark.implicits._

    // 装载参数二,即用户评分,该评分由评分器生成
    val userid=args(1).toInt;
    //删除该用户之前已经存在的电影推荐结果，为本次写入最新的推荐结果做准备
    DeleteFromMySQL.delete(userid)
    //从关系数据库中读取该用户对一些电影的个性化评分数据
    val personalRatingsLines:Array[String]=ReadFromMySQL.read(userid)
    val myRatings = loadRatings(personalRatingsLines)
    val myRatingsRDD = spark.sparkContext.parallelize(myRatings, 1)

    // 样本数据目录
    val movieLensHomeDir = args(0)

    val ratings = spark.sparkContext.textFile(new File(movieLensHomeDir,
      "ratings.dat").toString).map { line =>
      val fields = line.split("::")
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt,
        fields(2).toDouble))
    }


    val movies = spark.sparkContext.textFile(new File(movieLensHomeDir,
      "movies1.dat").toString).map { line =>
      val fields = line.split("::")
      (fields(0).toInt, fields(1).toString())
    }.collect().toMap

    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct().count()
    val numMovies = ratings.map(_._2.product).distinct().count()

    val numPartitions = 4


    val trainingDF = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD)
      .toDF()
      .repartition(numPartitions)


    val validationDF = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .toDF()
      .repartition(numPartitions)

    val testDF = ratings.filter(x => x._1 >= 8).values.toDF() //取评分时间除 10 的余数后值大于等于 8 分的作为测试样本
    val numTraining = trainingDF.count()
    val numValidation = validationDF.count()
    val numTest = testDF.count()

    val ranks = List(8, 12)
    val lambdas = List(0.1, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[ALSModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = 0.0
    var bestNumIter = 0
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      println("正在执行循环训练模型")
      val als = new ALS().setMaxIter(numIter).setRank(rank).setRegParam(lambda).setUserCol("user").setItemCol("product").setRatingCol("rating")
      val model = als.fit(trainingDF)

      val validationRmse = computeRmse(model, validationDF, numValidation)
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }


    val testRmse = computeRmse(bestModel.get, testDF, numTest)

    val meanRating = trainingDF.union(validationDF).select("rating").rdd.map{case Row(v : Double) => v}.mean
    val baselineRmse = math.sqrt(testDF.select("rating").rdd.map{case Row(v : Double) => v}.map(x => (meanRating - x) * (meanRating - x)).mean)

    val improvement = (baselineRmse - testRmse) / baselineRmse * 100

    val myRatedMovieIds = myRatings.map(_.product).toSet

    val candidates = spark.sparkContext.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq).map(Rating(userid,_,0.0))
      .toDF().select("user","product")

    val recommendations = bestModel.get
      .transform(candidates).select("user","product","prediction").rdd
      .map(x => Rating(x(0).toString.toInt,x(1).toString.toInt,x(2).toString.toDouble))
      .sortBy(-_.rating)
      .take(10)

    val rddForMySQL=recommendations.map(r=>r.user + "::"+ r.product + "::"+ r.rating+"::" + movies(r.product))
    InsertIntoMySQL.insert(rddForMySQL)
    var i = 1
    println("Movies recommended for you(用户 ID:推荐电影 ID:推荐分数:推荐电影名称):")
    recommendations.foreach { r =>
      println(r.user + ":" + r.product + ":" + r.rating + ":" + movies(r.product))
      i += 1
    }
    spark.sparkContext.stop()
  }

  def computeRmse(model: ALSModel, df: DataFrame, n: Long): Double = {
    import spark.implicits._
    val predictions = model.transform(df.select("user","product"))
    // 输出 predictionsAndRatings 预测和评分
    val predictionsAndRatings = predictions.select("user","product","prediction").rdd.map(x => ((x(0),x(1)),x(2)))
      .join(df.select("user","product","rating").rdd.map(x => ((x(0),x(1)),x(2))))
      .values
      .take(10)

    math.sqrt(predictionsAndRatings.map(x => (x._1.toString.toDouble  - x._2.toString.toDouble) * (x._1.toString.toDouble - x._2.toString.toDouble)).reduce(_ + _) / n)
  }


  def loadRatings(lines: Array[String]): Seq[Rating] = {
    val ratings = lines.map { line =>
      val fields = line.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.filter(_.rating > 0.0)
    if (ratings.isEmpty) {
      sys.error("No ratings provided.")
    } else {
      ratings.toSeq
    }
  }

}
