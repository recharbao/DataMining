package recommend

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


object MovieLensALS {
  case class Rating(user : Int, product : Int, rating : Double)

  // 创建SparkSession对象
  val spark=SparkSession.builder().appName("MovieLensALS").master("local[2]").getOrCreate()

  def main(args: Array[String]) {
    // 屏蔽
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    if (args.length != 5) {
      println("Usage: /usr/local/spark/bin/spark-submit --class recommend.MovieLensALS " +
        "Spark_Recommend_Dataframe.jar movieLensHomeDir personalRatingsFile bestRank bestLambda bestNumiter")
      sys.exit(1)
    }
    // 设置运行环境
    import spark.implicits._

    // 装载参数二,即用户评分,该评分由评分器生成
    val myRatings = loadRatings(args(1))
    // 创建RDD，并行化
    val myRatingsRDD = spark.sparkContext.parallelize(myRatings, 1)
    // 样本数据目录作为参数0
    val movieLensHomeDir = args(0)
    // 获取ratings.dat文件
    // 查看 File 源码可以看到 toString 返回文件的路径
    val ratings = spark.sparkContext.textFile(new File(movieLensHomeDir,
      "ratings.dat").toString).map { line => // 获取文件的每一行
      val fields = line.split("::") // 以 :: 分割
      // 时间戳 mod 10， 后面是一个对象
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt,
        fields(2).toDouble))
    }

    // map id -> name
    val movies = spark.sparkContext.textFile(new File(movieLensHomeDir,
      "movies.dat").toString).map { line =>
      val fields = line.split("::")
      (fields(0).toInt, fields(1).toString())
    }.collect().toMap


    val numRatings = ratings.count()

    // ratings.map() 对于rdd每个元素遍历_._2.user看上面(_(_,_,_))的形式
    val numUsers = ratings.map(_._2.user).distinct().count()
    //同上面
    val numMovies = ratings.map(_._2.product).distinct().count()

    // 做一个 cache
    val numPartitions = 4

    // 时间戳mode后小于6的values
    val trainingDF = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD)
      .toDF()
      .repartition(numPartitions) // 分区数量

    // validation 校验样本数据
    val validationDF = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .toDF() // rdd 转换为 DF
      .repartition(numPartitions)

    // test 测试样本数据
    val testDF = ratings.filter(x => x._1 >= 8).values.toDF()
    val numTraining = trainingDF.count()
    val numValidation = validationDF.count()
    val numTest = testDF.count()

    val ranks = List(8, 12)
    val lambdas = List(0.1, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[ALSModel] = None // 最好的模型
    var bestValidationRmse = Double.MaxValue // 最好的方根误差
    var bestRank = args(2).toInt  // 最好的隐语因子
    var bestLambda = args(3).toDouble // 最好的正则化参数
    var bestNumIter = args(4).toInt // 最好的迭代次数

    // 下面训练模型
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      // 输入参数
      val als = new ALS().setMaxIter(numIter).setRank(rank).setRegParam(lambda).setUserCol("user").setItemCol("product").setRatingCol("rating")
      val model = als.fit(trainingDF)

      val validationRmse = computeRmse(model, validationDF, numValidation) // 校验模型结果
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    val testRmse = computeRmse(bestModel.get, testDF, numTest)

    // 看前面的case class
    val meanRating = trainingDF.union(validationDF).select("rating").rdd.map{case Row(v : Double) => v}.mean
    val baselineRmse = math.sqrt(testDF.select("rating").rdd.map{case Row(v : Double) => v}.map(x => (meanRating - x) * (meanRating - x)).mean)

    val improvement = (baselineRmse - testRmse) / baselineRmse * 100

    val myRatedMovieIds = myRatings.map(_.product).toSet

    val candidates = spark.sparkContext.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq).map(Rating(1,_,0.0))
      .toDF().select("user","product")

    val recommendations = bestModel.get
      .transform(candidates).select("user","product","prediction").rdd
      .map(x => Rating(x(0).toString.toInt,x(1).toString.toInt,x(2).toString.toDouble))
      .sortBy(-_.rating)
      .take(10)
    var i = 1
    println("Movies recommended for you(用户 ID:推荐电影 ID:推荐分数:推荐电影名称):")
    recommendations.foreach { r =>
      println(r.user + ":" + r.product + ":" + r.rating + ":" + movies(r.product))
      i += 1
    }
    spark.sparkContext.stop()
  }

  // 模型计算
  def computeRmse(model: ALSModel, df: DataFrame, n: Long): Double = {
    import spark.implicits._
    val predictions = model.transform(df.select("user","product")) //调用预测的函数

    val predictionsAndRatings = predictions.select("user","product","prediction").rdd.map(x => ((x(0),x(1)),x(2)))
      .join(df.select("user","product","rating").rdd.map(x => ((x(0),x(1)),x(2))))
      .values
      .take(10)

    math.sqrt(predictionsAndRatings.map(x => (x._1.toString.toDouble  - x._2.toString.toDouble) * (x._1.toString.toDouble - x._2.toString.toDouble)).reduce(_ + _) / n)
  }

  // 加载数据
  def loadRatings(path: String): Seq[Rating] = {
    val lines = Source.fromFile(path).getLines()
    val ratings = lines.map { line =>
      val fields = line.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.filter(_.rating > 0.0) // 取 > 0
    if (ratings.isEmpty) {
      sys.error("No ratings provided.")
    } else {
      ratings.toSeq
    }
  }
}