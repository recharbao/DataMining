package recommend
import java.util.Properties
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
object InsertIntoMySQL
{
  def insert(array:Array[String]): Unit = {
    val spark= SparkSession.builder().appName("InsertIntoMySQL").master("local[2]").getOrCreate()
    val movieRDD = spark.sparkContext.parallelize(array).map(_.split("::"))
    val rowRDD = movieRDD.map(p => Row(p(0).trim.toInt, p(1).trim.toInt, p(2).trim.toFloat, p(3).trim))
    rowRDD.foreach(println)
    val schema = StructType(List(
      StructField("userid", IntegerType, true),
      StructField("movieid", IntegerType, true),
      StructField("rating", FloatType, true),
      StructField("moviename", StringType, true)))
    val movieDF=spark.createDataFrame(rowRDD,schema)
    val prop = new Properties()
    prop.put("user", "root")
    prop.put("password", "hadoop")
    prop.put("driver","com.mysql.jdbc.Driver")
    movieDF.write.mode("append").jdbc("jdbc:mysql://localhost:3306/movierecommend", "movierecommend.recommendresult", prop)
    movieDF.show()
  }
}