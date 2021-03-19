package recommend
import java.util.Properties

import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer


object ReadFromMySQL
{
  def read(userid:Int): Array[String] = {
    val spark= SparkSession.builder().appName("ReadFromMySQL").master("local[2]").getOrCreate()
    import spark.implicits._
    val personalRatingsDF = spark.
      read.format("jdbc").
      option("url", "jdbc:mysql://localhost:3306/movierecommend").
      option("driver","com.mysql.jdbc.Driver").
      option("dbtable", "personalratings").
      option("user", "root").
      option("password", "hadoop").load()
    personalRatingsDF.show()
    personalRatingsDF.createOrReplaceTempView("personalratings")
    val prDF=spark.sql("select * from personalratings where userid="+userid)
    val myrdd=prDF.rdd.map(r=>{r(0).toString+"::"+r(1).toString+"::"+r(2).toString+"::"+r(3).toString})
    val array=ArrayBuffer[String]()
    array++=myrdd.collect();
    println(array.length)
    val i=0;
    for (i <- 0 until array.length){println(array(i))}
    array.toArray
  }
}