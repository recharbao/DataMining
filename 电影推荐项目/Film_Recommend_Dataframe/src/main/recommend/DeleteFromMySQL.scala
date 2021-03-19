package recommend
import java.sql.{Connection, DriverManager, PreparedStatement}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object DeleteFromMySQL {
  // 以下是jdbc连接数据库并删除操作
  val url = "jdbc:mysql://localhost:3306/movierecommend?useUnicode=true&characterEncoding=UTF-8"
  val prop = new java.util.Properties
  prop.setProperty("user", "root")
  prop.setProperty("password", "hadoop")
  def delete(userid:Int): Unit = {
    var conn: Connection = null
    var ps: PreparedStatement = null
    val sql = "delete from recommendresult where userid="+userid
    conn = DriverManager.getConnection(url,prop)
    ps = conn.prepareStatement(sql)
    ps.executeUpdate()

    if (ps != null) {
      ps.close()
    }
    if (conn != null) {
      conn.close()
    }
  }

}
