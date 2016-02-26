package com.tetrisj

import com.tetrisj.Data.RawUserEvents
import org.apache.hadoop.io.Text
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{explode, monotonicallyIncreasingId,array}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by jeniag on 2/4/2016.
  */
object DeepTest {
  val eventsPath = "/home/jenia/Deep/events/"
  val visitsPath = "/home/jenia/Deep/visits/"
  val outputRoot = "/home/jenia/Deep"
  val minParamCount = 100
  val minDomainCount = 10

  def prepareEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val rawUserEventsRDD = sc.sequenceFile[Text, Text](eventsPath, 16).map { t =>
      RawUserEvents(t._1.toString,
        t._2.toString.split('\n').toList.map(Data.RawEvent.fromString))
    }

    //Find url parameters (for future vectors)

    val allParams = rawUserEventsRDD.flatMap(_.events).flatMap { event =>
      val params = Features.urlParams(event.requestUrl) ++ Features.urlParams(event.prevUrl)
      params.toSeq.map(p => (p, 1))
    }
    val topParams = allParams.reduceByKey(_ + _).filter { p => p._2 > minParamCount }.keys
    val topParamsDict = topParams.collect.zipWithIndex.toMap

    //Find domains  (for future vectors)
    val allDomains = rawUserEventsRDD.flatMap(_.events).flatMap { event =>
      Features.urlDomain(event.requestUrl)
    }

    val topDomains = allDomains.countByValue.filter { p => p._2 > minDomainCount }.keys
    val topDomainsDict = topDomains.zipWithIndex.toMap

    val eventsRDD = rawUserEventsRDD.map { ue =>
      Data.UserEvents(ue.userId,
        ue.events.map(e => Features.eventFeatures(e, topParamsDict, topDomainsDict)))
    }
    eventsRDD.toDF
  }

  def prepareVisits()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    val visitsData = sqlContext.read.json(visitsPath)
    visitsData.filter(visitsData("site").endsWith("*"))
  }

  def combineVistsAndEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    val events = prepareEvents()
    events.registerTempTable("events")

    val visits = prepareVisits()
    visits.registerTempTable("visits")

    val combined = visits
      .join(events, events("userId") === visits("user"))

    combined
  }


  def main(args: Array[String]): Unit = {

    System.setProperty("spark.master", "local[4]")
    System.setProperty("spark.app.name", "DeepVisit")
    val conf = new SparkConf()
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._


//    val combined = combineVistsAndEvents()
//    combined.write.parquet(outputRoot + "/combinedParquet")




    val combined = sqlContext.read.parquet(outputRoot + "/combinedParquet")
    combined.registerTempTable("combined")
    combined.printSchema()


    val exploded = combined.
      withColumn("visitId",monotonicallyIncreasingId).
      withColumn("event", explode($"events")).
      select("visitId","userId","event","timestamps")
      .withColumn("timestamp", explode($"timestamps"))
      .select("visitId","userId","event","timestamp")

    exploded.printSchema()
    exploded.registerTempTable("exploded")
    val visitEvents = sqlContext.sql("select *, abs(timestamp/1000.0-event.event.timestamp)<0.1 label from exploded")
    visitEvents.limit(1000000).write.option("mapred.output.compress","true").json(outputRoot + "/labeledJson")








//    case class Employee(firstName: String, lastName: String, email: String)
//    case class Department(id: String, name: String)
//    case class DepartmentWithEmployees(department: Department, employees: Seq[Employee])
//
//    val employee1 = new Employee("michael", "armbrust", "abc123@prodigy.net")
//    val employee2 = new Employee("chris", "fregly", "def456@compuserve.net")
//
//    val department1 = new Department("123456", "Engineering")
//    val department2 = new Department("123456", "Psychology")
//    val departmentWithEmployees1 = new DepartmentWithEmployees(department1, Seq(employee1, employee2))
//    val departmentWithEmployees2 = new DepartmentWithEmployees(department2, Seq(employee1, employee2))
//
//    val departmentWithEmployeesRDD = sc.parallelize(Seq(departmentWithEmployees1, departmentWithEmployees2))
//    departmentWithEmployeesRDD.toDF().saveAsParquetFile("dwe.parquet")
//
//    val departmentWithEmployeesDF = sqlContext.parquetFile("dwe.parquet")
//
//    // This would be replaced by explodeArray()
//    val explodedDepartmentWithEmployeesDF = departmentWithEmployeesDF.explode(departmentWithEmployeesDF("employees")) {
//      case Row(employee: Seq[Row]) => employee.map(employee =>
//        Employee(employee(0).asInstanceOf[String], employee(1).asInstanceOf[String], employee(2).asInstanceOf[String])
//      )
//    }

  }
}
