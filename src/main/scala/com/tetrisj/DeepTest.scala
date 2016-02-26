package com.tetrisj

import com.tetrisj.Data.RawUserEvents
import org.apache.hadoop.io.Text
import org.apache.spark.sql.SQLContext
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


//    val combined = combineVistsAndEvents()
//    combined.write.parquet(outputRoot + "/combinedParquet")


    val combined = sqlContext.read.parquet(outputRoot + "/combinedParquet")
    combined.registerTempTable("combined")
    combined.printSchema()
    combined.as[Data.CombinedVisit].take(5).foreach(println)




    //    val visits = sqlContext.read.json(visitsPath)
    //    visits.printSchema()
    //    visits.registerTempTable("visits")
    //
    //    val combined = visits.filter(visits("site").endsWith("*"))
    //      .join(events, events("userId") === visits("user"))
    //    val json = combined.toJSON
    //    json.take(10).foreach(println)
    //    delete
    //    json.saveAsTextFile(outputRoot + "/combined")
  }
}
