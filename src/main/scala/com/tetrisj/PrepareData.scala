package com.tetrisj

import com.tetrisj.Data.RawUserEvents
import org.apache.hadoop.io.Text
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{explode, monotonicallyIncreasingId, min}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf


/**
  * Created by jeniag on 2/4/2016.
  */
object PrepareData {
  val eventsPath = "/home/jenia/Deep/events/"
  val visitsPath = "/home/jenia/Deep/visits/"
  val outputRoot = "/home/jenia/Deep"
  val minParamCount = 1000
  val minDomainCount = 1000

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
    val visits = visitsData.filter(visitsData("site").endsWith("*"))
      .withColumn("visitId", monotonicallyIncreasingId)

    //UDF to extract minimum timestamp
    def minTimestamp = udf {
      (timestamps: scala.collection.mutable.WrappedArray[Long]) => timestamps.min
    }

    visits.withColumn("startTime", minTimestamp(visits("timestamps")) / 1000.0)

  }

  def combineVistsAndEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    val visits = prepareVisits()
    visits.registerTempTable("visits")

    val events = prepareEvents()
    events.registerTempTable("events")

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


    val combined = combineVistsAndEvents()
    //combined.write.parquet(outputRoot + "/combinedParquet")


    //val combined = sqlContext.read.parquet(outputRoot + "/combinedParquet")
    combined.registerTempTable("combined")
    combined.printSchema()
    combined.show(10)

    val exploded = combined
      .select("startTime", "visitId", "userId", "events", "timestamps")
      .withColumn("event", explode($"events"))
      .where("event.event.timestamp > startTime - 0.01")
      .select("visitId", "event", "timestamps")
      .withColumn("timestamp", explode($"timestamps"))
      .select("visitId", "event", "timestamp")

    exploded.printSchema()
    exploded.registerTempTable("exploded")
    val visitEvents = sqlContext.sql("select *, abs(timestamp/1000.0-event.event.timestamp)<0.01 label from exploded")

    visitEvents.limit(100000).write.option("mapred.output.compress", "true").json(outputRoot + "/labeledJson")
  }
}
