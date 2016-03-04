package com.tetrisj

import com.tetrisj.Data.RawUserEvents
import org.apache.hadoop.io.Text
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{explode, monotonicallyIncreasingId}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf


/**
  * Created by jeniag on 2/4/2016.
  */
object PrepareData {
  val eventsPath = "/home/jenia/Deep/events/"
  val visitsPath = "/home/jenia/Deep/visits/"
  val outputRoot = "/home/jenia/Deep"
  val minParamCount = 10000
  val minDomainCount = 0
  val maxVisitPageCount = 100
  val minVisitPageCount = 10

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
    val visits = visitsData
      .filter(visitsData("site").endsWith("*"))
      .filter(s"size(timestamps)<=$maxVisitPageCount and size(timestamps)>=$minVisitPageCount")
      .withColumn("visitId", monotonicallyIncreasingId)
      .select("user", "timestamps", "visitId", "sendingpage", "landingpage")

    visits

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

    System.setProperty("spark.master", "local[8]")
    System.setProperty("spark.app.name", "DeepVisit")
    System.setProperty("spark.driver.memory", "8g")
    System.setProperty("spark.sql.shuffle.partitions", "4096")

    val blockSize: Int = 4 * 1024 * 1024
    System.setProperty("dfs.blocksize", blockSize.toString)


    val conf = new SparkConf()
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._


    val combined = combineVistsAndEvents()
    combined
      .write
      .option("parquet.block.size", blockSize.toString)
      .option("spark.sql.parquet.compression.codec", "snappy")
      .parquet(outputRoot + "/combined")
  }
}
