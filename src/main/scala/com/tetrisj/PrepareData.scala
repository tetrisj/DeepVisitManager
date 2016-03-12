package com.tetrisj

import com.tetrisj.Data.{UserVisitConnections, VisitConnection, RawUserEvents}
import org.apache.hadoop.io.Text
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{explode, monotonicallyIncreasingId}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.udf

import scala.collection.mutable


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
  val maxUserEventCount = 400
  val minWord2VecCount = 1000

  def prepareEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val rawUserEventsRDD = sc.sequenceFile[Text, Text](eventsPath, 16).map { t =>
      RawUserEvents(t._1.toString,
        t._2.toString.split('\n').toList.map(Data.RawEvent.fromString))
    }

    val word2vec = new Word2Vec().setMinCount(minWord2VecCount)

    //Calculate word2vec
//    val allSeqs = rawUserEventsRDD.flatMap(_.events).flatMap { event =>
//      Seq(Features.urlSeq(event.requestUrl),
//        Features.urlSeq(event.referrerUrl),
//        Features.urlSeq(event.prevUrl))
//    }

//    val model = word2vec.fit(allSeqs)
//    model.save(sc,"/home/jenia/Deep/word2vecmodel")

    val model = Word2VecModel.load(sc,"/home/jenia/Deep/word2vecmodel")

    val eventsRDD = rawUserEventsRDD.map { ue =>
      Data.UserEvents(ue.userId,
        ue.events.map(e => Features.eventFeatures(e, model)))
    }

    eventsRDD.toDF

  }

  def prepareVisits()(implicit sc: SparkContext, sqlContext: SQLContext) = {

    val visitsData = sqlContext.read.json(visitsPath)
    val visits = visitsData
      .filter(visitsData("site").endsWith("*"))
      .filter(s"size(timestamps)<=$maxVisitPageCount and size(timestamps)>=$minVisitPageCount")
      .withColumn("visitId", monotonicallyIncreasingId)
      .select("user", "timestamps", "visitId", "sendingpage", "landingpage", "specialref")

    visits

  }


  def combineVistsAndEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val visits = prepareVisits()
    visits.registerTempTable("visits")

    val minTimestamp = udf { timestamps: mutable.WrappedArray[Long] => timestamps.min / 1000.0 }

    val visitConnections = visits
      .withColumn("landing_time", minTimestamp(visits("timestamps")))
      .filter("not isnull(sendingpage) and specialref!=5")
      .select("user", "sendingpage", "landingpage", "landing_time")

      .rdd.map { r => (r.getAs[String]("user"), VisitConnection(r.getAs[String]("sendingpage"), r.getAs[String]("landingpage"), r.getAs[Double]("landing_time"))) }
      .groupByKey()
      .map(t => UserVisitConnections(t._1, t._2.toList))
      .toDF

    visitConnections.printSchema()
    visitConnections.show()



    val events = prepareEvents()
    events.registerTempTable("events")



    val combined = visits
      .join(events, events("userId") === visits("user"))
      .filter(s"size(events) <= $maxUserEventCount")
    val combinedWithTransfers = combined
      .join(visitConnections, "userId")
      .select("timestamps", "events", "connections", "visitId")

    combinedWithTransfers.printSchema()
    combinedWithTransfers
  }

  def main(args: Array[String]): Unit = {

    System.setProperty("spark.master", "local[8]")
    System.setProperty("spark.app.name", "DeepVisit")
    System.setProperty("spark.driver.memory", "24g")
    System.setProperty("spark.sql.shuffle.partitions", "4096")

    val blockSize: Int = 1 * 1024 * 1024
    System.setProperty("dfs.blocksize", blockSize.toString)


    val conf = new SparkConf()
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    import org.apache.log4j.{Level, Logger}
    Logger.getLogger("org.apache.spark").setLevel(Level.INFO)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val combined = combineVistsAndEvents()

    combined
      .filter(combined("visitId") % 10 > 1)
      .write
      .option("parquet.block.size", blockSize.toString)
      .option("spark.sql.parquet.compression.codec", "gzip")
      .parquet(outputRoot + "/combined_train")

    combined
      .filter(combined("visitId") % 10 <= 1)
      .write
      .option("parquet.block.size", blockSize.toString)
      .option("spark.sql.parquet.compression.codec", "gzip")
      .parquet(outputRoot + "/combined_test")
  }
}
