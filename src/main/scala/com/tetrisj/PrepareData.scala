package com.tetrisj

import com.tetrisj.Data.{UserVisitConnections, VisitConnection, RawUserEvents}
import org.apache.hadoop.io.Text
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{monotonicallyIncreasingId}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.{udf, rand}


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
  val maxUserPageCount = 400
  val minWord2VecCount = 5

  def prepareEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val rawUserEventsRDD = sc.sequenceFile[Text, Text](eventsPath, 16).map { t =>
      RawUserEvents(t._1.toString,
        t._2.toString.split('\n').map(Data.RawEvent.fromString))
    }.filter(_.events.size < maxUserPageCount)

    val word2vec = new Word2Vec().setMinCount(minWord2VecCount).setNumIterations(3)

    //Calculate word2vec
//    val allSeqs = rawUserEventsRDD.flatMap(_.events).flatMap { event =>
//      Seq(Features.urlSeq(event.requestUrl),
//        Features.urlSeq(event.referrerUrl),
//        Features.urlSeq(event.prevUrl))
//    }
//
//    val urlModel = word2vec.fit(allSeqs)
//    urlModel.save(sc,"/home/jenia/Deep/word2vec_url_model")

//    val domains = rawUserEventsRDD.flatMap(_.events).map { event =>
//      Seq(Features.domain(event.requestUrl),
//        Features.domain(event.referrerUrl),
//        Features.domain(event.prevUrl))
//    }
//    val domainModel = word2vec.fit(domains)
//    domainModel.save(sc,"/home/jenia/Deep/word2vec_domain_model")

    val urlModel = Word2VecModel.load(sc, "/home/jenia/Deep/word2vec_url_model")
    val domainModel = Word2VecModel.load(sc, "/home/jenia/Deep/word2vec_domain_model")
    val urlModelVectors: mutable.Map[String, Array[Float]] = mutable.HashMap() ++ urlModel.getVectors
    val domainModelVectors: mutable.Map[String, Array[Float]] = mutable.HashMap() ++ domainModel.getVectors

    val eventsRDD = rawUserEventsRDD.map { ue =>
      Data.UserEvents(ue.userId,
        ue.events.map(e => Features.eventFeatures(e, urlModelVectors, domainModelVectors)))
    }

    eventsRDD.toDF

  }

  def prepareVisits()(implicit sc: SparkContext, sqlContext: SQLContext) = {

    val visitsData = sqlContext.read.json(visitsPath)
    val visits = visitsData
      .filter(visitsData("site").endsWith("*"))
      .withColumn("visitId", monotonicallyIncreasingId)
      .select("user", "timestamps", "visitId", "sendingpage", "landingpage", "specialref")

    visits

  }


  def combineVistsAndEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._

    val events = prepareEvents()
    events.registerTempTable("events")

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



    val combined = visits
      .join(events, events("userId") === visits("user"))
    val combinedWithTransfers = combined
      .join(visitConnections, "userId")
      .select("timestamps", "events", "connections")

    combinedWithTransfers
  }

  def main(args: Array[String]): Unit = {

    System.setProperty("spark.master", "local[4]")
    System.setProperty("spark.app.name", "DeepVisit")
    System.setProperty("spark.driver.memory", "22g")
    System.setProperty("spark.memory.storageFraction", "0.1")
    System.setProperty("spark.sql.shuffle.partitions", "512")

    val blockSize: Int = 1 * 1024 * 1024
    System.setProperty("dfs.blocksize", blockSize.toString)


    val conf = new SparkConf()
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    import org.apache.log4j.{Level, Logger}
    //Logger.getLogger("org.apache.spark").setLevel(Level.INFO)
    val combined = combineVistsAndEvents().withColumn("rand", rand())
    //Logger.getLogger("org.apache.spark").setLevel(Level.WARN)


    combined
      .filter(combined("rand") > 0.1)
      .write
      .option("parquet.block.size", blockSize.toString)
      .option("spark.sql.parquet.compression.codec", "gzip")
      .parquet(outputRoot + "/combined_train")

    combined
      .filter(combined("rand") < 0.1)
      .write
      .option("parquet.block.size", blockSize.toString)
      .option("spark.sql.parquet.compression.codec", "gzip")
      .parquet(outputRoot + "/combined_test")
  }
}
