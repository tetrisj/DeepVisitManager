package jeniag

import jeniag.Data.{RawUserEvents, UserVisits, Visit}
import org.apache.hadoop.io.Text
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.rand
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable


/**
  * Created by jeniag on 2/4/2016.
  */
object PrepareData {
  var eventsPath = "/similargroup/data/stats-advanced/year=16/month=06/day=01/*"
  var visitsPath = "/similargroup/data/advanced-analytics/daily/parquet-visits/year=16/month=06/day=01"
  var outputRoot = "/user/jeniag/deep"
  val maxUserPageCount = 800
  val minWord2VecCount = 5

  def prepareEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val rawUserEventsRDD = sc.sequenceFile[Text, Text](eventsPath, 16).map { t =>
      RawUserEvents(t._1.toString,
        t._2.toString.split('\n').map(Data.RawEvent.fromString))
    }.filter(rue => rue.events.exists(e => e.requestUrl.length < 3))


    val word2vec = new Word2Vec().setMinCount(minWord2VecCount).setNumIterations(8)

    //Calculate word2vec
    val allSeqs = rawUserEventsRDD.flatMap(_.events).flatMap { event =>
      Seq(Features.urlSeq(event.requestUrl),
        Features.urlSeq(event.referrerUrl),
        Features.urlSeq(event.prevUrl))
    }

    val urlModel = word2vec.fit(allSeqs)
    urlModel.save(sc,"/user/jeniag/deep/word2vec_url_model")

    val domains = rawUserEventsRDD.flatMap(_.events).map { event =>
      Seq(Features.domain(event.requestUrl),
        Features.domain(event.referrerUrl),
        Features.domain(event.prevUrl))
    }
    val domainSeqs = rawUserEventsRDD.map { rawEvents => rawEvents.events.map { event => Features.domain(event.requestUrl) }.toSeq }
    val domainCounts = domainSeqs.flatMap(x => x).countByValue()

    val domainIdMap = (domainCounts.keySet + Features.noDomainString).zipWithIndex.toMap

    val goodDomainsSet = domainCounts.filter(_._2 >= minWord2VecCount).keys.toSet
    val filteredDomainSeqs = domainSeqs.map(ds => ds.map{ domain =>
      if (goodDomainsSet.contains(domain)) domain
      else Features.noDomainString
    })

    val domainModel = word2vec.fit(filteredDomainSeqs)
    domainModel.save(sc,"/user/jeniag/deep/word2vec_domain_model")


    //Alternatively - load model from files
    //val urlModel = Word2VecModel.load(sc, "/home/jenia/Deep/word2vec_url_model")
    //val domainModel = Word2VecModel.load(sc, "/home/jenia/Deep/word2vec_domain_model")
    val urlModelVectors: mutable.Map[String, Array[Float]] = mutable.HashMap() ++ urlModel.getVectors
    val domainModelVectors: mutable.Map[String, Array[Float]] = mutable.HashMap() ++ domainModel.getVectors

    val eventsRDD = rawUserEventsRDD.filter(_.events.length < maxUserPageCount).map { ue =>
      Data.UserEvents(ue.userId,
        ue.events.map(e => Features.eventFeatures(e, urlModelVectors, domainModelVectors, domainIdMap)))
    }

    eventsRDD.toDF

  }


  def prepareVisits()(implicit sc: SparkContext, sqlContext: SQLContext) = {
    import sqlContext.implicits._
    val visitsData = sqlContext.read.parquet(visitsPath)
    visitsData.printSchema()
    val visits = visitsData
      .filter(visitsData("site").endsWith("*"))
      .select("user", "timestamps", "sendingpage", "landingpage", "specialref", "pages")
      .rdd.map { r => (r.getAs[String]("user"), Visit(r.getAs[String]("sendingpage"),
      r.getAs[String]("landingpage"),
      r.getAs[mutable.WrappedArray[Long]]("timestamps").map(_ / 1000.0).toList,
      r.getAs[mutable.WrappedArray[String]]("pages").toList
    ))
    }
      .groupByKey()
      .map(t => UserVisits(t._1, t._2.toList))
      .toDF

    visits
  }


  def combineVistsAndEvents()(implicit sc: SparkContext, sqlContext: SQLContext) = {


    val events = prepareEvents()
    val visits = prepareVisits()
    visits.printSchema()
    visits.show()
    val combined = events
      .join(visits, events("userId") === visits("userId"))
      .select("events","visits")
    combined.printSchema()
    combined

  }

  def main(args: Array[String]): Unit = {
    System.setProperty("spark.app.name", "DeepVisit")
    System.setProperty("spark.driver.memory", "20g")
    System.setProperty("spark.memory.storageFraction", "0.2")
    System.setProperty("spark.sql.shuffle.partitions", "256")
    eventsPath = args(0)
    visitsPath = args(1)

    val blockSize: Int = 1 * 1024 * 1024
    System.setProperty("dfs.blocksize", blockSize.toString)


    val conf = new SparkConf()
    implicit val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    //Logger.getLogger("org.apache.spark").setLevel(Level.INFO)
    val combined = combineVistsAndEvents().withColumn("rand", rand())
    //Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    combined.toJSON.take(1).foreach(println)

    combined
      .filter(combined("rand") >= 0.1)
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
