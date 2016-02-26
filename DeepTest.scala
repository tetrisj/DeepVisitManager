package com.similargroup.audience

import _root_.org.apache.hadoop.io.Text
import _root_.org.apache.spark.SparkConf
import _root_.org.apache.spark.SparkContext
import org.apache.hadoop.io.Text
import org.apache.spark.{SparkConf, SparkContext}


case class UserEvents(userId: String, events: List[Event]) {
  override def toString: String = s"UserEvents[$userId,$events]"
}

case class Event(timestamp: Double, requestUrl: String, referrerUrl: String, prevUrl: String)

object Event {
  def fromString(s: String) = {
    val fields = s.split('\t')
    val timestamp = fields(0).toDouble
    val referrerUrl = fields(10)
    val prevUrl = fields(11)
    val requestUrl = fields(12)

    Event(timestamp, requestUrl, referrerUrl, prevUrl)
  }
}

/**
  * Created by jeniag on 2/4/2016.
  */
object DeepTest {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Projects\\Deep\\winutils\\hadoop-2.6.0\\")
    System.setProperty("java.library.path", "C:\\Projects\\Deep\\winutils\\hadoop-2.6.0\\")
    System.setProperty("spark.master", "local[4]")
    System.setProperty("spark.app.name", "Audience Intelligence Calculation")
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val events = sc.sequenceFile[Text, Text]("C:\\Projects\\Deep\\events", 16).map { t =>
      UserEvents(t._1.toString,
        t._2.toString.split('\n').toList.map(Event.fromString))
    }.toDF()
    events.printSchema()
    events.registerTempTable("events")

    val visits = sqlContext.jsonFile("C:\\Projects\\Deep\\visits")
    visits.printSchema()
    visits.registerTempTable("visits")

    val combined = visits.filter(visits("site").endsWith("*"))
      .join(events, events("userId") === visits("user"))
    val json = combined.toJSON
    json.take(10).foreach(println)
    json.saveAsTextFile("C:\\Projects\\Deep\\combined")
  }
}
