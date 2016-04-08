package com.tetrisj

import org.apache.spark.mllib.linalg.SparseVector

/**
  * Created by jenia on 26/02/16.
  */
object Data {
  case class RawEvent(timestamp: Double, requestUrl: String, referrerUrl: String, prevUrl: String)
  case class EventFeature(requestVec: Array[Float], hrefVec: Array[Float], prevVec: Array[Float], timestamp:Double)
  case class EventWithFeatures(event:RawEvent, feature:EventFeature)


  case class RawUserEvents(userId: String, events: Array[RawEvent]) {
    override def toString: String = s"UserEvents[$userId,$events]"
  }
  object RawEvent {
    def fromString(s: String) = {
      val fields = s.split('\t')
      val timestamp = fields(0).toDouble
      val referrerUrl = Features.standartizeUrl(fields(10))
      val prevUrl = Features.standartizeUrl(fields(11))
      val requestUrl = Features.standartizeUrl(fields(12))

      RawEvent(timestamp, requestUrl, referrerUrl, prevUrl)
    }
  }

  case class UserEvents(userId: String, events: Array[EventWithFeatures])
  case class Visit(sendingPage: String, landingPage: String, timestamps: List[Double], pages: List[String])
  case class UserVisits(userId: String, visits: List[Visit])
}
