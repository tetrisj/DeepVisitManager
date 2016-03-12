package com.tetrisj

import org.apache.spark.mllib.linalg.SparseVector

/**
  * Created by jenia on 26/02/16.
  */
object Data {
  case class RawEvent(timestamp: Double, requestUrl: String, referrerUrl: String, prevUrl: String)
  case class EventWithFeatures(event:RawEvent, feature:List[Double])


  case class RawUserEvents(userId: String, events: List[RawEvent]) {
    override def toString: String = s"UserEvents[$userId,$events]"
  }
  object RawEvent {
    def fromString(s: String) = {
      val fields = s.split('\t')
      val timestamp = fields(0).toDouble
      val referrerUrl = fields(10)
      val prevUrl = fields(11)
      val requestUrl = fields(12)

      RawEvent(timestamp, requestUrl, referrerUrl, prevUrl)
    }
  }

  case class UserEvents(userId: String, events: List[EventWithFeatures])
  case class VisitConnection(sendingPage: String, landingPage: String, landingTime: Double)
  case class UserVisitConnections(userId: String, connections: List[VisitConnection])
}
