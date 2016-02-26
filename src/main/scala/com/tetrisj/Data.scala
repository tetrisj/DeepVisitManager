package com.tetrisj

import org.apache.spark.mllib.linalg.SparseVector

/**
  * Created by jenia on 26/02/16.
  */
object Data {
  case class RawEvent(timestamp: Double, requestUrl: String, referrerUrl: String, prevUrl: String)
  case class EventWithFeatures(event:RawEvent, feature:Map[Int,Double])

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
  case class CombinedVisit(sendingpage: String, landingpage:String, userId: String, timestamps: Array[Int], events:Array[EventWithFeatures])

//  root
//  |-- _corrupt_record: string (nullable = true)
//  |-- appsource: array (nullable = true)
//  |    |-- element: string (containsNull = true)
//  |-- country: long (nullable = true)
//  |-- issitereferral: boolean (nullable = true)
//  |-- keywords: string (nullable = true)
//  |-- landingpage: string (nullable = true)
//  |-- numpageviews: long (nullable = true)
//  |-- pages: array (nullable = true)
//  |    |-- element: string (containsNull = true)
//  |-- reftype: string (nullable = true)
//  |-- region: string (nullable = true)
//  |-- sendingpage: string (nullable = true)
//  |-- site: string (nullable = true)
//  |-- site2: string (nullable = true)
//  |-- source: string (nullable = true)
//  |-- specialref: long (nullable = true)
//  |-- timeonsite: double (nullable = true)
//  |-- timestamps: array (nullable = true)
//  |    |-- element: long (containsNull = true)
//  |-- user: string (nullable = true)
//  |-- userId: string (nullable = true)
//  |-- events: array (nullable = true)
//  |    |-- element: struct (containsNull = true)
//  |    |    |-- event: struct (nullable = true)
//  |    |    |    |-- timestamp: double (nullable = false)
//  |    |    |    |-- requestUrl: string (nullable = true)
//  |    |    |    |-- referrerUrl: string (nullable = true)
//  |    |    |    |-- prevUrl: string (nullable = true)
//  |    |    |-- feature: map (nullable = true)
//  |    |    |    |-- key: integer
//  |    |    |    |-- value: double (valueContainsNull = false)


}
