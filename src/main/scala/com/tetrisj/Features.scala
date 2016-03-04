package com.tetrisj

import java.nio.charset.Charset

import com.google.common.net.InternetDomainName
import org.apache.commons.httpclient.URI
import org.apache.http.client.utils.URLEncodedUtils
import org.apache.spark.mllib.linalg.{SparseVector, Vector}

import scala.util.hashing.MurmurHash3


/**
  * Created by jenia on 26/02/16.
  */

object Features {
  import collection.JavaConversions._
  val seed = 123891273
  def urlParams(u: String) = {
    try {
      val url = new URI(u, false)
      val paramPairs = URLEncodedUtils.parse(url.getQuery, Charset.forName("UTF-8"))
      paramPairs.map(_.getName)
    }catch {
      case e:Exception => Seq()
    }
  }

  def urlDomain(u: String) = {
    try {
      val url = new URI(u, false)
      val domain = InternetDomainName.from(url.getHost).topPrivateDomain.name()
      Some(domain)
    }catch{
      case e:Exception => None
    }
  }

  def urlFeatures(u: String, topParams: Map[String,Int], topDomains: Map[String,Int]) = {
    // domain   :  int
    // params   :  array of flags for each important parameter
    // hash     :  murmur3 hash of the complete url
    // features :  [params][domain][hash]
    // hash     :  murmur3(u)
    // returns features as indices and hash as Double

    val paramVectorSize = topParams.size

    try{
      val domain = urlDomain(u) match {
        case Some(d) => topDomains.getOrElse(d,-1).toDouble
        case None    => -2.0
      }
      val urlHash = MurmurHash3.stringHash(u,seed).toDouble
      val params = urlParams(u)
      val paramMap = params.flatMap(topParams.get).map(i => i->1.0).toMap
      paramMap + (paramVectorSize -> domain) + (paramVectorSize+1 -> urlHash)
    }catch {
      case e: Exception => Map[Int, Double]()
    }
  }

  def eventFeatures(e: Data.RawEvent, topParams: Map[String,Int], topDomains: Map[String,Int]) ={
    // [request features][href features][prev features][timestamp]
    val featureSize = topParams.size + 2
    val requestMap =  urlFeatures(e.requestUrl, topParams, topDomains)
    val hrefMap =  urlFeatures(e.referrerUrl, topParams, topDomains)
    val prevMap =  urlFeatures(e.prevUrl, topParams, topDomains)
    val features = requestMap ++
      hrefMap.map(kv => kv._1+featureSize -> kv._2) ++
      prevMap.map(kv => kv._1+featureSize*2 -> kv._2) +
      (featureSize*3-> e.timestamp)

    Data.EventWithFeatures(e, features)
  }

}
