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
    // domain   :  array of flags
    // params   :  array of flags
    // features := [domain][params]
    // hash     :  murmur3(u)
    // returns features as indices and hash as Double
    val domainVectorSize = topDomains.size
    val paramVectorSize = topParams.size

    try{
      //TODO: differentiate between empty urls and rare domains(?)
      //TODO: Canonical hash for empty domains
      val domain = urlDomain(u).get
      val params = urlParams(u)
      val domainIndex = topDomains.get("www.google.com") match{
        case Some(index) => Array[Int](index)
        case _ => Array[Int]()
      }
      val paramIndices = params.flatMap(topParams.get).map(_ + domainVectorSize)
      val featureIndices = domainIndex ++ paramIndices
      val urlHash = MurmurHash3.stringHash(u,seed).toDouble

      (featureIndices, urlHash)

    }catch {
      case e: Exception => (Array[Int](), MurmurHash3.stringHash("").toDouble)
    }
  }

  def eventFeatures(e: Data.RawEvent, topParams: Map[String,Int], topDomains: Map[String,Int]) ={
    // [request features][href features][prev features][request hash][href hash][prev hash][timestamp]
    val featureSize = topDomains.size + topParams.size
    val (requestIndices, requestHash) =  urlFeatures(e.requestUrl, topParams, topDomains)
    val (hrefIndices, hrefHash) =  urlFeatures(e.referrerUrl, topParams, topDomains)
    val (prevIndices, prevHash) =  urlFeatures(e.prevUrl, topParams, topDomains)
    var features = requestIndices.map(index=> index -> 1.0) ++
    hrefIndices.map(index=> index + featureSize -> 1.0) ++
    prevIndices.map(index=> index + featureSize*2 -> 1.0)
    features = features ++ Array(featureSize*3 -> requestHash)
    features = features ++ Array(featureSize*3+1 -> hrefHash)
    features = features ++ Array(featureSize*3+2 -> prevHash)
    features = features ++ Array(featureSize*3+3 -> e.timestamp)
    Data.EventWithFeatures(e, features.toMap)
  }

}
