package com.tetrisj

import java.nio.charset.Charset

import com.google.common.net.InternetDomainName
import org.apache.commons.httpclient.URI
import org.apache.http.client.utils.URLEncodedUtils
import org.apache.spark.mllib.feature.{Word2VecModel, Word2Vec}
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

  def urlTopDomain(u: String) = {
    try {
      val url = new URI(u, false)
      val domain = InternetDomainName.from(url.getHost).topPrivateDomain.name()
      Some(domain)
    }catch{
      case e:Exception => None
    }
  }

  def urlDomain(u: String) = {
    try {
      val url = new URI(u, false)
      val domain = InternetDomainName.from(url.getHost).name()
      Some(domain)
    }catch{
      case e:Exception => None
    }
  }

  def urlPath(u: String) = {
    try {
      val url = new URI(u, false)
      val path = url.getPath.split('/')
      path
    }catch{
      case e:Exception => Array[String]()
    }
  }

  def urlSeq(u:String) = {
    val domain = urlDomain(u)
    val path = urlPath(u)
    val params = urlParams(u)
    val res = (domain ++ path ++ params).toList
    res
  }

  val zeroVector = org.apache.spark.mllib.linalg.Vectors.zeros(100)

  def safeTransform(s: String)(implicit model: Word2VecModel) = {
    try{
      model.transform(s)
    }catch{
      case e:Exception => zeroVector
    }
  }


  def urlFeatures(u: String, model: Word2VecModel) = {
    val res = urlSeq(u).map(safeTransform).take(10).padTo(10, zeroVector)
    res
  }

  def eventFeatures(e: Data.RawEvent, model: Word2VecModel) ={
    // [request features][href features][prev features][timestamp]
    val requestMap =  urlFeatures(e.requestUrl, model)
    val hrefMap =  urlFeatures(e.referrerUrl, model)
    val prevMap =  urlFeatures(e.prevUrl, model)
    val featureVectors = requestMap ++ hrefMap ++ prevMap
    val features = featureVectors.flatMap(_.toArray)

    Data.EventWithFeatures(e, features)
  }

}
