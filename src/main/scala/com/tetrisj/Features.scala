package com.tetrisj

import java.nio.charset.Charset

import com.github.fommil.netlib.{F2jBLAS, BLAS}
import com.google.common.net.InternetDomainName
import com.tetrisj.Data.EventFeature
import org.apache.commons.httpclient.URI

import org.apache.http.client.utils.URLEncodedUtils
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, DenseVector, Vectors}
import org.scalatest.path

import scala.collection.mutable


/**
  * Created by jenia on 26/02/16.
  */

object Features {

  import collection.JavaConversions._

  val seed = 123891273
  val nUrlFeatures = 10
  val noDomainString = "<NO_DOMAIN>"
  @transient private var _f2jBLAS: NetlibBLAS = _
  @transient private var _nativeBLAS: NetlibBLAS = _

  // For level-1 routines, we use Java implementation.
  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }


  def domain(u: String) = {
    try {
      val url = new URI(u, false)
      val domain = url.getHost
      domain
    } catch {
      case e: Exception => noDomainString
    }
  }

  def urlSeq(u:String) = {
    try {
    val url = new URI(u, false)
    val domain = url.getHost
    val path = url.getPath.split('/')
    val params = URLEncodedUtils.parse(url.getQuery, Charset.forName("UTF-8")).map(_.getName).toArray
    val res = (domain +: path ++: params).toList
    res
    }catch {
      case e: Exception => List[String]()
    }
  }

  //Iterator directly for vectors for speed and to avoid generating long sequences in memory
  class UrlVectorsIterator(val u: String, model: mutable.Map[String, Array[Float]]) extends Iterator[Array[Float]] {
    var i = 0
    var _url: URI = null
    var _path: Array[String] = null
    var _params: scala.collection.mutable.Buffer[String] = null

    def url() = {
      if (_url == null)
        try {
          _url = new URI(u, false)
        }catch {
          case e: Exception => null
        }
      _url
    }

    def hostVector: Array[Float] = {
      if (url == null) zeroVector
      else model.getOrElse(url.getHost, zeroVector)
    }

    def path = {
      if (_path == null)
        _path = url.getPath.split('/')
      _path
    }

    def pathVector(i: Int) = {
      if (url == null) zeroVector
      else model.getOrElse(path(i - 1), zeroVector)
    }

    def params = {
      if (_params == null)
        _params = URLEncodedUtils.parse(url.getQuery, Charset.forName("UTF-8")).map(_.getName)
      _params
    }

    def paramsVector(i: Int) = {
      if (url == null) zeroVector
      else model.getOrElse(params(i - 1 - path.size), zeroVector)
    }

    @inline override def hasNext: Boolean = i < nUrlFeatures

    @inline override def next() = {
      try {
        if (url ==null) zeroVector
        else if (i == 0) hostVector
        else if (i - 1 < path.size) pathVector(i)
        else paramsVector(i)
      } catch {
        case e: Exception => zeroVector
      } finally {
        i += 1
      }
    }
  }

  /**
    * y += a * x
    */
  @inline private def axpy(a: Float, x: Array[Float], y: Array[Float]): Unit = {
    val n = x.size
    f2jBLAS.saxpy(n, a, x, 1, y, 1)
  }

  /**
    * x = a * x
    */
  def scal(a: Float, x: Array[Float]): Unit = {
    f2jBLAS.sscal(x.length, a, x, 1)
  }

  val zeroVector = Array.fill[Float](100)(0.0f)
  zeroVector.clone()


  def urlFeatures(u: String, urlModelMap: mutable.Map[String, Array[Float]], domainModelMap: mutable.Map[String, Array[Float]]) = {
    val sentenceVecs = new UrlVectorsIterator(u, urlModelMap)
    var featureCount = 0
    val urlVector = Array.fill[Float](200)(0.0f)
    sentenceVecs.foreach { v =>
      axpy(1.0f, v, urlVector)
      featureCount+=1
    }
    if (featureCount>0) {
      val scaleFactor = 1.0f/nUrlFeatures
      scal(scaleFactor, urlVector)
    }
    domainModelMap.get(domain(u)) match {
      case Some(v) => Array.copy(v,0,urlVector,100,100)
      case _ => None
    }
    urlVector
  }


  def eventFeatures(e: Data.RawEvent, urlModelMap: mutable.Map[String, Array[Float]], domainModelMap: mutable.Map[String, Array[Float]]) = {
    // [request features][href features][prev features][timestamp]
    val feature = EventFeature(
      urlFeatures(e.requestUrl, urlModelMap,domainModelMap),
      urlFeatures(e.referrerUrl, urlModelMap,domainModelMap),
      urlFeatures(e.prevUrl, urlModelMap, domainModelMap),
      e.timestamp
    )

    val res = Data.EventWithFeatures(e, feature)
    //println(t1 - t0)
    res

  }

}
