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


/**
  * Created by jenia on 26/02/16.
  */

object Features {

  import collection.JavaConversions._

  val seed = 123891273
  val nUrlFeatures = 10
  @transient private var _f2jBLAS: NetlibBLAS = _
  @transient private var _nativeBLAS: NetlibBLAS = _

  // For level-1 routines, we use Java implementation.
  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }


  def urlTopDomain(u: String) = {
    try {
      val url = new URI(u, false)
      val domain = InternetDomainName.from(url.getHost).topPrivateDomain.name()
      Some(domain)
    } catch {
      case e: Exception => None
    }
  }


  class UrlSeq(val u: String, model: Map[String, Array[Float]]) extends Iterator[Array[Float]] {
    var i = 0
    var _url: java.net.URI = null
    var _path: Array[String] = null
    var _params: scala.collection.mutable.Buffer[String] = null

    def url(): java.net.URI = {
      if (_url == null)
        try {
          //TODO: Try with a better url parser
          _url = new java.net.URI(u)
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
        _params = URLEncodedUtils.parse(url, "UTF-8").map(_.getName)
      _params
    }

    def paramsVector(i: Int) = {
      if (url == null) zeroVector
      else model.getOrElse(params(i - 1 - path.size), zeroVector)
    }

    override def hasNext: Boolean = i < nUrlFeatures

    override def next() = {
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
  private def axpy(a: Float, x: Array[Float], y: Array[Float]): Unit = {
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
  val scaleFactor = 1.0f/nUrlFeatures

  def urlFeatures(u: String, model: Map[String, Array[Float]]) = {
    val t0 = System.nanoTime()
    val sentenceVecs = new UrlSeq(u, model)

    val sum = zeroVector.clone()
    sentenceVecs.foreach { v =>
      axpy(1.0f, v, sum)
    }
    scal(scaleFactor, sum)
    sum

  }


  def eventFeatures(e: Data.RawEvent, modelMap: Map[String, Array[Float]]) = {
    // [request features][href features][prev features][timestamp]

    val t0 = System.nanoTime()
    val feature = EventFeature(
      urlFeatures(e.requestUrl, modelMap),
      urlFeatures(e.referrerUrl, modelMap),
      urlFeatures(e.prevUrl, modelMap),
      e.timestamp
    )

    val res = Data.EventWithFeatures(e, feature)
    val t1 = System.nanoTime()
    //println(t1 - t0)
    res

  }

}
