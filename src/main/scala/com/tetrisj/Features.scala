package com.tetrisj

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.{F2jBLAS, BLAS => NetlibBLAS}
import com.tetrisj.Data.EventFeature

import scala.collection.mutable


/**
  * Created by jenia on 26/02/16.
  */

object Features {

  val seed = 123891273
  val nUrlFeatures = 10
  val noDomainString = "<NO_DOMAIN>"
  @transient private var _f2jBLAS: NetlibBLAS = _
  @transient private var _nativeBLAS: NetlibBLAS = _

  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }

  def standartizeUrl(url:String) = {
    try {
      new UrlZ(url).getRawUrl
    }catch {
      case e: Exception => ""
    }
  }

  def domain(u: String) = {
    try {
      val url = new UrlZ(u)
      val domain = url.getFirstLevelDomain
      domain
    } catch {
      case e: Exception => noDomainString
    }
  }

  def urlSeq(u:String) = {
    try {
    val url = new UrlZ(u)
    val domain = url.getFirstLevelDomain
    val path = url.getPath.split("/")
    val params = url.getQueryParams.map(_.split('=')(0))
    val res = (domain +: path ++: params).toList
    res
    }catch {
      case e: Exception => List[String]()
    }
  }

  //Iterator directly for vectors for speed and to avoid generating long sequences in memory
  class UrlVectorsIterator(val u: String, model: mutable.Map[String, Array[Float]]) extends Iterator[Array[Float]] {
    var i = 0
    var _url: UrlZ = null
    var _path: Array[String] = null
    var _params: Array[String] = null

    def url() = {
      if (_url == null)
        try {
          _url = new UrlZ(u)
        }catch {
          case e: Exception => null
        }
      _url
    }

    def hostVector: Array[Float] = {
      if (url == null) zeroVector
      else model.getOrElse(url.getFirstLevelDomain, zeroVector)
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
        _params = url.getQueryParams.map(_.split("=")(0))
      _params
    }

    def paramsVector(i: Int) = {
      if (url == null) zeroVector
      else model.getOrElse(params(i - 1 - path.length), zeroVector)
    }

    @inline override def hasNext: Boolean = i < nUrlFeatures

    @inline override def next() = {
      try {
        if (url ==null) zeroVector
        else if (i == 0) hostVector
        else if (i - 1 < path.length) pathVector(i)
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
    val n = x.length
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


  def urlFeatures(u: String, urlModelMap: mutable.Map[String, Array[Float]], domainModelMap: mutable.Map[String, Array[Float]], domainIdMap:Map[String,Int]) = {
    val sentenceVecs = new UrlVectorsIterator(u, urlModelMap)
    var featureCount = 0
    val urlVector = Array.fill[Float](201)(0.0f)
    sentenceVecs.foreach { v =>
      axpy(1.0f, v, urlVector)
      featureCount+=1
    }
    if (featureCount>0) {
      val scaleFactor = 1.0f/nUrlFeatures
      scal(scaleFactor, urlVector)
    }
    val d = domain(u)
    val domainVector = domainModelMap.getOrElse(d, domainModelMap.get(noDomainString).get)
    Array.copy(domainVector,0,urlVector,100,100)
    urlVector.update(200, domainIdMap.getOrElse(d,-1).toFloat)
    urlVector
  }


  def eventFeatures(e: Data.RawEvent, urlModelMap: mutable.Map[String, Array[Float]], domainModelMap: mutable.Map[String, Array[Float]], domainIdMap:Map[String,Int]) = {
    // [request features][href features][prev features][timestamp]
    val feature = EventFeature(
      urlFeatures(e.requestUrl, urlModelMap,domainModelMap, domainIdMap),
      urlFeatures(e.referrerUrl, urlModelMap,domainModelMap, domainIdMap),
      urlFeatures(e.prevUrl, urlModelMap, domainModelMap, domainIdMap),
      e.timestamp
    )

    val res = Data.EventWithFeatures(e, feature)
    //println(t1 - t0)
    res

  }

}
