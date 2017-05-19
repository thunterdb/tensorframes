package org.tensorframes.impl_images

import java.nio.file.{Files, Paths}
import java.util

import org.apache.log4j.PropertyConfigurator

import scala.collection.JavaConverters._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.tfs_stubs.TensorFramesUDF
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.tensorflow.framework.GraphDef
import org.tensorframes.{Logging, Shape, ShapeDescription}
import org.tensorframes.impl.{SerializedGraph, SqlOps, TensorFlowOps}

/**
 * Small, python-accessible wrapper to load and register models in Spark.
 */
class ModelFactory {

}

// TODO: merge with the python factory eventually, this is essentially copy/paste
class PythonModelFactory() extends Logging {
  private var _shapeHints: ShapeDescription = ShapeDescription.empty
  // TODO: this object may leak because of Py4J -> do not hold to large objects here.
  private var _graph: SerializedGraph = null
  private var _graphPath: Option[String] = None
  private var _sqlCtx: SQLContext = null

  def initialize_logging(): Unit = initialize_logging("org/tensorframes/log4j.properties")

  /**
   * Performs some logging initialization before spark has the time to do it.
   *
   * Because of the the current implementation of PySpark, Spark thinks it runs as an interactive
   * console and makes some mistake when setting up log4j.
   */
  private def initialize_logging(file: String): Unit = {
    Option(this.getClass.getClassLoader.getResource(file)) match {
      case Some(url) =>
        PropertyConfigurator.configure(url)
      case None =>
        System.err.println(s"$this Could not load logging file $file")
    }
  }


  def shape(
      shapeHintsNames: util.ArrayList[String],
      shapeHintShapes: util.ArrayList[util.ArrayList[Int]]): this.type = {
    val s = shapeHintShapes.asScala.map(_.asScala.toSeq).map(x => Shape(x: _*))
    _shapeHints = _shapeHints.copy(out = shapeHintsNames.asScala.zip(s).toMap)
    this
  }

  def fetches(fetchNames: util.ArrayList[String]): this.type = {
    _shapeHints = _shapeHints.copy(requestedFetches = fetchNames.asScala)
    this
  }

  def graph(bytes: Array[Byte]): this.type = {
    _graph = SerializedGraph.create(bytes)
    this
  }

  def graphFromFile(filename: String): this.type = {
    _graphPath = Option(filename)
    this
  }

  def sqlContext(ctx: SQLContext): this.type = {
    _sqlCtx = ctx
    this
  }

  def inputs(
      placeholderPaths: util.ArrayList[String],
      fieldNames: util.ArrayList[String]): this.type = {
    require(placeholderPaths.size() == fieldNames.size(), (placeholderPaths.asScala, fieldNames.asScala))
    val map = placeholderPaths.asScala.zip(fieldNames.asScala).toMap
    _shapeHints = _shapeHints.copy(inputs = map)
    this
  }

  private def buildGraphDef(): GraphDef = {
    _graphPath match {
      case Some(p) =>
        val path = Paths.get(p)
        val bytes = Files.readAllBytes(path)
        TensorFlowOps.readGraphSerial(SerializedGraph.create(bytes))
      case None =>
        assert(_graph != null)
        TensorFlowOps.readGraphSerial(_graph)
    }
  }

  /**
   * Builds a java UDF based on the following input.
   *
   * This is take away from PythonInterface
   *
   */
  // TODO: merge with PythonInterface. It is easier to keep separate for the time being.
  def makeUDF(applyBlocks: Boolean): UserDefinedFunction = {
    SqlOps.makeUDF(buildGraphDef(), _shapeHints, applyBlocks=applyBlocks)
  }

  /**
   * Registers a TF UDF under the given name in Spark.
   * @param udfName the name of the UDF
   * @param blocked indicates that the UDF should be applied block-wise.
   * @return
   */
  def registerUDF(udfName: String, blocked: java.lang.Boolean): UserDefinedFunction = {
    assert(_sqlCtx != null)
    val udf = makeUDF(blocked)
    logger.warn(s"Registering udf $udfName -> $udf to session ${_sqlCtx.sparkSession}")
    TensorFramesUDF.registerUDF(_sqlCtx.sparkSession, udfName, udf)
  }
}
