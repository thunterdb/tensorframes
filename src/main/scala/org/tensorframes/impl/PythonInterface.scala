package org.tensorframes.impl

import java.util

import scala.collection.JavaConverters._

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.sql.{GroupedData, DataFrame, Row}
import org.apache.spark.sql.types.StructType
import org.tensorflow.framework.GraphDef
import org.tensorframes._

/**
 *
 * @param fieldName
 * @param shape the shape of blocks for this field in a dataframe.
 */
case class FieldInfo(fieldName: String, shape: util.ArrayList[Integer])

/**
  * Wrapper for python interop.
  */
private[tensorframes] trait PythonInterface { self: OperationsInterface with ExperimentalOperations =>
  import PythonInterface._

  def initialize_logging(): Unit = initialize_logging("org/tensorframes/log4j.properties")

  /**
    * Performs some logging initialization before spark has the time to do it.
    *
    * Because of the the current implementation of PySpark, Spark thinks it runs as an interactive
    * console and makes some mistake when setting up log4j.
    */
  private def initialize_logging(file: String): Unit = {
    Option(PythonInterface.getClass.getClassLoader.getResource(file)) match {
      case Some(url) =>
        PropertyConfigurator.configure(url)
      case None =>
        System.err.println(s"$this Could not load logging file $file")
    }
  }

  def map_blocks(dataframe: DataFrame, trim: Boolean): PythonOpBuilder = {
    if (trim) {
      new PythonOpBuilder(this, MapBlockTrimmed, dataframe)
    } else {
      new PythonOpBuilder(this, MapBlock, dataframe)
    }
  }

  def map_rows(dataFrame: DataFrame): PythonOpBuilder = {
    new PythonOpBuilder(this, MapRow, dataFrame)
  }

  def reduce_blocks(dataFrame: DataFrame): PythonOpBuilder = {
    new PythonOpBuilder(this, ReduceBlock, dataFrame)
  }

  def reduce_rows(dataFrame: DataFrame): PythonOpBuilder = {
    new PythonOpBuilder(this, ReduceRow, dataFrame)
  }

  def aggregate_blocks(groupedData: GroupedData): PythonOpBuilder = {
    new PythonOpBuilder(this, AggregateBlock, null, groupedData)
  }

  /**
   * More information about a dataframe, to help the python side build automatic placeholders.
 *
   * @param dataFrame
   */
  def extra_schema_info(dataFrame: DataFrame): util.List[FieldInfo] = {
    dataFrame.schema.flatMap { f =>
      val ci = ColumnInformation(f)
      ci.stf.map { s =>
        val a = new util.ArrayList(s.shape.dims.map(x => new Integer(x.toInt)).asJava)
        FieldInfo(ci.columnName, a)
      }
    }   .asJava
  }
}

class PythonOpBuilder(
    interface: OperationsInterface with ExperimentalOperations,
    op: PythonInterface.Operation,
    df: DataFrame = null,
    groupedData: GroupedData = null) {
  import PythonInterface._
  private var _shapeHints: ShapeDescription = ShapeDescription.empty
  private var _graph: GraphDef = null

  def shape(
      shapeHintsNames: util.ArrayList[String],
      shapeHintShapes: util.ArrayList[util.ArrayList[Int]]): this.type = {
    val s = shapeHintShapes.asScala.map(_.asScala.toSeq).map(x => Shape(x: _*))
    _shapeHints = _shapeHints.copy(out = shapeHintsNames.asScala.zip(s).toMap)
    this
  }

  def fetches(fetchNames: util.ArrayList[String]): this.type = {
    _shapeHints = _shapeHints.copy(requestedFetches = fetchNames.asScala.toSeq)
    this
  }

  def graph(bytes: Array[Byte]): this.type = graph(bytes, null)

  def graph(bytes: Array[Byte], graphId: String): this.type = {
    _graph = TensorFlowOps.readGraphSerial(bytes)
    if (graphId != null) {
      MemoizedGraphs.default.register(graphId, _graph)
    }
    this
  }

  def session(s: String): this.type = {
    _shapeHints = _shapeHints.copy(requestedSession = Some(s))
    this
  }

  def reuseGraph(graphId: String): Boolean = {
    MemoizedGraphs.default.registeredGraph(graphId) match {
      case Some(g) =>
        _graph = g
        true
      case None => false
    }
  }

  def buildRow(): DataFrame = op match {
    case ReduceBlock =>
      val allSchema = SchemaTransforms.reduceBlocksSchema(
        df.schema, _graph, _shapeHints)
      val r = interface.reduceBlocks(df, _graph, _shapeHints)
      wrapDF(r, allSchema.output)
    case ReduceRow =>
      val outSchema: StructType = SchemaTransforms.reduceRowsSchema(
        df.schema, _graph, _shapeHints)
      val r = interface.reduceRows(df, _graph, _shapeHints)
      wrapDF(r, outSchema)
    case x =>
      throw new Exception(s"Programming error: $x")
  }

  def buildDF(): DataFrame = op match {
    case MapBlock => interface.mapBlocks(df, _graph, _shapeHints)
    case MapBlockTrimmed => interface.mapBlocksTrimmed(df, _graph, _shapeHints)
    case MapRow => interface.mapRows(df, _graph, _shapeHints)
    case AggregateBlock => interface.aggregate(groupedData, _graph, _shapeHints)
    case x =>
      throw new Exception(s"Programming error: $x")
  }

  private def wrapDF(r: Row, schema: StructType): DataFrame = {
    val a = new util.ArrayList[Row]()
    a.add(r)
    df.sqlContext.createDataFrame(a, schema)
  }
}

private object PythonInterface {
  sealed trait Operation
  case object MapBlock extends Operation
  case object MapBlockTrimmed extends Operation
  case object MapRow extends Operation
  case object ReduceBlock extends Operation
  case object ReduceRow extends Operation
  case object AggregateBlock extends Operation
}
