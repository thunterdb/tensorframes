package org.tensorframes.impl

import java.io.File
import java.nio.file.{Files, Paths}

import com.jd.util.NativeUtils
import org.bytedeco.javacpp.{BytePointer, tensorflow => jtf}
import org.tensorflow.framework.GraphDef
import org.{tensorflow => tf}
import org.apache.spark.sql.types.NumericType
import org.tensorflow.{Graph, Session}
import org.tensorframes.test.ProtoConversions
import org.tensorframes.{Logging, Shape, ShapeDescription}

import scala.collection.JavaConverters._

/**
  * Contains a TensorFlow graph that has been serialized using protocol buffers.
  *
  * In order to limit the amount of memory being used by this class, it has the ability to dump its content onto
  * a disk file, and serve the data from this disk file.
  */
case class SerializedGraph private (
  private var _content: Option[Array[Byte]],
  private var file: Option[String]) extends Serializable with Logging {

  def content: Array[Byte] = file match {
    case Some(name) =>
      val p = Paths.get(name)
      Files.readAllBytes(p)
    case None =>
      _content.getOrElse {
        throw new Exception(s"Missing content for serialized graph $this")
      }
  }

  /**
    * Moves the graph description to a file, and drops the in-memory representation once it is safe to do so.
    */
  def evictContent(): Unit = this.synchronized {
    if (file.isDefined) {
      return // Nothing to do here
    }
    val bytes = _content.getOrElse {
      throw new Exception(s"Missing content for serialized graph $this")
    }
    val tempFile = File.createTempFile("tensorframes-graphs-", "-proto-bin")
    tempFile.deleteOnExit()
    SerializedGraph.logInfo(s"Evicting graph to temporary file $tempFile...")
    Files.write(tempFile.toPath, bytes)
    file = Some(tempFile.toString)
    _content = None
    SerializedGraph.logInfo(s"Done evicting graph graph: $this")
  }
}

object SerializedGraph extends Logging {
  // Stored in memory by default, so that the broadcast mechanism can send it around.
  def create(content: Array[Byte]): SerializedGraph = {
    require(content != null)
    new SerializedGraph(Some(content), None)
  }
}

/**
 * Some low-level tensorflow operations.
 */
object TensorFlowOps extends Logging {

  private[this] val lock = new Object

  lazy val _init = lock.synchronized {
    val x = new Exception().getStackTraceString
    logDebug("Starting TensorFlowOps...")
    logInfo("Starting TensorFlowOps... origin:"+x)
    jtf.InitMain("test", Array.empty[Int], null)
    logInfo("Starting TensorFlowOps... Done")
    true
  }

  def initTensorFlow(): Unit = {
    _init
  }

  def graphSerial(g: jtf.GraphDef): Array[Byte] = {
    val n = g.ByteSizeLong()
    assert(n < Int.MaxValue, s"Cannot serialize graphs of size more than ${Int.MaxValue} " +
      s"(trying to serialize a graph of size $n bytes")
    val arr = Array.fill[Byte](g.ByteSizeLong().toInt)(0)
    g.SerializeWithCachedSizesToArray(arr)
    arr
  }

  def graphSerial(g: GraphDef): SerializedGraph = {
    SerializedGraph.create(g.toByteString.toByteArray)
  }

  def readGraphSerial(arr: SerializedGraph): GraphDef = {
    GraphDef.parseFrom(arr.content)
  }

  def withSession[T](g: SerializedGraph)(f: tf.Session => T): T = {
    val graph2 = new Graph()
    graph2.importGraphDef(g.content)
    val session = new Session(graph2)
    try {
      f(session)
    } finally {
      session.close()
      graph2.close()
    }
  }


  def readGraph(sg: SerializedGraph): jtf.GraphDef = {
    val res = new jtf.GraphDef()
    val arr = sg.content
    val p = new BytePointer(arr.length)
    p.put(arr, 0, arr.length)
    jtf.ParseProtoUnlimited(res, p)
    res
  }


  def withSession[T](f: jtf.Session => T): T = {
    initTensorFlow()
    val options = new jtf.SessionOptions()
    val session = new jtf.Session(options)
    try {
      f(session)
    } finally {
      session.Close()
    }
  }

  def jtfShape(s: jtf.TensorShape): Shape = {
    val dims = (0 until s.dims()).map(s.dim_size).toArray
    Shape(dims)
  }

  def shape(sh: Shape): jtf.TensorShape = {
    val s = new jtf.TensorShape()
    sh.dims.foreach { dim =>
      s.AddDim(dim)
    }
    s
  }


  /**
    * Performs some analysis over the TF graph, by loading it into the TF runtime and extracting
    * the shapes of the various components in it.
    */
  def analyzeGraphTF(
      graphDef: GraphDef,
      shapeHints: ShapeDescription = ShapeDescription.empty): Seq[GraphNodeSummary] = {

    val nodes = graphDef.getNodeList.asScala
    val inputs: Set[String] = nodes
      .filter(n => n.getInputCount == 0 && n.getOp == "Placeholder")
      .map(_.getName).toSet
    // We identify a node with its output tensor.
    val outputs = shapeHints.requestedFetches.map(_.stripSuffix(":0")).toSet
    logDebug(s"Outputs: ${outputs}")

    // Test that the graph can be imported
    {
//      val g = new Graph()
      val ser = graphSerial(graphDef).content
      logInfo(s"analyzeGraphTF: the graph has size ${ser.length.toLong/1000000} MB")
//      g.importGraphDef(ser)
//      g.close()
    }

    nodes.flatMap { n =>
      val name = n.getName
      logTrace(s"Node $name")
      val isInput = inputs.contains(name)
      val isOutput = outputs.contains(name)
      if (isInput || isOutput) {
        // The shape stored in the graph seems buggy sometimes (when there are some unknowns)
        // Trust the one from the shape hints.
        val shapeOpt = shapeHints.out.get(name).orElse {
          // The name may include the default output slot
          // TODO(tjh) add a test for that
          shapeHints.out.get(name + ":0")
        } .orElse {
          if (n.getAttr.containsKey("shape")) {
            Some(Shape.from(n.getAttr.get("shape").getShape))
          } else {
            None
          }
        }
        logTrace(s"shape = $shapeOpt")
        val shape = shapeOpt.getOrElse {
          throw new Exception(s"Could not get the shape of node $name from the graph definition or from the shape hints")
        }
        val scalarType = SupportedOperations.opsFor(ProtoConversions.getDType(n)).sqlType
        Some(GraphNodeSummary(isInput, isInput, isOutput, scalarType, shape, name))
      } else { None }
    }
  }


  /**
   * Performs some analysis over the TF graph, by loading it into the TF runtime and extracting
   * the shapes of the various components in it.
   */
  def analyzeGraph(
      graphDef: GraphDef,
      shapeHints: ShapeDescription = ShapeDescription.empty): Seq[GraphNodeSummary] = {
    initTensorFlow()
//    logTrace(s"analyzeGraph: shapeHints=$shapeHints")
//    logTrace(s"analyzeGraph: graph=$graphDef")

    val nodes = graphDef.getNodeList.asScala
    val inputs: Set[String] = nodes
      .filter(n => n.getInputCount == 0 && n.getOp == "Placeholder")
      .map(_.getName).toSet
    // We identify a node with its output tensor.
    val outputs = shapeHints.requestedFetches.map(_.stripSuffix(":0")).toSet
    logDebug(s"Outputs: ${outputs}")

    withSession { session =>
      val g = readGraph(graphSerial(graphDef))
      val s1 = session.Extend(g)
      assert(s1.ok(), s1.error_message().getString)
      val options = new jtf.GraphConstructorOptions()
      val registry = jtf.OpRegistry.Global()
      val graph = new jtf.Graph(registry)
      val s2 = jtf.ConvertGraphDefToGraph(options, g, graph)
      val nodes = {
        val x = graph.nodes()
        var res: List[jtf.Node] = Nil
        val it = x.begin()
        while (it.notEquals(x.end())) {
          res ::= it.access()
          it.increment()
        }
        res
      }
      logDebug(s"Extracted ${nodes.size} nodes")
      // TODO: move this within the iterator, the nodes it attempts to access may have been deallocated at that point.
      nodes.filter(_.id() >= 2).map { node =>
        val id = node.id()
        val name = "?"
//        val name = node.name() // crash
        val op = "" //node.op_def().name().getString
//        val outputs = {
//          val l = node.output_types().size().toInt
//          (0 until 0).map { idx =>
//            node.output_type(idx)
//          }
//        }
        logDebug(s"Node: id=$id name=$name op=$op debug=${node.DebugString().getString}")
      }
      assert(s2.ok(), s2.error_message().getString)
    }
    nodes.flatMap { n =>
      val name = n.getName
      logTrace(s"Node $name")
      val isInput = inputs.contains(name)
      val isOutput = outputs.contains(name)
      if (isInput || isOutput) {
        // The shape stored in the graph seems buggy sometimes (when there are some unknowns)
        // Trust the one from the shape hints.
        val shapeOpt = shapeHints.out.get(name).orElse {
          // The name may include the default output slot
          // TODO(tjh) add a test for that
          shapeHints.out.get(name + ":0")
        } .orElse {
          if (n.getAttr.containsKey("shape")) {
            Some(Shape.from(n.getAttr.get("shape").getShape))
          } else {
            None
          }
        }
        logTrace(s"shape = $shapeOpt")
        val shape = shapeOpt.getOrElse {
          throw new Exception(s"Could not get the shape of node $name from the graph definition or from the shape hints")
        }
        val scalarType = SupportedOperations.opsFor(ProtoConversions.getDType(n)).sqlType
        Some(GraphNodeSummary(isInput, isInput, isOutput, scalarType, shape, name))
      } else { None }
    }
  }

  def stringVector(strings: Seq[String]): jtf.StringVector = {
    val o = new jtf.StringVector(strings.length)
    strings.indices.foreach { idx =>
      o.put(idx, strings(idx))
    }
    o
  }

}

/**
 * All the informations requested by TensorFrames to run on a graph node.
 *
 * @param isPlaceholder if the variable is a placeholder
 * @param isInput if the node is an input (no inner dependencies)
 * @param isOutput if it is an outpu (no node depends on it)
 * @param scalarType the scalar type of the final tensor associated to this node
 * @param shape the shape of the final tensor associated to this node
 * @param name the name of the node in the graph
 */
case class GraphNodeSummary(
    isPlaceholder: Boolean,
    isInput: Boolean,
    isOutput: Boolean,
    scalarType: NumericType,
    shape: Shape,
    name: String) extends Serializable
