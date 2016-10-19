package org.tensorframes.impl

import scala.util.control.NonFatal
import org.bytedeco.javacpp.{tensorflow => jtf}
import org.tensorflow.framework.GraphDef
import org.tensorframes.{Logging, ShapeDescription}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

/**
 * Builds some caches for the graph objects, to limit the cost of analysis and communication.
 */
trait MemoizedGraphs {
  def registeredGraph(graphId: String): Option[GraphDef]
  def register(grapdId: String, g: GraphDef): Unit
  def analyze(g: GraphDef, hints: ShapeDescription): Seq[GraphNodeSummary]
  def broadcastGraph(g: GraphDef): Broadcast[Array[Byte]]
  def graphId(g: GraphDef): Int = math.abs(g.hashCode())
}

object MemoizedGraphs {
  lazy val _impl = new MemoizedImpl()
  lazy val default: MemoizedGraphs = {
    _impl
    //MemoizedNoCache
  }

  def reset(): Unit = {
    _impl.reset()
  }
}

object MemoizedNoCache extends MemoizedGraphs {
  def sc: SparkContext = SparkContext.getOrCreate()
  override def registeredGraph(graphId: String): Option[GraphDef] = None
  override def register(grapdId: String, g: GraphDef): Unit = {}

  override def analyze(graph: GraphDef, shapeHints: ShapeDescription): Seq[GraphNodeSummary] = {
    TensorFlowOps.analyzeGraph(graph, shapeHints)
  }
  override def broadcastGraph(g: GraphDef): Broadcast[Array[Byte]] = {
    sc.broadcast(TensorFlowOps.graphSerial(g))
  }
}

class MemoizedImpl() extends MemoizedGraphs {
  var graphIds: Map[String, GraphDef] = Map.empty
  var graphs: Map[Int, GraphDef] = Map.empty
  var bcasts: Map[Int, Broadcast[Array[Byte]]] = Map.empty
  var analysis: Map[(Int, ShapeDescription), Seq[GraphNodeSummary]] = Map.empty

  def reset(): Unit = {
    graphIds = Map.empty
    graphs = Map.empty
    bcasts = Map.empty
    analysis = Map.empty
  }

  def sc = SparkContext.getOrCreate()
  override def registeredGraph(graphPythonId: String): Option[GraphDef] = graphIds.get(graphPythonId)
  override def register(graphPythonId: String, g: GraphDef): Unit = {
    graphIds += graphPythonId -> g
    val k = graphId(g)
    graphs += k -> g
  }
  override def analyze(g: GraphDef, hints: ShapeDescription): Seq[GraphNodeSummary] = {
    val gid = graphId(g)
    if (!graphs.contains(gid)) {
      graphs += gid -> g
    }
    val k = gid -> hints
    analysis.get(k).getOrElse {
      val res = TensorFlowOps.analyzeGraph(g, hints)
      analysis += k -> res
      res
    }
  }

  override def broadcastGraph(g: GraphDef): Broadcast[Array[Byte]] = {
    val k = graphId(g)
    bcasts.get(k).getOrElse {
      val res = sc.broadcast(TensorFlowOps.graphSerial(g))
      bcasts += k -> res
      res
    }
  }
}

object MemoizedSessions extends Logging {
  var savedSessions: Map[String, jtf.Session] = Map.empty

  def reset(): Unit = synchronized {
    savedSessions.foreach { case (name, session) =>
      logInfo(s"reset: closing session $name")
      try {
        session.close()
      } catch {
        case NonFatal(e) =>
          logInfo(s"ERROR Session $name did not close as expected" + e)
      }
    }
    savedSessions = Map.empty
  }

  def withSession[T](sessionName: Option[String],
                     graphDataBC: Broadcast[Array[Byte]])(fun: jtf.Session => T): T = synchronized {
    val res = sessionName match {
      case None =>
        logInfo(s"trying to enter regular session name:$sessionName bc:$graphDataBC")
        TensorFlowOps.withSession { session =>
          val graphData = graphDataBC.value
          val g = TensorFlowOps.readGraph(graphData)
          val s1 = session.Extend(g)
          assert(s1.ok(), s1.error_message().getString)
          fun(session)
        }
      case Some(name) =>
        val sess = savedSessions.get(name) match {
          case Some(s) =>
            logInfo(s"trying to enter reuse session name:$sessionName bc:$graphDataBC")
            s
          case None =>
            logInfo(s"trying to create session for reuse name:$sessionName bc:$graphDataBC")
            TensorFlowOps.initTensorFlow()
            val options = new jtf.SessionOptions()
            val graphData = graphDataBC.value
            val g = TensorFlowOps.readGraph(graphData)
            val session = new jtf.Session(options)
            val s1 = session.Extend(g)
            assert(s1.ok(), s1.error_message().getString)
            savedSessions += name -> session
            session
        }
        fun(sess)
    }
    logInfo(s"Session finished:$sessionName")
    res
  }
}