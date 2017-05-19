package org.tensorframes.impl_images

import java.net.URL
import java.nio.charset.Charset
import java.nio.file.{Files, Paths}
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.tfs_stubs.TFUDF
import org.tensorflow.framework.GraphDef
import org.tensorframes.impl.SqlOps
import org.tensorframes.{Logging, ShapeDescription}

import scala.collection.JavaConverters._

object FileOps extends Logging {
  val modelDownloaded = new AtomicBoolean(false)

  val localModelPath: String = {
    Paths.get(
      System.getProperty("java.io.tmpdir"), "inceptionv3-" + UUID.randomUUID().toString).toString
  }

  def downloadFile(url: String): Array[Byte] = {
    // A hack to install model by hijacking donwload file ...
    downloadAndInstallModel(localModelPath)

    val stream = new URL(url).openStream()
    val bytes = org.apache.commons.io.IOUtils.toByteArray(stream)
    stream.close()
    bytes
  }

  def downloadAndInstallModel(localPath: String): Unit = {
    def download(sourceUrl: String): Unit = {
      val url = new URL(sourceUrl)
      val localFile = Paths.get(localPath, FilenameUtils.getName(url.getFile)).toFile
      localFile.getParentFile.mkdirs()
      FileUtils.copyURLToFile(url, localFile)
    }

    // Download the model, but only do it once per process.
    if (!modelDownloaded.get) {
      modelDownloaded.synchronized {
        if (!modelDownloaded.get) {
          download("http://home.apache.org/~rxin/models/inceptionv3/classes.txt")
          download("http://home.apache.org/~rxin/models/inceptionv3/main.pb")
          download("http://home.apache.org/~rxin/models/inceptionv3/preprocessor.pb")
          modelDownloaded.set(true)
        }
      }
    }
  }
}

/**
 * Experimental class that contains the image-related code.
 */
object ImageOps extends Logging {

  def postprocessUDF(classes: Seq[String], threshold: Double): UserDefinedFunction = {
    def f2(input: Row): Array[String] = input match {
      case Row(x: Seq[Any]) =>
        assert(x.size == classes.size, s"output row has size ${x.size} but we have ${classes.size}")
        val x2 = x.map {
          case z: Float => z.toDouble
          case z: Double => z.toDouble
          case z => throw new Exception(s"Cannot cast element of type ${z.getClass}: $z")
        }
        val cs = classes.zip(x2).filter(_._2 > threshold).map(_._1)
        logger.debug(s"Found classes: $cs")
        cs.toArray
      case x =>
        throw new Exception(s"Expected row with array of double, got $x")
    }

    val returnType = ArrayType(StringType, containsNull = false)

    val udf2 = TFUDF.makeUDF(f2, returnType)
    udf2
  }

  def makeImageClassifier(): UserDefinedFunction = {
    makeImageClassifier(FileOps.localModelPath)
  }


  /**
   * Makes an image classifier from an existing model. The model must be exported from Python first.
   * @param directory the directory that contains the .protobuf files with the preprocessor and the main
   *                  model. It is going to look for:
   *                   - preprocessor.pb
   *                   - main.pb
   * @return a full classifier that is using by default the ImageNet set of labels.
   */
  def makeImageClassifier(directory: String): UserDefinedFunction = {
    val preprocessorGraph = {
      val bytes = Files.readAllBytes(Paths.get(directory + "/preprocessor.pb"))
      GraphDef.parseFrom(bytes)
    }
    val graph = {
      val bytes = Files.readAllBytes(Paths.get(directory + "/main.pb"))
      GraphDef.parseFrom(bytes)
    }
    val preproHints = ShapeDescription(Map(), List("output"), Map("image_input" -> "image_input"))
    val mainHints = ShapeDescription(Map(), List("prediction_vector"), Map("image_input" -> "output"))
    val hints = Files.readAllLines(
      Paths.get(directory+"/classes.txt"), Charset.defaultCharset()).asScala.map(_.split(" ").toSeq).flatMap {
      case Seq(id: String, name: String) => Some(id.toInt -> name)
      case _ => None
    }
    val maxKey = hints.map(_._1).max
    val m = hints.toMap

    val classes = (0 to maxKey).map { x => m.get(x).getOrElse("none_" + x) }
    makeImageClassifier(preprocessorGraph, preproHints, graph, mainHints, classes)
  }

  def makeImageClassifier(
      preprocessGraph: GraphDef,
      preprocessShapeHints: ShapeDescription,
      graph: GraphDef,
      shapeHints: ShapeDescription,
      classes: Seq[String]): UserDefinedFunction = {
    val udf0 = SqlOps.makeUDF(preprocessGraph, preprocessShapeHints, applyBlocks = false)
    val udf1 = SqlOps.makeUDF(graph, shapeHints, applyBlocks = false)
    val udf2 = postprocessUDF(classes, 0.7)
    TFUDF.pipeline(udf0, udf1, udf2)
  }
}
