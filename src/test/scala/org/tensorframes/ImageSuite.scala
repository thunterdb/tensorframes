package org.tensorframes

import org.scalatest.FunSuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.struct
import org.tensorframes.dsl.Implicits._
import org.tensorframes.dsl._
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.impl_images.ImageOps

// Some basic operations that stress shape transforms mostly.
class ImageSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging with GraphScoping {
  lazy val sql = sqlContext
  import Shape.Unknown

  val ops = new DebugRowOps
  val tfDataDir = "/tmp/tensorframes-model"
  val imageDir = "/tmp/tensorframes-images"

  testGraph("basic test") {
    import sql.implicits._
    val rdd = sc.binaryFiles(imageDir).map { case (name, s) => name -> s.toArray() }
    val df = rdd.toDF().toDF("uris", "data")
    // For now, the input name, shape, etc. is hardcoded and nod discovered.
    // Do not forget to wrap the content in a structure.
    val col = struct(df("data").alias("image_input")).alias("image_data")
    val images = df.select(df("uris"), col)
    images.show()

    val imageUdf = ImageOps.makeImageClassifier(tfDataDir)
    val df2 = images.select(images("uris"), imageUdf(images("image_data")))
    df2.show()
  }
}
