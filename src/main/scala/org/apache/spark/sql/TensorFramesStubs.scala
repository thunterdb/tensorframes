package org.apache.spark.sql

import org.apache.spark.sql.catalyst.expressions.ScalaUDF
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.DataType

class TFUDF(
    fun: Column => (Row => Row), returnType: DataType)
  extends UserDefinedFunction(null, returnType, None) {
  override def apply(exprs: Column*): Column = {
    assert(exprs.size == 1, exprs)
    val f = fun(exprs.head)
    Column(ScalaUDF(f, dataType, exprs.map(_.expr), Nil))
  }
}

class PipelinedUDF(udfs: Seq[UserDefinedFunction], returnType: DataType)
  extends UserDefinedFunction(null, returnType, None) {
  assert(udfs.nonEmpty)

  override def apply(exprs: Column*): Column = {
    val start = udfs.head.apply(exprs: _*)
    var rest = start
    for (udf <- udfs.tail) {
      rest = udf.apply(rest)
    }
    rest
  }
}

object TFUDF {
  def make1(fun: Column => (Row => Row), returnType: DataType): TFUDF = {
    new TFUDF(fun, returnType)
  }

  def makeUDF[U,V](f: U => V, returnType: DataType): UserDefinedFunction = {
    UserDefinedFunction(f, returnType, None)
  }

  def pipeline(udf1: UserDefinedFunction, udfs: UserDefinedFunction*): UserDefinedFunction = {
    if (udfs.isEmpty) {
      udf1
    } else {
      new PipelinedUDF(Seq(udf1) ++ udfs, udfs.last.dataType)
    }
  }
}
