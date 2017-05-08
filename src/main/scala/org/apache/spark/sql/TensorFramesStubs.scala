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
