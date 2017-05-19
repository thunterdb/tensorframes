
from __future__ import print_function

from pyspark import SparkContext
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql import Row
import tensorflow as tf

from tensorframes.sql import registerUDF, _java_api

class TestSql(object):

    @classmethod
    def setup_class(cls):
        print("setup ", cls)
        cls.sc = SparkContext('local[1]', cls.__name__)
        cls.sc.setLogLevel('INFO')

    @classmethod
    def teardown_class(cls):
        print("teardown ", cls)
        cls.sc.stop()

    def setUp(self):
        self.sql = SQLContext(TestSql.sc)
        self.api = _java_api(sqlCtx=self.sql)
        self.api.initialize_logging()
        print("setup")


    def teardown(self):
        print("teardown")

    def test_map_rows_sql_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # Let's register these computations in SQL.
            registerUDF(z, "map_rows_sql_1")
        # Here we go, for the SQL users, straight from PySpark.
        df2 = df.selectExpr("map_rows_sql_1(x).z AS z")
        print("df2 = %s" % df2)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2


    def test_map_blocks_sql_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # Let's register these computations in SQL.
            registerUDF(z, "map_blocks_sql_1", blocked=True)
        # Here we go, for the SQL users, straight from PySpark.
        df2 = df.selectExpr("map_blocks_sql_1(x).z AS z")
        print("df2 = %s" % df2)
        data2 = df2.collect()
        assert len(data2) == 5, data2
        assert data2[0].z == 3.0, data2


if __name__ == "__main__":
    # Some testing stuff that should not be executed
    with tf.Graph().as_default() as g:
        x_input = tf.placeholder(tf.double, shape=[2, 3], name="x_input")
        x = tf.reduce_sum(x_input, [0], name='x')
        print(g.as_graph_def())

    with tf.Graph().as_default() as g:
        x = tf.constant([1, 1], name="x")
        y = tf.reduce_sum(x, [0], name='y')
        print(g.as_graph_def())

    with tf.Graph().as_default() as g:
        tf.constant(1, name="x1")
        tf.constant(1.0, name="x2")
        tf.constant([1.0], name="x3")
        tf.constant([1.0, 2.0], name="x4")
        print(g.as_graph_def())
