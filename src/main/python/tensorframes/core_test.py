from pyspark import SparkContext
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql import Row
import tensorflow as tf

import tensorframes as tfs
from tensorframes.core import _java_api

class TestCore(object):

    @classmethod
    def setup_class(cls):
        print "setup ", cls
        cls.sc = SparkContext('local[1]', cls.__name__)

    @classmethod
    def teardown_class(cls):
        print "teardown ", cls
        cls.sc.stop()

    def setUp(self):
        self.sql = SQLContext(TestCore.sc)
        self.api = _java_api()
        self.api.initialize_logging()
        print "setup"


    def teardown(self):
        print "teardown"

    def test_schema(self):
        data = [Row(x=float(x)) for x in range(100)]
        df = self.sql.createDataFrame(data)
        tfs.print_schema(df)

    def test_map_blocks_1(self):
        data = [Row(x=float(x)) for x in range(10)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default() as g:
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_rows_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default() as g:
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # The resulting dataframe
            df2 = tfs.map_rows(z, df)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2


    def test_reduce_rows_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default() as g:
            # The placeholder that corresponds to column 'x'
            x_1 = tf.placeholder(tf.double, shape=[], name="x_1")
            x_2 = tf.placeholder(tf.double, shape=[], name="x_2")
            # The output that adds 3 to x
            x = tf.add(x_1, x_2, name='x')
            # The resulting number
            res = tfs.reduce_rows(x, df)
        assert res == sum([r.x for r in data])

    # This test fails
    def test_reduce_blocks_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default() as g:
            # The placeholder that corresponds to column 'x'
            x_input = tf.placeholder(tf.double, shape=[None], name="x_input")
            # The output that adds 3 to x
            x = tf.reduce_sum(x_input, name='x')
            # The resulting dataframe
            res = tfs.reduce_blocks(x, df)
        assert res == sum([r.x for r in data])

    def test_map_blocks_trimmed_1(self):
        data = [Row(x=float(x)) for x in range(3)]
        df = self.sql.createDataFrame(data)
        with tf.Graph().as_default() as g:
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output discards the input and return a single row of data
            z = tf.constant([2], name='z')
            # The resulting dataframe
            df2 = tfs.map_blocks(z, df, trim=True)
        data2 = df2.collect()
        assert data2[0].z == 2, data2

    def test_map_rows_unsafe_1(self):
        data = [Row(x=float(x)) for x in range(3)]
        df = self.sql.createDataFrame(data)
        g = tf.Graph()
        with g.as_default():
            # The placeholder that corresponds to column 'x'
            x = tf.placeholder(tf.double, shape=[], name="x")
            # The output discards the input and return a single row of data
            z = tf.constant(2, name='z')
        df2 = tfs.map_rows(z, df, session="session", cache_graph=True)


if __name__ == "__main__":
    # Some testing stuff that should not be executed
    with tf.Graph().as_default() as g:
        x_input = tf.placeholder(tf.double, shape=[2, 3], name="x_input")
        x = tf.reduce_sum(x_input, [0], name='x')
        print g.as_graph_def()

    with tf.Graph().as_default() as g:
        x = tf.constant([1, 1], name="x")
        y = tf.reduce_sum(x, [0], name='y')
        print g.as_graph_def()

    with tf.Graph().as_default() as g:
        tf.constant(1, name="x1")
        tf.constant(1.0, name="x2")
        tf.constant([1.0], name="x3")
        tf.constant([1.0, 2.0], name="x4")
        print g.as_graph_def()


if __name__ == "foo":
    from pyspark import SparkContext
    from pyspark.sql import DataFrame, SQLContext
    from pyspark.sql import Row
    import tensorflow as tf
    import tensorframes as tfs
    sc.setLogLevel("DEBUG")
    data = [Row(x=float(x)) for x in range(3)]
    df = sqlContext.createDataFrame(data)

