

import logging

from pyspark import RDD, SparkContext
from pyspark.sql import SQLContext, Row, DataFrame

from .core import _check_fetches, _get_graph, _add_graph, _add_shapes, _add_inputs

#__all__ = ['registerUDF']


_sc = None
_sql = None
logger = logging.getLogger('tensorframes')


def _java_api(javaClassName = "org.tensorframes.impl_images.PythonModelFactory", sqlCtx = None):
    """
    Loads the PythonInterface object (lazily, because the spark context needs to be initialized
    first).
    """
    global _sc, _sql
    if _sc is None:
        _sc = SparkContext._active_spark_context
        logger.info("Spark context = " + str(_sc))
        if not sqlCtx:
            _sql = SQLContext(_sc)
        else:
            _sql = sqlCtx
    _jvm = _sc._jvm
    # You cannot simply call the creation of the the class on the _jvm due to classloader issues
    # with Py4J.
    return _jvm.Thread.currentThread().getContextClassLoader().loadClass(javaClassName) \
        .newInstance()


def registerUDF(fetches, name, feed_dict=None, blocked=False):
    """ Registers a transform as a SQL UDF in Spark that can then be embedded inside SQL queries.

    Note regarding performance and resource management: registering a TensorFlow object as a SQL UDF
    will open resources both on the driver (registration, etc.) and also in each of the workers: the
    resources used by the TensorFlow programs (GPU memory, memory buffers, etc.) will not be released
     until the end of the Spark application, or until another UDF is registered. The only exact way
    to release all the resources is currently to restart the cluster.

    :param fetches:
    :param name: the name of the UDF that will be used in SQL
    :param feed_dict
    :param blocked: if set to true, the tensorflow program is expected to manipulate blocks of data at the same time.
      If the number of rows returned is different than the number of rows ingested, an error is raised.
    :return: nothing
    """
    fetches = _check_fetches(fetches)
    # We are not dealing for now with registered expansions, but this is something we should add later.
    graph = _get_graph(fetches)
    builder = _java_api()
    _add_graph(graph, builder)
    ph_names = _add_shapes(graph, builder, fetches)
    _add_inputs(builder, feed_dict, ph_names)
    # Set the SQL context for the builder.
    builder.sqlContext(_sql._ssql_ctx)
    builder.registerUDF(name, blocked)
