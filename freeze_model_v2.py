import tensorflow as tf

meta_path = 'longleaf_model/snapshot-469500.meta' # Your .meta file
output_node_names = ['pose/locref_pred/block4/BiasAdd', 'pose/part_pred/block4/BiasAdd']    # Output nodes

with tf.compat.v1.Session() as sess:
    # Restore the graph
    saver = tf.compat.v1.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.compat.v1.train.latest_checkpoint('longleaf_model/'))

    # Freeze the graph
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names,)

    # Save the frozen graph
    with open('longleaf_model/longleaf_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())