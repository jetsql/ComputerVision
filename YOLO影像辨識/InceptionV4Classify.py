from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import configparser
import logging
import os
import time
import numpy as np
import ast
import sys
import pandas as pd
import tensorflow as tf
from PIL import Image
from logging.config import fileConfig
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.core.protobuf import saver_pb2


def init_logger():
    logging.log_file = 'inception_v4_predict.log'
    fileConfig('logging.ini')


class InceptionV4Classify():
    FLAGS = argparse.ArgumentParser().parse_args('')
    logger = None
    model_class_pd = None
    classes_num = 0
    TOTAL_RESULT = list()

    def __init_logger__(self):
        if self.logger is None:
            init_logger()
            self.logger = logging.getLogger('main')


    def do_main(self, argv):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--gpu_enable',
            type=str,
            default='N'
        )
        parser.add_argument(
            '--gpu_id',
            type=str,
            default=''
        )
        parser.add_argument(
            '--cpu_nums',
            type=int,
            default=1
        )
        parser.add_argument(
            '--intra_threads',
            type=int,
            default=2
        )
        parser.add_argument(
            '--model_path',
            type=str,
            default=''
        )
        parser.add_argument(
            '--image_path',
            type=str,
            default=''
        )
        parser.add_argument(
            '--result_file',
            type=str,
            default=''
        )

        sys.path.append('slim.zip')
        _FLAGS, unparsed = parser.parse_known_args(args=argv)
        self.__init_logger__()
        self.run_inference_on_image(tFLAGS=_FLAGS)
        sys.path.remove('slim.zip')


    def __setup_environ__(self):
        self.logger.info('inception_v4: setup_environ')
        self.logger.info('inception_v4: setup_environ : gpu_enable = %s' % self.FLAGS.gpu_enable)
        if self.FLAGS.gpu_enable is True:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1': INFO, '2' : WARN}
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.FLAGS.gpu_id


    def __initial_eval_FLAGS__(self):
        self.FLAGS.total_images = None
        self.FLAGS.intra_threads = 2
        self.FLAGS.inter_threads = 1
        self.FLAGS.num_cpu_core = 2
        self.FLAGS.max_num_batches = None
        self.FLAGS.batch_size = 1
        self.FLAGS.distort_image = False
        self.FLAGS.eval_image_size = None
        self.FLAGS.num_preprocessing_threads = 1
        self.FLAGS.model_name = 'inception_v4'
        self.FLAGS.preprocessing_name = None
        self.FLAGS.moving_average_decay = None
        self.FLAGS.checkpoint_path = ''
        self.FLAGS.master = ''
        self.FLAGS.dataset_dir = ''
        self.FLAGS.eval_dir = ''
        self.FLAGS.result_file = ''

    def __initial_FLAGS__(self, tFLAGS=None):
        self.__initial_eval_FLAGS__()

        if tFLAGS.gpu_enable == 'Y':
            self.FLAGS.gpu_enable = True
            self.FLAGS.gpu_id = tFLAGS.gpu_id
            self.FLAGS.gpu_memory_fraction = ast.literal_eval(tFLAGS.gpu_memory_fraction)
        else:
            self.FLAGS.gpu_enable = False

        self.FLAGS.model_name = 'inception_v4'
        self.FLAGS.num_cpu_core = tFLAGS.cpu_nums
        self.FLAGS.intra_threads = tFLAGS.intra_threads
        self.FLAGS.inter_threads = 1 # 控制多個運算符之間的並行
        self.FLAGS.moving_average_decay = None
        self.FLAGS.labels_offset = 0

        self.FLAGS.model_path = tFLAGS.model_path
        self.FLAGS.image_path = tFLAGS.image_path
        self.FLAGS.result_file = tFLAGS.result_file

        if os.path.exists('%s/frozen_inference_graph.pb' % self.FLAGS.model_path) is True or os.path.exists('%s/frozen_inference_graph.enc' % self.FLAGS.model_path) is True:
            self.FLAGS.is_run_frozen_file = True
        else:
            self.FLAGS.is_run_frozen_file = False

        ################################################
        # get hyper parameters for preprocessing
        ################################################
        self.FLAGS.distort_image = True
        self.FLAGS.central_fraction = 1
        self.FLAGS.default_image_size = None

        if os.path.exists('%s/hyper_param.ini' % tFLAGS.model_path) is True:
            config = configparser.ConfigParser()
            config.read('%s/hyper_param.ini' % tFLAGS.model_path)
            self.FLAGS.eval_image_size = config.getint(config.default_section, 'image_size', fallback=self.FLAGS.eval_image_size)
            self.FLAGS.central_fraction = config.getfloat(config.default_section, 'central_fraction', fallback=self.FLAGS.central_fraction)
        ################################################

        self.__setup_environ__()


    def __initial_variables__(self):
        self.TOTAL_RESULT[:] = []
        #######################
        # Load Labels as Dict #
        #######################
        if self.model_class_pd is not None:
            del self.model_class_pd
        self.model_class_pd = pd.read_csv('%s/labels.txt' % self.FLAGS.model_path)
        self.classes_num = len(self.model_class_pd)


    def __create_input_image_op__(self):
        self.logger.info('__create_input_image_op__ ...')

        eval_image_size = self.FLAGS.eval_image_size or self.FLAGS.default_image_size
        self.logger.info('__create_input_image_op__ : eval_image_size = %d' % eval_image_size)

        #########################################################
        image_op = tf.placeholder(dtype=tf.float32, shape=[eval_image_size, eval_image_size, 3], name="image/encoded")
        name_op = tf.placeholder(dtype=tf.string, shape=(), name="image/filename")
        self.batch_size_tensor = tf.placeholder_with_default(1, shape=[])

        images_op, names_op = tf.train.batch(
            [image_op, name_op],
            batch_size=self.batch_size_tensor,
            num_threads=1,
            capacity=2 * self.FLAGS.batch_size
        )

        images_op = tf.placeholder_with_default(images_op, shape=(None, eval_image_size, eval_image_size, 3))
        #########################################################

        return images_op, names_op

    def __initial_network_graph__(self, num_classes=2):
        from nets import nets_factory
        self.logger.info("__initial_network_graph__ ...")
        tStart = time.time()

        self.persist_graph = tf.Graph()
        with self.persist_graph.as_default():
            self.logger.info('create_network_fn ...')
            network_fn = nets_factory.get_network_fn(
                name='inception_v4',
                num_classes=(num_classes - self.FLAGS.labels_offset),
                is_training=False)
            self.FLAGS.default_image_size = network_fn.default_image_size
            self.persist_default_image_size = self.FLAGS.default_image_size
            self.images_tensor, self.names_tensor = self.__create_input_image_op__()
            self.logits_tensor, self.endpoint = network_fn(self.images_tensor)

        self.logger.info('initial_network_graph done')
        self.logger.info("__initial_network_graph__ : %f sec" % (time.time() - tStart))


    def __initial_network_graph_via_pb__(self, num_classes=2):
        self.logger.info("__initial_network_graph_via_pb__ ...")
        tStart = time.time()

        self.persist_graph = tf.Graph()
        with self.persist_graph.as_default():
            od_graph_def = tf.GraphDef()
            if os.path.isfile('%s/frozen_inference_graph.pb' % self.FLAGS.model_path):
                with tf.gfile.GFile('%s/frozen_inference_graph.pb' % self.FLAGS.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            else:
                raise Exception('Model File Not Found')

        self.logger.info("__initial_network_graph_via_pb__ : %f sec" % (time.time() - tStart))

        tStart = time.time()
        self.logger.info('__create_session_and_restore_checkpoint__ ...')
        tf_config = None
        if self.FLAGS.gpu_enable is False:
            tf_config = tf.ConfigProto(device_count={'CPU': self.FLAGS.num_cpu_core},
                                       intra_op_parallelism_threads=self.FLAGS.intra_threads,  # 控制運算符內部的並行
                                       inter_op_parallelism_threads=self.FLAGS.inter_threads,  # 控制多個運算符之間的並行
                                       allow_soft_placement=True,
                                       )
        else:
            tf_config = tf.ConfigProto(allow_soft_placement=True)
            tf_config.gpu_options.per_process_gpu_memory_fraction = self.FLAGS.gpu_memory_fraction
        ##
        self.persist_sess = tf.Session(graph=self.persist_graph, config=tf_config)
        self.logger.info("__create_session_and_restore_checkpoint__ : %f sec" % (time.time() - tStart))


    def __create_session_and_restore_checkpoint__(self):
        self.logger.info('__create_session_and_restore_checkpoint__ ...')
        tStart = time.time()

        if self.is_initial_sess is False:
            with self.persist_graph.as_default():
                tf_global_step = self.slim.get_or_create_global_step()

                if self.FLAGS.moving_average_decay:
                    variable_averages = tf.train.ExponentialMovingAverage(self.FLAGS.moving_average_decay, tf_global_step)
                    variables_to_restore = variable_averages.variables_to_restore(self.slim.get_model_variables())
                    variables_to_restore[tf_global_step.op.name] = tf_global_step
                else:
                    variables_to_restore = self.slim.get_variables_to_restore()

                ################################################################
                saver = tf.train.Saver(
                    variables_to_restore or variables.get_variables_to_restore(),
                    write_version=saver_pb2.SaverDef.V1)

                ################################################################
                tf_config = tf.ConfigProto(device_count={'CPU': self.FLAGS.num_cpu_core},
                                           intra_op_parallelism_threads=self.FLAGS.intra_threads,  # 控制運算符內部的並行
                                           inter_op_parallelism_threads=self.FLAGS.inter_threads,  # 控制多個運算符之間的並行
                                           allow_soft_placement=True,
                                           )
                self.logger.info('create session ...')
                self.persist_sess = tf.Session(config=tf_config)

                self.logger.info('session run global_variables_initializer ...')
                self.persist_sess.run(tf.global_variables_initializer())

                ###########################################################################
                ckpt = tf.train.get_checkpoint_state(self.FLAGS.checkpoint_path)
                if ckpt is None:
                    raise Exception('No Checkpoint Found')

                self.logger.info('restore checkpoint_path = %s' % ckpt.model_checkpoint_path)
                tStart0 = time.time()
                saver.restore(self.persist_sess, ckpt.model_checkpoint_path)
                self.logger.info('restore checkpoint done : %f sec' % (time.time() - tStart0))
            ###########################################################################
            self.is_initial_sess = True
        self.logger.info("__create_session_and_restore_checkpoint__ : %f sec" % (time.time() - tStart))


    def __create_image_preprocessing__(self, preprocessing_name, x_image):
        from preprocessing import preprocessing_factory

        eval_image_size = self.FLAGS.eval_image_size or self.FLAGS.default_image_size
        print('eval_image_size is %d' % eval_image_size)
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,
                                                                         is_training=False)
        image = image_preprocessing_fn(x_image, eval_image_size, eval_image_size,
                                       distort_image=self.FLAGS.distort_image,
                                       central_fraction=self.FLAGS.central_fraction)
        return image


    def __convert_image_format__(self, image_file):
        image = Image.open(image_file)
        if image.format == 'BMP':
            rgb_img = image.convert('RGB')
            rgb_img.save(image_file, format='png', compress_level=0)
            rgb_img.close()
        elif image.mode != 'RGB' and image.mode != 'RGBA':
            rgb_img = image.convert('RGB')
            rgb_img.save(image_file, format=image.format)
            rgb_img.close()
        image.close()


    def __decode_jpeg(self, sess, image_data, mformat='JPEG'):
        # Initializes function that decodes RGB JPEG data.
        _decode_data = tf.placeholder(dtype=tf.string)
        _decoder_ = None
        if mformat == 'JPEG':
            _decoder_ = tf.image.decode_jpeg(_decode_data, channels=3)
        elif mformat == 'PNG':
            _decoder_ = tf.image.decode_png(_decode_data, channels=3)
        elif mformat == 'BMP':
            _decoder_ = tf.image.decode_bmp(_decode_data, channels=3)

        image = sess.run(_decoder_, feed_dict={_decode_data: image_data})
        return image


    def __read_image_data__(self, img_file):
        self.logger.info('start to read_image_data')
        tStart = time.time()
        tf_config = tf.ConfigProto(device_count={'CPU': self.FLAGS.num_cpu_core},
                                   intra_op_parallelism_threads=self.FLAGS.intra_threads,  # 控制運算符內部的並行
                                   inter_op_parallelism_threads=self.FLAGS.inter_threads,  # 控制多個運算符之間的並行
                                   allow_soft_placement=True,
                                   )

        self.__convert_image_format__(img_file)

        images = []
        names = []

        with self.persist_graph.as_default():
            with tf.Session(config=tf_config) as sess:
                image_data = tf.gfile.FastGFile(img_file, 'rb').read()
                decoded_img = self.__decode_jpeg(sess, image_data)
                image_reshape = tf.reshape(decoded_img, [decoded_img.shape[0], decoded_img.shape[1], 3])
                image_reshape = self.__create_image_preprocessing__(preprocessing_name=self.FLAGS.model_name,
                                                                    x_image=image_reshape)  # -->(299, 299, 3)
                images.append(sess.run(image_reshape))
                names.append(os.path.basename(img_file))


        self.logger.info("__read_image_data__ : %f sec" % (time.time() - tStart))
        return images, names


    def run_inference_on_image(self, tFLAGS=None):
        self.logger.info('Start run_inference_on_image ...')

        self.__initial_FLAGS__(tFLAGS=tFLAGS)

        self.__initial_variables__()

        #######################################################################
        self.logger.info('run_inference_on_image classes_num %d' % self.classes_num)
        if self.FLAGS.is_run_frozen_file is False:
            self.logger.info('using building-network ...')
            self.__initial_network_graph__(num_classes=self.classes_num)
            self.__create_session_and_restore_checkpoint__()
        else:
            self.logger.info('using frozen_inference_graph ...')
            self.__initial_network_graph_via_pb__(num_classes=self.classes_num)

        #######################################################################
        if self.FLAGS.is_run_frozen_file is False:
            self.logger.info('using ckpt file ...')
            self.logger.info('__read_image_data__ ...')
            image_datas, image_names = self.__read_image_data__(self.FLAGS.image_path)
            self.__just_run_many__(self.persist_sess, image_datas, image_names)
        else:
            self.logger.info('using pb file ...')
            self.__just_run_many_for_pb__(self.persist_sess, self.FLAGS.image_path)

        #######################################################################
        self.logger.info('Finished')


    def __just_run_many_for_pb__(self, session, img_file):
        self.logger.info("__just_run_many_for_pb__ ...")

        tStart = time.time()

        pred_probs = []
        predictions = []
        image_name_list = []
        # for ocsvm
        feature_regions_arr = []

        softmax_tensor = session.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')

        ##
        image_name = os.path.basename(img_file)
        self.__convert_image_format__(image_file=img_file)
        width, _ = Image.open(img_file).size

        if width > 0:
            ##
            image_name_list.append(image_name)

            tStart = time.time()
            image_data = tf.gfile.FastGFile(img_file, 'rb').read()
            self.logger.info("__read_image_data__ : %f sec" % (time.time() - tStart))

            ########################################################################
            feature_vector = None
            self.logger.info('image session.run start')
            pred_values = session.run([softmax_tensor], feed_dict={'input:0': image_data})
            self.logger.info("image session.run done : %f sec" % (time.time() - tStart))
            ########################################################################

            pred_value = np.squeeze(pred_values[0])
            num_top_predictions = 1
            top_k = pred_value.argsort()[-num_top_predictions:][::-1]

            ########################################################################
            for node_id in top_k:
                predictions.append(node_id)
                pred_probs.append(pred_value)

        self.__summary_predict_result__(pred_probs, predictions, image_name_list)
        self.__gen_prediction_report__(prediction_file=self.FLAGS.result_file)
        self.logger.info("__just_run_many_for_pb__ : %f sec" % (time.time() - tStart))


    def __just_run_many__(self, session, image_data_list, image_name_list):
        self.logger.info("__just_run_many__ ...")
        tStart = time.time()

        pred_prob = tf.nn.softmax(self.logits_tensor)
        predictions = tf.argmax(self.logits_tensor, 1)

        self.logger.info('__just_run_many__: session.run start')
        pred_values = session.run([pred_prob, predictions, self.names_tensor],
                                  feed_dict={self.images_tensor: image_data_list,
                                             self.names_tensor: image_name_list,
                                             self.batch_size_tensor: self.FLAGS.batch_size})
        self.logger.info("__just_run_many__ : session.run done : %f sec" % (time.time() - tStart))

        self.__summary_predict_result__(pred_values[0], pred_values[1], pred_values[2])
        self.__gen_prediction_report__(prediction_file=self.FLAGS.result_file)
        self.logger.info("__just_run_many__ : %f sec" % (time.time() - tStart))


    def __summary_predict_result__(self, probabilities, predictions, image_names):
        """
        :param probabilities: [[0.54521137, 0.45478866], [...], ...]
        :param predictions: [0,1,...] 標示哪個 label 為預測結果
        :param image_names: [b'xxxxx', ...]
        :return:
        """
        tStart = time.time()
        _how_many_results = len(predictions)
        for k in range(_how_many_results):
            if isinstance(image_names[k], str) is True:
                _only_file_name = image_names[k]
            else:
                _only_file_name = image_names[k].decode()

            _prediction_label_idx = predictions[k]

            single_result = list()
            single_result.append(self.FLAGS.model_name)
            # image_file_name
            single_result.append(_only_file_name)

            # first_pred_class, first_pred_probability
            first_clazz, first_pred_prob = self.__get_first_predicate_clazz__(probabilities[k], _prediction_label_idx)
            single_result.append(self.__remove_capability__(first_clazz))
            single_result.append(first_pred_prob)

            # ped empty final_judge box
            single_result.append(0)
            single_result.append(0)
            single_result.append(0)
            single_result.append(0)

            print(single_result)
            self.TOTAL_RESULT.append(single_result)

        self.logger.info("__summary_predict_result__ : %f sec" % (time.time() - tStart))


    def __gen_prediction_report__(self, prediction_file=None):
        tStart = time.time()

        col_name = []
        col_name.extend(['network_name', 'image_file', 'predict_class', 'predict_probability',
                         'box_xmin', 'box_ymin', 'box_xmax', 'box_ymax'])

        df_result = pd.DataFrame(self.TOTAL_RESULT, columns=col_name)
        df_result.to_csv(prediction_file, index=False)
        self.logger.info("__gen_prediction_report__ : %f sec" % (time.time() - tStart))


    def __remove_capability__(self, class_name=''):
        i_start = class_name.index("[", 0)
        i_stop = class_name.index("]", i_start)
        return class_name[i_stop + 1:]


    def __get_first_predicate_clazz__(self, probabilities, prediction_idx):
        _clazz = self.model_class_pd[self.model_class_pd['class_id'] == prediction_idx].iloc[0]['class_name']
        return _clazz, probabilities[prediction_idx]


if __name__ == '__main__':
    
    sys.argv.append('--gpu_enable=N')
    sys.argv.append('--gpu_id=0')
    sys.argv.append('--cpu_nums=1')
    sys.argv.append('--intra_threads=2')  # 控制運算符內部的並行

    sys.argv.append('--model_path=D:\\MyProject\\test_inference\\inceptionV4')
    sys.argv.append('--image_path=D:\\MyProject\\test_inference\\Images\\48b7588201cdd27b4f78fd81fad1b69d.jpg')
    sys.argv.append('--result_file=D:\\MyProject\\test_inference\\Predict_Result\\inceptionV4_predict_result.csv')

    inception_v4 = InceptionV4Classify()
    inception_v4.do_main(sys.argv)
