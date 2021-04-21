from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import configparser
import logging
import os
import time
import operator
import cv2
import glob
import ast
import sys
from PIL import Image
from logging.config import fileConfig
import pandas as pd
import tensorflow as tf
import numpy as np


def init_logger():
    logging.log_file = 'yolo_v3_predict.log'
    fileConfig('logging.ini')


class YoloV3Classify():
    FLAGS = argparse.ArgumentParser().parse_args('')
    logger = None
    model_class_pd = None
    TOTAL_RESULT = list()
    infer_model = None

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

        sys.path.append('keras_object_detection.zip')
        _FLAGS, unparsed = parser.parse_known_args(args=argv)
        self.__init_logger__()
        self.run_inference_on_image(tFLAGS=_FLAGS)
        sys.path.remove('keras_object_detection.zip')


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
        self.FLAGS.model_name = 'yolov3'
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

        self.FLAGS.model_name = 'yolo_v3'
        self.FLAGS.num_cpu_core = tFLAGS.cpu_nums
        self.FLAGS.intra_threads = tFLAGS.intra_threads
        self.FLAGS.inter_threads = 1  # 控制多個運算符之間的並行

        self.FLAGS.model_path = tFLAGS.model_path
        self.FLAGS.image_path = tFLAGS.image_path
        self.FLAGS.result_file = tFLAGS.result_file

        self.FLAGS.use_background_score = False
        self.FLAGS.moving_average_decay = None
        self.FLAGS.labels_offset = 0

        ################################################
        if os.path.exists('%s/hyper_param.ini' % tFLAGS.model_path) is False:
            raise Exception('file not found: %s/hyper_param.ini' % tFLAGS.model_path)

        self.FLAGS.default_config_file = '%s/hyper_param.ini' % tFLAGS.model_path

        self.__setup_environ__()


    def __setup_environ__(self):
        self.logger.info('yolo_v3: setup_environ')
        self.logger.info('yolo_v3: setup_environ : gpu_enable = %s' % self.FLAGS.gpu_enable)
        if self.FLAGS.gpu_enable is True:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1': INFO, '2' : WARN}
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.FLAGS.gpu_id


    def __initial_variables__(self):
        self.TOTAL_RESULT[:] = []

        #######################
        # Load Labels as Dict #
        #######################
        if self.model_class_pd is not None:
            del self.model_class_pd
        self.model_class_pd =  pd.read_csv('%s/labels.txt' % self.FLAGS.model_path)
        self.classes_num = len(self.model_class_pd)


    def __initial_network_graph__(self):
        self.logger.info("__initial_network_graph__ ...")
        tStart = time.time()

        if os.path.isfile('%s/imt_yolo3.h5' % self.FLAGS.model_path) is False and os.path.isfile('%s/imt_yolo3.enc' % self.FLAGS.model_path) is False:
            raise Exception('H5 Model Not Found')
        else:
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
            self.persist_sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)
            from keras import backend as K
            K.set_session(self.persist_sess)
            K.get_session().run(tf.global_variables_initializer())

            if os.path.isfile('%s/imt_yolo3.h5' % self.FLAGS.model_path):
                from keras.models import load_model
                self.infer_model = load_model('%s/imt_yolo3.h5' % self.FLAGS.model_path)
            else:
                raise Exception('Model File Not Found')

            self.logger.info('initial_network_graph done')

        self.logger.info("__initial_network_graph__ : %f sec" % (time.time() - tStart))


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


    def run_inference_on_image(self, tFLAGS=None):
        self.logger.info('Start run_inference_on_image ...')

        self.__initial_FLAGS__(tFLAGS=tFLAGS)

        self.__initial_variables__()

        #######################################################################
        self.logger.info('run_inference_on_image classes_num %d' % self.classes_num)
        self.logger.info('using h5 file ...')
        self.__initial_network_graph__()
        self.__run_yolo_inference__()

        #######################################################################
        self.logger.info('Finished')


    def __run_yolo_inference__(self):
        self.logger.info("__run_yolov3_inference__ ...")

        eval_result_path ='%s/eval_result.csv' % os.path.dirname(os.path.realpath(self.FLAGS.result_file))
        tStart = time.time()

        dic_images = {}

        image_file = self.FLAGS.image_path
        image_id = os.path.basename(image_file)

        self.__convert_image_format__(image_file=image_file)
        width, _ = Image.open(image_file).size
        dic_images[image_id] = image_file

        self.logger.info("__run_AUO_yolo_v3.do_inference__ : %d" % len(dic_images))

        import AUO_yolo_v3
        AUO_yolo_v3.do_inference(FLAGS=self.FLAGS,
                                 do_initialize=True,
                                 logger=self.logger,
                                 model=self.infer_model,
                                 dic_image_path=dic_images,
                                 output_path=eval_result_path)

        self.logger.info("__run_yolov3_inference__ : judged images: %d" % len(dic_images))
        self.logger.info("__run_yolov3_inference__ : %f sec" % (time.time() - tStart))

        self.logger.info("__summary_predict_result__")

        self.__summary_predict_result__(eval_result_path)

        self.logger.info("__gen_prediction_report__")
        self.__gen_prediction_report__(prediction_file=self.FLAGS.result_file)


    def __summary_predict_result__(self, eval_result_path):
        tStart = time.time()
        eval_result_pd = pd.read_csv(eval_result_path)
        eval_result_grouped = eval_result_pd.groupby(by='image_seq', as_index=False)

        self.logger.info("eval_result_grouped : %d" % len(eval_result_grouped))
        for (g_seq, data_by_file) in eval_result_grouped:
            for _, top_row in data_by_file.iterrows():
                single_result = list()
                single_result.append(self.FLAGS.model_name)
                # image_file
                single_result.append(top_row['image_file'])

                # final judge class_label
                _predict_class_ = int(top_row['predict_class'])

                # predict_class
                _predict_label_ = self.model_class_pd[self.model_class_pd['class_id'] == _predict_class_].iloc[0]['class_name']
                single_result.append(self.__remove_capability__(_predict_label_))
                # predict_probability
                single_result.append(top_row['predict_probability'])
                # box_xmin
                single_result.append(top_row['box_xmin'])
                # box_ymin
                single_result.append(top_row['box_ymin'])
                # box_xmax
                single_result.append(top_row['box_xmax'])
                # box_ymax
                single_result.append(top_row['box_ymax'])

                self.TOTAL_RESULT.append(single_result)

        self.logger.info("__summary_predict_result__ : %f sec" % (time.time() - tStart))


    def __gen_prediction_report__(self, prediction_file=None):
        col_name = []
        col_name.extend(['network_name', 'image_file', 'predict_class', 'predict_probability',
                         'box_xmin', 'box_ymin', 'box_xmax', 'box_ymax'])

        df_result = pd.DataFrame(self.TOTAL_RESULT, columns=col_name)
        df_result.to_csv(prediction_file, index=False)
        self.logger.info("__gen_prediction_report__: %s" % prediction_file)


    def __remove_capability__(self, class_name=''):
        i_start = class_name.index("[", 0)
        i_stop = class_name.index("]", i_start)
        return class_name[i_stop + 1:]



if __name__ == '__main__':

    sys.argv.append('--gpu_enable=N')
    sys.argv.append('--gpu_id=0')
    sys.argv.append('--cpu_nums=1')
    sys.argv.append('--intra_threads=2')  # 控制運算符內部的並行

    sys.argv.append('--model_path=D:\\MyProject\\test_inference\\yoloV3')
    sys.argv.append('--image_path=D:\\MyProject\\test_inference\\Images\\48b7588201cdd27b4f78fd81fad1b69d.jpg')
    sys.argv.append('--result_file=D:\\MyProject\\test_inference\\Predict_Result\\yoloV3_image_classifier_predict_result.csv')

    yoloV3 = YoloV3Classify()
    yoloV3.do_main(sys.argv)

