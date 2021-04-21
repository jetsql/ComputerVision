import logging
import argparse
import os
import time
import sys
import ast
import json
import operator
import numpy as np
import pandas as pd
from PIL import Image
from logging.config import fileConfig
import tensorflow as tf


def init_logger():
    logging.log_file = 'ssd_predict.log'
    fileConfig('logging.ini')


class SSDClassify():
    FLAGS = argparse.ArgumentParser().parse_args('')
    logger = None
    model_class_pd = None
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

        sys.path.append('object_detection.zip')
        _FLAGS, unparsed = parser.parse_known_args(args=argv)
        self.__init_logger__()
        self.run_inference_on_image(tFLAGS=_FLAGS)
        sys.path.remove('object_detection.zip')


    def __initial_FLAGS__(self, tFLAGS=None):

        if tFLAGS.gpu_enable == 'Y':
            self.FLAGS.gpu_enable = True
            self.FLAGS.gpu_id = tFLAGS.gpu_id
            self.FLAGS.gpu_memory_fraction = ast.literal_eval(tFLAGS.gpu_memory_fraction)
        else:
            self.FLAGS.gpu_enable = False

        self.FLAGS.model_name = 'ssd'
        self.FLAGS.num_cpu_core = tFLAGS.cpu_nums
        self.FLAGS.intra_threads = tFLAGS.intra_threads
        self.FLAGS.inter_threads = 1 # 控制多個運算符之間的並行

        self.FLAGS.model_path = tFLAGS.model_path
        self.FLAGS.image_path = tFLAGS.image_path
        self.FLAGS.result_file = tFLAGS.result_file

        self.__setup_environ__()


    def __setup_environ__(self):
        self.logger.info('ssd: setup_environ')
        self.logger.info('ssd: setup_environ : gpu_enable = %s' % self.FLAGS.gpu_enable)
        if self.FLAGS.gpu_enable is True:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1': INFO, '2' : WARN}
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.FLAGS.gpu_id


    def __initial_variables__(self):
        from object_detection.utils import label_map_util
        self.TOTAL_RESULT[:] = []

        ssd_label_file = '%s/aoi_label_map.pbtxt' % self.FLAGS.model_path
        if os.path.isfile(ssd_label_file) is False:
            ssd_label_file = '%s/label_map.pbtxt' % self.FLAGS.model_path

        #######################
        label_map = label_map_util.load_labelmap(ssd_label_file)
        max_num_classes = max([item.id for item in label_map.item])
        self.classes_num = max_num_classes

        categories = label_map_util.convert_label_map_to_categories(label_map=label_map,
                                                                    max_num_classes=max_num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        ##########################################
        if self.model_class_pd is not None:
            del self.model_class_pd

        # Reading from labels.txt
        self.model_class_pd = pd.read_csv('%s/labels.txt' % self.FLAGS.model_path)


    def __initial_network_graph__(self):
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

        self.is_initial_graph = True
        self.logger.info('initial_network_graph loaded : %f sec' % (time.time() - tStart))


    def __create_session_and_restore_checkpoint__(self):
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
        self.is_initial_sess = True

        self.logger.info("create_session_and_restore_checkpoint : %f sec" % (time.time() - tStart))


    def run_inference_on_image(self, tFLAGS=None):
        self.logger.info('Start run_inference_on_image ...')

        self.__initial_FLAGS__(tFLAGS=tFLAGS)

        self.__initial_variables__()

        self.__initial_network_graph__()

        self.__create_session_and_restore_checkpoint__()

        tStart01 = time.time()
        self.logger.info('start detect_objects ...')
        image_file = self.FLAGS.image_path
        self.__convert_image_format__(image_file=image_file)
        self.detect_objects(image_file, self.persist_sess, self.persist_graph)
        self.logger.info('Finished detect_objects : %f sec' % (time.time() - tStart01))

        self.__gen_prediction_report__(prediction_file=self.FLAGS.result_file)
        #######################################################################
        self.logger.info('Finished')


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

    def detect_objects(self, image_file, sess, detection_graph):
        """

        :param image_file:
        :param sess:
        :param detection_graph:
        :return:
        """
        image = Image.open(image_file)

        width, height = image.size
        if width > 4000 or width == 0:
            self.logger.info('image width > 4000 or 0')
        else:
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # About 0.9 sec/image
            tStart01 = time.time()
            image_np = self.load_image_into_numpy_array(image)
            self.logger.info('load_image_into_numpy_array : %f sec' % (time.time() - tStart01))

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            all_scores = None
            try:
                all_scores = detection_graph.get_tensor_by_name('all_scores:0')
            except KeyError as ex:
                try:
                    self.logger.error('can not found tensor(all_scores:0), try (all_score:0)')
                    all_scores = detection_graph.get_tensor_by_name('all_score:0')
                except KeyError as ex:
                    self.logger.error(ex)

            tStart01 = time.time()
            if all_scores is not None:
                (boxes, scores, classes, num_detections, all_scores) = sess.run(
                    [boxes, scores, classes, num_detections, all_scores],
                    feed_dict={image_tensor: image_np_expanded})
                self.logger.info('sess.run : %f sec' % (time.time() - tStart01))
            else:
                all_scores = np.array(np.zeros((self.classes_num + 1, 10), dtype=np.float64), ndmin=3)
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

            self.__summary_predict_result__(*image.size,
                                            np.squeeze(all_scores),
                                            np.squeeze(boxes),
                                            np.squeeze(scores),
                                            np.squeeze(classes).astype(np.int32),
                                            image_file)

            from object_detection.utils import visualization_utils as vis_util
            if False:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    min_score_thresh=0.0,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                img = Image.fromarray(image_np, 'RGB')
                img.show()

    def __summary_predict_result__(self, im_width, im_height, all_scores, boxes, scores, classes, image_file):
        """

        :param im_width:
        :param im_height:
        :param all_scores: 有包含 score0.[[] [] [] ...]
                           第一層內的 [] 數量 = class label count + 1
                           第二層內 elements 數量 = max_total_detections
        :param boxes:
        :param scores:
        :param classes:
        :param image_file:
        :return:
        """
        _only_file_name = os.path.basename(image_file)

        self.logger.info('%s, predict#: %d' % (image_file, len(scores)))
        for idx, pred_score in enumerate(scores):
            _class_id = classes[idx] - 1

            adj_box = boxes[idx]
            pred_clazz = self.model_class_pd[self.model_class_pd['class_id'] == _class_id].iloc[0]['class_name']

            single_result = list()
            single_result.append(self.FLAGS.model_name)
            # image_file_name
            single_result.append(_only_file_name)

            # first_pred_class, first_pred_probability
            single_result.append(self.__remove_capability__(pred_clazz))
            single_result.append(pred_score)

            # final_judge box
            box_coord = self.to_absolute_coordinates(im_width, im_height, *adj_box)
            single_result.append(box_coord[0])
            single_result.append(box_coord[1])
            single_result.append(box_coord[2])
            single_result.append(box_coord[3])

            self.TOTAL_RESULT.append(single_result)


    def to_absolute_coordinates(self, im_width, im_height, ymin, xmin, ymax, xmax):
        """

        :param im_width:
        :param im_height:
        :param ymin:
        :param xmin:
        :param ymax:
        :param xmax:
        :return: tuple
        """
        (left, top, right, bottom) = (xmin * im_width, ymin * im_height,
                                      xmax * im_width, ymax * im_height)
        return left, top, right, bottom


    def load_image_into_numpy_array(self, image):
        if image.mode != 'RGB':
            rgb_img = image.convert('RGB')
            (im_width, im_height) = rgb_img.size
            # read 2K -> 0.04 sec
            return np.asarray(rgb_img).reshape((im_height, im_width, 3)).astype(np.uint8)
        else:
            (im_width, im_height) = image.size
            # read 2K -> 0.04 sec
            return np.asarray(image).reshape((im_height, im_width, 3)).astype(np.uint8)


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


if __name__ == '__main__':

    sys.argv.append('--gpu_enable=N')
    sys.argv.append('--gpu_id=0')
    sys.argv.append('--cpu_nums=1')
    sys.argv.append('--intra_threads=2')# 控制運算符內部的並行
    
    sys.argv.append('--model_path=D:\\MyProject\\test_inference\\ssd')
    sys.argv.append('--image_path=D:\\MyProject\\test_inference\\Images\\48b7588201cdd27b4f78fd81fad1b69d.jpg')
    sys.argv.append('--result_file=D:\\MyProject\\test_inference\\Predict_Result\\ssd_image_classifier_predict_result.csv')

    ssd = SSDClassify()
    ssd.do_main(sys.argv)

