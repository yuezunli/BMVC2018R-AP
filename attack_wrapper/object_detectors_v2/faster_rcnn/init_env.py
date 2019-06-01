
import sys, os

pwd = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = pwd + '/../../../object_detectors/pytorch-faster-rcnn/'
sys.path.insert(0, DETECTOR_DIR)
ROOT_DIR = pwd + '/../../../'
sys.path.append(ROOT_DIR)
