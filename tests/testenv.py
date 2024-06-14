import os
import os.path
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
CURRDIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(CURRDIR, ".."))
