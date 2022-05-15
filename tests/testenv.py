import os
# os.environ['TF_KERAS'] = '1'
# os.environ['TF_EAGER'] = '0'
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "../..")
