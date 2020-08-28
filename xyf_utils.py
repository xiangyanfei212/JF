import logging
import inspect
import os
import shutil

import numpy as np


def logging_config(folder=None, name=None, level=logging.INFO, console_level=logging.DEBUG):
    """
    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level

    Returns
    -------
    """""
    # folder = os.path.join('/cuit/xiangyanfei/videopred-master-V1', folder)
    # print('folder: ', folder)
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]

    if folder is None:
        folder = os.path.join(os.getcwd(), name)
            
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Remove all the current handlers     
    for handler in logging.root.handlers:                                                                   
        logging.root.removeHandler(handler)       
    
    logging.root.handlers = []
    
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to %s" % logpath)
                                                                                    
    # 如果日志存在，删除重建
    # if os.path.exists(logpath):
    #    os.remove(logpath)
    
    
    logging.root.setLevel(level)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Initialze the file logging
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    # Initialze the console logging
    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(formatter)
    logging.root.addHandler(logconsole)
    
    return folder
