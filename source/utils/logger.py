import logging


def getlogger(name, log_file='app.log', log_level=logging.INFO):
    # 创建一个日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger