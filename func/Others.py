# -*- coding: utf-8 -*-
"""
@author: YuZhu
"""


def timeRecord(type='total', show=True):
    import time
    """
    功能：打印消耗的时间
    输入：（打印时间类型）
    输出：所消耗的时间
    备注：默认打印消耗的总时间
    """
    if not hasattr(timeRecord, 'time0'):
        timeRecord.time0 = time.time()  # 记录起始时间
        timeRecord.time1 = timeRecord.time0
        timeRecord.time2 = timeRecord.time0
        return None
    timeRecord.time2 = time.time()  # 记录当前时间
    if type == 'part':
        timeUsed = timeRecord.time2 - timeRecord.time1
    else:
        timeUsed = timeRecord.time2 - timeRecord.time0
    if show:
        print("Running " + type + " time: %g" % timeUsed)
    timeRecord.time1 = timeRecord.time2  # 记录上次时间
    return timeUsed
