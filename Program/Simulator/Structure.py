#!/usr/bin/env python3
import logging

class dotdict(dict):
    '''
        Add tab completion for dict class.
        Example:
            x = dotdict({'abc': 5})
            x.a<tab> will show x.abc
    '''
    def __getattr__(self, name):
        if type(self[name]) is dict:
            return dotdict(self[name])
        return self[name]
    def __dir__(self):
        return self.keys()


def date_convert(date):
    '''
    str/int/datetype to date type
    '''
    import datetime
    if type(date) == int: date = str(date)
    if type(date) == str:
        if   len(date) == 10: return datetime.datetime.strptime(date, '%Y-%m-%d')
        elif len(date) == 8:  return datetime.datetime.strptime(date, '%Y%m%d')
        logging.error("Can't convert {0} to date type. example: date = 20180101".format(date))

    else:
        return date
