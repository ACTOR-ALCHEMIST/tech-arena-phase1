"""Consistent rolling windows from an explicit first usable information date."""
from run import timestamp


def days(config, cutoff, kind):
    if config.get(kind+'_start'):
        span = timestamp(cutoff)-timestamp(config[kind+'_start'])
        if span <= 0 or span%86400:
            raise ValueError('Training origins must start on an earlier UTC day')
        return span//86400
    return config.get(kind+'_days')
