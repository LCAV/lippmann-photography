import os


def make_dirs_safe(func, file, *args, **kwargs,):
    """ Make directory of input path, if it does not exist yet. """
    dirname = os.path.dirname(file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    func(file, *args, **kwargs,)
