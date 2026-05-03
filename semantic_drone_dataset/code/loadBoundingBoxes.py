import pickle

def load(path = '../training_set/gt/bounding_box/bounding_boxes/person/imgIdToBBoxArray.p'):
    '''
    Reads the specified file from the disk and tries to interpret it as a
    pickle file.

    If called with default arguments and the training set is included then it
    will return a dictionary mapping the image id ('123') to a array of bounding
    boxes for persons. 

    e.g. {'123':[numpy.array([[ xMinBox1,  yMinBox1],
                              [ xMaxBox1,  yMaxBox1]]),
                 numpy.array([[ xMinBox2,  yMinBox2],
                              [ xMaxBox2,  yMaxBox2]])]}

    Args:
        path: The path to the file that should be loaded.
              The default is the path to the pickle file containing a
              dictionary mapping the image id to an array of the contained
              bounding boxes for persons in the training set.
    Returns:
        The contents of the pickle file specified.
    '''
    with open(path, 'rb') as pickleFile:
        return pickle.load(pickleFile)

if __name__ == '__main__':
    print('This is not a standalone script.')
    print('Intended usage is an import into a separate python script.')

    print ('\nHelper functions:\n\n  load(path):')
    print(load.__doc__)
