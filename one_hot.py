import numpy as np

def one_hot(labels, n_classes):
    """ Konvertierung von ganzzahligen Labels [1, num_classes] in eine 1-aus-k-Kodierung.
    
        Params:
            labels (np.ndarray): 1d Array mit ganzzahligen Labels.
            n_classes (int): Anzahl der Klassen im Datensatz.
        
        Returns:
            np.ndarray (Shape labels.shape[0] x n_classes) 1-aus-k/Kodierung der ganzzahligen Labels.
    """
    ret = np.zeros((labels.shape[0], n_classes), dtype=float)
    ret[np.arange(labels.shape[0]), labels] = 1
    return ret