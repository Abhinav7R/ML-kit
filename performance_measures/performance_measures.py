import numpy as np

class Performance_Measures:
    def __init__(self):
        pass

    # accuracy = total correct predictions / total predictions

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    # below are helper functions for precision, recall and f1 score

    def _precision(self, tp, fp):
        return tp / (tp + fp) if (tp + fp) != 0 else 0
    
    def _recall(self, tp, fn):
        return tp / (tp + fn) if (tp + fn) != 0 else 0
    
    def _f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    def precision(self, y_true, y_pred, average='macro'):
        # average precision across all classes - macro
        # total tp / total tp + total fp - micro

        classes = np.unique(y_true)
        total_precision = 0
        tp_sum = 0
        fp_sum = 0
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            total_precision += self._precision(tp, fp)
            tp_sum += tp
            fp_sum += fp
            
        if average == 'macro':
            return total_precision / len(classes) if len(classes) != 0 else 0
        elif average == 'micro':
            return self._precision(tp_sum, fp_sum) 
        else:
            raise ValueError(f"Unknown average type: {average}")
        
    def recall(self, y_true, y_pred, average='macro'):
        # average recall across all classes - macro
        # total tp / total tp + total fn - micro

        classes = np.unique(y_true)
        total_recall = 0
        tp_sum = 0
        fn_sum = 0
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            total_recall += self._recall(tp, fn)
            tp_sum += tp
            fn_sum += fn
            
        if average == 'macro':
            return total_recall / len(classes) if len(classes) != 0 else 0
        elif average == 'micro':
            return self._recall(tp_sum, fn_sum) 
        else:
            raise ValueError(f"Unknown average type: {average}")
    
    def f1_score(self, y_true, y_pred, average='macro'):
        if average == 'macro':
            
            # macro f1 score = average of f1 scores of all classes

            F = 0
            classes = np.unique(y_true)
            for c in classes:
                p = self.precision(y_true, y_pred, average='macro')
                r = self.recall(y_true, y_pred, average='macro')
                F += self._f1_score(p, r)
            return F / len(classes) if len(classes) != 0 else 0
        
        elif average == 'micro':

            # micro f1 score = 2 * (micro precision * micro recall) / (micro precision + micro recall)

            p = self.precision(y_true, y_pred, average='micro')
            r = self.recall(y_true, y_pred, average='micro')
            return self._f1_score(p, r)
        else:
            raise ValueError(f"Unknown average type: {average}")
        
    def mean_square_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def standard_deviation(self, y_true, y_pred):
        return np.std(y_true - y_pred)
    
    def variance(self, y_true, y_pred):
        return np.var(y_true - y_pred)