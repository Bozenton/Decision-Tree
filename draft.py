import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")

class Kmeans1d():
    def __init__(self, epsilon=1e-2, max_step=20):
        self.stop_epsilon = epsilon
        self.MAX_STEP = max_step
    
    def fit(self, data: np.ndarray, k):
        assert len(data.shape) == 1, "The input data of Kmeans1d should be 1 dim"
        # initialization
        low_boundary = np.min(data)
        up_boundary = np.max(data)
        # randomly generate k points as initial center for each class
        center_points = low_boundary + (up_boundary-low_boundary)*np.random.rand(k)
        center_points = np.sort(center_points)
        center_points_last = np.ones(k)*np.inf
        record = pd.DataFrame({'data': data, 'cluster': np.zeros(data.shape)})
        
        cnt = 0
        while not self.stopping_condition(center_points, center_points_last) and cnt<self.MAX_STEP:
            cnt = cnt + 1
            center_points_last = center_points
            # step1: with fixed center points, cluster the samples
            for i, sample in enumerate(data):
                cluster_class = np.argmin(self.squaredEuclideanDistance(center_points, sample))
                record.iloc[i, 1] = np.int16(cluster_class)
            
            # step2: compute new center for each class
            for idx, center in enumerate(center_points_last):
                record_filtered = record.query("cluster==@idx")
                if record_filtered.empty:
                    new_center = center
                else:
                    new_center = record_filtered['data'].mean()
                center_points[idx] = new_center
        return center_points, record
    
    def stopping_condition(self, points, points_last):
        distance = self.squaredEuclideanDistance(points, points_last)
        if np.all(distance < self.stop_epsilon):
            return True 
        else:
            return False
    
    def auto_fit(self, data:np.ndarray, max_k=5, max_each=5):
        best_k = 0
        best_score = -np.inf
        best_cps = None # cps: center points
        best_rcd = None # rcd: record
        all_score = np.zeros(max_k-1)
        for i in np.arange(2, max_k+1):
            best_score_for_this_k = -np.inf
            best_rcd_for_this_k = None 
            best_cps_for_this_k = None
            for j in np.arange(2, max_each+1):
                
                print(f"Trying k={i} for the {j}th time")
                
                cps, rcd = self.fit(data, k=i)
                score = self.silhouette(rcd)
                if score > best_score_for_this_k:
                    best_score_for_this_k = score
                    best_rcd_for_this_k = rcd
                    best_cps_for_this_k = cps
            all_score[i-2] = best_score_for_this_k
            if best_score_for_this_k > best_score:
                best_score = best_score_for_this_k
                best_cps = best_cps_for_this_k
                best_rcd = best_rcd_for_this_k
        
        return best_score, best_cps, best_rcd, all_score
        
    @staticmethod
    def squaredEuclideanDistance(x1, x2):
        return np.square(x1-x2)

    @staticmethod
    def silhouette(data:pd.DataFrame):
        clusters = np.sort(data['cluster'].unique())
        a = np.zeros(data['data'].count())
        b = np.zeros(data['data'].count())
        s = np.zeros(data['data'].count())
        for i, sample in enumerate(data.values):
            num = sample[0]
            cls = sample[1]
            
            # compute a(i)
            # a measure of how well i is assigned to its cluster 
            # (the smaller the value, the better the assignment).
            a_mask = np.int8(data['cluster'].values == cls)
            a_mask_sum = a_mask.sum()
            distance_to_all = Kmeans1d.squaredEuclideanDistance(num, data['data'].values)
            if a_mask_sum == 1:
                a[i] = 0
            elif a_mask_sum <= 0:
                raise ValueError("Cannot find sample with this class. Something wrong happened")
            else:
                a[i] = np.dot(distance_to_all, a_mask) / (a_mask_sum-1)
                
            # compute b(i)
            b_min = np.inf
            for c in clusters:
                if c == cls:
                    continue
                b_mask = np.int8(data['cluster'].values == c)
                assert b_mask.sum()>0, "Something wrong happened"
                temp_b = np.dot(distance_to_all, b_mask) / b_mask.sum()
                if temp_b < b_min:
                    b_min = temp_b
            b[i] = b_min
            
            if b[i] == np.inf:
                print(data)
                raise ValueError(f"WTF, b_min={b_min}")
            
            # compute s(i)
            if a_mask_sum > 1:
                try:
                    s[i] = 1.*(b[i]-a[i]) / np.max([a[i], b[i]])
                except RuntimeWarning:
                    print(f"Something wrong happened, a={a[i]}, b={b[i]}")
            else:
                s[i] = 0
            
        # Thus the mean s(i) over all data of the entire dataset is a measure of 
        # how appropriately the data have been clustered.
        return np.mean(s)

if __name__ == '__main__':
    data_path = '.\data\iris.data'
    iris_data = pd.read_csv(data_path, 
            names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'], 
            delimiter=',')
    
    kk = Kmeans1d()
    try:
        best_score, best_cps, best_rcd, all_score = \
                kk.auto_fit(iris_data['petal width'].values, max_k=8, max_each=10)
    except ValueError:
        print("May be dividing by NaN, this may be caused by null subsets in clustering")
    print(best_score)
    print(best_cps)
    print(all_score)
    