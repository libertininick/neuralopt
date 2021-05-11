import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import torch
import umap


def get_common_discourse_vectors(X, n_components):
    """Computes the top N principal components of a matrix of 
    (token|word|sentence|document) embeddings. 
    
    These top components are assumed to be the "common discourse" 
    components of the embedding space.
    
    Singular value decomposition (SVD) is used vs. PCA,  because SVD 
    does not center the data before computing the decomposition. 
    
    Args:
        X (ndarray): Matrix of embeddings [n_obs, emb_dim]
        n_components: Number of common discourse vectors to return
        
    Returns:
        components (ndarray): Common discourse vectors [n_components, emb_dim]
    """
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=0)
    svd.fit(X)
    
    return svd.components_


def _remove_vectors(X, v):
    return X - X.dot(v.T).dot(v)


def precision_recall_f1(v_a, v_b, k=5):
    manifold_a = Manifold(k).fit(v_a)
    manifold_b = Manifold(k).fit(v_b)
    
    predictions_ab = manifold_a.predict(v_b)  # Do points b lie in manifold a (precision)
    predictions_ba = manifold_b.predict(v_a)  # Do points a lie in manifold b (recall)
    
    precision = np.mean(predictions_ab)
    recall = np.mean(predictions_ba)
    
    f1 = 2*precision*recall/(precision + recall)

    return precision, recall, f1


class SequenceVectorizer():
    """Riff on "Simple but tough-to-beat baseline..." for combining 
    an encoded sequence into a single vector representation

    Example:
        batch_encodings = next(iter(DataLoader(datasets['train'], batch_size=1000, shuffle=True)))
        sv = SequenceVectorizer(enc_func=model.encode, n_common=2, n_ref_samples=500, seed=1234)
        sv.fit(batch_encodings['historical_seq_emb'], batch_encodings['historical_seq_LCH'])
        v = sv.vectorize_seq(batch_encodings['future_seq_emb'], batch_encodings['future_seq_LCH'])
    """
    def __init__(self, enc_func, n_common, n_ref_samples, seed=None):
        """
        Args:
            enc_func (PriceSeriesFeaturizer.encode): From a pretrained instance of PriceSeriesFeaturizer
            n_common (int): Number of "common" components to remove
            n_ref_samples (int): Number of reference samples to use for density estimation and inverse density weighting
            seed (int): Random state
        """
        self.enc_func = enc_func
        self.n_common = n_common
        self.n_ref_samples = n_ref_samples
        self.rnd = np.random.RandomState(seed)
        
        self.common_components = None
        self.ref_samples = None
        self.knn_median_min = None

    def _encode(self,  x_emb, x_LCH):
        with torch.no_grad():
            encodings = self.enc_func(x_emb, x_LCH)
        return np.concatenate(encodings.numpy(), axis=0)
    
    def fit(self, x_emb, x_LCH):
        # Encode
        enc = self._encode(x_emb, x_LCH)

        # Fit common components
        self.common_components = get_common_discourse_vectors(enc, self.n_common)

        # Remove common components
        enc_adj = _remove_vectors(enc, self.common_components)
        
        # Sample reference vectors
        idxs = self.rnd.choice(np.arange(len(enc_adj)), size=self.n_ref_samples , replace=False)
        self.ref_samples = enc_adj[idxs, :]

        # Cosine distance to reference vectors
        dists = cdist(enc_adj, self.ref_samples, metric='cosine')
        knn_median = np.median(dists, axis=-1)
        self.knn_median_min = np.min(knn_median)

        return self

    def vectorize_seq(self, x_emb, x_LCH):
    
        batch_size, seq_len = x_emb.shape[:2]
        
        # Encode
        enc = self._encode(x_emb, x_LCH)
            
        # Remove common components
        enc_adj = _remove_vectors(enc, self.common_components)
        
        # Cosine distance to reference vectors
        dists = cdist(enc_adj, self.ref_samples, metric='cosine')
        
        # Median distance
        knn_median = np.median(dists, axis=-1)
        
        # Inverse density weights
        inv_density_wts = np.maximum(0, knn_median/self.knn_median_min - 1)
        inv_density_wts = inv_density_wts.reshape(batch_size, seq_len)
        inv_density_wts = inv_density_wts/np.sum(inv_density_wts, axis=-1, keepdims=True)
        
        # Positional weights
        rhs_wts = np.arange(seq_len) + 1
        rhs_wts = rhs_wts/np.sum(rhs_wts)
        lhs_wts = rhs_wts[::-1]
        
        # Combined weights
        lhs_wts = inv_density_wts*lhs_wts[None,:]
        lhs_wts = lhs_wts/np.sum(lhs_wts, axis=-1, keepdims=True)
        rhs_wts = inv_density_wts*rhs_wts[None,:]
        rhs_wts = rhs_wts/np.sum(rhs_wts, axis=-1, keepdims=True)
        
        # Weighted averages
        x_enc_adj = enc_adj.reshape(batch_size, seq_len, -1)
        lhs_avg = np.tensordot(lhs_wts, x_enc_adj, axes=(1,1)).diagonal().T
        rhs_avg = np.tensordot(rhs_wts, x_enc_adj, axes=(1,1)).diagonal().T
        
        # Concat lhs and rhs to form vector
        v = np.concatenate((lhs_avg, rhs_avg), axis=-1)

        # Normalize
        v = v/np.linalg.norm(v, 2)
        
        return v


class Manifold():
    """Used to approximate the manifold of a sample of points from 
    either a real distribution of examples or a generated distribution of examples.
    
    Improved Precision and Recall Metric for Assessing Generative Models 
    https://arxiv.org/pdf/1904.06991.pdf
    
    """
    def __init__(self, k):
        self.k = k
        self.neighborhood = NearestNeighbors(n_neighbors=k, metric='cosine')
        self.ref_points = None
        self.dists_k, self.neighbor_k = None, None
        
    def fit(self, x):
        """Fit neighborhood distances to reference points
        """
        self.ref_points = x
        self.neighborhood.fit(x)
        dists, neigbor_idxs = self.neighborhood.kneighbors(x)
        self.dists_k, self.neighbor_k = dists[:, -1], neigbor_idxs[:, -1]
        return self
    
    def plot_manifold(self, figsize=(10,10)):
        """Plot approx manifold in 2D
        """
        # Plotting style
        plt.style.use('seaborn-colorblind')
        mpl.rcParams['axes.grid'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['figure.facecolor'] = 'white'
        mpl.rcParams['axes.facecolor'] = 'white'

        reducer = umap.UMAP(
            n_components=2,
            random_state=1234
        )

        ref_points_2d = reducer.fit_transform(self.ref_points)
        means, stds = np.mean(ref_points_2d, axis=0), np.std(ref_points_2d, axis=0)
        ref_points_2d = (ref_points_2d - means[None,:])/stds[None,:]
        
        
        dists_2d, _ = (
            NearestNeighbors(n_neighbors=self.k)
            .fit(ref_points_2d)
            .kneighbors(ref_points_2d)
        )
        dists_k_2d = dists_2d[:, -1]
        
        fig, ax = plt.subplots(figsize=figsize)
        _ = ax.scatter(*ref_points_2d.T, marker='.')

        for (x,y), r in zip(ref_points_2d, dists_k_2d):
            ax.add_patch(Circle((x,y), r, alpha=0.1))
            
        return fig, ax
    
    def predict(self, x):
        """Predict if point lies within the manifold
        
        Specifically, if the distance of x_i to any point in ref_points
        is <= the neighborhood distance for the reference point (dists_k_j),
        then x_i is predicted to lie within the manifold approximated by
        the reference points.
        """
        
        # Distance of every point in x to every point in ref_points
        dists_x_to_ref = cdist(x, self.ref_points, metric='cosine')
        
        # Check if any distance falls within a reference point's neighborhood
        dist_diffs = dists_x_to_ref - self.dists_k[None, :]
        in_manifold = np.any(dist_diffs <= 0, axis=-1).astype(int)
        
        return in_manifold