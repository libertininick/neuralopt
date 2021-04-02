import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
import torch


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
        
        return v
