import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx,
    kNN,
)

from extract_patches import extract_patches


class ImageSegmenter:
    def __init__(self, k_fg=5, k_bg=8, mode='kmeans'):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        
        # Number of clusters in FG/BG
        self.k_fg = k_fg
        self.k_bg = k_bg
        
        self.mode= mode

        # During evaluation, this will be replaced by a generator with different
        # random seeds. Use this generator, whenever you require random numbers,
        # otherwise your score might be lower due to stochasticity
        self.rng = np.random.default_rng(42)
        
    def extract_features_(self, sample_dd):
        """ Extract features from the RGB image """
        
        img = sample_dd['img']
        H, W, C = img.shape
        p = 3
        # Extract intensities as features
        features = extract_patches(img, p)
        features = features.reshape(-1, C*p*p)
        return features
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        # Extract features of the foreground
        fg_mask = sample_dd['scribble_fg'] > 0
        bg_mask = sample_dd['scribble_bg'] > 0

        features = self.extract_features_(sample_dd)
        features_fg = features[fg_mask.reshape(-1)]
        features_bg = features[bg_mask.reshape(-1)]
        
        # Stack the foreground and background features
        features_fg_bg = np.vstack((features_fg, features_bg))
        
        centroids_fg = kmeans_fit(features_fg, self.k_fg, self.rng)
        centroids_bg = kmeans_fit(features_bg, self.k_bg, self.rng)

        # Combine centroids for distance computation
        all_centroids = np.vstack((centroids_fg, centroids_bg))
        
        labels = kmeans_predict_idx(features, all_centroids)

        # Create a mask for the foreground
        fg_mask = labels < self.k_fg
        
        return fg_mask.reshape(H, W).astype(np.uint8)

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")