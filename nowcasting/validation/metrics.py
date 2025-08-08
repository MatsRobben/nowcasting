# ==============================================================================
# Original Code Source:
# This code is a slightly modified version from the repository accompanying
# the paper "Improving precipitation nowcasting for high-intensity events using deep 
# generative models with balanced loss and temperature data: a case study in the Netherlands" by
# Charlotte Van Nooten and colleagues.
# Repository: https://github.com/charlottecvn/precipitationnowcasting-generativemodels-highevents
# ==============================================================================

import numpy as np
import os
from typing import List, Tuple, Optional, Union

from pysteps.verification.detcatscores import det_cat_fct_init, det_cat_fct_accum, det_cat_fct_compute, det_cat_fct_merge
from pysteps.verification.detcontscores import det_cont_fct_init, det_cont_fct_accum, det_cont_fct_compute
from pysteps.verification.spatialscores import fss_init, fss_accum, fss_compute

def centercrop_tensor(tensor: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Center-crops a 2D (or higher-dimensional where the last two dims are spatial)
    NumPy array to a specified `size`.

    Parameters:
        tensor : np.ndarray
            The input tensor. The last two dimensions are assumed to be
            height and width, respectively.
        size : Tuple[int, int], default (64, 64)
            A tuple (height, width) specifying the desired output size of the crop.

    Returns:
        np.ndarray
            The center-cropped tensor.
    """
    height, width = tensor.shape[-2:]
    center = (height//2, width//2)

    height_range = (center[0] - size[0]//2, center[0] + size[0]//2)
    width_range = (center[1] - size[1]//2, center[1] + size[1]//2)

    return tensor[height_range[0]:height_range[1], width_range[0]:width_range[1]]

class Evaluator:
    '''
    This functions is used to validate a models predictions.
    Categorical scores and continues errors are accumulated in dictionaries.
    When validation is finished, the metrics can be computed by using these dictionaries.
    nowcast_method: indicates with what model the nowcast were made
    thresholds: precipitation threshold to use for the categorical scores
    leadtimes: the leadtimes of the predictions
    save_after_n_sample: if higher than 0, the evaluator will save its dictionary after it has seen n samples. 
    scales: The spatial scales used to compute FSS at different spatial levels
    '''
    def __init__(self,
                 nowcast_method: str,
                 dir: Optional[str] = None,
                 thresholds: List[float] = [0, 0.5, 1, 2, 5, 10, 30],
                 leadtimes: List[int] = [5, 10, 15, 30, 60, 90],
                 save_after_n_samples: int = 0,
                 scales: List[int] = [1, 2, 4, 16, 32, 64]):
        """
        Initializes the Evaluator with specified verification settings.

        Parameters:
            nowcast_method : str
                A string identifier for the nowcasting model being evaluated
                (e.g., "DGMR", "PySTEPS", "Earthformer"). Used for organizing saved results.
            dir : Optional[str], default None
                An optional subdirectory name to organize results. If None, results
                are saved directly under `./results/scores/{nowcast_method}`.
            thresholds : List[float], default [0, 0.5, 1, 2, 5, 10, 30]
                List of precipitation thresholds (e.g., mm/h) for categorical and FSS scores.
            leadtimes : List[int], default [5, 10, 15, 30, 60, 90]
                List of forecast leadtimes (in minutes) for which to compute metrics.
            save_after_n_samples : int, default 0
                If greater than 0, accumulated scores will be saved to disk
                every `n` full samples processed. A full sample means predictions
                for all defined leadtimes have been verified once.
            scales : List[int], default [1, 2, 4, 16, 32, 64]
                List of spatial scales (in pixels) for which to compute FSS.
        """
        self.nowcast_method = nowcast_method
        self.dir = dir
    
        ## Create dictionaries to compute the model errors
        # dict for each threshold and leadtime combination. shape = (n_leadtimes, n_thresholds))
        self.cat_dicts = np.array([[det_cat_fct_init(thr) for thr in thresholds] for _ in leadtimes])
        # For the MSE and MAE create dictonary per leadtime to accumulate the errors
        self.cont_dicts = np.array([det_cont_fct_init() for _ in leadtimes])
        
        self.fss_dicts = np.array([[[fss_init(thr = thr, scale = scale) 
                                     for scale in scales] 
                                    for thr in thresholds]
                                   for _ in leadtimes])
        
        self.leadtimes = leadtimes
        self.thresholds = thresholds
        self.scales = scales

        self.save_after_n_samples = save_after_n_samples
        self.n_verifies = 0

    def verify(self, y: np.ndarray, y_pred: np.ndarray, leadtime: int, centercrop: bool = True):
        """
        Accumulates verification metrics for a single observation-prediction pair
        at a specific leadtime.

        Parameters:
            y : np.ndarray
                The observed precipitation field (ground truth).
                Expected shape: (Height, Width).
            y_pred : np.ndarray
                The predicted precipitation field.
                Expected shape: (Height, Width).
            leadtime : int
                The leadtime (in minutes) corresponding to the provided `y` and `y_pred`.
                Must be present in `self.leadtimes`.
            centercrop : bool, default True
                If True, `y` and `y_pred` will be center-cropped to (64, 64)
                before accumulating scores.
        """
        if centercrop:
            y = centercrop_tensor(y)
            y_pred = centercrop_tensor(y_pred)

        index = self.leadtimes.index(leadtime)
        for cat_dict in self.cat_dicts[index]:
            det_cat_fct_accum(cat_dict, obs = y, pred = y_pred)
        det_cont_fct_accum(self.cont_dicts[index], obs = y, pred = y_pred)
        
        for i, fss_thr in enumerate(self.fss_dicts[index]):
            for j, fss_scale in enumerate(fss_thr):
                fss_accum(self.fss_dicts[index, i, j], X_o = y, X_f = y_pred)
                
        # Make checkpount if model went through n samples
        self.n_verifies +=1
        # verify is called for each lead time
        n_samples = self.n_verifies/len(self.leadtimes)
        if  self.save_after_n_samples > 0 and n_samples % self.save_after_n_samples == 0:
            self.save_accum_scores(n_samples)

    def get_scores(self) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Computes the final categorical, continuous, and FSS scores from the
        accumulated statistics.

        Returns:
            Tuple[List[dict], List[dict], List[dict]]
                A tuple containing three lists of dictionaries:
                - cat_scores: List of dictionaries, each representing categorical scores
                              (POD, CSI, FAR, BIAS) for a specific leadtime and threshold.
                              Each dict includes 'leadtime' and 'threshold' keys.
                - cont_scores: List of dictionaries, each representing continuous scores
                               (MSE, MAE) for a specific leadtime. Each dict includes
                               a 'leadtime' key.
                - fss_scores: List of dictionaries, each representing the FSS value
                              for a specific leadtime, threshold, and scale. Each dict
                              includes 'leadtime', 'threshold', and 'scale' keys.
        """
        cat_scores = []
        for i, lt in enumerate(self.leadtimes):
            for j, thr in enumerate(self.thresholds):
                cat_score = det_cat_fct_compute(self.cat_dicts[i,j], scores = ['POD', 'CSI', 'FAR', 'BIAS'])
                cat_score['threshold'] = thr
                cat_score['leadtime'] = lt
                cat_scores.append(cat_score)

        cont_scores = [det_cont_fct_compute(cont_dict, scores = ['MSE', 'MAE']) for cont_dict in self.cont_dicts]
        for lt, cont_score in zip(self.leadtimes, cont_scores):
            cont_score['leadtime'] = lt
            
        # Compute FSS scores
        fss_scores = []
        for i, lt in enumerate(self.leadtimes):
            for j, thr in enumerate(self.thresholds):
                for k, scale in enumerate(self.scales):
                    fss_val = fss_compute(self.fss_dicts[i, j, k])
                    fss_scores.append({
                        'leadtime': lt,
                        'threshold': thr,
                        'scale': scale,
                        'FSS': fss_val
                    })
                    
        return cat_scores, cont_scores, fss_scores

    def save_accum_scores(self, n_samples: int):
        """
        Saves the current accumulated score dictionaries to disk.
        This allows for resuming evaluation or analyzing partial results.

        Parameters:
            n_samples : int
                The number of full samples that have been processed when saving.
                Used in the filename to indicate progress.
        """
        if self.dir is None:
            dir = f'./results/scores/{self.nowcast_method}'
        else:
            dir = f'./results/scores/{self.dir}/{self.nowcast_method}'

        os.makedirs(dir, exist_ok=True)

        np.save(f'{dir}/cat_dicts', self.cat_dicts)
        np.save(f'{dir}/cont_dicts', self.cont_dicts)
        np.save(f'{dir}/fss_dicts', self.fss_dicts)
        np.save(f'{dir}/n_sample', n_samples)


    def load_accum_scores(self):
        """
        Loads previously saved accumulated score dictionaries from disk.
        This is useful for resuming an interrupted evaluation process.
        """
        if self.dir is None:
            dir = f'./results/scores/{self.nowcast_method}'
        else:
            dir = f'./results/scores/{self.dir}/{self.nowcast_method}'

        self.cat_dicts = np.load(f'{dir}/cat_dicts.npy', allow_pickle=True)
        self.cont_dicts = np.load(f'{dir}/cont_dicts.npy', allow_pickle=True)
        self.fss_dicts = np.load(f'{dir}/fss_dicts.npy', allow_pickle=True)
        self.n_verifies = 3 * np.load(f'{dir}/n_sample.npy')