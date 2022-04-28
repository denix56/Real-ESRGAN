from PIL import Image
import joblib
from joblib import Parallel, delayed
import tqdm
from tqdm.auto import tqdm
import contextlib
import argparse
import glob
import os


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('--output', type=str, default='data/DF2K/DF2K_LR_bicubic', help='Output folder')
    parser.add_argument('--n_thread', type=int, default=1, help='Number of threads')
    parser.add_argument('--scale', type=int, default=4, help='Downscale factor')
    args = parser.parse_args()

    def worker_func(in_path, out_dir, scale):
        img = Image.open(in_path)
        h, w = img.size
        img = img.resize(size=(h//scale, w//scale), resample=Image.BICUBIC)

        basename = os.path.basename(in_path)
        img.save(os.path.join(out_dir, basename))


    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    os.makedirs(args.output, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Creating bicubic lr images", total=len(path_list))) as progress_bar:
        Parallel(n_jobs=args.n_thread)(delayed(worker_func)(path, args.output, args.scale) for path in path_list)