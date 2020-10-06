from dacapo.evaluate import evaluate_affs
from dacapo.models import Model
from dacapo.tasks.post_processors import Watershed
import gunpowder as gp
import torch

import lsd

import time


class LSD_Affinities(Model):
    def __init__(self, data, model, post_processor=None):

        self.dims = data.raw.spatial_dims

        super(LSD_Affinities, self).__init__(
            model.input_shape, model.fmaps_in, self.dims
        )

        self.neighborhood = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        conv = torch.nn.Conv3d
        num_shape_descriptors = 10
        lsd = [
            model,
            conv(model.fmaps_out, num_shape_descriptors, (1,) * self.dims),
            torch.nn.Sigmoid(),  # Maybe
        ]

        affs = [
            model,
            conv(model.fmaps_out, self.dims, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.affs = torch.nn.Sequential(*affs)

        self.lsd = torch.nn.Sequential(*lsd)
        self.prediction_channels = num_shape_descriptors
        self.target_channels = num_shape_descriptors

        if post_processor is None:
            self.post_processor = Watershed()
        else:
            self.post_processor = post_processor

    def add_target(self, gt: gp.ArrayKey, target: gp.ArrayKey):

        return (
            lsd.gp.AddLocalShapeDescriptor(gt, target)
            # ensure affs are float
            # gp.Normalize(target, factor=1.0)
        )

    def add_affinities(self, gt, target):
        return (
            gp.AddAffinities(
                affinity_neighborhood=self.neighborhood, labels=gt, affinities=target
            )
            +
            # ensure affs are float
            gp.Normalize(target, factor=1.0)
        )

    def forward(self, x):
        lsd = self.lsd(x)
        affinities = self.affs(x)
        torch.cat([affinities, lsd], dim=1)
        return lsd

    def evaluate(self, predictions, gt, targets, return_results):
        reconstructions = self.post_processor.enumerate(predictions)

        for parameters, reconstruction in reconstructions:

            print(f"Evaluation post-processing with {parameters}...")
            start = time.time()
            # This could be factored out.
            # keep evaulate as a super class method
            # over-write evaluate_reconstruction
            ret = evaluate_affs(reconstruction, gt, return_results=return_results)

            print(f"...done ({time.time() - start}s)")

        if return_results:
            scores, results = ret
            yield parameters, scores, results
        else:
            yield parameters, ret