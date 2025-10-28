import numpy as np
from typing import Sequence


from eo_flood_ops.thresholding_model import ThresholdingModel
from eo_flood_ops.model_utils import (
    GroundTruthMeasurement,
    LaplaceDepthSolverConfig,
    flood_extent_to_depth_solve,
    _TrainExample,
    flood_height_raster
)


class ManifoldModel:
    def __init__(
        self,
        dem: np.ndarray,
        scale: float,
        laplace_config: LaplaceDepthSolverConfig,
        force_tolerance: float,
        force_local_region_width: int,
        flood_agree_threshold: float,
    ) -> None:
        """Initializes a ManifoldModel object.

        Args:
          dem: A 2D array representing the DEM, in the shape of the ground truth
            images.
          scale: The scale of the DEM and the ground truth images.
          laplace_config: The Laplace solver config to use.
          force_tolerance: see `CatScanForceCalculator.tolerance`.
          force_local_region_width: see `CatScanForceCalculator.local_region_width`.
          agree_threshold: Used for the flood-fill algorithm. See
            `flood_height_raster` for more details.
        """
        self.dem = dem
        self.scale = scale
        self.laplace_config = laplace_config
        self.force_tolerance = force_tolerance
        self.force_local_region_width = force_local_region_width
        self.flood_agree_threshold = flood_agree_threshold
        self.thresholding_model = ThresholdingModel()

    def train(
        self, minumum_ratios: list, ground_truth: Sequence[GroundTruthMeasurement]
    ):
        print("Training an inner thresholding model used for flood-fill.")
        self.thresholding_model.train(minumum_ratios, ground_truth)

        print("Running flood extent to depth on ground truth examples..")
        sorted_ground_truth = sorted(ground_truth, key=lambda gt: gt.gauge_measurement)
        self._train_examples = []
        for gt in sorted_ground_truth:
            gauge_level = gt.gauge_measurement
            print(
                "Running flood extent to depth algorithm for image at gauge_level",
                gauge_level,
            )
            height_raster = flood_extent_to_depth_solve(
                gt.ground_truth,
                self.dem,
                self.scale,
                self.laplace_config,
                self.force_tolerance,
                self.force_local_region_width,
            )
            self._train_examples.append(
                _TrainExample(height_raster=height_raster, gauge_level=gauge_level)
            )

    def _interpolate_between(
        self, level: float, example_below: _TrainExample, example_above: _TrainExample
    ) -> np.ndarray:
        """Linearly interpolates between two train examples.

        Performs the piecewise linear interpolation on the train examples.

        Args:
          level: The gauge level to be used.
          example_below: The train example which is closest to `level` from below.
          example_above: The train example which is closest to `level` from above.

        Returns:
          The water height raster which is the linear interpolation between the
          provided train examples.
        """
        level_below = example_below.gauge_level
        level_above = example_above.gauge_level
        level_ratio = (level - level_below) / (level_above - level_below)
        return (
            level_ratio * example_above.height_raster
            + (1 - level_ratio) * example_below.height_raster
        )

    def _infer_piecewise_linear_manifold(self, gauge_level: float) -> np.ndarray:
        """Infers piecewise linear water height manifold given gauge level.

        The method performs piecewise linear interpolation between the saved train
        examples.

        Args:
          gauge_level: The gauge level to infer for.

        Returns:
          The inferred low-resolution water height manifold.
        """
        train_levels = [example.gauge_level for example in self._train_examples]
        index = np.searchsorted(train_levels, gauge_level)

        if index == 0:
            # Lower than train event.
            lowest_example = self._train_examples[0]
            return lowest_example.height_raster

        elif index == len(train_levels):
            # Extreme event.
            highest_example = self._train_examples[-1]
            # Create a new height raster by adding the difference between the current
            # measurement and the (previous) highest measurement to the height raster
            # of the (previous) highest measurement.
            return highest_example.height_raster + (gauge_level - train_levels[-1])
        else:
            return self._interpolate_between(
                level=gauge_level,
                example_below=self._train_examples[index - 1],
                example_above=self._train_examples[index],
            )

    def infer(self, gauge_level: float):
        reference_inundation_map = self.thresholding_model.infer(gauge_level)
        manifold = self._infer_piecewise_linear_manifold(gauge_level)
        return flood_height_raster(
            interpolated_height_raster=manifold,
            dem=self.dem,
            inundation_map=reference_inundation_map,
            agree_threshold=self.flood_agree_threshold,
        )
