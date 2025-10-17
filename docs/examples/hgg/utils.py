from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor

from tumortwin.postprocessing import (
    plot_imaging_summary,
    plot_patient_timeline
)
from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.types import CropSettings, CropTarget
from tumortwin.types.hgg_data import HGGPatientData


DATA_FOLDER = Path(__file__).resolve().parent.joinpath("input_files")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_patient_data(
    folder_name: str, 
    info_fname: str,
    num_visits_calibration: int, 
    plot: bool=False
) -> Tuple[HGGPatientData, List, List, Tensor]:
    """Reads in data for a particular patient.
    
    num_visits_calibration: number of visit images to match (including 
    initial visit).
    """

    PATIENT_INFO_PATH = Path(DATA_FOLDER, folder_name, info_fname)
    IMAGE_PATH = Path(DATA_FOLDER, folder_name)

    crop_settings = CropSettings(
        crop_to=CropTarget.ROI_ENHANCE, 
        padding=10, 
        visit_index=-1
    )

    patient_data = HGGPatientData.from_file(
        PATIENT_INFO_PATH, 
        image_dir=IMAGE_PATH, 
        crop_settings=crop_settings
    )

    measured_cellularity_maps = [
        ADC_to_cellularity(
            visit.adc_image, 
            visit.roi_enhance_image, 
            visit.roi_nonenhance_image
        )
        for visit in patient_data.visits
    ]

    target_timepoints = [
        visit.time 
        for visit in patient_data.visits[:num_visits_calibration]
    ]
    target_solution = torch.stack([
        torch.from_numpy(m.array)
        for m in measured_cellularity_maps[:num_visits_calibration]
    ])

    if plot:
        plot_patient_timeline(patient_data)
        plot_imaging_summary(patient_data)

    return patient_data, measured_cellularity_maps, target_timepoints, target_solution