# Import Required Modules

import os
import argparse
import numpy as np
import pandas as pd
import reducers as r
from tqdm import tqdm
from skimage.io import imread

##############################################################################

DEFAULT_INPUT_FOLDER = '.../'
DEFAULT_OUTPUT_FOLDER = '.../'
DEFAULT_SURVIVAL_RATE_FILE = 'GBM_Logs.csv'
DEFAULT_IMAGE_SHAPE = (1024, 1024, 3)

SURVIVAL_RATE_FILE_HEADER_NAMES = [
    "number",
    r.PATIENT_ID,
    "survival_rate_days",
    "survival_rate_months",
    "age_years",
    "primary_diagnosis",
]

ENCODINGS = [
    {
        r.PIXEL_TYPE: "Leading_Edge_LE_(Teal_or_Blue_Areas)",
        r.PIXEL_VALUE: (33, 143, 166),
    },
    {
        r.PIXEL_TYPE: "Infiltrating_Tumor_IT_(Purple_Areas)",
        r.PIXEL_VALUE: (210, 5, 208)
    },
    {
        r.PIXEL_TYPE: "Cellular_Tumor_CT_(Green_Areas)",
        r.PIXEL_VALUE: (5, 208, 4)
    },
    {
        r.PIXEL_TYPE: "Necrosis_CTne_(Black_Areas)",
        r.PIXEL_VALUE: (5, 5, 5)
    },
    {
        r.PIXEL_TYPE: "Perinecrotic_Zone_CTpnz_(Light_Blue_Areas)",
        r.PIXEL_VALUE: (37, 209, 247)
    },
    {
        r.PIXEL_TYPE: "Pseudopalisading_Cells_Around_Necrosis_CTpan_(Sea_Green_Areas)",
        r.PIXEL_VALUE: (6, 208, 170)
    },
    {
        r.PIXEL_TYPE: "Microvascular_Proliferation_CTmvp_(Red_Areas)",
        r.PIXEL_VALUE: (255, 102, 0)
    },
    {
        r.PIXEL_TYPE: "Background",
        r.PIXEL_VALUE: (255, 255, 255)
    }
]


def run(
        input_folder,
        output_folder,
        dry_run,
        verbose,
):
    if not os.path.exists(input_folder):
        raise Exception('Input directory must exist: "{input_folder}"'.format(input_folder=input_folder))
    if not os.path.exists(output_folder):
        if verbose:
            print('Creating output directory "{output_folder}"'.format(output_folder=output_folder))
        if not dry_run:
            os.makedirs(output_folder, exist_ok=True)
    survival_rate_file = os.path.join(input_folder, DEFAULT_SURVIVAL_RATE_FILE)
    if not os.path.exists(survival_rate_file):
        raise Exception('Survival rate file must exist: "{survival_rate_file}"'.format(survival_rate_file=survival_rate_file))

    survival_rate_per_patient = pd.read_csv(survival_rate_file, header=0, names=SURVIVAL_RATE_FILE_HEADER_NAMES)[[r.PATIENT_ID, 'survival_rate_days', 'survival_rate_months']]
    average_pixel_type_per_patient = pd.DataFrame()
    average_pixel_type_per_patient_reducer = r.AveragePixelTypePerPatientReducer(ENCODINGS)
    for patient_id in tqdm(os.listdir(input_folder), total = len(os.listdir(input_folder))):
        patient_folder = os.path.join(input_folder, patient_id)
        if os.path.isdir(patient_folder):
            patient_tissue_blocks = np.zeros((len(os.listdir(patient_folder)),) + DEFAULT_IMAGE_SHAPE)
            for i, patient_tissue_block in enumerate(os.listdir(patient_folder)):
                patient_tissue_block_file = os.path.join(patient_folder, patient_tissue_block)
                patient_tissue_blocks[i, :, :, :] = imread(patient_tissue_block_file)
            tissue_blocks_by_patient_id = pd.DataFrame([
                {
                    r.PATIENT_ID: patient_id,
                    r.SEGMENTATION: patient_tissue_blocks
                }
            ])
            _average_pixel_type_per_patient = average_pixel_type_per_patient_reducer \
                .calculate_average_pixel_type_per_patient(tissue_blocks_by_patient_id)
            average_pixel_type_per_patient = average_pixel_type_per_patient.append(_average_pixel_type_per_patient)
    average_pixel_type_per_patient = average_pixel_type_per_patient.drop('Background', axis=1)
    average_pixel_type_per_patient \
        .merge(survival_rate_per_patient, how="left", on=r.PATIENT_ID) \
        .to_csv(output_folder + 'average_pixel_value_per_patient.csv', index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Computers Average Pixel Values Per Patient')
    arg_parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    arg_parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    arg_parser.add_argument("--dry-run", action='store_true')
    arg_parser.add_argument("--verbose", action='store_true')

    args = arg_parser.parse_args()

    run(input_folder=args.input_folder,
        output_folder=args.output_folder,
        dry_run=args.dry_run,
        verbose=args.verbose)
