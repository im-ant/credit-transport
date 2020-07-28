# ==========
# CSV 2 Tensorboard for easier visualization
# ==========

import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def write_loop(args):
    """
    Assumes structure
    :param args:
    :return:
    """
    file_wc = '*progress.csv'
    step_col_name = 'Diagnostics/CumSteps'
    val_col_list = ['ReturnAverage', 'ReturnStd', 'DiscountedReturnAverage',
                    'lossAverage', 'lossStd', 'lossMax',
                    'init_q_minAverage', 'init_q_maxAverage',
                    'final_q_minAverage', 'final_q_maxAverage']

    # Find all the progress files in this experiment
    for csv_path in Path(args.in_dir).rglob(file_wc):
        # Make the parent directory name as so (heuristically)
        log_dir_name = '_'.join(csv_path.parts[-3:-1])
        log_dir_path = os.path.join(args.out_dir, log_dir_name)

        #
        print('In file:', csv_path)
        print('Out file:', log_dir_path)

        # Write file to log
        write_file(csv_path, log_dir_path,
                   step_col_name=step_col_name,
                   val_col_list=val_col_list)


def write_file(csv_path, logger_dir_path,
               step_col_name='Diagnostics/CumSteps',
               val_col_list=['ReturnAverage']):
    # Open csv and logger
    csvfile = open(csv_path, newline='\n')
    reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    logger = SummaryWriter(log_dir=logger_dir_path)

    # Count total rows for tqdm
    num_rows_count = sum(1 for row in reader)
    csvfile.seek(0)  # reset index
    next(reader)  # skip header

    # Write each row to logger
    for row_dict in tqdm(reader, total=num_rows_count):
        # Get step for current row
        row_step = row_dict[step_col_name]

        # Iterate over columns and write
        for k in val_col_list:
            try:
                # Get value
                val = float(row_dict[k])
                logger.add_scalar(str(k), val, global_step=row_step)
            except ValueError:
                pass
            except KeyError:
                pass  # TODO throw warning instead

    # Close
    csvfile.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--in_dir', type=str,
        default='/network/tmp1/chenant/ant/cred_transport/long_arms/07-25/exp2_sanity',
        help='path to the directory containing input csv files'
    )

    parser.add_argument(
        '--out_dir', type=str,
        default='./tmp_out',
        help='path to the output dir to write the tensorboards'
    )

    #
    args = parser.parse_args()
    print(args)

    # ==
    # Write
    write_loop(args)
