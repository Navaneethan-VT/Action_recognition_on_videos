"""
Copyright(c) 2019-2022 Deep Safety GmbH.

All rights not expressly granted by the Licensor remain reserved.

This unpublished material is proprietary to Deep Safety GmbH.

Proprietary software is computer software licensed under exclusive legal right
of the copyright holder. The receipt or possession of this source code and /
or related information does not convey or imply any rights to use, reproduce,
disclose or distribute its contents, or to manufacture, use, or sell anything
that it may describe, in whole or in part unless prior written permission is
obtained from Deep Safety GmbH.

The methods and techniques described herein are considered trade secrets and /
or confidential. You shall not disclose such Confidential Information and
shall use it only in accordance with the terms of the license agreement you
entered into with Deep Safety GmbH.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES.
"""

import pandas as pd


def transpose_csv(input_file):
    pd.read_csv(input_file, header=None).T.to_csv(input_file, header=False, index=False)


def main():
    input_csv = "/home/navaneethan/working/putting_on_shoes_train.csv"
    transpose_csv(input_csv)


if __name__ == '__main__':
    main()
