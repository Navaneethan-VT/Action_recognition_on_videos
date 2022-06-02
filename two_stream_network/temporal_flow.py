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

import numpy as np
import torch
import torch.nn as nn


class Temporal_conv(nn.Module):

    def __init__(self, config):
        super(Temporal_conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(3, 96, (1, 7, 7), stride=(1,)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),

            nn.Conv3d(96, 256, (1, 5, 5), stride=(1,)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),

            nn.Conv3d(256, 512, (1, 3, 3), stride=(1,)),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.Conv3d(512, 512, (1, 3, 3), stride=(1,)),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.Conv3d(512, 512, (1, 3, 3), stride=(1,)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        ones = torch.ones(1, 3, config["clip_len"]-1, config["crop_size"], config["crop_size"])
        size = self.layers(ones)
        size = np.prod(size.shape[1:])
        self.fc1 = nn.Linear(int(size), 4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(4096, 2048)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.connection = nn.Linear(2048, 1024)
        self.connection2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.connection(x)
        x = self.connection2(x)

        return x
