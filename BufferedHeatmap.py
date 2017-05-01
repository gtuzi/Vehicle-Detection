# Copyright (c) 2017, Gerti Tuzi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Gerti Tuzi nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




import numpy as np
from utils import  add_heat, apply_threshold
class BufferedHeatmap(object):

    def __init__(self, shape):
        """
        
        :param shape: heatmap shape 
        """
        self.nbuff = 3
        self.hmshape = shape
        self.initMaps()


    def initMaps(self):
        self.heatmaps = np.zeros(shape=(self.nbuff,) + self.hmshape, dtype=np.int32)
        self.initmaps = np.zeros(shape=(self.nbuff, 1), dtype=np.bool)



    def onBoxes(self, bboxes):
        """
            Generate maps from bounding boxes
        :param bboxes: 
        :return: 
        """
        self._add_heatmap(bboxes)


    def getThreshBuffHeatMap(self, thresh = 2):
        """
        Return thresholded buffered heat map results.
        Existing maps are added together. Then threshold is applied
        :param: threshold to use
        :return: 
        """
        heatmap_cum = np.sum(self.heatmaps, axis=0)
        return apply_threshold(heatmap_cum, threshold=thresh)

    def _add_heatmap(self, bboxes):
        """
            Add new heat map by shifting up older maps and inserting
            new maps at the bottom of the buffer. 
            Initialization flags are updated accordingly
        :param bboxes: box list
        :return: 
        """
        # Accumulate new heatmaps
        # Shift existing entries up
        for i in range(0, self.nbuff - 1):
            self.heatmaps[i, :, :] = self.heatmaps[i + 1, :, :]
            self.initmaps[i] = self.initmaps[i + 1]

        # Insert new map
        newmap = np.zeros_like(self.heatmaps[-1, :, :])
        self.heatmaps[-1, :, :] = add_heat(newmap, bbox_list=bboxes)
        # Flag that we have initialized it
        self.initmaps[-1] = True
