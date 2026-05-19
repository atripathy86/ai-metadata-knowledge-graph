###
# Copyright (2024) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

"""
Shared vocabulary lists and utility functions used across task, model, dataset,
and pipeline similarity modules.
"""

import re
import glob
import os

task_category_vocab = [
    'recognition', 'regression', 'reconstruction', 'segmentation', 'detection',
    'generation', 'harmonization', 'translation', 'classification', 'adaptation',
    'search', 'analysis', 'extraction', 'retrieval', 'annotation', 'generalization',
    'augmentation', 'anonymization', 'prediction', 'correlation', 'fusion', 'matching',
    'synthesis', 'understanding', 'testing', 'parsing', 'identification', 'transfer',
    'spotting', 'estimation', 'resolution', 'clustering', 'separation', 'localization',
    'summarization', 'recommendation', 'expansion', 'labeling', 'imaging',
    'interpretation', 'captioning', 'selection', 'assessment', 'registration',
    'forecasting', 'planning', 'tracking', 'inference', 'grounding', 'disambiguation',
    'reasoning', 'comprehension', 'reading', 'reduction', 'completion', 'compression',
    'decomposition', 'learning', 'sampling', 'verification', 'animation',
    'interpolation', 'visualization', 'propagation', 'mining', 'surveillance',
    'diagnosis', 'ranking', 'optimization', 'anomaly', 'linking',
]

image_vocab = [
    '2d', '3d', 'image', 'visual', 'depth', 'pixel', 'voxel', 'RBG', 'action',
    'object', 'facial', 'pose', 'grayscale', 'texture', 'pattern', 'face', 'scene',
    'imagery', 'imaging', 'image-based', 'vision', 'computer-vision', 'computer vision',
]
text_vocab = [
    'text', 'word', 'language', 'lingual', 'dialogue', 'dialog', 'corpus', 'sentence',
    'reading', 'news', 'reviews', 'grammar', 'grammatical', 'natural-language-processing',
    'nlp', 'natural language processing', 'textual', 'translation', 'question', 'answering',
    'conversational', 'conversation', 'entity', 'document', 'paragraph', 'paraphrase',
]
video_vocab = ['video', 'video-based', 'motion']
audio_vocab = ['audio', 'voice', 'speech', 'sound', 'headphone', 'music', 'spoken']
multi_vocab = ['multi', 'cross', 'multimodal', 'crossmodal', 'multi-modal']
super_class = ['image', 'text', 'video', 'audio']


def find_file_path(filename, search_directory="."):
    for file_path in glob.iglob(f"{search_directory}/**/{filename}", recursive=True):
        return os.path.abspath(file_path)
    return None


def create_tokens(tid):
    return [t.lower() for t in re.split(r'[-\s]+', tid) if t]


def compute_IOU(tokens1, tokens2):
    return len(set(tokens1).intersection(tokens2)) / len(set(tokens1).union(tokens2))


def compute_category(item_tokens):
    tokens = list(item_tokens)
    inter = list(set(tokens).intersection(set(task_category_vocab)))
    return ','.join(inter) if inter else 'none'


def compute_modality(item_tokens):
    tokens = list(item_tokens)
    modality_list = []
    if set(tokens).intersection(set(image_vocab)):
        modality_list.append('image')
    elif set(tokens).intersection(set(text_vocab)):
        modality_list.append('text')
    elif set(tokens).intersection(set(audio_vocab)):
        modality_list.append('audio')
    elif set(tokens).intersection(set(video_vocab)):
        modality_list.append('video')
    elif set(tokens).intersection(set(multi_vocab)):
        modality_list.append('multimodal')
    if len(set(modality_list).intersection(set(super_class))) > 1:
        modality_list.append('multimodal')
    return ','.join(modality_list) if modality_list else 'none'
