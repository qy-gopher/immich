import os
import sys
from threading import Lock
import numpy as np
import cv2
from PIL import Image
from typing import Union, List

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
import bert_tokenizer as bert

processing_lock = Lock()

img_onnx_model_path = "/cache/rknn/clip_images.rknn"
txt_onnx_model_path = "/cache/rknn/clip_text.rknn"

clip_img_model = None
clip_txt_model = None

_tokenizer = bert.FullTokenizer()


def img_preprocess(image):
    CROP_SIZE = 224
    IMAGE_SIZE = [224, 224]

    img_bgr = np.array(image)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    if h < CROP_SIZE:
        padh = (CROP_SIZE - h) // 2
        img = np.pad(img, ((padh, CROP_SIZE - h - padh), (0, 0), (0, 0)), mode='constant').astype(np.float32)
    if w < CROP_SIZE:
        padw = (CROP_SIZE - w) // 2
        img = np.pad(img, ((0, 0), (padw, CROP_SIZE - w - padw), (0, 0)), mode='constant').astype(np.float32)
    if h > CROP_SIZE and w > CROP_SIZE:
        start_x = (w - CROP_SIZE) // 2
        start_y = (h - CROP_SIZE) // 2
        img = img[start_y:start_y+CROP_SIZE, start_x:start_x+CROP_SIZE]
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]),)
    img = np.expand_dims(img, 0)

    return img


def tokenize_numpy(texts: Union[str, List[str]], context_length: int = 52) -> np.ndarray:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 52 as the context length
    Returns
    -------
    A two-dimensional numpy array containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = np.array(tokens)

    return result


def load_img_model():
    global clip_img_model
    if clip_img_model is None:

        from rknnlite.api import RKNNLite
        clip_img_model = RKNNLite()
        clip_img_model.load_rknn(img_onnx_model_path)
        clip_img_model.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)


def process_image(img):
    input = img_preprocess(img)

    tmp = []
    with processing_lock:
        load_img_model()
        tmp = clip_img_model.inference(inputs=[input])[0][0]

    l2_norm = np.linalg.norm(tmp)
    output = tmp / l2_norm

    return output


def load_txt_model():
    global clip_txt_model
    if clip_txt_model is None:

        from rknnlite.api import RKNNLite
        clip_txt_model = RKNNLite()
        clip_txt_model.load_rknn(txt_onnx_model_path)
        clip_txt_model.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)


def process_txt(txt):
    input = tokenize_numpy(txt, 52)

    tmp = []
    with processing_lock:
        load_txt_model()
        tmp = clip_txt_model.inference(inputs=[input])[0][0]

    l2_norm = np.linalg.norm(tmp)
    output = tmp / l2_norm

    return output

