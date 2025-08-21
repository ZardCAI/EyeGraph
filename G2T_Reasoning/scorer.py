import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        # (Cider(), "Cider"),
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            # print(method)
            score, scores = scorer.compute_score(gts, res, verbose=0)
        # except TypeError:
        except Exception:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def load_templates(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        templates = [line.strip() for line in f.readlines()]
    return templates

class Scorer:
    def __init__(self):
        template_file = 'template_ds_fill.txt'
        true_templates = load_templates(template_file)
        self.true_dict = {i + 1: [true_templates[i]] for i in range(10)}  # key为1-10，value为模板
        self.weights = {
            'BLEU-1': 0.1,
            'BLEU-4': 0.5,
            'METEOR': 0.2,
            'ROUGE-L': 0.15
        }
    
    def _score_report(self, report):
        pred_dict = {i + 1: [report] for i in range(10)}
        eval_res = compute_scores(self.true_dict, pred_dict)
        # print(eval_res)
        # b1, b4, meteor, rouge = eval_res['BLEU_1'], eval_res['BLEU_4'], eval_res['METEOR'], eval_res['ROUGE_L']
        composite_score = (
            self.weights['BLEU-1'] * eval_res['BLEU_1'] +
            self.weights['BLEU-4'] * eval_res['BLEU_4'] +
            self.weights['METEOR'] * eval_res['METEOR'] +
            self.weights['ROUGE-L'] * eval_res['ROUGE_L']
        )
        return composite_score
    
