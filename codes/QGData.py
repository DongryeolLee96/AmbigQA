import os
import json
import gzip
import re
import pickle as pkl
import string
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from QAData import QAData, AmbigQAData
from DataLoader import MySimpleQADataset, MySimpleQADatasetForPair, MyDataLoader
from util import decode_span_batch

# for evaluation
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
from pycocoevalcap.bleu.bleu import Bleu
from rouge import Rouge
from transformers import BartTokenizer

TEST_IDS = [2,
 3,
 6,
 7,
 9,
 11,
 12,
 15,
 17,
 18,
 21,
 22,
 23,
 27,
 28,
 29,
 30,
 31,
 32,
 34,
 38,
 42,
 43,
 44,
 46,
 48,
 50,
 51,
 52,
 57,
 59,
 60,
 61,
 66,
 69,
 76,
 77,
 78,
 79,
 80,
 81,
 88,
 90,
 92,
 94,
 96,
 97,
 98,
 100,
 102,
 109,
 110,
 112,
 116,
 119,
 121,
 122,
 125,
 127,
 128,
 130,
 131,
 134,
 135,
 136,
 137,
 139,
 142,
 144,
 152,
 153,
 156,
 158,
 159,
 161,
 163,
 164,
 167,
 168,
 169,
 171,
 174,
 176,
 179,
 182,
 184,
 185,
 190,
 193,
 196,
 197,
 198,
 200,
 201,
 202,
 206,
 208,
 210,
 219,
 220,
 223,
 224,
 226,
 227,
 228,
 231,
 233,
 235,
 237,
 239,
 240,
 241,
 244,
 245,
 246,
 247,
 250,
 252,
 253,
 255,
 256,
 259,
 260,
 264,
 266,
 270,
 271,
 274,
 275,
 276,
 277,
 278,
 280,
 282,
 283,
 284,
 285,
 286,
 287,
 288,
 290,
 291,
 293,
 294,
 297,
 300,
 302,
 304,
 306,
 307,
 308,
 310,
 311,
 314,
 317,
 318,
 320,
 322,
 324,
 325,
 327,
 329,
 330,
 335,
 339,
 340,
 341,
 343,
 344,
 345,
 346,
 347,
 348,
 349,
 350,
 351,
 354,
 355,
 356,
 357,
 358,
 360,
 362,
 363,
 364,
 365,
 367,
 368,
 369,
 371,
 374,
 375,
 378,
 379,
 380,
 384,
 386,
 398,
 401,
 402,
 404,
 405,
 406,
 407,
 408,
 410,
 412,
 419,
 420,
 424,
 425,
 429,
 430,
 431,
 432,
 435,
 438,
 439,
 441,
 443,
 445,
 446,
 449,
 450,
 451,
 454,
 456,
 458,
 459,
 460,
 461,
 462,
 464,
 470,
 473,
 474,
 475,
 476,
 479,
 480,
 481,
 483,
 485,
 489,
 490,
 492,
 494,
 496,
 498,
 500,
 503,
 511,
 516,
 518,
 519,
 520,
 522,
 523,
 524,
 525,
 529,
 531,
 532,
 533,
 534,
 536,
 541,
 542,
 543,
 544,
 545,
 551,
 552,
 554,
 556,
 557,
 559,
 561,
 578,
 580,
 581,
 583,
 584,
 586,
 587,
 589,
 593,
 596,
 598,
 600,
 601,
 603,
 604,
 605,
 610,
 611,
 612,
 613,
 614,
 615,
 616,
 617,
 618,
 621,
 622,
 623,
 624,
 625,
 626,
 627,
 629,
 631,
 632,
 633,
 635,
 636,
 637,
 639,
 641,
 644,
 646,
 647,
 649,
 650,
 653,
 654,
 655,
 656,
 657,
 658,
 659,
 660,
 661,
 663,
 667,
 668,
 670,
 671,
 676,
 677,
 678,
 679,
 681,
 685,
 686,
 687,
 689,
 692,
 693,
 694,
 695,
 697,
 703,
 704,
 705,
 707,
 711,
 712,
 713,
 714,
 719,
 720,
 722,
 725,
 727,
 730,
 732,
 733,
 735,
 736,
 738,
 742,
 743,
 744,
 745,
 748,
 749,
 752,
 758,
 761,
 762,
 763,
 765,
 767,
 768,
 771,
 774,
 775,
 776,
 778,
 779,
 781,
 782,
 785,
 786,
 790,
 792,
 794,
 795,
 797,
 798,
 799,
 801,
 803,
 806,
 807,
 808,
 810,
 813,
 822,
 823,
 825,
 827,
 828,
 829,
 830,
 831,
 832,
 833,
 834,
 837,
 839,
 840,
 843,
 847,
 848,
 849,
 854,
 855,
 856,
 857,
 858,
 860,
 861,
 863,
 865,
 867,
 868,
 869,
 870,
 876,
 878,
 880,
 881,
 882,
 884,
 886,
 887,
 889,
 890,
 891,
 892,
 893,
 895,
 896,
 902,
 903,
 904,
 905,
 906,
 907,
 908,
 910,
 911,
 913,
 914,
 915,
 916,
 920,
 921,
 922,
 924,
 925,
 928,
 930,
 932,
 935,
 939,
 940,
 942,
 944,
 946,
 947,
 948,
 949,
 952,
 957,
 959,
 963,
 964,
 966,
 969,
 971,
 973,
 974,
 978,
 979,
 981,
 986,
 988,
 991,
 992,
 993,
 994,
 995,
 997]

def parse_clarification_question(cq):
    temp = cq.split("--")
    if len(temp) != 2:
        return "invalid form", ["invalid form"]
    ap, option_string = temp
    def _extract_ap(ap):
        if "Could you clarify '" in ap:
            temp = "could you clarify '"
            ap = ap[len(temp):-1]
        elif "Could you clarify'" in ap:
            temp = "could you clarify'"
            ap = ap[len(temp):-1]
        if "be more specific" in ap or len(ap) == 0:
            ap = "SPECIFY"
        return ap
    def _extract_options(option_string):
        options = []
        flag = False
        def _is_valid(ch, future):
            escapes = ["', '", "' ,'", "','",
                       "', or '", "',or '", "', or'", "',or'",
                       "', '", "','", "' ,'",
                       "' or '", "'or '", "' or'",
                       "'?", "' ?"]
            if any([future.startswith(es) for es in escapes]):
                return False
            return True
        for idx, ch in enumerate(option_string):
            if "'" in ch and not flag:
                flag = True
                options.append("")
            elif flag and _is_valid(ch, option_string[idx:]):
                options[-1] += ch
                continue
            elif flag:
                flag=False
        return options
    ap = _extract_ap(ap)
    options = _extract_options(option_string)
    return ap, options


class QGData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(QGData, self).__init__(logger, args, data_path, is_training, passages)
        self.qg_tokenizer = PTBTokenizer()
        self.metric = "Bleu"
        if not self.is_training:
            self.qg_tokenizer = PTBTokenizer()

    def load_dpr_data(self):
        dpr_retrieval_path = "out/dpr/{}_predictions.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type)
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = dpr_retrieval_path.replace(".json", "_{}_qg.json".format(postfix))
        assert "Bart" in postfix
        return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)

    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
        else:
            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                self.passages.load_tokenized_data("bart", all=True)
            if "train_for_inference" not in dpr_retrieval_path:
                dpr_retrieval_path = dpr_retrieval_path.replace("train", "train_for_inference")
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
                assert len(dpr_passages)==len(self)
            assert self.args.psg_sel_dir is not None
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference") if "for_inference" not in self.data_type else self.data_type,
                                          "_20200201" if self.args.wiki_2020 else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
                assert len(fg_passages)==len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)
            bos_token_id = self.tokenizer.bos_token_id

            def _get_tokenized_answer(idx):
                tokens = self.tokenized_data[2][idx]
                if 0 in self.tokenized_data[3][idx]:
                    tokens = tokens[:self.tokenized_data[3][idx].index(0)]
                assert tokens[0]==tokens[1]==self.tokenizer.bos_token_id and tokens[-1]==self.tokenizer.eos_token_id
                return tokens[2:-1]

            def _included(tokens, curr_input_ids, end_of_answer):
                for i in range(end_of_answer, 1024-len(tokens)+1):
                    if curr_input_ids[i:i+len(tokens)]==tokens:
                        return True
                return False

            has_valid = []
            new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], [], [], []
            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

                # create multiple inputs
                answer_input_ids_list, answer_attention_mask_list, is_valid_list = [], [], []
                for answer_idx in range(*curr_metadata):
                    end_of_answer = decoder_input_ids[answer_idx].index(self.tokenizer.eos_token_id)+1
                    answer_input_ids = decoder_input_ids[answer_idx][:end_of_answer]
                    answer_attention_mask = decoder_attention_mask[answer_idx][:end_of_answer]
                    offset = 0
                    while len(answer_input_ids)<1024:
                        assert dpr_input_ids[offset][0] == bos_token_id
                        assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                        assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                        answer_input_ids += dpr_input_ids[offset][1:]
                        answer_attention_mask += dpr_attention_mask[offset][1:]
                        offset += 1
                    assert len(answer_input_ids)==len(answer_attention_mask)
                    answer_input_ids_list.append(answer_input_ids[:1024])
                    answer_attention_mask_list.append(answer_attention_mask[:1024])
                    is_valid_list.append(_included(
                        decoder_input_ids[answer_idx][2:end_of_answer-1],
                        answer_input_ids, end_of_answer))

                has_valid.append(any(is_valid_list))
                if self.is_training:
                    if not any(is_valid_list):
                        is_valid_list = [True for _ in is_valid_list]
                    new_metadata.append((len(new_input_ids), len(new_input_ids)+sum(is_valid_list)))
                    new_input_ids += [answer_input_ids for answer_input_ids, is_valid in
                                      zip(answer_input_ids_list, is_valid_list) if is_valid]
                    new_attention_mask += [answer_attention_mask for answer_attention_mask, is_valid in
                                           zip(answer_attention_mask_list, is_valid_list) if is_valid]
                else:
                    index = is_valid_list.index(True) if any(is_valid_list) else 0
                    new_metadata.append((len(new_input_ids), len(new_input_ids)+1))
                    new_input_ids.append(answer_input_ids_list[index])
                    new_attention_mask.append(answer_attention_mask_list[index])
                new_decoder_input_ids.append(curr_input_ids)
                new_decoder_attention_mask.append(curr_attention_mask)

            assert len(new_input_ids)==len(new_attention_mask)==new_metadata[-1][-1]
            self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
            with open(dpr_tokenized_path, "w") as f:
                json.dump(self.tokenized_data, f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))
            self.logger.info("%.1f%% questions have at least one answer mentioned in passages" % (100*np.mean(has_valid)))


    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        self.dataset = MySimpleQADataset(input_ids,
                                            attention_mask,
                                            decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                            decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                            in_metadata=metadata,
                                            out_metadata=None,
                                            is_training=self.is_training,
                                            answer_as_prefix=self.args.nq_answer_as_prefix)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, is_training=self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, n_paragraphs=None):
        assert len(predictions)==len(self), (len(predictions), len(self))
        bleu = []

        # first, tokenize
        data_to_tokenize = {}
        for i, (d, pred) in enumerate(zip(self.data, predictions)):
            data_to_tokenize["ref.{}".format(i)] = [{"caption": d["question"]}]
            data_to_tokenize["gen.{}".format(i)] = [{"caption": pred if type(pred)==str else pred[0]}]
        all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)
        for i in range(len(self.data)):
            reference = {"sent": [normalize_answer(text) for text in all_tokens["ref.{}".format(i)]]}
            generated = {"sent": [normalize_answer(text) for text in all_tokens["gen.{}".format(i)]]}
            bleu.append(Bleu(4).compute_score(reference, generated)[0][-1])
        return np.mean(bleu)

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        save_path = os.path.join(self.args.output_dir, "{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 and not self.args.ambigqa else ""))
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))


class AmbigQGData(AmbigQAData, QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGData, self).__init__(logger, args, data_path, is_training, passages)

        with open("/".join(data_path.split("/")[:-2]) + "/nqopen/{}.json".format(self.data_type), "r") as f:
            orig_data = json.load(f)
            id_to_orig_idx = {d["id"]:i for i, d in enumerate(orig_data)}

        self.ref_questions = []
        self.ref_answers = []
        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                questions.append([[q.strip() for q in pair["question"].split("|")] for pair in annotation["qaPairs"]])
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])
            self.ref_questions.append(questions)
            self.ref_answers.append(answers)
            self.data[i]["orig_idx"] = id_to_orig_idx[d["id"]]


        self.SEP = "<SEP>"
        self.qg_tokenizer = PTBTokenizer()
        self.metric = "EDIT-F1"
        if not self.is_training:
            self.qg_tokenizer = PTBTokenizer()

    # override
    def load_dpr_data(self):
        task = "qg"
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix

        data_type_for_dpr_retrieval_path = self.data_type+"_2020" if self.args.wiki_2020 else self.data_type
        data_type_dpr_tokenized_path = self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type
        dpr_retrieval_path = Path(self.args.psg_sel_dir) / f"ambigqa_{data_type_for_dpr_retrieval_path}.json"
        data_name = f"ambigqa_predictions_{postfix}_{task}"
        dpr_tokenized_path = f"out/dpr/{data_type_dpr_tokenized_path}_{data_name}.json"

        self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)

        # in attention_mask, 1 means answer + passages, 2 means prompt, 3 means other answers
        do_include_prompt=True
        do_include_others=True
        new_input_ids, new_attention_mask = [], []
        for input_ids, attention_mask in zip(self.tokenized_data[0], self.tokenized_data[1]):
            _input_ids = [_id for _id, mask in zip(input_ids, attention_mask)
                          if mask==1 or (do_include_prompt and mask==2) or (do_include_others and mask==3)]
            _attention_mask = [1 for mask in attention_mask
                          if mask==1 or (do_include_prompt and mask==2) or (do_include_others and mask==3)]
            assert len(_input_ids)==len(_attention_mask)
            while len(_input_ids)<1024:
                _input_ids.append(self.tokenizer.pad_token_id)
                _attention_mask.append(0)
            new_input_ids.append(_input_ids[:1024])
            new_attention_mask.append(_attention_mask[:1024])
        self.tokenized_data[0] = new_input_ids
        self.tokenized_data[1] = new_attention_mask


    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):

        self.logger.info(dpr_tokenized_path)

        if self.is_training and self.args.consider_order_for_multiple_answers:
            dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_ordered.json")

        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
            return

        import itertools
        self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
        if self.passages.tokenized_data is None:
            self.passages.load_tokenized_data("bart", all=True)

        with open(dpr_retrieval_path.format(self.data_type).replace("train", "train_for_inference"), "r") as f:
            dpr_passages = json.load(f)
        assert self.args.psg_sel_dir is not None

        psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          self.data_type.replace("train", "train_for_inference"),
                                          "_20200201" if self.args.wiki_2020 else ""))
        self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
        with open(psg_sel_fn, "r") as f:
            fg_passages = json.load(f)
            assert len(fg_passages)==len(dpr_passages)
            dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]

        # added to convert original DPR data to AmbigQA DPR data
        dpr_passages = [dpr_passages[d["orig_idx"]] for d in self.data]

        assert len(dpr_passages)==len(self)
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int


        def _get_tokenized_answer(idx):
            tokens = decoder_input_ids[idx]
            if 0 in decoder_attention_mask[idx]:
                tokens = tokens[:decoder_attention_mask[idx].index(0)]
            assert tokens[0]==tokens[1]==bos_token_id and tokens[-1]==eos_token_id
            return tokens[2:-1]

        def _included(tokens, curr_input_ids):
            for i in range(len(curr_input_ids)+1):
                if curr_input_ids[i:i+len(tokens)]==tokens:
                    return True
            return False

        new_input_ids, new_attention_mask = [], []
        new_output, new_metadata = [], []
        chosen_list = []
        for idx, (curr_input_ids, curr_attention_mask, dpr_ids) in tqdm(enumerate(
                zip(input_ids, attention_mask, dpr_passages))):
            if self.ref_questions[idx] is None:
                continue

            end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
            q_input_ids = curr_input_ids[:end_of_question]

            p_input_ids, p_attention_mask = [], []
            dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
            dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
            offset = 0
            while len(p_input_ids)<1024:
                assert dpr_input_ids[offset][0] == bos_token_id
                assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                p_input_ids += dpr_input_ids[offset][1:]
                p_attention_mask += dpr_attention_mask[offset][1:]

            tokenized_ref_answers_list, is_valid_list, n_missing_list = [], [], []
            for ref_questions, ref_answers, ref_metadata in zip(self.ref_questions[idx],
                                                                self.ref_answers[idx],
                                                                metadata[idx]):
                # ref_metadata: [(0, 1), (1, 4)]
                assert type(ref_metadata[0][0])==int
                assert [len(ref_answer)==end-start for ref_answer, (start, end)
                        in zip(ref_answers, ref_metadata)]
                tokenized_ref_answers = [[_get_tokenized_answer(i) for i in range(*m)] for m in ref_metadata]
                is_valid = [[_included(tokens, p_input_ids) for tokens in _tokens] for _tokens in tokenized_ref_answers]
                n_missing = np.sum([not any(v) for v in is_valid])
                tokenized_ref_answers_list.append(tokenized_ref_answers)
                is_valid_list.append(is_valid)
                n_missing_list.append(n_missing)

            min_n_missing = np.min(n_missing_list)
            annotation_indices = [ann_idx for ann_idx in range(len(n_missing_list))
                                  if n_missing_list[ann_idx]==min_n_missing]

            def _form_data(annotation_idx):
                ref_questions = self.ref_questions[idx][annotation_idx]
                ref_answers = self.ref_answers[idx][annotation_idx]
                tokenized_ref_answers = tokenized_ref_answers_list[annotation_idx]
                assert len(ref_questions)==len(ref_answers)==len(tokenized_ref_answers)==len(is_valid_list[annotation_idx])
                final_ref_questions, final_ref_answers = [], []
                chosen_indices = []
                for (ref_question, ref_answer, tok_ref_answer, is_valid) in \
                        zip(ref_questions, ref_answers, tokenized_ref_answers, is_valid_list[annotation_idx]):
                    assert len(ref_answer)==len(tok_ref_answer)==len(is_valid)
                    chosen_idx = is_valid.index(True) if True in is_valid else 0
                    chosen_indices.append(chosen_idx)
                    final_ref_questions.append(ref_question[0])
                    final_ref_answers.append(tok_ref_answer[chosen_idx])
                for i, final_ref_question in enumerate(final_ref_questions):
                    input_ids = [bos_token_id, bos_token_id] + final_ref_answers[i]
                    attention_mask = [1 for _ in input_ids]
                    input_ids += [sep_token_id] + q_input_ids
                    attention_mask += [2 for _ in range(len(q_input_ids)+1)]
                    for j, answer in enumerate(final_ref_answers):
                        if j==i: continue
                        input_ids += [sep_token_id] + answer
                        attention_mask += [3 for _ in range(len(answer)+1)]
                    input_ids += p_input_ids
                    attention_mask += p_attention_mask
                    assert len(input_ids)==len(attention_mask)
                    new_input_ids.append(input_ids)
                    new_attention_mask.append(attention_mask)
                    new_output.append(final_ref_question)
                return chosen_indices

            start = len(new_output)
            if self.is_training:
                start = len(new_output)
                for annotation_idx in annotation_indices:
                    _form_data(annotation_idx)
            else:
                annotation_idx = annotation_indices[0]
                chosen_indices = _form_data(annotation_idx)
                chosen_list.append({"annotation_idx": annotation_idx,
                                    "answer_idx": chosen_indices})
            assert len(new_output)-start > 0
            new_metadata.append((start, len(new_output)))

        if self.is_training:
            new_output = self.tokenizer.batch_encode_plus(new_output, max_length=32, pad_to_max_length=True)
            new_decoder_input_ids = new_output["input_ids"]
            new_decoder_attention_mask = new_output["attention_mask"]
        else:
            new_decoder_input_ids, new_decoder_attention_mask = None, None

        self.tokenized_data = [new_input_ids, new_attention_mask,
                               new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
        if not self.is_training:
            self.tokenized_data.append(chosen_list)
        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)
        self.logger.info("Finish saving tokenized DPR data")


    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data[:5]
        self.dataset = MySimpleQADatasetForPair(input_ids,
                                                attention_mask,
                                                decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                                decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                                metadata=metadata,
                                                is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset


    # override
    def evaluate(self, predictions, n_paragraphs=None):
        metadata, chosen_list = self.tokenized_data[-2:]
        assert np.sum([ref_questions is not None for ref_questions in self.ref_questions])==len(metadata)
        assert len(predictions)==metadata[-1][-1] and len(chosen_list)==len(metadata)
        # first, tokenize
        data_to_tokenize = {}
        indices = []
        offset = 0
        for i, (d, ref_questions, ref_answers) in enumerate(zip(self.data,  self.ref_questions, self.ref_answers)):
            if ref_questions is None: continue
            data_to_tokenize["prompt.{}".format(i)] = [{"caption": d["question"]}]
            ann_idx = chosen_list[offset]["annotation_idx"]
            answer_idx = chosen_list[offset]["answer_idx"]
            start, end = metadata[offset]
            assert len(ref_questions[ann_idx])==len(ref_answers[ann_idx])==len(answer_idx)==end-start
            indices.append((i, len(answer_idx)))
            for j, (ref_question, pred, a_idx) in enumerate(
                    zip(ref_questions[ann_idx], predictions[start:end], answer_idx)):
                assert type(ref_question)==list
                data_to_tokenize["gen.{}.{}".format(i, j)] = [{"caption": pred if type(pred)==str else pred[0]}]
                data_to_tokenize["ref.{}.{}".format(i, j)] = [{"caption": ref} for ref in ref_question]
            offset += 1

        assert offset==len(metadata)
        all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}

        bleu, f1s = [], []
        def _get_qg_metrics(gens, refs, prompt, metrics):
            return np.mean([get_qg_metrics(gen, ref, prompt, metrics) for gen, ref in zip(gens, refs)])

        for (i, n) in indices:
            curr_bleu, curr_f1s = [], []
            for j in range(n):
                e = get_qg_metrics(_get("gen.{}.{}".format(i, j)),
                                   _get("ref.{}.{}".format(i, j)),
                                   _get("prompt.{}".format(i)),
                                   metrics=["bleu4", "edit-f1"])
                curr_bleu.append(e["bleu4"])
                curr_f1s.append(e["edit-f1"])
            bleu.append(np.mean(curr_bleu))
            f1s.append(np.mean(curr_f1s))
        self.logger.info("BLEU=%.1f\tEDIT-F1=%.1f" % (100*np.mean(bleu), 100*np.mean(f1s)))
        return np.mean(f1s)


class CAmbigQGData(AmbigQGData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQGData, self).__init__(logger, args, data_path, is_training, passages)

        data_path = Path(data_path)
        with open(data_path.parent.parent / "nqopen" / f"{self.data_type}.json", "r") as f:
            orig_data = json.load(f)
            id_to_orig_idx = {d["id"]:i for i, d in enumerate(orig_data)}

        if args.ablation == "with_predicted_answers":
            with open(self.args.predicted_answers_path, "r") as f:
                predictions = json.load(f)
            predicted_answers = []
            for prediction in predictions:
                a = list(set([text.strip() for text in prediction.split(self.SEP)]))
                predicted_answers.append([[i] for i in a])
                # for each question, answers are given as a list of list
                # For each semantically distinct answer set, there exists one answer.
            err_msg = f"data len is {len(self.data)} but predicted answers len is {len(predicted_answers)}"
            assert len(self.data) == len(predicted_answers), err_msg

        self.ref_questions = []
        self.ref_answers = []
        self.ref_cq = []
        # we will only consider questions with multiple answers
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                questions.append([[q.strip() for q in pair["question"].split("|")] for pair in annotation["qaPairs"]])
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                   all([type(answer)==list for answer in answers]) and \
                   all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])

            if args.ablation == "with_predicted_answers":
                answers = [predicted_answers[i]]
                questions = [[questions[0][0]] * len(answers[0])]  # dummy

            self.ref_questions.append(questions)
            self.ref_answers.append(answers)
            self.ref_cq.append(d["clarification_question"])
            self.data[i]["orig_idx"] = id_to_orig_idx[d["id"]]

        self.SEP = "<SEP>"
        self.qg_tokenizer = PTBTokenizer()
        self.metric = "EDIT-F1"
        if not self.is_training:
            self.qg_tokenizer = PTBTokenizer()

    # override
    def load_dpr_data(self):
        task = "cqg" if len(self.args.ablation) == 0 else f"cqg_{self.args.ablation}"
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix

        data_type_for_dpr_retrieval_path = self.data_type+"_2020" if self.args.wiki_2020 else self.data_type
        dpr_retrieval_path = Path(self.args.psg_sel_dir) / f"ambigqa_{data_type_for_dpr_retrieval_path}.json"

        data_type_dpr_tokenized_path = self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type
        data_name = f"ambigqa_predictions_{postfix}_{task}"
        dpr_tokenized_path = Path(self.args.output_dir) / f"{data_type_dpr_tokenized_path}_{data_name}.json"
        # dpr_tokenized_path: model inputs and targets.

        # Here, prepare model inputs and targets by concatenation (question(+multiple answers) + passages)
        self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)

        # in attention_mask, 1 means answer + passages, 2 means prompt, 3 means other answers
        do_include_prompt=True
        do_include_others=True
        new_input_ids, new_attention_mask = [], []

        # Update attention masks and trim the inputs up to max_token_nums.
        max_token_nums = self.args.max_token_nums
        for input_ids, attention_mask in zip(self.tokenized_data[0], self.tokenized_data[1]):
            _input_ids = [_id for _id, mask in zip(input_ids, attention_mask)
                          if mask==1 or (do_include_prompt and mask==2) or (do_include_others and mask==3)]
            _attention_mask = [1 for mask in attention_mask
                               if mask==1 or (do_include_prompt and mask==2) or (do_include_others and mask==3)]
            assert len(_input_ids)==len(_attention_mask)
            while len(_input_ids) < max_token_nums:
                _input_ids.append(self.tokenizer.pad_token_id)
                _attention_mask.append(0)
            new_input_ids.append(_input_ids[:max_token_nums])
            new_attention_mask.append(_attention_mask[:max_token_nums])
        self.tokenized_data[0] = new_input_ids
        self.tokenized_data[1] = new_attention_mask


    # override
    # CONCAT: questions, answers + passages ==> resulting in model inputs and targets (=decoder inputs).
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        # If not saved, load passages and concat them with questions, resulting in input
        # dpr_tokenized_path: model inputs and targets.
        if self.is_training and self.args.consider_order_for_multiple_answers:
            dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_ordered.json")
        self.logger.info(dpr_tokenized_path)

        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading passage-augmented ready-to-use data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
            return

        self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
        self.logger.info("Loading passage selection from DPR reader: {}".format(dpr_retrieval_path))
        with open(dpr_retrieval_path) as f:  # For each question, which passages we need to read. = reranking results...
            dpr_passages = json.load(f)

        if self.passages.tokenized_data is None:
            self.passages.load_tokenized_data("bart", all=True)  # 2200M passages.. TAKES LONG. / use subset options?

        assert len(dpr_passages)==len(self)

        # NOTE: two more variables for clarification questions.
        input_ids, attention_mask, cq_input_ids, cq_attention_mask, answer_input_ids, answer_attention_mask, metadata = self.tokenized_data
        assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        assert type(bos_token_id)==type(eos_token_id)==type(sep_token_id)==int

        def _get_tokenized_answer(idx):
            tokens = answer_input_ids[idx]
            if 0 in answer_attention_mask[idx]:
                tokens = tokens[:answer_attention_mask[idx].index(0)]
            assert tokens[0]==tokens[1]==bos_token_id and tokens[-1]==eos_token_id
            return tokens[2:-1]

        def _included(tokens, curr_input_ids):
            for i in range(len(curr_input_ids)+1):
                if curr_input_ids[i:i+len(tokens)]==tokens:
                    return True
            return False

        new_input_ids, new_attention_mask = [], []
        new_output, new_metadata = [], []
        chosen_list = []

        # Concatenation...
        for idx, (curr_input_ids, curr_attention_mask, dpr_ids) in tqdm(enumerate(
                zip(input_ids, attention_mask, dpr_passages))):
            if self.ref_questions[idx] is None:
                continue

            # Extract question tokens
            end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
            q_input_ids = curr_input_ids[:end_of_question]

            # Aggregate passages
            p_input_ids, p_attention_mask = [], []
            dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
            dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
            offset = 0
            # Concat passages up to 1024 tokens.
            while len(p_input_ids)<1024:
                assert dpr_input_ids[offset][0] == bos_token_id
                assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                p_input_ids += dpr_input_ids[offset][1:]
                p_attention_mask += dpr_attention_mask[offset][1:]
                offset += 1

            # Concat answers that are valid (=must belong to the concated passages)
            tokenized_ref_answers_list, is_valid_list, n_missing_list = [], [], []
            for ref_answers, ref_metadata in zip(self.ref_answers[idx], metadata[idx]):  # For each turker (=annotator)...
                # ref_metadata: [(0, 1), (1, 4), (4, 8)]
                # Each range indicates answer_input_ids' indices that are semantically same for a dq.
                # For DQ0, #answers=1 / For DQ1, #answers=3 / For DQ2, #answers=4
                # Specifically, answer_input_ids[1], answer_input_ids[2], and answer_input_ids[3] are semantically same
                # answers for the DQ1 of the idx-th AQ annotated by a turker.
                assert type(ref_metadata[0][0])==int
                assert [len(ref_answer)==end-start for ref_answer, (start, end)
                        in zip(ref_answers, ref_metadata)]
                tokenized_ref_answers = [[_get_tokenized_answer(i) for i in range(*m)] for m in ref_metadata]
                is_valid = [[_included(tokens, p_input_ids) for tokens in _tokens] for _tokens in tokenized_ref_answers]
                # is_valid = [[True], [True, True, False], [False, False, False, False]]
                # the last answer set is disjoint with passages. In this case, n_missing=1
                n_missing = np.sum([not any(v) for v in is_valid])
                tokenized_ref_answers_list.append(tokenized_ref_answers)
                is_valid_list.append(is_valid)
                n_missing_list.append(n_missing)

            ## Originally, select DQ-A pairs by annotators who give answers that are minimally missing on passages.
            #min_n_missing = np.min(n_missing_list)
            #annotation_indices = [ann_idx for ann_idx in range(len(n_missing_list))
            #                      if n_missing_list[ann_idx]==min_n_missing]

            # We select DQ-A pairs by annotators who give answers as many as possible.
            annotation_idx = max([(ann_idx, questions) for ann_idx, questions in enumerate(self.ref_questions[idx])],
                                 key=lambda x: len(x[1]))[0]

            def _form_data(annotation_idx):
                # in attention_mask, 1 means answer + passages, 2 means prompt
                # prompt(=ambiguous question) + answers + passages
                # 222222222222222222222222222 + 1111111 + 11111111
                ref_answers = self.ref_answers[idx][annotation_idx]
                tokenized_ref_answers = tokenized_ref_answers_list[annotation_idx]
                assert len(ref_answers)==len(tokenized_ref_answers)==len(is_valid_list[annotation_idx])

                # Among answers for the each disambiguated question, we only use the first valid answer
                #  (if not exists, just the first one).
                final_ref_answers = []
                chosen_indices = []
                for (ref_answer, tok_ref_answer, is_valid) in \
                        zip(ref_answers, tokenized_ref_answers, is_valid_list[annotation_idx]):
                    assert len(ref_answer)==len(tok_ref_answer)==len(is_valid)
                    chosen_idx = is_valid.index(True) if True in is_valid else 0
                    chosen_indices.append(chosen_idx)
                    final_ref_answers.append(tok_ref_answer[chosen_idx])

                # NOW CONCATENATION
                # question
                input_ids = [bos_token_id, bos_token_id] + q_input_ids
                attention_mask = [2 for _ in input_ids]
                # answers
                if self.args.ablation != "without_answers":
                    for j, answer in enumerate(final_ref_answers):
                        input_ids += [sep_token_id] + answer
                        attention_mask += [1 for _ in range(len(answer)+1)]
                input_ids += [self.tokenizer.eos_token_id]
                attention_mask += [1]
                # passages
                input_ids += p_input_ids
                attention_mask += p_attention_mask
                assert len(input_ids)==len(attention_mask)

                new_input_ids.append(input_ids)
                new_attention_mask.append(attention_mask)
                new_output.append(cq_input_ids[idx])  # Set CQ as the target.
                return chosen_indices

            start = len(new_output)
            if self.is_training:
                start = len(new_output)
                _form_data(annotation_idx)
            else:
                chosen_indices = _form_data(annotation_idx)
                chosen_list.append({"annotation_idx": annotation_idx,
                                    "answer_idx": chosen_indices})
            assert len(new_output)-start > 0
            new_metadata.append((start, len(new_output)))

        if self.is_training:
            new_output = self.tokenizer.batch_encode_plus(new_output, max_length=128, pad_to_max_length=True)
            new_decoder_input_ids = new_output["input_ids"]
            new_decoder_attention_mask = new_output["attention_mask"]
        else:
            new_decoder_input_ids, new_decoder_attention_mask = None, None
        self.tokenized_data = [new_input_ids, new_attention_mask,
                               new_decoder_input_ids, new_decoder_attention_mask, new_metadata]
        if not self.is_training:
            self.tokenized_data.append(chosen_list)
        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)
        self.logger.info("Finish saving tokenized DPR data")

    def load_tokenized_data(self, tokenizer):
        # Read tokenized questions and answers. If not exists, save new ones.
        self.tokenizer = tokenizer
        if self.args.ablation != "with_predicted_answers":
            postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        else:
            postfix = tokenizer.__class__.__name__.replace("zer", "zed") + "_with_predicted_answers"
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix
        assert ("Bart" in postfix and self.args.append_another_bos) \
            or ("Bart" not in postfix and not self.args.append_another_bos)
        data_path = Path(self.data_path)

        data_format = ".tsv" if data_path.name.endswith(".tsv") else ".json"
        properties = ("-uncased" if self.args.do_lowercase else "",
                      "-xbos" if self.args.append_another_bos else "",
                      postfix)

        # questions and multiple answers only. (NOT PASSAGES)
        preprocessed_name = data_path.name.replace(data_format, "%s%s-%s.json" % properties)
        preprocessed_path = data_path.parent / preprocessed_name
        # preprocessed_path stores:
        #   tokenized questions, e.g., ambiguous questions or clarification questions,
        #   and answers
        # THESE ARE NOT MODEL INPUTS. NEED TO BE CONCATED WITH (tokenized) PASSAGAGES, which we do in load_dpr_data().

        if self.load and preprocessed_path.exists():
            self.logger.info("Loading pre-tokenized data (without passages) from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print("Start tokenizing...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                         for d in self.data]
            clarification_questions = [d.get("clarification_question", "NO NEED") for d in self.data]

            if self.args.ablation != "with_predicted_answers":
                answers = [d["answer"] for d in self.data]
            else:
                answers = [d["predicted_answer"] for d in self.data]

            answers, metadata = self.flatten(answers)
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=32)
            clarification_question_input = tokenizer.batch_encode_plus(clarification_questions,
                                                                       pad_to_max_length=True,
                                                                       max_length=128)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            cq_input_ids, cq_attention_mask = clarification_question_input["input_ids"], \
                                              clarification_question_input["attention_mask"]
            answer_input_ids, answer_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask, cq_input_ids, cq_attention_mask,
                              answer_input_ids, answer_attention_mask, metadata]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f)
        self.tokenized_data = tokenized_data  # THIS IS NOT OVER. in self.load_dpr_data(), IT WILL BE UPDATED.
        if not self.args.dpr:
            # Here, not only load passages (+ tokenize if not exist), but also concat inputs and passages.
            self.load_dpr_data()

    def evaluate(self, predictions, n_paragraphs=None, metrics=["EM", "BLEU4", "N-EDIT"], return_dict=False,
                 test_ids=TEST_IDS, dev=True):
        metadata, chosen_list = self.tokenized_data[-2:]
        assert np.sum([ref_questions is not None for ref_questions in self.ref_questions]) == len(metadata)
        assert len(predictions) == metadata[-1][-1] and len(chosen_list) == len(metadata)

        # first, tokenize
        data_to_tokenize = {}
        indices, ref_opt_lens, gen_opt_lens, gen_num_options = [], [], [], []
        offset = 0
        for i, (d, ref_questions, ref_answers) in enumerate(
                zip(self.data, self.ref_questions, self.ref_answers)):
            if ref_questions is None: continue
            ref_cq = self.ref_cq[offset]
            gen_cq = predictions[offset]
            ref_ap, ref_options = parse_clarification_question(ref_cq)
            gen_ap, gen_options = parse_clarification_question(gen_cq)

            gen_num_options.append(len(gen_options))
            data_to_tokenize[f"ref_ap.{i}"] = [{"caption": ref_ap}]
            data_to_tokenize[f"gen_ap.{i}"] = [{"caption": gen_ap}]

            for j, ref_opt in enumerate(ref_options):
                data_to_tokenize[f"ref_options.{i}.{j}"] = [{"caption": ref_opt}]
            ref_opt_lens.append(len(ref_options))

            for j, gen_opt in enumerate(gen_options):
                data_to_tokenize[f"gen_options.{i}.{j}"] = [{"caption": gen_opt}]
            gen_opt_lens.append(len(gen_options))

            indices.append(i)
            offset += 1
        all_tokens = self.qg_tokenizer.tokenize(data_to_tokenize)

        def _get(key):
            return {'sent': [normalize_answer(value) for value in all_tokens[key]]}


        res = defaultdict(list, [])

        for idx, (ref_opt_len, gen_opt_len, i) in enumerate(zip(ref_opt_lens, gen_opt_lens, indices)):
            if not dev and idx not in test_ids:
                continue
            if dev and idx in test_ids:
                continue
            e = get_ap_metrics(_get(f"ref_ap.{i}"),
                               _get(f"gen_ap.{i}"),
                               metrics=metrics)
            for metric in metrics:
                res[f"ap-{metric}"].append(e[metric])
            e = get_options_metrics([_get(f"ref_options.{i}.{j}") for j in range(ref_opt_len)],
                                    [_get(f"gen_options.{i}.{j}") for j in range(gen_opt_len)],
                                    metrics=metrics)
            for metric, val in e.items():
                res[f"option-{metric}"].append(val)
            res["#options"] = gen_num_options

        res_log = "\n"
        for metric, v in res.items():
            res_log += f"{metric}: {np.mean(v)*100:.2f}%\n"
        self.logger.info(res_log)
        if return_dict:
            res_ = {}
            for k, v in res.items():
                res_[k] = np.mean(v)*100
            return res_
        return 0

    def save_predictions(self, predictions):
        checkpoint = Path(self.args.checkpoint)
        ckpt_name = checkpoint.name.split(".")[0]  # drop .th
        psg_data = "_20200201" if self.args.wiki_2020 and not self.args.ambigqa else ""

        pred_file_name = f"{self.data_type}{psg_data}_predictions_by_{ckpt_name}.json"
        save_path = Path(self.args.output_dir) / pred_file_name
        if self.args.ablation != "":
            save_path = save_path.parent / save_path.name.replace("predictions", f"predictions_{self.args.ablation}")
        if self.args.save_psg_sel_only:
            save_path = save_path.parent / save_path.name.replace("predictions.json", "psg_sel.json")
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))


def get_bleu(question, generated):
    return dict(zip([f'BLEU{i}' for i in range(1, 5)], Bleu(4).compute_score(question, generated)[0]))


def get_sim_fn(metric):
    if metric == "EM":
        return lambda x, y: x['sent'][0] == y['sent'][0]
    elif metric == "ROUGE-L":
        return lambda x, y: Rouge().get_scores(x['sent'], y['sent'])[0]['rouge-l']['f']
    elif metric == "N-EDIT":
        def _get_edits(tokens1, tokens2):
            allCommon = []
            while True:
                commons = list(set(tokens1) & set(tokens2))
                if len(commons) == 0:
                    break
                allCommon += commons
                for c in commons:
                    ind1, ind2 = tokens1.index(c), tokens2.index(c)
                    tokens1 = tokens1[:ind1] + tokens1[ind1 + 1:]
                    tokens2 = tokens2[:ind2] + tokens2[ind2 + 1:]
            deleted = ["[DELETED]" + token for token in tokens1]
            added = ["[ADDED]" + token for token in tokens2]
            common = ["[FIXED]" + token for token in allCommon]
            return deleted + added  # +common

        def fn(x, y):
            x_, y_ = x['sent'][0].split(" "), y['sent'][0].split(" ")
            return 1 - len(_get_edits(x_, y_)) / (len(x_) + len(y_))

        return fn
    elif "BLEU" in metric:
        return lambda x, y: get_bleu(x, y)[metric]


def get_ap_metrics(ref, gen, metrics):
    res = {}
    for metric in metrics:
        sim_fn = get_sim_fn(metric)
        res[metric] = sim_fn(ref, gen)
    return res


def get_options_metrics(refs, gens, metrics):
    res = {}
    for metric in metrics:
        sim_fn = get_sim_fn(metric)

        I_s = {}
        for m, gen in enumerate(gens):
            if len(gen['sent'][0]) == 0:
                gen['sent'] = ["$"]
            I_s[m] = max([(n, sim_fn(gen, ref)) for n, ref in enumerate(refs)], key=lambda x: x[1])
        numerator = sum([v[1] for k, v in I_s.items()])
        prec = numerator / len(gens)
        rec = numerator / len(refs)
        f1 = 2 * prec * rec / (prec + rec + 0.00000001)
        res[f"{metric}-precision"] = prec
        res[f"{metric}-recall"] = rec
        res[f"{metric}-F1"] = f1
    return res

