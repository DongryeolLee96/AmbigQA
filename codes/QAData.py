import os
import json
import gzip
import re
import pickle as pkl
import string
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from DataLoader import MySimpleQADataset, MyQADataset, MyDataLoader
from util import decode_span_batch

# for evaluation
from ambigqa_evaluate_script import normalize_answer, get_exact_match, get_f1, get_qg_metrics
from pycocoevalcap.bleu.bleu import Bleu

from transformers import AutoTokenizer
test_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

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


class QAData(object):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        self.data_path = data_path
        self.passages = passages
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            self.data_type = "train" if is_training else "train_for_inference"
        else:
            raise NotImplementedError()

        with open(self.data_path, "r") as f:
            self.data = json.load(f)

        if "data" in self.data:
            self.data = self.data["data"]

        if "answers" in self.data[0]:
            self.data = [{"id": d["id"], "question": d["question"], "answer": d["answers"]} for d in self.data]

        if args.debug:
            self.data = self.data[:40]
        assert type(self.data)==list

        if not args.ambigqa:
            id2answer_path = os.path.join("/".join(self.data_path.split("/")[:-1]),
                                          "{}_id2answers.json".format(self.data_type.replace("train_for_inference", "train")))
            with open(id2answer_path, "r") as f:
                id2answers = json.load(f)
            for i, d in enumerate(self.data):
                if is_training:
                    self.data[i]["answer"] += id2answers[d["id"]]
                else:
                    self.data[i]["answer"] = id2answers[d["id"]]

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.metric = "EM"
        self.tokenizer = None
        self.tokenized_data = None
        self.dpr_tokenized_data = None
        self.dataset = None
        self.dataloader = None

    def __len__(self):
        return len(self.data)

    def get_answers(self):
        return [d["answer"] for d in self.data]

    def decode(self, tokens):
        if type(tokens[0])==list:
            return [self.decode(_tokens) for _tokens in tokens]
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip().replace(" - ", "-").replace(" : ", ":")

    def decode_span(self, outputs, n_paragraphs):
        assert len(self.data)==len(self.tokenized_data["positive_input_ids"])==\
                len(self.tokenized_data["positive_input_mask"])==len(outputs), \
                (len(self.data), len(self.tokenized_data["positive_input_ids"]),
                len(self.tokenized_data["positive_input_mask"]), len(outputs))
        return decode_span_batch(list(zip(self.tokenized_data["positive_input_ids"],
                                          self.tokenized_data["positive_input_mask"])),
                                 outputs,
                                 tokenizer=self.tokenizer,
                                 max_answer_length=self.args.max_answer_length,
                                 n_paragraphs=n_paragraphs,
                                 topk_answer=self.args.topk_answer,
                                 verbose=self.args.verbose,
                                 n_jobs=self.args.n_jobs,
                                 save_psg_sel_only=self.args.save_psg_sel_only)

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix
        assert ("Bart" in postfix and self.args.append_another_bos) \
            or ("Bart" not in postfix and not self.args.append_another_bos)
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if self.args.append_another_bos else "",
                    postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                tokenized_data = json.load(f)
        else:
            print ("Start tokenizing...")
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
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
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length="Bart" in postfix,
                                                       max_length=20)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            tokenized_data = [input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask, metadata]
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump(tokenized_data, f)
        self.tokenized_data = tokenized_data
        if not self.args.dpr:
            self.load_dpr_data()

    def load_dpr_data(self):
        data_type = self.data_type.replace("train_for_inference", "train")
        dpr_retrieval_path = "out/dpr/{}_predictions.json".format(
            data_type+"_20200201" if self.args.wiki_2020 else data_type)
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        dpr_tokenized_path = "out/dpr/{}_predictions_{}.json".format(
            self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type, postfix)
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        elif "Bert" in postfix or "Albert" in postfix:
            return self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)
        else:
            raise NotImplementedError()

    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        self.logger.info("{}\n{}".format(dpr_retrieval_path, dpr_tokenized_path))
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                input_ids, attention_mask = json.load(f)

        else:
            with open(dpr_retrieval_path, "r") as f:
                dpr_passages = json.load(f)
                assert len(dpr_passages)==len(self)
            assert self.args.psg_sel_dir is not None
            data_type = self.data_type.replace("train", "train_for_inference") \
                if "for_inference" not in self.data_type else self.data_type
            psg_sel_fn = os.path.join(self.args.psg_sel_dir,
                                      "{}{}_psg_sel.json".format(
                                          data_type,
                                          "_20200201" if self.args.wiki_2020 else ""))
            self.logger.info("Loading passage selection from DPR reader: {}".format(psg_sel_fn))
            with open(psg_sel_fn, "r") as f:
                fg_passages = json.load(f)
                assert len(fg_passages)==len(dpr_passages)
                dpr_passages = [[psgs[i] for i in fg_psgs] for psgs, fg_psgs in zip(dpr_passages, fg_passages)]

            self.logger.info("Start processing DPR data")
            if self.passages.tokenized_data is None:
                subset = set([p_idx for retrieved in dpr_passages for p_idx in retrieved])
                self.passages.load_tokenized_data("bart", subset=subset, all=True)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            assert len(dpr_passages)==len(input_ids)==len(attention_mask)
            bos_token_id = self.tokenizer.bos_token_id

            def _get_tokenized_answer(idx):
                tokens = decoder_input_ids[idx]
                if 0 in decoder_attention_mask[idx]:
                    tokens = tokens[:decoder_attention_mask[idx].index(0)]
                assert tokens[0]==tokens[1]==bos_token_id and tokens[-1]==self.tokenizer.eos_token_id
                return tokens[2:-1]

            for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                    input_ids, attention_mask, metadata, dpr_passages)):
                dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
                dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
                offset = 0
                end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
                input_ids[idx] = curr_input_ids[:end_of_question]
                attention_mask[idx] = curr_attention_mask[:end_of_question]

                while len(input_ids[idx])<1024:
                    assert dpr_input_ids[offset][0] == bos_token_id
                    assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                    assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                    input_ids[idx] += dpr_input_ids[offset][1:]
                    attention_mask[idx] += dpr_attention_mask[offset][1:]
                    offset += 1
                assert len(input_ids)==len(attention_mask)
                input_ids[idx] = input_ids[idx][:1024]
                attention_mask[idx] = attention_mask[idx][:1024]

            with open(dpr_tokenized_path, "w") as f:
                json.dump([input_ids, attention_mask], f)
            self.logger.info("Finish saving tokenized DPR data at {}".format(dpr_tokenized_path))

        self.tokenized_data[0] = input_ids
        self.tokenized_data[1] = attention_mask

        if self.is_training and self.args.discard_not_found_answers:
            self.discard_not_found_answers()

    def discard_not_found_answers(self):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
        new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], [], [], []

        skipped_idxs = []

        self.logger.info("Discarding training examples where retrieval fails...")

        def _get_tokenized_answer(idx):
            tokens = self.tokenized_data[2][idx]
            if 0 in self.tokenized_data[3][idx]:
                tokens = tokens[:self.tokenized_data[3][idx].index(0)]
            assert tokens[0]==tokens[1]==self.tokenizer.bos_token_id and tokens[-1]==self.tokenizer.eos_token_id
            return tokens[2:-1]

        for idx, (curr_input_ids, curr_attention_mask, curr_metadata) in enumerate(zip(
                input_ids, attention_mask, metadata)):
            end_of_question = curr_input_ids.index(self.tokenizer.eos_token_id)+1
            def _included(tokens):
                for i in range(end_of_question, 1024-len(tokens)+1):
                    if curr_input_ids[i:i+len(tokens)]==tokens:
                        return True
                return False

            valid_answer_idxs = [answer_idx for answer_idx in range(curr_metadata[0], curr_metadata[1])
                                    if _included(_get_tokenized_answer(answer_idx))]
            if len(valid_answer_idxs)==0:
                skipped_idxs.append(idx)
                continue
            new_input_ids.append(curr_input_ids)
            new_attention_mask.append(curr_attention_mask)
            new_decoder_input_ids += [decoder_input_ids[i] for i in valid_answer_idxs]
            new_decoder_attention_mask += [decoder_attention_mask[i] for i in valid_answer_idxs]
            new_metadata.append([len(new_decoder_input_ids)-len(valid_answer_idxs), len(new_decoder_input_ids)])

        self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids, new_decoder_attention_mask, new_metadata]

        print (len(input_ids), len(new_input_ids), len(skipped_idxs))

    def load_dpr_data_bert(self, dpr_retrieval_path, dpr_tokenized_path):
        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
            return
        self.logger.info("Start processing DPR data")
        with open(dpr_retrieval_path, "r") as f:
            dpr_passages = json.load(f)

        if self.args.ambigqa:
            # added to convert original DPR data to AmbigQA DPR data
            dpr_passages = [dpr_passages[d["orig_idx"]] for d in self.data]
        elif self.is_training:
            with open(os.path.join(self.args.dpr_data_dir, "data/gold_passages_info/nq_train.json"), "r") as f:
                gold_titles = [d["title"] for d in json.load(f)["data"]]
                assert len(gold_titles)==len(self)

        input_ids, attention_mask, answer_input_ids, _, metadata = self.tokenized_data
        eos_token_id = self.tokenizer.eos_token_id if "Albert" in dpr_tokenized_path else self.tokenizer.sep_token_id
        assert eos_token_id is not None
        assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        if self.passages.tokenized_data is None:
            subset = set([p_idx for retrieved in dpr_passages for p_idx in retrieved])
            self.passages.load_tokenized_data("albert" if "Albert" in dpr_tokenized_path else "bert",
                                              subset=subset, all=True)
        features = defaultdict(list)
        max_n_answers = self.args.max_n_answers
        oracle_exact_matches = []
        flatten_exact_matches = []
        positive_contains_gold_title = []
        for i, (q_input_ids, q_attention_mask, retrieved) in \
                tqdm(enumerate(zip(input_ids, attention_mask, dpr_passages))):
            # Inside one ambiguous question....
            assert len(q_input_ids)==len(q_attention_mask)==32
            q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
            assert 3<=len(q_input_ids)<=32
            p_input_ids = [self.passages.tokenized_data["input_ids"][p_idx] for p_idx in retrieved]
            p_attention_mask = [self.passages.tokenized_data["attention_mask"][p_idx] for p_idx in retrieved]
            a_input_ids = [answer_input_ids[idx][1:-1] for idx in range(metadata[i][0], metadata[i][1])] # semantically same

            detected_spans = []
            for _p_input_ids in p_input_ids:
                detected_spans.append([])
                for _a_input_ids in a_input_ids:
                    decoded_a_input_ids = self.decode(_a_input_ids)
                    for j in range(len(_p_input_ids)-len(_a_input_ids)+1):
                        if _p_input_ids[j:j+len(_a_input_ids)]==_a_input_ids:
                            detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+len(_a_input_ids)-1))
                        elif "Albert" in dpr_tokenized_path and \
                                _p_input_ids[j]==_a_input_ids[0] and \
                                13 in _p_input_ids[j:j+len(_a_input_ids)]:
                            k = j + len(_a_input_ids)+1
                            while k<len(_p_input_ids) and np.sum([_p_input_ids[z]!=13 for z in range(j, k)])<len(_a_input_ids):
                                k += 1
                            if decoded_a_input_ids==self.decode(_p_input_ids[j:k]):
                                detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+k-1))
            if self.args.ambigqa and self.is_training:
                positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0][:20]
                negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
                if len(positives)==0:
                    continue
            elif self.is_training:
                gold_title = normalize_answer(gold_titles[i])
                _positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0]
                if len(_positives)==0:
                    continue
                positives = [j for j in _positives if normalize_answer(self.decode(p_input_ids[j][:p_input_ids[j].index(eos_token_id)]))==gold_title]
                positive_contains_gold_title.append(len(positives)>0)
                if len(positives)==0:
                    positives = _positives[:20]
                negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
            else:
                positives = [j for j in range(len(detected_spans))]
                negatives = []
            for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
                        "positive_start_positions", "positive_end_positions", "positive_answer_mask",
                        "negative_input_ids", "negative_input_mask", "negative_token_type_ids"]:
                features[key].append([])

            def _form_input(p_input_ids, p_attention_mask):
                assert len(p_input_ids)==len(p_attention_mask)
                assert len(p_input_ids)==128 or (len(p_input_ids)<=128 and np.sum(p_attention_mask)==len(p_attention_mask))
                if len(p_input_ids)<128:
                    p_input_ids += [self.tokenizer.pad_token_id for _ in range(128-len(p_input_ids))]
                    p_attention_mask += [0 for _ in range(128-len(p_attention_mask))]
                input_ids = q_input_ids + p_input_ids + [self.tokenizer.pad_token_id for _ in range(32-len(q_input_ids))]
                attention_mask = [1 for _ in range(len(q_input_ids))]  + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                token_type_ids = [0 for _ in range(len(q_input_ids))] + p_attention_mask + [0 for _ in range(32-len(q_input_ids))]
                return input_ids, attention_mask, token_type_ids

            for idx in positives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["positive_input_ids"][-1].append(input_ids)
                features["positive_input_mask"][-1].append(attention_mask)
                features["positive_token_type_ids"][-1].append(token_type_ids)
                detected_span = detected_spans[idx]
                features["positive_start_positions"][-1].append(
                    [s[0] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_end_positions"][-1].append(
                    [s[1] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
                features["positive_answer_mask"][-1].append(
                    [1 for _ in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
            for idx in negatives:
                input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
                features["negative_input_ids"][-1].append(input_ids)
                features["negative_input_mask"][-1].append(attention_mask)
                features["negative_token_type_ids"][-1].append(token_type_ids)
            # for debugging
            for p_input_ids, starts, ends, masks in zip(features["positive_input_ids"][-1],
                                                    features["positive_start_positions"][-1],
                                                    features["positive_end_positions"][-1],
                                                    features["positive_answer_mask"][-1]):
                if np.sum(masks)==0: continue
                assert len(starts)==len(ends)==len(masks)==max_n_answers
                decoded_answers = [self.tokenizer.decode(p_input_ids[start:end+1]) for start, end, mask in zip(starts, ends, masks) if mask]
                ems = [get_exact_match(decoded_answer, self.data[i]["answer"]) for decoded_answer in decoded_answers]
                oracle_exact_matches.append(np.max(ems))
                flatten_exact_matches += ems
        print ("oracle exact matches", np.mean(oracle_exact_matches))
        print ("flatten exact matches", np.mean(flatten_exact_matches))
        print ("positive contains gold title", np.mean(positive_contains_gold_title))
        print (len(positive_contains_gold_title))
        self.tokenized_data = features

        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if isinstance(self.tokenized_data, dict):
            self.dataset = MyQADataset(self.tokenized_data,
                                       is_training=self.is_training,
                                       train_M=self.args.train_M,
                                       test_M=self.args.test_M)
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data
            self.dataset = MySimpleQADataset(input_ids,
                                             attention_mask,
                                             decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                             decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                             in_metadata=None,
                                             out_metadata=metadata,
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
        if self.args.save_psg_sel_only:
            return [-1]
        if n_paragraphs is None:
            ems = []
            for (prediction, dp) in zip(predictions, self.data):
                if type(prediction)==list:
                    prediction = prediction[0]
                if type(prediction)==dict:
                    prediction = prediction["text"]
                ems.append(get_exact_match(prediction, dp["answer"]))
            return ems
        ems = defaultdict(list)
        for (prediction, dp) in zip(predictions, self.data):
            assert len(n_paragraphs)==len(prediction)
            for pred, n in zip(prediction, n_paragraphs):
                if type(pred)==list:
                    pred = pred[0]
                if type(pred)==dict:
                    pred = pred["text"]
                ems[n].append(get_exact_match(pred, dp["answer"]))
        for n in n_paragraphs:
            self.logger.info("n_paragraphs=%d\t#M=%.2f" % (n, np.mean(ems[n])*100))
        return ems[n_paragraphs[-1]]

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        save_path = os.path.join(self.args.output_dir, "{}{}_predictions.json".format(
            self.data_type if self.args.prefix is None else self.args.prefix,
            "_20200201" if self.args.wiki_2020 and not self.args.ambigqa else ""))
        if self.args.save_psg_sel_only:
            save_path = save_path.replace("predictions.json", "psg_sel.json")
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))


class AmbigQAData(QAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(AmbigQAData, self).__init__(logger, args, data_path, is_training, passages)

        data_path = Path(data_path)
        with open(data_path.parent.parent / "nqopen" / f"{self.data_type}.json", "r") as f:
            orig_data = json.load(f)
            id_to_orig_idx = {d["id"]:i for i, d in enumerate(orig_data)}

        for i, d in enumerate(self.data):
            answers = []
            for annotation in d["annotations"]:
                assert annotation["type"] in ["singleAnswer", "multipleQAs"]
                if annotation["type"]=="singleAnswer":
                    answers.append([list(set(annotation["answer"]))])
                else:
                    answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
            assert type(answers)==list and \
                all([type(answer)==list for answer in answers]) and \
                all([type(_a)==str for answer in answers for _answer in answer for _a in _answer])
            self.data[i]["answer"] = answers
            self.data[i]["orig_idx"] = id_to_orig_idx[d["id"]]

        self.metric = "F1"
        self.SEP = "<SEP>"

    # override
    def flatten(self, answers):
        new_answers, metadata = [], []
        for _answers in answers:
            assert type(_answers)==list
            metadata.append([])
            for answer in _answers:
                metadata[-1].append([])
                for _answer in answer:
                    assert len(_answer)>0, _answers
                    assert type(_answer)==list and type(_answer[0])==str, _answers
                    metadata[-1][-1].append((len(new_answers), len(new_answers)+len(_answer)))
                    new_answers += _answer
        return new_answers, metadata

    # override
    def load_dpr_data(self):
        task = "qa" if len(self.args.ablation) == 0 else f"qa_{self.args.ablation}"
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")

        data_type_for_dpr_retrieval_path = self.data_type+"_2020" if self.args.wiki_2020 else self.data_type
        dpr_retrieval_path = Path(self.args.psg_sel_dir) / f"ambigqa_{data_type_for_dpr_retrieval_path}.json"

        data_type_dpr_tokenized_path = self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type
        data_name = f"ambigqa_predictions_{postfix}_{task}"
        dpr_tokenized_path = os.path.join(self.args.output_dir, f"{data_type_dpr_tokenized_path}_{data_name}.json")
        # dpr_tokenized_path: model inputs and targets.


        ################################
        if "Bart" in postfix:
            return self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)

        metadata, new_metadata = self.tokenized_data[-1], []
        for curr_metadata in metadata:
            new_metadata.append((curr_metadata[0][0][0], curr_metadata[-1][-1][-1]))
        self.tokenized_data[-1] = new_metadata
        return self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)

    # override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if self.is_training and self.args.consider_order_for_multiple_answers:
            dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_ordered.json")

        self.logger.info(dpr_tokenized_path)

        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
            return

        self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
        self.logger.info("Loading passage selection from DPR reader: {}".format(dpr_retrieval_path))
        with open(dpr_retrieval_path) as f:  # For each question, which passages we need to read. = reranking results...
            dpr_passages = json.load(f)

        if self.passages.tokenized_data is None:
            self.passages.load_tokenized_data("bart", all=True)  # 2200M passages.. TAKES LONG.

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

        new_input_ids, new_attention_mask = [], []
        if self.is_training:
            new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], []
        else:
            new_decoder_input_ids, new_decoder_attention_mask, new_metadata = None, None, None
        for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                input_ids, attention_mask, metadata, dpr_passages)):

            dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
            # print(dpr_ids)
            # print(self.passages.tokenized_data["input_ids"][0])
            # print(len(self.passages.tokenized_data["input_ids"]))
            
            dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]

            # creating input_ids is done in the same way as NQ-open.
            offset = 0
            end_of_question = curr_input_ids.index(eos_token_id)+1
            input_ids[idx] = curr_input_ids[:end_of_question]
            attention_mask[idx] = curr_attention_mask[:end_of_question]
            while len(input_ids[idx])<1024:
                assert dpr_input_ids[offset][0] == bos_token_id
                assert len(dpr_input_ids[offset])==len(dpr_attention_mask[offset])
                assert np.sum(dpr_attention_mask[offset])==len(dpr_attention_mask[offset])
                input_ids[idx] += dpr_input_ids[offset][1:]
                attention_mask[idx] += dpr_attention_mask[offset][1:]
                offset += 1

            if self.is_training:
                # now, re-creating decoder_input_ids and metadata
                def _included(tokens):
                    for i in range(end_of_question, 1024-len(tokens)+1):
                        if input_ids[idx][i:i+len(tokens)]==tokens:
                            return True
                    return False
                def _valid(tokens_list):
                    offset = 0
                    for i in range(end_of_question, 1024):
                        if input_ids[idx][i:i+len(tokens_list[offset])]==tokens_list[offset]:
                            offset += 1
                            if offset==len(tokens_list):
                                return True
                    return False

                for _curr_metadata in curr_metadata:
                    found_answers = []
                    for start, end in _curr_metadata:
                        _answers = []
                        for j in range(start, end):
                            answer = _get_tokenized_answer(j)
                            if not _included(answer): continue
                            if answer in _answers: continue
                            _answers.append(answer)
                        if len(_answers)>0:
                            found_answers.append(_answers)

                    if len(found_answers)==0:
                        continue

                    decoder_offset = len(new_decoder_input_ids)
                    cnt = 0
                    for _answers in itertools.product(*found_answers):
                        _answers = list(_answers)
                        if self.args.consider_order_for_multiple_answers and not _valid(_answers):
                            continue
                        answers = [bos_token_id, bos_token_id]
                        for j, answer in enumerate(_answers):
                            if j>0: answers.append(sep_token_id)
                            answers += answer
                        answers.append(eos_token_id)
                        answers = answers[:30]
                        new_decoder_input_ids.append(
                            answers + [pad_token_id for _ in range(30-len(answers))])
                        new_decoder_attention_mask.append(
                            [1 for _ in answers] + [0 for _ in range(30-len(answers))])
                        cnt += 1
                        if cnt==100:
                            break
                assert decoder_offset+cnt==len(new_decoder_input_ids)
                if cnt==0:
                    continue
                new_metadata.append([decoder_offset, decoder_offset+cnt])

            new_input_ids.append(input_ids[idx][:1024])
            new_attention_mask.append(attention_mask[idx][:1024])

        self.tokenized_data = [new_input_ids, new_attention_mask, new_decoder_input_ids,
                               new_decoder_attention_mask, new_metadata]
        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)
        self.logger.info("Finish saving tokenized DPR data")

        # Update attention masks and trim the inputs up to max_token_nums.
        new_input_ids, new_attention_mask = [], []
        max_token_nums = self.args.max_token_nums
        for input_ids, attention_mask in zip(self.tokenized_data[0], self.tokenized_data[1]):
            assert len(input_ids)==len(attention_mask)
            # while len(input_ids) < max_token_nums:
            #     input_ids.append(self.tokenizer.pad_token_id)
            #     attention_mask.append(0)
            new_input_ids.append(input_ids[:max_token_nums])
            new_attention_mask.append(attention_mask[:max_token_nums])
        self.tokenized_data[0] = new_input_ids
        self.tokenized_data[1] = new_attention_mask


    # override
    def evaluate(self, predictions, n_paragraphs=None):
        assert len(predictions)==len(self), (len(predictions), len(self))
        f1s, f1s_wo_dupli, f1s_multi, f1s_multi_wo_dupli = [], [], [], []
        if self.args.is_seq2seq:
            for (prediction, dp) in zip(predictions, self.data):
                prediction1 = [text.strip() for text in prediction.split(self.SEP)]
                prediction2 = list(set(prediction1))
                f1 = np.max([get_f1(answer, prediction1) for answer in dp["answer"]])  # among best one by annotations
                f1_wo_dupli = np.max([get_f1(answer, prediction2) for answer in dp["answer"]])

                f1s.append(f1)
                f1s_wo_dupli.append(f1_wo_dupli)
                if any(annot['type'] != 'singleAnswer' for annot in dp["annotations"]) > 0:
                    f1s_multi.append(f1)
                    f1s_multi_wo_dupli.append(f1_wo_dupli)
            res = f"F1={np.mean(f1s)*100:.2f}, F1 w/o dupli={np.mean(f1s_wo_dupli)*100:.2f}"
            res += f", F1 only multi={np.mean(f1s_multi)*100:.2f}"
            res += f", F1 only multi w/0 dupli={np.mean(f1s_multi_wo_dupli)*100:.2f}"
            self.logger.info(res)
        else:
            for (prediction, dp) in zip(predictions, self.data):
                preds = []
                if type(prediction[0])==list:
                    prediction = prediction[-1]
                for p in prediction:
                    if normalize_answer(p["text"]) not in preds:
                        if p["log_softmax"]>np.log(0.05) or len(preds)==0:
                            preds.append(normalize_answer(p["text"]))
                        if p["log_softmax"]<=np.log(0.05) or len(preds)==3:
                            break
                f1s.append(np.max([get_f1(answer, preds) for answer in dp["answer"]]))
        return f1s


class CAmbigQAData(AmbigQAData):
    def __init__(self, logger, args, data_path, is_training, passages=None):
        super(CAmbigQAData, self).__init__(logger, args, data_path, is_training, passages)
        data_path = Path(data_path)

        with open(data_path.parent.parent / "nqopen" / f"{self.data_type}.json", "r") as f:
            orig_data = json.load(f)
            id_to_orig_idx = {d["id"]:i for i, d in enumerate(orig_data)}

        # we will only consider questions with multiple answers
        self.ref_questions, self.ref_answers, self.ref_cq = [], [], []
        for i, d in enumerate(self.data):
            if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                self.ref_questions.append(None)
                self.ref_answers.append(None)
                continue
            questions, answers = [], []
            for annotation in d["annotations"]:
                questions.append([[q.strip() for q in pair["question"].split("|")] for pair in annotation["qaPairs"]])
                answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])  # corresponding to answers by index matching
            assert type(answers)==list and \
                   all([type(answer)==list for answer in answers]) and \
                   all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])

            self.ref_questions.append(questions)
            self.ref_answers.append(answers)
            self.ref_cq.append(d["clarification_question"])
            self.data[i]["answer"] = answers
            self.data[i]["orig_idx"] = id_to_orig_idx[d["id"]]

        self.SEP = "<SEP>"
        self.metric = "F1"
        self.data_ = None  # original predict file.

        if "with_dqs_by" in data_path.name and "auto" not in data_path.name:
            assert len(self.args.original_predict_file) > 0, "Need to set original_predict_file"
            with open(self.args.original_predict_file, "r") as f:
                self.data_ = json.load(f)
            if "data" in self.data_:
                self.data_ = self.data_["data"]
            if "answers" in self.data_[0]:
                self.data_ = [{"id": d["id"], "question": d["question"], "answer": d["answers"]}
                              for d in self.data_]
            self.ref_questions_, self.ref_answers_, self.ref_cq_ = [], [], []
            for i, d in enumerate(self.data_):
                if not all([ann["type"]=="multipleQAs" for ann in d["annotations"]]):
                    self.ref_questions_.append(None)
                    self.ref_answers_.append(None)
                    continue
                questions, answers = [], []
                for annotation in d["annotations"]:
                    questions.append([[q.strip() for q in pair["question"].split("|")] for pair in annotation["qaPairs"]])
                    answers.append(d["clarification_answers"])  # corresponding to converted cq options
                assert type(answers)==list and \
                       all([type(answer)==list for answer in answers]) and \
                       all([type(_a)==str for answer in answers+questions for _answer in answer for _a in _answer])
                self.ref_questions_.append(questions)
                self.ref_answers_.append(answers)
                self.ref_cq_.append(d["clarification_question"])
                self.data_[i]["orig_idx"] = id_to_orig_idx[d["id"]]
                self.data_[i]["answer"] = answers

    def load_tokenized_data(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        assert "Bart" in postfix or "Bert" in postfix or "Albert" in postfix
        assert ("Bart" in postfix and self.args.append_another_bos) \
               or ("Bart" not in postfix and not self.args.append_another_bos)

        data_path = Path(self.data_path)
        name = data_path.name.replace(
            ".tsv" if data_path.name.endswith(".tsv") else ".json",
            "{}{}-{}.json".format("-uncased" if self.args.do_lowercase else "", "-xbos" if self.args.append_another_bos else "", postfix))
        preprocessed_path = data_path.parent / name

        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
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

    # BART override
    def load_dpr_data_bart(self, dpr_retrieval_path, dpr_tokenized_path):
        if self.is_training and self.args.consider_order_for_multiple_answers:
            dpr_tokenized_path = dpr_tokenized_path.replace(".json", "_ordered.json")

        self.logger.info(dpr_tokenized_path)

        if os.path.exists(dpr_tokenized_path):
            self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
            with open(dpr_tokenized_path, "r") as f:
                self.tokenized_data = json.load(f)
            return

        self.logger.info("Start processing DPR data from {}".format(dpr_retrieval_path))
        self.logger.info("Loading passage selection from DPR reader: {}".format(dpr_retrieval_path))
        with open(dpr_retrieval_path) as f:  # For each question, which passages we need to read. = reranking results...
            dpr_passages = json.load(f)

        bos_token_id = self.tokenizer.bos_token_id
        # class EmptyPassage:
        #     def __init__(self, default):
        #         self.default = default
        #     def __getitem__(self, key):
        #         return [self.default] * 100

        if self.passages.tokenized_data is None:
            # self.passages.tokenized_data = {"input_ids": EmptyPassage(bos_token_id), "attention_mask": EmptyPassage(1)}
            self.passages.load_tokenized_data("bart", all=True)  # 2200M passages.. TAKES LONG.


        assert len(dpr_passages)==len(self)

        # NOTE: two more variables for clarification questions.
        input_ids, attention_mask, _, _, answer_input_ids, answer_attention_mask, metadata = self.tokenized_data
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
        new_output, new_output_mask, new_metadata = [], [], []
        chosen_list = []

        new_decoder_input_ids, new_decoder_attention_mask, new_metadata = [], [], []
        for idx, (curr_input_ids, curr_attention_mask, curr_metadata, dpr_ids) in enumerate(zip(
                input_ids, attention_mask, metadata, dpr_passages)):  # input_ids and attention_mask are not used
            if self.ref_questions[idx] is None:
                continue

            # Extract question tokens
            # end_of_question = curr_input_ids.index(eos_token_id)+1
            # q_input_ids = curr_input_ids[:end_of_question]
            # q_input_attention_mask = curr_attention_mask[:end_of_question]

            # Aggregate passages
            p_input_ids, p_attention_mask = [], []
            dpr_input_ids = [self.passages.tokenized_data["input_ids"][_id] for _id in dpr_ids]
            dpr_attention_mask = [self.passages.tokenized_data["attention_mask"][_id] for _id in dpr_ids]
            offset = 0
            while len(p_input_ids)<1024:
                assert dpr_input_ids[offset][0] == bos_token_id, f"not bos_token: {dpr_input_ids[offset]}"
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

            min_n_missing = np.min(n_missing_list)
            annotation_indices = [ann_idx for ann_idx in range(len(n_missing_list))
                                  if n_missing_list[ann_idx]==min_n_missing]

            def _form_data(annotation_idx, new_output, new_output_mask):
                ref_questions = self.ref_questions[idx][annotation_idx]
                tokenized_ref_answers = tokenized_ref_answers_list[annotation_idx]
                checks = len(ref_questions), len(tokenized_ref_answers), len(is_valid_list[annotation_idx])
                assert len(ref_questions)==len(tokenized_ref_answers)==len(is_valid_list[annotation_idx])," ".join([str(i) for i in checks])
                final_ref_questions = []
                chosen_indices = []
                for (ref_question, tok_ref_answer, is_valid) in \
                        zip(ref_questions, tokenized_ref_answers, is_valid_list[annotation_idx]):
                    assert len(tok_ref_answer)==len(is_valid)
                    chosen_idx = is_valid.index(True) if True in is_valid else 0
                    chosen_indices.append(chosen_idx)
                    final_ref_questions.append(ref_question[0])

                    answer = [bos_token_id] * 2 + tok_ref_answer[chosen_idx] + [eos_token_id]
                    len_answer = min(len(answer), self.args.max_answer_length)
                    answer = answer[:self.args.max_answer_length] + [pad_token_id] * (self.args.max_answer_length-len_answer)
                    answer_mask = [1] * len_answer + [0] * (self.args.max_answer_length-len_answer)
                    new_output.append(answer)
                    new_output_mask.append(answer_mask)

                # print(tokenized_ref_answers)
                # 1/0

                tokenized_ref_questions = self.tokenizer.batch_encode_plus(final_ref_questions)
                dq_input_ids = tokenized_ref_questions["input_ids"]
                dq_attention_mask = tokenized_ref_questions["attention_mask"]
                for i, elts in enumerate(zip(dq_input_ids, dq_attention_mask)):
                    input_ids, attention_mask = elts
                    input_ids += p_input_ids
                    attention_mask += p_attention_mask
                    assert len(input_ids)==len(attention_mask)
                    new_input_ids.append(input_ids)
                    new_attention_mask.append(attention_mask)
                return chosen_indices

            start = len(new_output)
            if self.is_training:
                start = len(new_output)
                for annotation_idx in annotation_indices:
                    _form_data(annotation_idx, new_output, new_output_mask)
            else:
                annotation_idx = annotation_indices[0]
                chosen_indices = _form_data(annotation_idx, new_output, new_output_mask)
                chosen_list.append({"annotation_idx": annotation_idx,
                                    "answer_idx": chosen_indices})
            assert len(new_output)-start > 0
            assert len(new_output)==len(new_input_ids)
            new_metadata.append((start, len(new_output)))

        if self.is_training:
            assert len(new_input_ids) == len(new_output)
        else:
            new_output, new_output_mask = None, None

        self.tokenized_data = [new_input_ids, new_attention_mask, new_output, new_output_mask, new_metadata]

        if not self.is_training:
            self.tokenized_data.append(chosen_list)

        with open(dpr_tokenized_path, "w") as f:
            json.dump(self.tokenized_data, f)
        self.logger.info("Finish saving tokenized DPR data")

    #TODO: HERE!!
    def load_dpr_data_bert(self, dpr_retrieval_path, dpr_tokenized_path):
        raise NotImplementedError
        # if os.path.exists(dpr_tokenized_path):
        #     self.logger.info("Loading DPR data from {}".format(dpr_tokenized_path))
        #     with open(dpr_tokenized_path, "r") as f:
        #         self.tokenized_data = json.load(f)
        #     return
        # self.logger.info("Start processing DPR data")
        # self.logger.info("Loading passage selection from DPR reader: {}".format(dpr_retrieval_path))
        # with open(dpr_retrieval_path) as f:  # For each question, which passages we need to read. = reranking results...
        #     dpr_passages = json.load(f)
        # if self.is_training:
        #     with open(os.path.join(self.args.dpr_data_dir, "data/gold_passages_info/nq_train.json"), "r") as f:
        #         gold_titles = [d["title"] for d in json.load(f)["data"]]
        #         assert len(gold_titles)==len(self)
        #
        # #TODO: Two more variables for CAmbigQA / BART return 7 vars but BERT return 5. BUT dealing with later..
        # input_ids, attention_mask, _, _, answer_input_ids, answer_attention_mask, metadata = self.tokenized_data
        # eos_token_id = self.tokenizer.eos_token_id if "Albert" in dpr_tokenized_path.name else self.tokenizer.sep_token_id
        # assert eos_token_id is not None
        # assert len(dpr_passages)==len(input_ids)==len(attention_mask)==len(metadata)
        #
        # # class EmptyPassage:
        # #     def __getitem__(self, key):
        # #         return [1] * 100
        #
        # if self.passages.tokenized_data is None:
        #     # self.passages.tokenized_data = {"input_ids": EmptyPassage(), "attention_mask": EmptyPassage()}
        #     subset = set([p_idx for retrieved in dpr_passages for p_idx in retrieved])
        #     self.passages.load_tokenized_data("albert" if "Albert" in dpr_tokenized_path.name else "bert",
        #                                       subset=subset, all=True)
        # features = defaultdict(list)
        # max_n_answers = self.args.max_n_answers
        # oracle_exact_matches = []
        # flatten_exact_matches = []
        # positive_contains_gold_title = []
        #
        # # def _extract_title_from_passage(tokenized_passage):
        # #     tokenized_title = tokenized_passage[:tokenized_passage.index(eos_token_id)]
        # #     title = self.decode(tokenized_title)
        # #     return normalize_answer(title)
        #
        # for i, (q_input_ids, q_attention_mask, retrieved) in \
        #         tqdm(enumerate(zip(input_ids, attention_mask, dpr_passages))):
        #     if self.ref_questions[i] is None:
        #         continue
        #     assert len(q_input_ids)==len(q_attention_mask)==32
        #     q_input_ids = [in_ for in_, mask in zip(q_input_ids, q_attention_mask) if mask]
        #     assert 3<=len(q_input_ids)<=32
        #
        #     # dq_input_ids =
        #     assert 3<=len(q_input_ids)<=32
        #
        #     p_input_ids = [self.passages.tokenized_data["input_ids"][p_idx] for p_idx in retrieved]
        #     p_attention_mask = [self.passages.tokenized_data["attention_mask"][p_idx] for p_idx in retrieved]
        #
        #     # print(len(metadata), metadata[:5])
        #     # print(len(answer_input_ids), answer_input_ids[:5])
        #     # print([idx for idx in range(metadata[i][0], metadata[i][1])])
        #
        #     for ref_answers, ref_questions, ref_metadata in zip(self.ref_answers[i], self.ref_questions[i], metadata[i]):  # For each turker (=annotator)...
        #         # ref_questions = self.ref_questions[idx][annotation_idx]
        #         # final_ref_questions, final_ref_answers = [], []
        #         # chosen_indices = []
        #         # for (ref_question, ref_answer, tok_ref_answer, is_valid) in \
        #         #         zip(ref_questions, ref_answers, tokenized_ref_answers, is_valid_list[annotation_idx]):
        #         #     assert len(ref_answer)==len(tok_ref_answer)==len(is_valid)
        #         #     chosen_idx = is_valid.index(True) if True in is_valid else 0
        #         #     chosen_indices.append(chosen_idx)
        #         #     final_ref_questions.append(ref_question[0])
        #         #     final_ref_answers.append(tok_ref_answer[chosen_idx])
        #         # tokenized_ref_questions = self.tokenizer.batch_encode_plus(final_ref_questions)
        #         # dq_input_ids = tokenized_ref_questions["input_ids"]
        #         # dq_attention_mask = tokenized_ref_questions["attention_mask"]
        #
        #         tokenized_ref_questions = self.tokenizer.encode_plus(ref_questions, max_length=32, truncation=True)["input_ids"]
        #         for dq_input_ids, metadata_ in zip(tokenized_ref_questions, ref_metadata): # For each semantically same answers,
        #             a_input_ids = [answer_input_ids[idx][1:-1] for idx in range(metadata_[0], metadata_[1])]
        #             detected_spans = []
        #             for _p_input_ids in p_input_ids:
        #                 detected_spans.append([])
        #                 for _a_input_ids in a_input_ids:  # a_input_ids: set of semantically same answers
        #                     decoded_a_input_ids = self.decode(_a_input_ids)
        #                     for j in range(len(_p_input_ids)-len(_a_input_ids)+1):
        #                         if _p_input_ids[j:j+len(_a_input_ids)]==_a_input_ids:
        #                             detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+len(_a_input_ids)-1))
        #                         elif "Albert" in dpr_tokenized_path and \
        #                                 _p_input_ids[j]==_a_input_ids[0] and \
        #                                 13 in _p_input_ids[j:j+len(_a_input_ids)]:
        #                             k = j + len(_a_input_ids)+1
        #                             while k<len(_p_input_ids) and np.sum([_p_input_ids[z]!=13 for z in range(j, k)])<len(_a_input_ids):
        #                                 k += 1
        #                             if decoded_a_input_ids==self.decode(_p_input_ids[j:k]):
        #                                 detected_spans[-1].append((j+len(q_input_ids), j+len(q_input_ids)+k-1))
        #
        #             negatives = [j for j, spans in enumerate(detected_spans) if len(spans)==0][:50]
        #             if self.is_training:
        #                 positives = [j for j, spans in enumerate(detected_spans) if len(spans)>0][:20]
        #                 if len(positives)==0:
        #                     continue
        #             else:
        #                 # positives = [j for j in range(len(detected_spans))]
        #                 positives = [j for j in range(100)]
        #                 negatives = []
        #             for key in ["positive_input_ids", "positive_input_mask", "positive_token_type_ids",
        #                         "positive_start_positions", "positive_end_positions", "positive_answer_mask",
        #                         "negative_input_ids", "negative_input_mask", "negative_token_type_ids"]:
        #                 features[key].append([])
        #
        #             def _form_input(p_input_ids, p_attention_mask):
        #                 assert len(p_input_ids)==len(p_attention_mask)
        #                 assert len(p_input_ids)==128 or (len(p_input_ids)<=128 and np.sum(p_attention_mask)==len(p_attention_mask))
        #                 if len(p_input_ids) < 128:
        #                     p_input_ids += [self.tokenizer.pad_token_id for _ in range(128-len(p_input_ids))]
        #                     p_attention_mask += [0 for _ in range(128-len(p_attention_mask))]
        #                 input_ids = dq_input_ids + p_input_ids + [self.tokenizer.pad_token_id for _ in range(32-len(dq_input_ids))]
        #                 attention_mask = [1 for _ in range(len(dq_input_ids))] + p_attention_mask + [0 for _ in range(32-len(dq_input_ids))]
        #                 token_type_ids = [0 for _ in range(len(dq_input_ids))] + p_attention_mask + [0 for _ in range(32-len(dq_input_ids))]
        #                 return input_ids, attention_mask, token_type_ids
        #
        #             for idx in positives:
        #                 input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
        #                 features["positive_input_ids"][-1].append(input_ids)
        #                 features["positive_input_mask"][-1].append(attention_mask)
        #                 features["positive_token_type_ids"][-1].append(token_type_ids)
        #                 detected_span = detected_spans[idx]
        #                 features["positive_start_positions"][-1].append(
        #                     [s[0] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
        #                 features["positive_end_positions"][-1].append(
        #                     [s[1] for s in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
        #                 features["positive_answer_mask"][-1].append(
        #                     [1 for _ in detected_span[:max_n_answers]] + [0 for _ in range(max_n_answers-len(detected_span))])
        #             for idx in negatives:
        #                 input_ids, attention_mask, token_type_ids = _form_input(p_input_ids[idx], p_attention_mask[idx])
        #                 features["negative_input_ids"][-1].append(input_ids)
        #                 features["negative_input_mask"][-1].append(attention_mask)
        #                 features["negative_token_type_ids"][-1].append(token_type_ids)
        #             # for debugging
        # #             for p_input_ids, starts, ends, masks in zip(features["positive_input_ids"][-1],
        # #                                                         features["positive_start_positions"][-1],
        # #                                                         features["positive_end_positions"][-1],
        # #                                                         features["positive_answer_mask"][-1]):
        # #                 if np.sum(masks)==0: continue
        # #                 assert len(starts)==len(ends)==len(masks)==max_n_answers
        # #                 decoded_answers = [self.tokenizer.decode(p_input_ids[start:end+1]) for start, end, mask in zip(starts, ends, masks) if mask]
        # #                 ems = [get_exact_match(decoded_answer, self.data[i]["answer"]) for decoded_answer in decoded_answers]
        # #                 oracle_exact_matches.append(np.max(ems))
        # #                 flatten_exact_matches += ems
        # # print("oracle exact matches", np.mean(oracle_exact_matches))
        # # print("flatten exact matches", np.mean(flatten_exact_matches))
        # # print("positive contains gold title", np.mean(positive_contains_gold_title))
        # # print(len(positive_contains_gold_title))
        # self.tokenized_data = features
        #
        # with open(dpr_tokenized_path, "w") as f:
        #     json.dump(self.tokenized_data, f)

    # override
    def load_dpr_data(self):
        if len(self.args.original_predict_file) == 0:
            task = "cqa"
        else:
            original_predict_file = Path(self.args.original_predict_file)
            info = original_predict_file.name.split(".json")[0]
            task = f"cqa_{info}"
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        data_type_for_dpr_retrieval_path = self.data_type+"_2020" if self.args.wiki_2020 else self.data_type
        dpr_retrieval_path = Path(self.args.psg_sel_dir) / f"ambigqa_{data_type_for_dpr_retrieval_path}.json"

        if "dev" in self.data_path:
            name = Path(self.data_path).name
            data_type = name.split(".")[0]
        else:
            data_type = self.data_type
        data_type_dpr_tokenized_path = data_type+"_20200201" if self.args.wiki_2020 else data_type
        # data_type_dpr_tokenized_path = self.data_type+"_20200201" if self.args.wiki_2020 else self.data_type
        data_name = f"ambigqa_predictions_{postfix}_{task}"
        dpr_tokenized_path = Path(self.args.output_dir) / f"{data_type_dpr_tokenized_path}_{data_name}.json"
        # dpr_tokenized_path: model inputs and targets.

        # Here, prepare model inputs and targets by concatenation (question(+multiple answers) + passages)
        if "Bart" in postfix:
            self.load_dpr_data_bart(dpr_retrieval_path, dpr_tokenized_path)
        elif "Bert" in postfix or "Albert" in postfix:
            # metadata, new_metadata = self.tokenized_data[-1], []
            # for curr_metadata in metadata:
            #     new_metadata.append((curr_metadata[0][0][0], curr_metadata[-1][-1][-1]))
            # self.tokenized_data[-1] = new_metadata
            self.load_dpr_data_bert(dpr_retrieval_path, dpr_tokenized_path)

        # in attention_mask, 1 means answer + passages, 2 means prompt, 3 means other answers
        # Update attention masks and trim the inputs up to max_token_nums.
        new_input_ids, new_attention_mask = [], []
        max_token_nums = self.args.max_token_nums
        for input_ids, attention_mask in zip(self.tokenized_data[0], self.tokenized_data[1]):
            assert len(input_ids)==len(attention_mask)
            # while len(input_ids) < max_token_nums:
            #     input_ids.append(self.tokenizer.pad_token_id)
            #     attention_mask.append(0)
            new_input_ids.append(input_ids[:max_token_nums])
            new_attention_mask.append(attention_mask[:max_token_nums])
        self.tokenized_data[0] = new_input_ids
        self.tokenized_data[1] = new_attention_mask

    def load_dataset(self, tokenizer, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data(tokenizer)
        if isinstance(self.tokenized_data, dict):  # BERT
            self.dataset = MyQADataset(self.tokenized_data,
                                       is_training=self.is_training,
                                       train_M=self.args.train_M,
                                       test_M=self.args.test_M)
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = self.tokenized_data[:5]
            self.dataset = MySimpleQADataset(input_ids,
                                             attention_mask,
                                             decoder_input_ids if self.is_training or self.args.nq_answer_as_prefix else None,
                                             decoder_attention_mask if self.is_training or self.args.nq_answer_as_prefix else None,
                                             in_metadata=None,
                                             out_metadata=None,  #FLATTEN!!!, originally, metadata.
                                             is_training=self.is_training,
                                             answer_as_prefix=self.args.nq_answer_as_prefix)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    # override
    #TODO: evaluation macro / instance-wise
    def evaluate(self, predictions, n_paragraphs=None, return_dict=False, test_ids=TEST_IDS, dev=True):
        chosen_list = self.tokenized_data[-1]
        num_dqs = sum([len(i["answer_idx"]) for i in chosen_list])
        assert len(predictions) == num_dqs, (len(predictions), num_dqs)

        indices = [idx for idx, a in enumerate(self.ref_answers) if a]
        dq_answers = [self.ref_answers[i] for i in indices]

        if self.data_ is None:
            orig_dq_answers = [self.data[i]["answer"] for i in indices]  # multiple annotations...
        else:
            orig_dq_answers = [self.data_[i]["answer"] for i in indices]  # multiple annotations...
        orig_dq_answers = [[[[normalize_answer(ans) for ans in sem_same_set] for sem_same_set in annot] for annot in ex] for ex in orig_dq_answers]
        # orig_dq_answers = self.ref_answers
        # orig_dq_answers = [[[[normalize_answer(ans) for ans in sem_same_set] for sem_same_set in annot] for annot in ex] for ex in orig_dq_answers]

        assert len(dq_answers) == len(chosen_list), (len(dq_answers), len(chosen_list))

        ref_ans_aq_wisely = []
        pred_ans_aq_wisely = []
        cur = 0

        for chosen_info, dq_ans in zip(chosen_list, dq_answers):
            dq_ans = dq_ans[chosen_info["annotation_idx"]]
            num_preds = len(chosen_info["answer_idx"])
            #TODO: why len(dq_ans) < num_preds ????????
            ref_ans_aq_wisely.append([[normalize_answer(ans) for ans in sem_same_set] for sem_same_set in dq_ans[:num_preds]])
            pred_ans_aq_wisely.append([normalize_answer(ans) for ans in predictions[cur: cur + num_preds]])
            cur += num_preds

        res = {"EM": [], "F1": [], "#options": [], "Rec": [], "Prec": []}
        if self.args.is_seq2seq:
            assert len(pred_ans_aq_wisely) == len(ref_ans_aq_wisely)
            assert len(orig_dq_answers) == len(ref_ans_aq_wisely)
            for idx, (preds, refs, orig_refs) in enumerate(zip(pred_ans_aq_wisely, ref_ans_aq_wisely, orig_dq_answers)):
                if not dev and idx not in test_ids:
                    continue
                if dev and idx in test_ids:
                    continue
                ems = []
                for pred, ref in zip(preds, refs):
                    em = get_exact_match(ref, [pred], False)
                    ems.append(em)
                res["EM"].append(np.average(em))
                f1prs = [get_f1(answer, preds, is_equal=lambda x, y: get_exact_match(x, y, False), return_p_and_r=True) for answer in orig_refs]
                f1s = [i[0] for i in f1prs]
                recs = [i[1] for i in f1prs]
                precs = [i[2] for i in f1prs]
                idx = max([(idx, f1) for idx, f1 in enumerate(f1s)], key=lambda x: x[1])[0]

                res["Rec"].append(recs[idx])
                res["Prec"].append(precs[idx])
                res["F1"].append(f1s[idx])
                res["#options"].append(len(list(set(preds))))

        else:  # TODO: for BERT
            raise NotImplementedError()
        #     for (prediction, dp) in zip(predictions, self.data):
        #         preds = []
        #         if type(prediction[0]) == list:
        #             prediction = prediction[-1]
        #         for p in prediction:
        #             if normalize_answer(p["text"]) not in preds:
        #                 if p["log_softmax"] > np.log(0.05) or len(preds) == 0:
        #                     preds.append(normalize_answer(p["text"]))
        #                 if p["log_softmax"] <= np.log(0.05) or len(preds) == 3:
        #                     break
        #         f1s.append(np.max([get_f1(answer, preds) for answer in dp["answer"]]))
        # #         return f1s

        res_log = "\n"
        for metric, v in res.items():
            res_log += f"{metric}: {np.mean(v)*100:.2f}%\n"
        self.logger.info(res_log)

        if return_dict:
            res_ = {}
            for k, v in res.items():
                res_[k] = np.mean(v) * 100
            return res_
        return res

    def save_predictions(self, predictions):
        checkpoint = Path(self.args.checkpoint)
        ckpt_name = checkpoint.name.split(".")[0]  # drop .th

        data_path = Path(self.data_path)
        data_name = data_path.name.split(".")[0]  # drop .json   dev_cq.json / dev_cq_predicted_with_... /
        psg_data = "_20200201" if self.args.wiki_2020 and not self.args.ambigqa else ""
        pred_file_name = f"{data_name}{psg_data}_predictions_by_{ckpt_name}.json"

        save_path = Path(self.args.output_dir) / pred_file_name
        if self.args.save_psg_sel_only:
            save_path = save_path.parent / save_path.name.replace("predictions.json", "psg_sel.json")
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))

    # def evaluate(self, predictions, n_paragraphs=None):
    #     assert len(predictions)==len(self), (len(predictions), len(self))
    #     if self.args.save_psg_sel_only:
    #         return [-1]
    #     if n_paragraphs is None:
    #         ems = []
    #         for (prediction, dp) in zip(predictions, self.data):
    #             if type(prediction)==list:
    #                 prediction = prediction[0]
    #             if type(prediction)==dict:
    #                 prediction = prediction["text"]
    #             ems.append(get_exact_match(prediction, dp["answer"]))
    #         return ems
    #     ems = defaultdict(list)
    #     for (prediction, dp) in zip(predictions, self.data):
    #         assert len(n_paragraphs)==len(prediction)
    #         for pred, n in zip(prediction, n_paragraphs):
    #             if type(pred)==list:
    #                 pred = pred[0]
    #             if type(pred)==dict:
    #                 pred = pred["text"]
    #             ems[n].append(get_exact_match(pred, dp["answer"]))
    #     for n in n_paragraphs:
    #         self.logger.info("n_paragraphs=%d\t#M=%.2f" % (n, np.mean(ems[n])*100))
    #     return ems[n_paragraphs[-1]]
