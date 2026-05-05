# -*- coding: utf-8 -*-

from dataUtils import DataManager
from tensorboard_logger import configure, log_value
import numpy as np
import random, logging, argparse, time, json
import torch
from evaluation import evaluateClassification
from bridgeModel import bridgeModel


parser = argparse.ArgumentParser()
parser.add_argument('--voc_size', type=int, default=30000) # 4500
parser.add_argument('--dim_input', type=int, default=300, choices=[100, 200, 300])
parser.add_argument('--dim_hidden', type=int, default=256, choices=[128, 256, 300, 512, 768])
parser.add_argument('--dim_bert', type=int, default=768)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_class', type=int, default=2, choices=[2, 6, 7])
parser.add_argument('--bidirectional', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout_rate', type=float, default=0.5)
parser.add_argument('--cell_dropout_rate', type=float, default=0.5)
parser.add_argument('--final_dropout_rate', type=float, default=0.5)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_length_sen', type=int, default=100)
parser.add_argument('--iter_num', type=int, default=32*150)
parser.add_argument('--per_checkpoint', type=int, default=32)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--data_dir', type=str, default='./debatev2/spacy/')
parser.add_argument('--breakpoint', type=int, default=56)
parser.add_argument('--path_wordvec', type=str, default='glove.840B.300d.txt')
parser.add_argument('--name_model', type=str, default='BertAtt')
parser.add_argument('--name_dataset', type=str, default='match')
parser.add_argument('--linear_dropout_rate', type=float, default=0.1)
parser.add_argument('--lr_bert', type=float, default=5e-05)
parser.add_argument('--multi_dim', type=int, default=20)
parser.add_argument('--tokenizer', type=str, default='spacy') # nltk, spacy, bert
parser.add_argument('--margin', type=float, default=0.5, help='Margin used in contrastive loss')
parser.add_argument('--supcon', type=int, default=1, help='Whether to use supervised contrastive loss (1=True, 0=False)')
parser.add_argument('--predict_dir', type=str, default='./predict/')
parser.add_argument('--predict', type=int, default=0)
parser.add_argument('--model_dir', type=str, default='./models/')
parser.add_argument('--t_sne', type=int, default=0) # together with --predict
parser.add_argument('--save_model', type=int, default=0)

FLAGS = parser.parse_args()
print(FLAGS)

np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.backends.cudnn.enabled = False
FLAGS.device = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(FLAGS.seed)
    FLAGS.device = torch.device('cuda') # not a specific one
else:
    FLAGS.device = torch.device('cpu')
def get_sequential_batch(data_train, step, batch_size):
    num_data = len(data_train)
    num_batches = (num_data + batch_size - 1) // batch_size  # 向上取整
    batch_index = step % num_batches
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, num_data)
    return data_train[start_idx:end_idx]


def train(model, datamanager, data_train, sample=False):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = datamanager.gen_batched_data(selected_data)
    loss, _ = model.stepTrain(batched_data)  # (loss_val, prob_np)
    return loss

num_loss = 1  # hanya Focal Loss (sarcasm)

def evaluate(model, datamanager, data_):
    loss_accum = 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    y_pred_sarcasm, y_true_sarcasm = [], []

    while st < min(len(data_), 1e4):
        selected_data = data_[st:ed]
        batched_data = datamanager.gen_batched_data(selected_data)
        _loss, prob_np = model.stepTrain(batched_data, inference=True)
        y_true_sarcasm.extend(batched_data['sarcasms'])
        # prob_np: [B, 2] – ambil argmax sebagai prediksi kelas
        y_pred_sarcasm.extend(prob_np.tolist())
        loss_accum += _loss
        st, ed = ed, ed + FLAGS.batch_size
        times += 1

    loss_avg = loss_accum / max(times, 1)

    dict_evaluation = dict()
    dict_evaluation['sarcasm'] = evaluateClassification(y_true_sarcasm, y_pred_sarcasm)

    return loss_avg, dict_evaluation

class mainModel(object):
    def __init__(self):
        logging.basicConfig(
            filename='log/%s%s%s.log' % (FLAGS.name_dataset, FLAGS.name_model, time.strftime('%m-%d_%H.%M', time.localtime())),
            level=logging.DEBUG,
            format='%(asctime)s %(filename)s %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S'
        )

        
        
        print(f"当前数据集路径: {FLAGS.data_dir}")
        print(f"当前数据集名称: {FLAGS.name_dataset}")
        
        
        
        
#         11111111删了一个valid
        self.dataset_name = ['train', 'test']
        
        self.datamanager = DataManager(FLAGS)
        self.data = {}

        for tmp in self.dataset_name:
            self.data[tmp] = self.datamanager.load_data(FLAGS.data_dir, f'{tmp}.txt')
        vocab, embed, vocab_dict = self.datamanager.build_vocab('%s%s' % (FLAGS.data_dir, FLAGS.path_wordvec), self.data[self.dataset_name[0]] + self.data[self.dataset_name[1]] )
        

        logging.info('model parameters: %s' % str(FLAGS))
        logging.info(f'Use device: {str(FLAGS.device)}')

        self.model = bridgeModel(FLAGS,
                vocab=vocab,
                embed=embed
                )

    def train(self, breakpoint=-1):
        configure(
            'summary/%s%s%s' % (FLAGS.name_dataset, FLAGS.name_model, time.strftime('%m-%d_%H.%M', time.localtime())),
            flush_secs=3)
        if breakpoint > 0:
            self.model.load_model('./models/%s%s' % (FLAGS.name_dataset, FLAGS.name_model), FLAGS.breakpoint)

        start_iter = 0 if breakpoint < 0 else (breakpoint * FLAGS.per_checkpoint + 1)
        loss_step, time_step = 0.0, 0  # scalar accumulator
        start_time = time.time()

        max_test_acc = 0.0
        best_test_step = -1

        from tqdm import tqdm

        for step in tqdm(range(start_iter, FLAGS.iter_num), desc="Training Progress"):
            if step % FLAGS.per_checkpoint == 0:
                time_step = time.time() - start_time
                logging.info('-' * 50)
                logging.info('Time of iter training %.2f s' % time_step)
                logging.info('On iter step %s: global step %d  Loss-step %.6f' % (
                    step / FLAGS.per_checkpoint, step, loss_step))
                if FLAGS.save_model:
                    self.model.save_model('%s%s%s' % (FLAGS.model_dir, FLAGS.name_dataset, FLAGS.name_model),
                                          int(step / FLAGS.per_checkpoint))

                for name in self.dataset_name:
                    loss_avg, dict_eva = evaluate(self.model, self.datamanager, self.data[name])
                    log_value(f'Loss-{name}-all', loss_avg, int(step / FLAGS.per_checkpoint))
                    logging.info(f'In dataset {name}: Loss (FocalLoss) = {loss_avg:.6f}')
                    for i, tmp in enumerate(['sarcasm']):
                        log_value(f'Loss-{tmp}-{name}', loss_avg, int(step / FLAGS.per_checkpoint))
                        log_value(f'Acc-{tmp}-{name}', dict_eva[tmp]['acc'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'F1-micro-{tmp}-{name}', dict_eva[tmp]['f1_micro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'F1-macro-{tmp}-{name}', dict_eva[tmp]['f1_macro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'pre-{tmp}-{name}', dict_eva[tmp]['pre_macro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'rec-{tmp}-{name}', dict_eva[tmp]['rec_macro'],
                                  int(step / FLAGS.per_checkpoint))
                        log_value(f'AUC-{tmp}-{name}', dict_eva[tmp]['auc'],
                                  int(step / FLAGS.per_checkpoint))
                        logging.info(
                            f"\t\tAcc is {dict_eva[tmp]['acc']:.4f}, F1-micro is {dict_eva[tmp]['f1_micro']:.4f}")
                        logging.info(
                            f"\t\tF1-macro is {dict_eva[tmp]['f1_macro']:.4f}, AUC is {dict_eva[tmp]['auc']:.4f}")
                        logging.info(
                            f"\t\tPre-macro is {dict_eva[tmp]['pre_macro']:.4f}, Rec_macro is {dict_eva[tmp]['rec_macro']:.4f}")
                        logging.info(f"\t\tFor {tmp}, C_M is \n{dict_eva[tmp]['c_m']}")
                        

                    if name == 'test':
                        acc = dict_eva['sarcasm']['acc'] 
                     
                        if acc > max_test_acc:
                            max_test_acc = acc
                            best_test_step = step
                        logging.info(f'testacc: {max_test_acc:.4f}')

                start_time = time.time()
                loss_step = 0.0  # reset scalar accumulator
            train_data = self.data[self.dataset_name[0]]
            batch_data = get_sequential_batch(train_data, step, FLAGS.batch_size)
            batched_data = self.datamanager.gen_batched_data(batch_data)
            loss, _ = self.model.stepTrain(batched_data)
            loss_step += loss / FLAGS.per_checkpoint

            # loss_step += train(self.model, self.datamanager, self.data[self.dataset_name[0]]) / FLAGS.per_checkpoint

        print(f"\n🎯 Best test accuracy (sarcasm): {max_test_acc:.4f} at step {best_test_step}")


if __name__ == '__main__':
    mm = mainModel()
    mm.train(FLAGS.breakpoint)