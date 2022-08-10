from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
import utils as utils
import inspect
#from torchcrf import CRF

class KmerCNN(nn.Module):
    def __init__(self, vecs, num_kmers, num_channels, num_convs, num_fcs, vec_sizes, fc_size=128, conv_features=100,
                 pool_size=2, filter_size=3, dilation=1, padding=0, stride=1, use_gpu=False):
        super(KmerCNN, self).__init__()
        #Save parameters
        self.FC_SIZE = fc_size
        self.NUM_CHANNELS = num_channels
        #print(f"NUM CHANNELS = {self.NUM_CHANNELS}\nvec_sizes = {vec_sizes}\nvecs size = {vecs.size()}\nvecs = {vecs}")
        self.NUM_CONVS = num_convs
        self.NUM_FCS = num_fcs
        #print(f"num fcs = {num_fcs}\nnum convs = {num_convs}")
        self.POOL_SIZE = pool_size
        self.FILTER_SIZE = filter_size
        self.NUM_KMERS = num_kmers
        self.DILATION = dilation
        self.PADDING = padding
        self.STRIDE = stride
        self.VEC_SIZES = vec_sizes
        self.MAX_SIZE = max(vec_sizes)
        self.CONV_FEATURES = conv_features
        self.DEVICE = torch.device("cuda" if use_gpu else "cpu")
        self.GROUPS = self.NUM_CHANNELS

        if len(self.VEC_SIZES) != self.NUM_CHANNELS:
            print(f"ERROR: self.VEC_SIZES ({len(self.VEC_SIZES)}) != self.NUM_CHANNELS ({self.NUM_CHANNELS}). Exiting")
            exit()
        #Embeddings
        self.embeddings = nn.ModuleList()
        if num_channels == 1:
            self.embeddings.append(nn.Embedding.from_pretrained(vecs))
        else:
            start_pos = 0
            vocab_size = vecs.size(0)
            for i in range(self.NUM_CHANNELS):
                v = vecs[:, start_pos:start_pos + vec_sizes[i]]
                #print(f"{i} v size= {v.size()}\n{i} v = {v}")
                padded = torch.zeros(vocab_size, self.MAX_SIZE)
                padded[:, :self.VEC_SIZES[i]] = v
                self.embeddings.append(nn.Embedding.from_pretrained(padded, freeze=True))
                start_pos += vec_sizes[i]

        #Convolutions
        self.convolutions = nn.ModuleList([nn.Conv2d(self.NUM_CHANNELS, self.CONV_FEATURES, self.FILTER_SIZE,
                                          groups=self.GROUPS, dilation=self.DILATION,
                                          padding=self.PADDING, stride=self.STRIDE)])
        self.convolutions.extend([nn.Conv2d(self.CONV_FEATURES, self.CONV_FEATURES, self.FILTER_SIZE,
                                            groups=1, dilation=self.DILATION,
                                            padding=self.PADDING, stride=self.STRIDE)
                                  for _ in range(1, self.NUM_CONVS)])
        self.conv_bns = nn.ModuleList([nn.BatchNorm2d(self.CONV_FEATURES) for _ in range(self.NUM_CONVS)])
        
        #FCs
        fc_input = 0
        h = self.NUM_KMERS
        w = self.MAX_SIZE
        for j in range(self.NUM_CONVS):
            (h, w) = utils.output_size(h, w, self.PADDING, self.DILATION, self.FILTER_SIZE, self.STRIDE)
            (h, w) = utils.output_size(h, w, self.PADDING, self.DILATION, self.POOL_SIZE, self.POOL_SIZE)
        fc_input += h * w * self.CONV_FEATURES
        self.fc_sizes = [fc_input]
        self.fc_sizes.extend([self.FC_SIZE if i == 0 else int(self.FC_SIZE / (2 * i)) for i in range(self.NUM_FCS - 1)])
        self.fc_sizes.append(1)
        print(f" fc sizes = {self.fc_sizes}")
        self.fcs = nn.ModuleList([nn.Linear(self.fc_sizes[i], self.fc_sizes[i+1]) for i in range(self.NUM_FCS)])
        self.fc_bns = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(self.NUM_FCS - 1)])


    def forward(self, kmers):
        (_, batch_size) = kmers.size() # (num_kmers, batch)
        kmers = kmers.transpose(0, 1) # (batch, num_kmers)
        if self.NUM_CHANNELS > 1:
            emb_chans = [self.embeddings[i](kmers) for i in range(self.NUM_CHANNELS)]
            #print(f"emb chans shape = ({len(emb_chans)}, {len(emb_chans[0])}, {len(emb_chans[0][0])}, {len(emb_chans[0][0][0])})")
            emb_chans = torch.stack(emb_chans, 1).to(self.DEVICE)
            #print(f"emb chans size = {emb_chans.size()}")
        else:
            emb_chans = self.embeddings[0](kmers).unsqueeze(1) # (batch, 1, num_kmers, embedding_size
        #print(f"\nemb chans size = {emb_chans.size()}")
        conv_input = emb_chans
        for i in range(self.NUM_CONVS):
            conv_input = F.max_pool2d(F.relu(self.convolutions[i](conv_input)), self.POOL_SIZE, self.POOL_SIZE) # (batch, CONV_FEATURES, ~1/2 num_kmers, ~1/2 embedding_size)
            conv_input = self.conv_bns[i](conv_input)
            #print(f"conv_input size = {conv_input.size()}")
        #print(f"conv input size = {conv_input.size()}")
        fc_input = conv_input.view(batch_size, 1, self.fc_sizes[0]) # (batch, FC_sizes[0])
        for i in range(self.NUM_FCS - 1):
            fc_input = F.relu(self.fcs[i](fc_input)) # (batch, FC2_IN)
            fc_input = self.fc_bns[i](fc_input)
        final_fc_out = self.fcs[self.NUM_FCS - 1](fc_input)
        #print(f"final_fc_out = {final_fc_out}\nfinal_fc_out size = {final_fc_out.size()}")
        output = torch.sigmoid(final_fc_out)
        #BCEWithLogitsLoss does the sigmoiding and BCE together, more stable
        #x = self.fc4(x)


        #BCEWithLogitsLoss does the sigmoiding and BCE together, more stable
        #x = self.fc4(x)
        #print(f"output = {output}\noutput size = {output.size()}")
        return output


class KmerMulticlassCNN(nn.Module):
    def __init__(self, vecs, num_kmers, num_channels, num_convs, num_fcs, vec_sizes, curr_classes, fc_size=128,
                 conv_features=100, pool_size=2, filter_size=3, dilation=1, padding=0, stride=1, use_gpu=False):
        super(KmerMulticlassCNN, self).__init__()
        # Save parameters
        self.FC_SIZE = fc_size
        self.NUM_CHANNELS = num_channels
        self.CURR_CLASSES = curr_classes
        #print(f"NUM CHANNELS = {self.NUM_CHANNELS}\nvec_sizes = {vec_sizes}\nvecs size = {vecs.size()}\nvecs = {vecs}")
        self.NUM_CONVS = num_convs
        self.NUM_FCS = num_fcs
        # print(f"num fcs = {num_fcs}\nnum convs = {num_convs}")
        self.POOL_SIZE = pool_size
        self.FILTER_SIZE = filter_size
        self.NUM_KMERS = num_kmers
        self.DILATION = dilation
        self.PADDING = padding
        self.STRIDE = stride
        self.VEC_SIZES = vec_sizes
        self.MAX_SIZE = max(vec_sizes)
        self.CONV_FEATURES = conv_features * self.NUM_CHANNELS
        self.DEVICE = torch.device("cuda" if use_gpu else "cpu")

        if len(self.VEC_SIZES) != self.NUM_CHANNELS:
            print(f"ERROR: self.VEC_SIZES ({len(self.VEC_SIZES)}) != self.NUM_CHANNELS ({self.NUM_CHANNELS}). Exiting")
            exit()
        # Embeddings
        self.embeddings = nn.ModuleList()
        if num_channels == 1:
            self.embeddings.append(nn.Embedding.from_pretrained(vecs))
        else:
            start_pos = 0
            vocab_size = vecs.size(0)
            for i in range(self.NUM_CHANNELS):
                v = vecs[:, start_pos:start_pos + vec_sizes[i]]
                #print(f"{i} v size= {v.size()}\n{i} v = {v}")
                padded = torch.zeros(vocab_size, self.MAX_SIZE)
                padded[:, :self.VEC_SIZES[i]] = v
                self.embeddings.append(nn.Embedding.from_pretrained(padded))
                start_pos = vec_sizes[i]

        # Convolutions
        self.convolutions = nn.ModuleList([nn.Conv2d(self.NUM_CHANNELS, self.CONV_FEATURES, self.FILTER_SIZE,
                                                     groups=self.NUM_CHANNELS, dilation=self.DILATION,
                                                     padding=self.PADDING, stride=self.STRIDE)])
        self.convolutions.extend([nn.Conv2d(self.CONV_FEATURES, self.CONV_FEATURES, self.FILTER_SIZE,
                                            groups=self.NUM_CHANNELS, dilation=self.DILATION,
                                            padding=self.PADDING, stride=self.STRIDE)
                                  for _ in range(1, self.NUM_CONVS)])
        self.conv_bns = nn.ModuleList([nn.BatchNorm2d(self.CONV_FEATURES) for _ in range(self.NUM_CONVS)])

        # FCs
        fc_input = 0
        h = self.NUM_KMERS
        w = self.MAX_SIZE
        for j in range(self.NUM_CONVS):
            (h, w) = utils.output_size(h, w, self.PADDING, self.DILATION, self.FILTER_SIZE, self.STRIDE)
            (h, w) = utils.output_size(h, w, self.PADDING, self.DILATION, self.POOL_SIZE, self.POOL_SIZE)
        fc_input += h * w * self.CONV_FEATURES
        self.fc_sizes = [fc_input]
        self.fc_sizes.extend([self.FC_SIZE if i == 0 else int(self.FC_SIZE / (2 * i)) for i in range(self.NUM_FCS - 1)])
        #TODO CHANGE TO SUPPORT DIFFERENT TAXA, TESTING PHYLUM NOW
        self.fc_sizes.append(self.CURR_CLASSES)
        #print(f" fc sizes = {self.fc_sizes}")
        self.fcs = nn.ModuleList([nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]) for i in range(self.NUM_FCS)])
        self.fc_bns = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(self.NUM_FCS - 1)])

    def forward(self, kmers):
        (_, batch_size) = kmers.size()  # (num_kmers, batch)
        kmers = kmers.transpose(0, 1)  # (batch, num_kmers)
        if self.NUM_CHANNELS > 1:
            emb_chans = [self.embeddings[i](kmers) for i in range(self.NUM_CHANNELS)]
            # print(f"emb chans shape = ({len(emb_chans)}, {len(emb_chans[0])}, {len(emb_chans[0][0])}, {len(emb_chans[0][0][0])})")
            emb_chans = torch.stack(emb_chans, 1).to(self.DEVICE)
            # print(f"emb chans size = {emb_chans.size()}")
        else:
            emb_chans = self.embeddings[0](kmers).unsqueeze(1)  # (batch, 1, num_kmers, embedding_size
        # print(f"\nemb chans size = {emb_chans.size()}")
        conv_input = emb_chans
        for i in range(self.NUM_CONVS):
            conv_input = F.max_pool2d(F.relu(self.convolutions[i](conv_input)), self.POOL_SIZE,
                                      self.POOL_SIZE)  # (batch, CONV_FEATURES, ~1/2 num_kmers, ~1/2 embedding_size)
            conv_input = self.conv_bns[i](conv_input)
            # print(f"conv_input size = {conv_input.size()}")
        fc_input = conv_input.view(batch_size, 1, self.fc_sizes[0])  # (batch, FC_sizes[0])

        for i in range(self.NUM_FCS - 1):
            fc_input = F.relu(self.fcs[i](fc_input))  # (batch, FC2_IN)
            fc_input = self.fc_bns[i](fc_input)
        final_fc_out = self.fcs[self.NUM_FCS - 1](fc_input)
        # print(f"final_fc_out = {final_fc_out}\nfinal_fc_out size = {final_fc_out.size()}")
        output = final_fc_out
        # BCEWithLogitsLoss does the sigmoiding and BCE together, more stable
        # x = self.fc4(x)

        # BCEWithLogitsLoss does the sigmoiding and BCE together, more stable
        # x = self.fc4(x)
        # print(f"output = {output}\noutput size = {output.size()}")
        return output


class KmerFreqCNN(nn.Module):
    def __init__(self, in_size, num_convs, num_fcs, fc_size=128, conv_features=100,
                 pool_size=2, filter_size=3, dilation=1, padding=0, stride=1):
        super(KmerFreqCNN, self).__init__()
        # Save parameters
        self.FC_SIZE = fc_size
        self.NUM_CONVS = num_convs
        self.NUM_FCS = num_fcs
        self.POOL_SIZE = pool_size
        self.FILTER_SIZE = filter_size
        self.IN_SIZE = in_size
        self.DILATION = dilation
        self.PADDING = padding
        self.STRIDE = stride
        self.CONV_FEATURES = conv_features


        # Convolutions
        self.convolutions = nn.ModuleList([nn.Conv1d(1, self.CONV_FEATURES, self.FILTER_SIZE,
                                                     groups=1, dilation=self.DILATION,
                                                     padding=self.PADDING, stride=self.STRIDE)])
        self.convolutions.extend([nn.Conv1d(self.CONV_FEATURES, self.CONV_FEATURES, self.FILTER_SIZE,
                                           groups=1, dilation=self.DILATION,
                                           padding=self.PADDING, stride=self.STRIDE)
                                      for _ in range(1, self.NUM_CONVS)])

        # FCs
        fc_input = 0
        h = self.IN_SIZE
        w = self.IN_SIZE
        for j in range(self.NUM_CONVS):
            (h, w) = utils.output_size(h, w, self.PADDING, self.DILATION, self.FILTER_SIZE, self.STRIDE)
            (h, w) = utils.output_size(h, w, self.PADDING, self.DILATION, self.POOL_SIZE, self.POOL_SIZE)
        fc_input += h * self.CONV_FEATURES
        self.fc_sizes = [fc_input]
        self.fc_sizes.extend([self.FC_SIZE if i == 0 else int(self.FC_SIZE / (2 * i)) for i in range(self.NUM_FCS - 1)])
        self.fc_sizes.append(1)
        self.fcs = nn.ModuleList([nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]) for i in range(self.NUM_FCS)])

    def forward(self, kmers):
        (batch_size, _) = kmers.size()  # (num_kmers, batch)
        conv_input = kmers.unsqueeze(1).float()  # (batch, 1, num_kmers)
        #print(f"\nconv_input size = {conv_input.size()}")
        for i in range(self.NUM_CONVS):
            conv_input = F.max_pool1d(F.relu(self.convolutions[i](conv_input)), self.POOL_SIZE,
                                      self.POOL_SIZE)  # (batch, CONV_FEATURES, ~1/2 num_kmers, ~1/2 embedding_size)
            #print(f"conv_input size = {conv_input.size()}")
        fc_input = conv_input.view(batch_size, self.fc_sizes[0])  # (batch, FC_sizes[0])

        for i in range(self.NUM_FCS - 1):
            fc_input = F.relu(self.fcs[i](fc_input))  # (batch, FC2_IN)
        output = torch.sigmoid(self.fcs[self.NUM_FCS - 1](fc_input))
        # BCEWithLogitsLoss does the sigmoiding and BCE together, more stable
        # x = self.fc4(x)


        # BCEWithLogitsLoss does the sigmoiding and BCE together, more stable
        # x = self.fc4(x)
        #print(f"output = {output}\noutput size = {output.size()}")
        return output


"""
class KmerSepCNN(nn.Module):
    def __init__(self, vecs, num_kmers, num_channels, num_convs, num_fcs, pool_size=2, filter_size=3, dilation=1, padding=0, stride=1):
        super(KmerSepCNN, self).__init__()
        #TODO CHANNEL STUFF
        self.CONV_FEATURES = 100 
        self.FC_SIZE = 128
        self.embeddings = []
        for v in vecs:
            self.embeddings.append((Embedding.from_pretrained(v), v.size(1)))
        self.POOL_SIZE = pool_size
        self.FILTER_SIZE = filter_size
        self.NUM_KMERS = num_kmers
        self.DILATION = dilation
        self.PADDING = padding
        self.STRIDE = stride
        self.NUM_CONVS = num_convs
        self.NUM_CHANNELS = num_channels

        self.convolutions = []
        self.fc_ins = []
        self.FC1_IN = 0
        for i in range(self.NUM_CHANNELS):
            h = self.NUM_KMERS
            w = self.embeddings[i][1]
            for j in range(self.NUM_CONVS):
                conv_h_out, conv_w_out = utils.output_size(h, w, self.PADDING,
                                                           self.DILATION, self.FILTER_SIZE, self.STRIDE)
                h, w = utils.output_size(conv_h_out, conv_w_out, self.PADDING,
                                         self.DILATION, self.POOL_SIZE, self.POOL_SIZE)
            self.FC1_IN += h * w * self.CONV_FEATURES
            


        (self.CONV2_H_OUT, self.CONV2_W_OUT) = utils.output_size(self.POOL1_H_OUT, self.POOL1_W_OUT, self.PADDING,
                                                                 self.DILATION, self.FILTER_SIZE, self.STRIDE)
        (self.POOL2_H_OUT, self.POOL2_W_OUT) = utils.output_size(self.CONV2_H_OUT, self.CONV2_W_OUT, self.PADDING,
                                                                 self.DILATION, self.POOL_SIZE, self.POOL_SIZE)
        self.FC1_IN = self.POOL2_H_OUT * self.POOL2_W_OUT * self.CONV2_CHANS
        self.FC2_IN = self.FC_SIZE
        self.FC3_IN = int(self.FC2_IN / 2)
        self.FC4_IN = int(self.FC3_IN / 2)
        #self.FC2_IN = int(self.FC1_IN / 2)
        #self.FC3_IN = int(self.FC2_IN / 2)
        #self.FC4_IN = int(self.FC3_IN / 2)

        self.conv1 = nn.Conv2d(1, self.CONV1_CHANS, self.FILTER_SIZE, dilation=self.DILATION, padding=self.PADDING,
                               stride=self.STRIDE)
        self.pool = nn.MaxPool2d(self.POOL_SIZE, self.POOL_SIZE)
        self.conv2 = nn.Conv2d(self.CONV1_CHANS, self.CONV2_CHANS, self.FILTER_SIZE, dilation=self.DILATION,
                               padding=self.PADDING, stride=self.STRIDE)
        self.fc1 = nn.Linear(self.FC1_IN, self.FC2_IN)
        self.fc2 = nn.Linear(self.FC2_IN, self.FC3_IN)
        self.fc3 = nn.Linear(self.FC3_IN, self.FC4_IN)
        self.fc4 = nn.Linear(self.FC4_IN, 1)


"""

class KmerLSTM_CRF(nn.Module):
    def __init__(self, emb_vecs, hid_size, num_taxa, num_levels, dropout, use_gpu):
        super(KmerLSTM_CRF, self).__init__()
        self.EMB_SIZE = emb_vecs.size(1)
        self.HID_SIZE = hid_size
        self.TAXA_SIZE = num_taxa
        self.NUM_LEVELS = num_levels
        self.DROPOUT = dropout
        self.word_embs = nn.Embedding.from_pretrained(emb_vecs)
        self.use_gpu = use_gpu
        self.bilstm = nn.LSTM(self.EMB_SIZE,
                              self.HID_SIZE // 2,
                              num_layers=self.NUM_LEVELS,
                              bidirectional=True,
                              batch_first=False)
        self.crf = CRF(self.TAXA_SIZE,
                       batch_first=False)
        self.hid2taxa = nn.Linear(hid_size, self.TAXA_SIZE)


    def forward(self, kmers, tags):
        #print(f"sent size = {sent.size()}")
        (sent_size, batch_size) = kmers.size()
        #print(f"batch size = {batch_size} sent_size = {sent_size}")
        print(f"kmers = {kmers.size()} tags = {tags.size()}")
        #print(f"sent size = {sent.size()} tags size = {tags.size()} casing size = {casing.size()}")
        lstm_ems = self.get_lstm_ems(kmers, batch_size).byte()
        #print(f"lstm_ems = {lstm_ems}")
        print(f"tags = {tags}")
        tags = tags.repeat(sent_size, 1).byte()
        print(f"new tags = {tags}")
        print(f"lstm_emissions = {lstm_ems.size()}\ntags = {tags.size()}")
        neg_log_likelihood = -self.crf(lstm_ems, tags)
        #print(f"log_like = {neg_log_likelihood}")
        return neg_log_likelihood


    def init_hidden(self, batch_size):
        # (# layers * # directions, size of batch (1), size of hidden layer)
        # TODO TEST RANDN VS ZEROS
        h_0 = torch.zeros(2, batch_size, self.HID_SIZE // 2)
        c_0 = torch.zeros(2, batch_size, self.HID_SIZE // 2)

        if self.use_gpu:  # Transfer vectors to the GPU if using GPU
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        return (h_0, c_0)


    def get_lstm_ems(self, kmers, batch_size):
        kmers_embed = self.word_embs(kmers)
        #print(f"sent_embed size = {sent_embed.size()}")
        #print(f"embedded sent shape = {sent_embed.size()}")
        #TODO CHANGE TO INIT_HIDDEN
        lstm_out, _ = self.bilstm(kmers_embed, self.init_hidden(batch_size))
        #print(f"lstm_out size = {lstm_out.size()}")
        lstm_preds = self.hid2taxa(lstm_out)
        #print(f"lstm_tags size = {lstm_preds.size()}")
        return lstm_preds


    def get_preds(self, kmers):
        (sent_size, batch_size) = kmers.size()
        kmers = kmers.transpose(0, 1)
        lstm_ems = self.get_lstm_ems(kmers, batch_size)
        tags_pred = self.decode_viterbi(lstm_ems)
        return tags_pred


    def decode_viterbi(self, lstm_ems):
        return self.crf.decode(lstm_ems)

