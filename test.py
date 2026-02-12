import argparse
import math

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence

from model_sakt_draw import SAKT
from utils import *
from draw_heatmap import heatmap


def get_data(df, max_length):
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    return data


def prepare_batches(data, batch_size):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch
    Output:
        batches (list of lists of torch Tensor)
    """
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])
    return batches


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def test(val_data, model, batch_size):
    val_batches = prepare_batches(val_data, batch_size)
    model.eval()
    for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
        item_inputs = item_inputs.cuda()
        skill_inputs = skill_inputs.cuda()
        label_inputs = label_inputs.cuda()
        item_ids = item_ids.cuda()
        skill_ids = skill_ids.cuda()
        preds, weight = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
        preds = torch.sigmoid(preds).cpu()
        val_auc = compute_auc(preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TMSAKT.')
    parser.add_argument('--savedir', type=str, default='save/sakt')
    parser.add_argument('--dataset', type=str, default='statics')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=140)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=10)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    test_data = get_data(test_df, args.max_length)
    test_batches = prepare_batches(test_data, args.batch_size)

    num_items = int(test_df["item_id"].max() + 1)
    num_skills = int(test_df["skill_id"].max() + 1)

    model = SAKT(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob).cuda()

    old_path = os.path.join(args.savedir, 'statics,batch_size=128,max_length=200,encode_pos=False,max_pos=10')
    old_model = torch.load(old_path)
    new_path = os.path.join(args.savedir, 'params.pkl')
    torch.save(old_model.state_dict(), new_path)
    state_dict = torch.load(new_path)
    model.load_state_dict(state_dict)
    test_preds = np.empty(0)

    # Predict on test set
    model.eval()
    i = 1
    for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
        item_inputs = item_inputs.cuda()
        skill_inputs = skill_inputs.cuda()
        label_inputs = label_inputs.cuda()
        item_ids = item_ids.cuda()
        skill_ids = skill_ids.cuda()
        with torch.no_grad():
            preds, weights = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])
            # for i in range(weights.size(0)):
            #     weight = weights[i].cpu().numpy()
            #     heatmap(weight)
            weight = weights[0].cpu().numpy()
            heatmap(weight, i)
            i += 1

    # Write predictions to csv
    test_df["SAKT"] = test_preds
    #test_df.to_csv(f'data/{args.dataset}/heatmap1.csv', sep="\t", index=False)
    print("auc_test = ", roc_auc_score(test_df["correct"], test_preds))
