from collections import Counter
import os
import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_text_answer(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_text_question(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, max_length=32):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.max_length = 24

        # answerの辞書を作成
        self.answer2idx = {}
        self.idx2answer = {}

        # tokenizerとvocaburary
        self.tokenizer = get_tokenizer("basic_english")
        counter = Counter()
        for question in self.df["question"]:
            question = process_text_question(question)
            counter.update(self.tokenizer(question))

        self.vocaburary = vocab(counter, specials=('<unk>', '<PAD>', '<BOS>', '<EOS>'))
        self.vocaburary.set_default_index(self.vocaburary["<unk>"])

        # 後でEmbeddingに渡すために語彙サイズを得ておく
        self.vocaburary_size = len(self.vocaburary)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text_answer(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)


    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像,質問,回答）を取得

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をtokenizerにかけたもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question_text = self.df["question"][idx]
        question = [self.vocaburary[token] for token in self.tokenizer(question_text)][:self.max_length - 2]
        question = [self.vocaburary["<BOS>"]] + question + [self.vocaburary["<EOS>"]]

        if self.answer:
            answers = [self.answer2idx[process_text_answer(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

# dataloaderにbatch毎の処理を行わせる
def collate_batch_train(batch):
    image_list, question_list, answers_list, mode_answer_idx_list = [], [], [], []
    for image, question, answers, mode_answer_idx in batch:
        image_list.append(image)
        answers_list.append(answers)
        mode_answer_idx_list.append(mode_answer_idx)
        question_list.append(question)
    # torch.stackでないとerrorが出る
    # ref: https://qiita.com/sekishoku/items/a578103786e77f31d0e1
    return torch.stack(image_list), pad_sequence(question_list, batch_first=True, padding_value=1).to(dtype=torch.int), torch.stack(answers_list), torch.tensor(mode_answer_idx_list)

def collate_batch_test(batch):
    image_list, question_list = [], []
    for image, question in batch:
        image_list.append(image)
        question_list.append(question)
    return torch.stack(image_list), pad_sequence(question_list, batch_first=True, padding_value=1).to(dtype=torch.int)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
class Conv2dNormReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding="same", stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
            nn.GELU(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        x = self.norm1(inputs)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + inputs
        z = self.norm2(x)
        z = self.mlp(z)
        return z + x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, seq_length: int, hidden_dim: int, num_heads: int, mlp_dim:int, num_layers: int = 8, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        self.encoders = nn.Sequential()
        for _ in range(num_layers):
            self.encoders.append(EncoderBlock(hidden_dim, num_heads, mlp_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        # inputs: (batch_size, seq_length, hidden_dim)
        inputs = inputs + self.pos_embedding
        return self.norm(self.encoders(self.dropout(inputs)))


class VisionTransformer(nn.Module):
    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            hidden_dim: int,
            mlp_dim: int,
            num_heads: int = 16,
            in_channels: int = 3
        ):
        super().__init__()
        # ref: https://github.com/pytorch/vision/blob/4d1ff711581c54ad21bfee7c41c62a145a3a4cfc/torchvision/models/vision_transformer.py#L192
        self.conv_proj = Conv2dNormReLU(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        seq_length = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1
        self.seq_length = seq_length

        self.encoder = VisionTransformerEncoder(
            seq_length,
            hidden_dim,
            num_heads,
            mlp_dim,
            num_layers
        )

    def _process_input(self, inputs):
        n, c, h, w = inputs.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(inputs)
        x = torch.reshape(x, shape=(n, self.hidden_dim, n_h * n_w))
        # (n, hidden_dim, n_h * n_w) -> (n, n_h * n_w, hidden_dim)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, inputs):
        x = self._process_input(inputs)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # class token
        return x[:, 0]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, encoder_num: int = 6, n_head: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.transformer = nn.Sequential()
        for _ in range(encoder_num):
            self.transformer.append(nn.TransformerEncoderLayer(embedding_dim, n_head))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # defaultでTransformerEncoderLayerのactivation functionはReLUになっている
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs):
        x = self.embedding(inputs)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.transformer(x)
        # class token
        return x[:, 0]

class VQAModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_answer: int,
            image_size: int = 224,
            patch_size: int = 16,
            num_layers_vit: int = 12,
            hidden_dim: int = 512,
            mlp_dim: int = 1024,

    ):
        super().__init__()
        self.vit = VisionTransformer(
            image_size,
            patch_size,
            num_layers_vit,
            hidden_dim,
            mlp_dim,

        )
        self.text_encoder = TextEncoder(
            vocab_size,
            hidden_dim,
        )

        # MLP Head
        self.fc1 = nn.Linear(hidden_dim * 2, mlp_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_dim, n_answer)

        # 重みの初期化
        nn.init.kaiming_uniform_(self.fc1.weight.data)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight.data)
        self.fc2.bias.data.zero_()


    def forward(self, image, question):
        image_class_token = self.vit(image)  # 画像のclass token
        question_class_token = self.text_encoder(question)  # テキストのclass token

        x = torch.cat([image_class_token, question_class_token], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        # pin_memory=Trueとセットでnon_blocking=Trueとして高速化させる
        image, question, answer, mode_answer = \
            image.to(device, non_blocking=True), question.to(device, non_blocking=True), answers.to(device, non_blocking=True), mode_answer.to(device, non_blocking=True)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    # 訓練の高速化をする
    # ref: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_batch_train,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch_test,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    torch.set_float32_matmul_precision("high")
    model = torch.compile(VQAModel(vocab_size=train_dataset.vocaburary_size, n_answer=len(train_dataset.answer2idx)).to(device), mode="default")

    # optimizer / criterion
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    # loggingにしたほうが良いが今回はそこまでしない
    print("Start training...")
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】", f"train time: {train_time:.2f} [s]", f"train loss: {train_loss:.4f}", f"train acc: {train_acc:.4f}", f"train simple acc: {train_simple_acc:.4f}")
        if (epoch + 1) % 10 == 0:
            model_path = f"model_assets/model_e{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint: file {model_path} saved.")

    print("Finished training!")

    model.eval()
    submission = []
    with torch.no_grad():
        for image, question in test_loader:
            image, question = image.to(device), question.to(device)
            pred = model(image, question)
            pred = pred.argmax(1).cpu().item()
            submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
