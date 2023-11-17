import argparse, sys, glob
import torch
from transformers import AutoTokenizer, AutoModel
import util

from huggingface_hub import login
login("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")

optim = torch.optim.AdamW(model.parameters(), lr=lr)
torch.manual_seed(42)
np.random.seed(42)

def check_version():
    import transformers
    if(transformers.__version__ is not "4.30.2"):
        sys.stderr.write("error: transformers version {transformers.__version__} is not 4.30.2, please install correct version")
        sys.exit(1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings
    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


def pretrain(model, tokenizer, train_data, val_data, lr, epoch, batch_size):
    val_batch = tokenizer(val_data, return_tensors = 'pt', padding=True, truncation=True, max_length=512)

def main(model_name, data_dir, logger):
    lr = 5.9574e-05
    epochs = 10 #28
    batch_size = 512
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True) #using DNABERT-2 since DNABERT-6's tokenizer is not very explainable
    model = = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_ids = [0, 1, 2, 3]
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        model.to(device)

    # Read and load data
    train_data, val_data, test_data = [], [], []
    file_list = ['Uniprot_Prokaryotes.csv', 'Uniprot_Eukaryotes.csv', 'Swissprot_Eukaryotes.csv', 'Swissprot_Prokaryotes.csv']
    for file_path in file_list:
        file_path = "{data_dir}/{file_path}"
        # Here we need to read Uniprot data first and then swiss prot, so model learn correct info in the latter stages of learning
        logger.info("reading from file {file_path}")
        train_temp, val_temp, test_temp = util.dataload(file_path)
        train_data.extend(train_temp['Sequence'])
        val_data.extend(val_temp['Sequence'])
        test_data.extend(test_temp['Sequence'])
    
    pretrain(model, tokenizer train_data, val_data, lr, epoch, batch_size)

    
if(__name__) == ('__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="model name", required=True)
    parser.add_argument("-d", "--data_dir", help="data absolute directory", required=True)
    parser.add_argument("-id","--job_id", help="job id", required=True)
    args = parser.parse_args()
    logging.basicConfig(filename="log/{}_{}.log".format(args.job_id, 'train'),
                    format="%(asctime)s [%(levelname)s]: %(message)s",
                    filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    check_version()  
    main(args.model_name, args.data_dir, logger)