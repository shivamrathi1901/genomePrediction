import argparse, sys, os, logging
import torch
from transformers import AutoTokenizer, AutoModel
import util
import numpy as np
import torch.nn as nn


# from huggingface_hub import login
# login("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")


torch.manual_seed(42)
np.random.seed(42)

def check_version():
    import transformers
    if(transformers.__version__ != "4.30.2"):
        sys.stderr.write("error: transformers version {} is not 4.30.2, please install correct version".format(transformers.__version__))
        sys.exit(1)

def prerequisite():
    create_dir = ['models', 'log']
    for dir in create_dir:
        if(not os.path.isdir(dir)):
            os.makedirs(dir)


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


def pretrain(model_name, train_data, val_data, job_id, scratch_model, scratch_token):
    # logger.info("{} and {}".format(type(val_data), len(val_data)))
    lr = 5.9574e-05
    epochs = 10 #28
    batch_size = 64
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True) #using DNABERT-2 since DNABERT-6's tokenizer is not very explainable
    if(scratch_model):
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(f"models/{model_name}", trust_remote_code=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_ids = [0, 1, 2, 3]
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = device_ids)
        # model = torch.nn.parallel.DistributedDataParallel(model)
        
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    training_losses = []
    val_losses = []
    val_batch = tokenizer(val_data, return_tensors = 'pt', padding=True, truncation=True, max_length=512)
    val_labels = val_batch['input_ids'].clone().detach()
    val_mask = val_batch['attention_mask'].clone().detach()
    val_input_ids = val_labels.detach().clone()
    val_encodings = {'input_ids': val_input_ids, 'attention_mask': val_mask, 'labels': val_labels}

    val_dataset = Dataset(val_encodings)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    

    min_avg_loss = 9999
    logger.info("Training Started!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for epoch in range(epochs):
        logger.info("epoch {} started..........................................................".format(epoch))
        mean_train_loss = 0
        mean_val_loss = 0
        logger.info("creating training batch for epoch {}".format(epoch))
        batch = tokenizer(train_data, return_tensors = 'pt', padding=True, truncation=True, max_length=512)
        # logger.info("training batch for epoch {} created \n {}".format(epoch, batch))
        # labels = torch.tensor(batch['input_ids'])
        # mask = torch.tensor(batch['attention_mask'])
        labels = batch['input_ids'].clone().detach()
        mask = batch['attention_mask'].clone().detach()
        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2) * (input_ids != 3)
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            # mask input_ids
            input_ids[i, selection] = 4  # our custom [MASK] token == 4

        encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
        dataset = Dataset(encodings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        # logger.info("Model parameters: {} \n\t next : {}".format(list(model.parameters()), next(model.parameters())))
        train_loss = 0
        counter = 0
        for batch in loader:
            counter += 1
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # logger.info("Args passed : {} \n\t {} \n\t {}".format(input_ids, attention_mask, labels))
            loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
            train_loss = loss
            if torch.cuda.device_count() > 1:
                train_loss.sum().backward()
            else:
                train_loss.backward()
            util.get_gpu_utilization(logger)
            # logger.info(f'Epoch {epoch},batch {counter} mean Loss : {train_loss.mean().item()}')  
            mean_train_loss += train_loss.mean().item()
        mean_train_loss = mean_train_loss/counter
        training_losses.append(mean_train_loss)
        logger.info(f'Epoch {epoch}, Training Mean Loss : {mean_train_loss}')
        counter = 0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                counter += 1
                val_input_ids = val_batch['input_ids'].to(device)
                val_attention_mask = val_batch['attention_mask'].to(device)
                val_labels = val_batch['labels'].to(device)
                # Process for validation
                loss, logits = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                val_loss += loss.mean().item()
        mean_val_loss = val_loss/counter
        val_losses.append(mean_val_loss)
        logger.info(f'Epoch {epoch}, Validation Mean Loss: {mean_val_loss}')
        if min_avg_loss > (0.5*mean_train_loss + 0.5*mean_val_loss):
            min_avg_loss = 0.5*mean_train_loss + 0.5*mean_val_loss
            logger.info("Saving best run model for value = {}".format(min_avg_loss))
            if torch.cuda.device_count() > 1:
                model.module.save_pretrained("models/{}".format(model_name))
            else:
                model.save_pretrained("models/{}".format(model_name))
        else:
            logger.info("Not saving model, because loss hasn't decreased")
    logger.info("Training completed, exiting!!")
    util.plot(training_losses, val_losses, job_id)
    return 0.5*mean_train_loss + 0.5*mean_val_loss



def main(model_name, data_dir, logger, job_id):
    

    # Read and load data
    train_data, val_data, test_data = [], [], []
    file_list = ['Uniprot_Eukaryotes.csv'] #, 'Uniprot_Eukaryotes.csv', 'Swissprot_Eukaryotes.csv', 'Swissprot_Prokaryotes.csv', 'Swissprot_Prokaryotes.csv'
    for file_path in file_list:
        file_path = "{}/{}".format(data_dir, file_path)
        # Here we need to read Uniprot data first and then swiss prot, so model learn correct info in the latter stages of learning
        logger.info("reading from file {}".format(file_path))
        train_temp, val_temp, test_temp = util.dataload(file_path)
        train_data.extend(train_temp['Sequence'].values.tolist())
        val_data.extend(val_temp['Sequence'].values.tolist())
        test_data.extend(test_temp['Sequence'].values.tolist())
    
    pretrain(model_name, train_data, val_data, job_id, scratch_model=False, scratch_token=True)

    
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
    prerequisite() 
    main(args.model_name, args.data_dir, logger, args.job_id)