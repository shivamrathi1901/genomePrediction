from transformers import AutoTokenizer
import argparse, sys, os, logging, glob
import pandas as pd
import gc


def check_version():
    import transformers
    if(transformers.__version__ != "4.30.2"):
        sys.stderr.write("error: transformers version {transformers.__version__} is not 4.30.2, please install correct version")
        sys.exit(1)

def get_training_corpus(sequences):

    for i in range(0, len(sequences), 1000):
        # for seq in sequences[i : i + 1000]:
        #     if(not isinstance(seq, str)):
        #         logger.info("seq is : {}".format(seq))
        #         sys.exit(1)
        yield sequences[i : i + 1000]

def tokenize(sequences, model_name, logger):
    # not required as we are using static tokenizer
    # if(os.path.exists("models/{}".format(model_name))):
    #     tokenizer = AutoTokenizer.from_pretrained("models/{}".format(model_name), trust_remote_code=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    training_corpus = get_training_corpus(sequences)
    logger.info("now training tokenizer")
    tokenizer.train_new_from_iterator(training_corpus, 4096) #, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    # if(not os.path.exists(model_name)):
    #   os.mkdir(model_name)
    tokenizer.save_pretrained("models/{}".format(model_name))
    logger.info("tokenizer is trained and saved")

def read_all_sequences(data_dir):
    rawdata = pd.DataFrame(columns=[['Sequence']])
    for file in glob.glob("{}/Swissprot_*.csv".format(data_dir)):
        logger.info("reading file : {}".format(file))
        temp = pd.read_csv(file)
        temp = temp[['Sequence']]
        frames = [rawdata, temp]
        rawdata = pd.concat(frames)
        # rawdata = rawdata.sample(frac=0.5, random_state=42)
    del frames
    gc.collect()
    rawdata = rawdata[['Sequence']]
    rawdata = rawdata.dropna()
    return rawdata['Sequence'].values.tolist()


def main(model_name, data_dir, logger):
    sequences = read_all_sequences(data_dir)
    logger.info("Sequence read is of size {}".format(len(sequences)))
    tokenize(sequences, model_name, logger)

if(__name__) == ('__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="model name", required=True)
    parser.add_argument("-d", "--data_dir", help="data absolute directory", required=True)
    parser.add_argument("-id","--job_id", help="job id", required=True)
    args = parser.parse_args()
    logging.basicConfig(filename="log/{}_{}.log".format(args.job_id, 'tokenize'),
                    format="%(asctime)s [%(levelname)s]: %(message)s",
                    filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    main(args.model_name, args.data_dir, logger)
