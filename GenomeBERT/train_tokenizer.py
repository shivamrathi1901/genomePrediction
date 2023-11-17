from transformers import AutoTokenizer
import argparse, sys, os, logging, glob
import pandas as pd



def check_version():
    import transformers
    if(transformers.__version__ is not "4.30.2"):
        sys.stderr.write("error: transformers version {transformers.__version__} is not 4.30.2, please install correct version")
        sys.exit(1)

def tokenize(sequences, model_name):
    # not required as we are using static tokenizer
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    tokenizer.train(sequences, vocab_size=4096, min_frequency=2) #, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    if(not os.path.exists(model_name)):
      os.mkdir(model_name)
    tokenizer.save_model("models/{model_name}")

def read_all_sequences(data_dir):
    rawdata = pd.DataFrame(columns=[['Sequence']])
    for file in glob.glob("*.csv"):
        temp = pd.read_csv(file)
        temp = temp[['Sequence']]
        rawdata.append(temp)
    return rawdata['Sequence'].values.tolist()


def main(model_name, data_dir, logger):
    sequences = read_all_sequences(data_dir)
    logger.info("sequences are read, now training tokenizer")
    tokenize(sequences, model_name)

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
    main(args.model_name, args.data_dir, logger)
