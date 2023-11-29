from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForMaskedLM
import random, logging, sys, glob
from sklearn.metrics import classification_report
import pandas as pd
import util
tokenizer_dir = "zhihan1996/DNABERT-2-117M" #default tokenizer
model_name = "zhihan1996/DNABERT-2-117M"
model_name = sys.argv[1]
tokenizer_dir = sys.argv[2]
meta = "Test: Swiss trained tokenizer with Model trained till 15th epoch"
# with open(f"{model_name}/meta.txt", 'w') as file:
#     file.write(meta)
logging.basicConfig(filename="log/{}_{}.log".format(sys.argv[3], 'test'),
                    format="%(asctime)s [%(levelname)s]: %(message)s",
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.info(meta)
def run_test(sequences, model, tokenizer):
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    result = []
    expected = []
    for i in range(len(sequences)):
        idx = random.randint(0, len(sequences[i]) - 1)
        test_seq = sequences[i][:idx] + fill.tokenizer.mask_token + sequences[i][idx + 4:]
        expected.append(sequences[i][idx: idx + 4])
        # sequences[i] = test_seq
    # Directly access the model outputs without modifying the tuple
        fill_output = fill(test_seq)
        # logger.info(fill_output[0]['token_str'])
        resp = fill_output[0]['token_str']
        result.append(fill_output[0]['token_str'])
    # logger.info(f"result len : {len(result)}  expected len : {len(expected)}")
    return result, expected

def chunk_sequences(file_path, chunk_size, model, tokenizer):
    result, expect = [], []
    for file in glob.glob(file_path):
        reader = pd.read_csv(file, chunksize=chunk_size)
        for chunk in reader:
            sequences = chunk['Sequence'].tolist()
            results, expected = run_test(sequences, model, tokenizer)
            result.extend(results)
            expect.extend(expected)

        # Process the results and expected values as needed
        #for res, exp in zip(results, expected):
        #    print(f"Expected: {exp}  Response: {res}")
        #    if(len(expected)%100==0):
        #        logger.info("processing {}".format(len(expected)))
        #    print(classification_report(results,expected))
    overlap_acc = 0
    for str1, str2 in zip(result, expect):
        overlap_string = util.find_largest_overlapping_substring(str1, str2)
        overlap_acc += len(overlap_string)/len(str2)

    logger.info(f"Avg overlap accuraccy : {overlap_acc/len(expect) *100}%")

    print(classification_report(result,expect))

if __name__ == '__main__':
    csv_file_path = "./data/test/Swissprot_*.csv"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"Loading Tokenizer from {tokenizer_dir} and loading model from {model_name}")
    chunk_sequences(csv_file_path, chunk_size=50, model=model, tokenizer=tokenizer)

# import pandas as pd
# from transformers import AutoTokenizer, AutoModel, pipeline
# import transformers
# csv_file_path = "test.csv"
# model_name = "GenomeBERT"
# expected = []
# #tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# #model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
# sequence = "ATGATTGTCGGCATTCTCACCACGCTGGCTACGCTGGCCACACTCGCAGCTAGTGTGCCTCTAGAGGAGCGGCAAGCTTGCTCAAGCGTCTGGGGCCAATGTGGTGGCCAGAATTGGTCGGGTCCGACTTGCTGTGCTTCCGGAAGCACATGCGTCTACTCCAACGACTATTACTCCCAGTGTCTTCCCGGCGCTGCAAGCTCAAGCTCGTCCACGCGCGCCGCGTCGACGACTTCTCGAGTATCCCCCACAACATCCCGGTCGAGCTCCGCGACGCCTCCACCTGGTTCTACTACTACCAGAGTACCTCCAGTCGGATCGGGAACCGCTACGTATTCAGGCAACCCTTTTGTTGGGGTCACTCCTTGGGCCAATGCATATTACGCCTCTGAAGTTAGCAGCCTCGCTATTCCTAGCTTGACTGGAGCCATGGCCACTGCTGCAGCAGCTGTCGCAAAGGTTCCCTCTTTTATGTGGCTAGATACTCTTGACAAGACCCCTCTCATGGAGCAAACCTTGGCCGACATCCGCACCGCCAACAAGAATGGCGGTAACTATGCCGGACAGTTTGTGGTGTATGACTTGCCGGATCGCGATTGCGCTGCCCTTGCCTCGAATGGCGAATACTCTATTGCCGATGGTGGCGTCGCCAAATATAAGAACTATATCGACACCATTCGTCAAATTGTCGTGGAATATTCCGATATCCGGACCCTCCTGGTTATTGAGCCTGACTCTCTTGCCAACCTGGTGACCAACCTCGGTACTCCAAAGTGTGCCAATGCTCAGTCAGCCTACCTTGAGTGCATCAACTACGCCGTCACACAGCTGAACCTTCCAAATGTTGCGATGTATTTGGACGCTGGCCATGCAGGATGGCTTGGCTGGCCGGCAAACCAAGACCCGGCCGCTCAGCTATTTGCAAATGTTTACAAGAATGCATCGTCTCCGAGAGCTCTTCGCGGATTGGCAACCAATGTCGCCAACTACAACGGGTGGAACATTACCAGCCCCCCATCGTACACGCAAGGCAACGCTGTCTACAACGAGAAGCTGTACATCCACGCTATTGGACCTCTTCTTGCCAATCACGGCTGGTCCAACGCCTTCTTCATCACTGATCAAGGTCGATCGGGAAAGCAGCCTACCGGACAGCAACAGTGGGGAGACTGGTGCAATGTGATCGGCACCGGATTTGGTATTCGCCCATCCGCAAACACTGGGGACTCGTTGCTGGATTCGTTTGTCTGGGTCAAGCCAGGCGGCGAGTGTGACGGCACCAGCGACAGCAGTGCGCCACGATTTGACTCCCACTGTGCGCTCCCAGATGCCTTGCAACCGGCGCCTCAAGCTGGTGCTTGGTTCCAAGCCTACTTTGTGCAGCTTCTCACAAACGCAAACCCATCGTTCCTGTAA"
# fill = pipeline('fill-mask', model=model, tokenizer="zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# print(transformers.__version__)
# idx = 10
# masked_token = sequence[idx]
# test_seq = sequence[:idx] + fill.tokenizer.mask_token + sequence[idx + 1:]
# # expected.append(sequence[idx])
# print(test_seq)
# resp = fill(test_seq)
# print(resp)
