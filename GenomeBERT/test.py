from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForMaskedLM
import random, logging, sys, glob
from sklearn.metrics import classification_report
import pandas as pd

logging.basicConfig(filename="log/{}_{}.log".format(sys.argv[2], 'test'),
                    format="%(asctime)s [%(levelname)s]: %(message)s",
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def run_test(sequences, model, tokenizer):
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    result = []
    expected = []

    for i in range(len(sequences)):
        idx = random.randint(0, len(sequences[i]) - 1)
        test_seq = sequences[i][:idx] + fill.tokenizer.mask_token + sequences[i][idx + 4:]
        expected.append(sequences[i][idx: idx + 4])
        sequences[i] = test_seq

    # Directly access the model outputs without modifying the tuple
        fill_output = fill(test_seq)
        logger.info(fill_output[0]['token_str'])
        resp = fill_output[0]['token_str']

        result.append(fill_output[0]['token_str'])

    return result, expected

def chunk_sequences(file_path, chunk_size, model, tokenizer):
    result, expect = [], []
    for file in glob.glob(file_path):
        reader = pd.read_csv(file, chunksize=chunk_size)
        for chunk in reader:
            sequences = chunk['Sequence'].tolist()
            results, expected = run_test(sequences, model, tokenizer)
            result.extend(results)
            expect.append(expected)

        # Process the results and expected values as needed
        #for res, exp in zip(results, expected):
        #    print(f"Expected: {exp}  Response: {res}")
        #    if(len(expected)%100==0):
        #        logger.info("processing {}".format(len(expected)))
        #    print(classification_report(results,expected))
    print(classification_report(result,expect))

if __name__ == '__main__':
    csv_file_path = "./test/Swissprot_*.csv"
    model_name = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    chunk_sequences(csv_file_path, chunk_size=2, model=model, tokenizer=tokenizer)

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
