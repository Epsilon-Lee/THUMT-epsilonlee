import argparse

parser = argparse.ArgumentParser("Reverse target file sentence order.")
parser.add_argument("--train_tgt", type=str, required=True)
args = parser.parse_args()
with open(args.train_tgt, 'r') as f_read, open(args.train_tgt + ".reverse", 'w') as f_write:
    for line in f_read:
        words = line.strip().split()
        words.reverse()
        new_line = ' '.join(words) + '\n'
        f_write.write(new_line)
print("Finished!")