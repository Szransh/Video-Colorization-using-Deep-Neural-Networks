import pandas as pd
file1 = open("/Users/harsh/m.txt","a")
metric = df = pd.DataFrame(columns = ['Generator Total Loss', 'Discriminator Total Loss', 'Genrator BCE LOSS'])
gen_total = []
disc_total = []
gen_bce = []

with open("/Users/harsh/m.txt", 'r') as f:
    lines = f.readlines()

for idx, l in enumerate(lines):
    if idx % 5 == 0:
        continue
    if idx % 5 == 1:
        continue
    if idx % 5 == 2:
        gen_total.append(l.split()[2])
    if idx % 5 == 3:
        disc_total.append(l.split()[2])
    if idx % 5 == 4:
        gen_bce.append(l.split()[3])

# print(gen_total)
# print(disc_total)
# print(gen_bce)
metric = pd.DataFrame(list(zip(gen_total, disc_total, gen_bce)),
               columns =['Generator Total Loss', 'Discriminator Total Loss', 'Genrator BCE LOSS'])
metric.to_csv("metric_epoch_51-150.csv")