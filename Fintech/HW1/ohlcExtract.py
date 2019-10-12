import csv
import sys
OHLC = [None,None,None,None]
with open(sys.argv[1], newline='', encoding='BIG5') as f:
    reader = csv.reader(f)
    due_month = 202000
    for row in reader:
        if row[1] == "TX     " and int(row[3]) >= 84500 and int(row[3]) <= 134500 and '/' not in row[2] and int(row[2]) <= due_month:
            if int(row[2]) < due_month:
                due_month = int(row[2])
                OHLC = [row[4],row[4],row[4],row[4]]
            else:
                if row[4] > OHLC[1]:
                    OHLC[1] = row[4]
                if row[4] < OHLC[2]:
                    OHLC[2] = row[4]
                OHLC[3] = row[4]
print(OHLC[0], OHLC[1], OHLC[2], OHLC[3])
