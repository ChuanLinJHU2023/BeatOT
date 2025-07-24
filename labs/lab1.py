from methods import *
number_experiments = 10

for case_number in range(4):
    for i in range(number_experiments):
        method_tradition(case_number=case_number)

for case_number in range(4):
    for i in range(number_experiments):
        method_BCD_or_CD(if_BCD=True, case_number=case_number)

for case_number in range(4):
    for i in range(number_experiments):
        method_BCD_or_CD(if_BCD=False, case_number=case_number, num_epochs=10000)

for case_number in range(4):
    for i in range(number_experiments):
        method_OT(case_number=case_number)