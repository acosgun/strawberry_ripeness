all_data_in_format.csv details.
The data is smoothed and noise is removed. Water content is also added to it with five potential classes. Best to use class_4. (earlier there was five class but class 2 and clas 5 were same so I removed one of them)

1. First column is name of the sample
2. column name 350 (2nd col) to name 2300 is the wavelenth
3. WC colymn is water content value ((fresh weight-dry weight)/(fresh weight))
4. Class_1: 0 for WC<90, 1 for WC>90
5. Class_2: 0 for WC<90, 1 for WC>91, 2 for 90<WC<91
6. Class_3: 0 for 87-88, 1 for 88-89, 2 for 89-90, 3 for 90-91, 4 for 91-92, 5 for 92-93, 6 for 93-94
7. Class_4: 0 for 87-89, 1 for 89-91, 2 for 91-94
..........................................................