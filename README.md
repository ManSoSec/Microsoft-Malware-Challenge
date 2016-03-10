# Microsoft-Malware-Challenge

The source code of feature extraction is not completely ready. But, The classification part is ready and you can run whole_system.py to see the results. 

Confusion Matrix related to 5 Fold cross validation :

![Confusion Matrix](https://github.com/ManSoSec/Microsoft-Malware-Challenge/blob/master/Dataset/Confusion matrix on MicMalChal xgb.png)

The final submitted file to kaggle website is in submissions folder.

The implementation is related to our paper at CODASPY'16:

https://www.researchgate.net/publication/283986464_Novel_Feature_Extraction_Selection_and_Fusion_for_Effective_Malware_Family_Classification

BibTex :

@inproceedings{Ahmadi:2016:NFE:2857705.2857713,
 author = {Ahmadi, Mansour and Ulyanov, Dmitry and Semenov, Stanislav and Trofimov, Mikhail and Giacinto, Giorgio},
 title = {Novel Feature Extraction, Selection and Fusion for Effective Malware Family Classification},
 booktitle = {Proceedings of the Sixth ACM on Conference on Data and Application Security and Privacy},
 series = {CODASPY '16},
 year = {2016},
 isbn = {978-1-4503-3935-3},
 location = {New Orleans, Louisiana, USA},
 pages = {183--194},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/2857705.2857713},
 doi = {10.1145/2857705.2857713},
 acmid = {2857713},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {classification, computer security, machine learning, malware family, microsoft malware classification challenge, windows malware},
}

Requirements :
numpy, pandas, xgboost, scikit-learn, hickle, pickle, numba, matplotlib


Thanks from Dmitry, Stanislav, and Mikhail for their support.
