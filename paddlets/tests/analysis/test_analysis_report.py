# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
sys.path.append(".")
import pandas as pd
import numpy as np

import os
import time
import unittest
from unittest import TestCase

from paddlets import TimeSeries, TSDataset
from paddlets.analysis import AnalysisReport


class TestAnalysisReport(TestCase):
    def setUp(self):
        """
        unittest function
        """
        periods = 100
        df = pd.DataFrame(
            [1 for i in range(periods)],
            index=pd.date_range('2022-01-01', periods=periods, freq='1D'),
            columns=['target']
        )
        ts = TSDataset.load_from_dataframe(df, target_cols="target")
        ts['target2'] = ts['target'] + 1

        self.tsdataset = ts

        super().setUp()

    def test_init(self):
        ###case1
        rp1 = AnalysisReport(self.tsdataset)

        ###case2
        analyzers_names = ["max", "summary"]
        rp2 = AnalysisReport(self.tsdataset, analyzers_names)

        ###case3
        analyzers_names = ["max", "summary", "fft"]
        params = {
            "fft": {
                "norm": False
            }
        }
        rp3 = AnalysisReport(self.tsdataset, analyzers_names, params)

        ###case4 badcase
        analyzers_names = ["max", "summary", "abb"]
        with self.assertRaises(ValueError):
            rp4 = AnalysisReport(self.tsdataset, analyzers_names, params)

    def test_export_pdf_report(self):
        ###case1
        ap1 = AnalysisReport(self.tsdataset)
        ap1.export_docx_report(path="/tmp",file_name="rp1.docx")
        file_object = open("/tmp/rp1.docx")

        assert file_object is not None

        ###case2
        analyzers_names = ["max", "summary"]
        ap2 = AnalysisReport(self.tsdataset, analyzers_names)
        ap2.export_docx_report(path="/tmp",file_name="rp2.docx")
        file_object = open("/tmp/rp2.docx")

        assert file_object is not None

        ###case3
        analyzers_names = ["max", "summary", "fft"]
        params = {
            "fft": {
                "norm": False
            }
        }
        ap3 = AnalysisReport(self.tsdataset, analyzers_names, params)
        ap3.export_docx_report(path="/tmp",file_name="rp3.docx")
        file_object = open("/tmp/rp3.docx")

        assert file_object is not None

        ###case4, bad_case, file path do not exist
        with self.assertRaises(ValueError):
            ap4 = AnalysisReport(self.tsdataset)
            ap4.export_docx_report(path="./bad_path", file_name="rp1.docx")

    def test_json_pdf_report(self):
        ###case1
        ap1 = AnalysisReport(self.tsdataset)
        json_rp = ap1.export_json_report()

        json = {'max': {'heading': 'MAX',
                        'description': 'Maximum values of given columns',
                        'analysis_results': '{"target":1,"target2":2}'
                        },
                'summary': {'heading': 'SUMMARY',
                            'description': 'Specified statistical indicators, currently support: numbers, mean,                 variance, minimum, 25% median, 50% median, 75% median, maximum value, missing percentage, stationarity p value',
                            'analysis_results': '{"target":{"missing":0.0,"count":100.0,"mean":1.0,"std":0.0,"min":1.0,"25%":1.0,"50%":1.0,"75%":1.0,"max":1.0},"target2":{"missing":0.0,"count":100.0,"mean":2.0,"std":0.0,"min":2.0,"25%":2.0,"50%":2.0,"75%":2.0,"max":2.0}}'}}
        self.assertEqual(str(json_rp), str(json))


        ###case2
        ap2 = AnalysisReport(self.tsdataset)
        json_rp = ap2.export_json_report(log=False)
        json = {'max': {'heading': 'MAX', 
                        'description': 'Maximum values of given columns', 
                        'analysis_results': '{"target":1,"target2":2}'},
                'summary': {'heading': 'SUMMARY', 
                            'description': 'Specified statistical indicators, currently support: numbers, mean,                 variance, minimum, 25% median, 50% median, 75% median, maximum value, missing percentage, stationarity p value',
                            'analysis_results': '{"target":{"missing":0.0,"count":100.0,"mean":1.0,"std":0.0,"min":1.0,"25%":1.0,"50%":1.0,"75%":1.0,"max":1.0},"target2":{"missing":0.0,"count":100.0,"mean":2.0,"std":0.0,"min":2.0,"25%":2.0,"50%":2.0,"75%":2.0,"max":2.0}}'}}
        self.assertEqual(str(json_rp), str(json))

        #case3
        analyzers_names = ["max", "summary", "fft"]
        ap3 = AnalysisReport(self.tsdataset, analyzers_names)
        json_rp = ap3.export_json_report()
        json = {'fft': {'heading': 'FFT', 
                        'description': 'Frequency domain analysis of signal based on fast Fourier transform.',
                        'analysis_results': '{"target_x":{"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12,"13":13,"14":14,"15":15,"16":16,"17":17,"18":18,"19":19,"20":20,"21":21,"22":22,"23":23,"24":24,"25":25,"26":26,"27":27,"28":28,"29":29,"30":30,"31":31,"32":32,"33":33,"34":34,"35":35,"36":36,"37":37,"38":38,"39":39,"40":40,"41":41,"42":42,"43":43,"44":44,"45":45,"46":46,"47":47,"48":48,"49":49},"target_amplitude":{"0":1.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0,"16":0.0,"17":0.0,"18":0.0,"19":0.0,"20":0.0,"21":0.0,"22":0.0,"23":0.0,"24":0.0,"25":0.0,"26":0.0,"27":0.0,"28":0.0,"29":0.0,"30":0.0,"31":0.0,"32":0.0,"33":0.0,"34":0.0,"35":0.0,"36":0.0,"37":0.0,"38":0.0,"39":0.0,"40":0.0,"41":0.0,"42":0.0,"43":0.0,"44":0.0,"45":0.0,"46":0.0,"47":0.0,"48":0.0,"49":0.0},"target_phase":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0,"16":0.0,"17":0.0,"18":0.0,"19":0.0,"20":0.0,"21":0.0,"22":0.0,"23":0.0,"24":0.0,"25":0.0,"26":0.0,"27":0.0,"28":0.0,"29":0.0,"30":0.0,"31":0.0,"32":0.0,"33":0.0,"34":0.0,"35":0.0,"36":0.0,"37":0.0,"38":0.0,"39":0.0,"40":0.0,"41":0.0,"42":0.0,"43":0.0,"44":0.0,"45":0.0,"46":0.0,"47":0.0,"48":0.0,"49":0.0},"target2_x":{"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12,"13":13,"14":14,"15":15,"16":16,"17":17,"18":18,"19":19,"20":20,"21":21,"22":22,"23":23,"24":24,"25":25,"26":26,"27":27,"28":28,"29":29,"30":30,"31":31,"32":32,"33":33,"34":34,"35":35,"36":36,"37":37,"38":38,"39":39,"40":40,"41":41,"42":42,"43":43,"44":44,"45":45,"46":46,"47":47,"48":48,"49":49},"target2_amplitude":{"0":2.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0,"16":0.0,"17":0.0,"18":0.0,"19":0.0,"20":0.0,"21":0.0,"22":0.0,"23":0.0,"24":0.0,"25":0.0,"26":0.0,"27":0.0,"28":0.0,"29":0.0,"30":0.0,"31":0.0,"32":0.0,"33":0.0,"34":0.0,"35":0.0,"36":0.0,"37":0.0,"38":0.0,"39":0.0,"40":0.0,"41":0.0,"42":0.0,"43":0.0,"44":0.0,"45":0.0,"46":0.0,"47":0.0,"48":0.0,"49":0.0},"target2_phase":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0,"16":0.0,"17":0.0,"18":0.0,"19":0.0,"20":0.0,"21":0.0,"22":0.0,"23":0.0,"24":0.0,"25":0.0,"26":0.0,"27":0.0,"28":0.0,"29":0.0,"30":0.0,"31":0.0,"32":0.0,"33":0.0,"34":0.0,"35":0.0,"36":0.0,"37":0.0,"38":0.0,"39":0.0,"40":0.0,"41":0.0,"42":0.0,"43":0.0,"44":0.0,"45":0.0,"46":0.0,"47":0.0,"48":0.0,"49":0.0}}'},
                'max': {'heading': 'MAX', 
                        'description': 'Maximum values of given columns', 
                        'analysis_results': '{"target":1,"target2":2}'},
                'summary': {'heading': 'SUMMARY', 
                            'description': 'Specified statistical indicators, currently support: numbers, mean,                 variance, minimum, 25% median, 50% median, 75% median, maximum value, missing percentage, stationarity p value',
                            'analysis_results': '{"target":{"missing":0.0,"count":100.0,"mean":1.0,"std":0.0,"min":1.0,"25%":1.0,"50%":1.0,"75%":1.0,"max":1.0},"target2":{"missing":0.0,"count":100.0,"mean":2.0,"std":0.0,"min":2.0,"25%":2.0,"50%":2.0,"75%":2.0,"max":2.0}}'}}
        self.assertEqual(str(json_rp), str(json))

if __name__ == "__main__":
    unittest.main()
