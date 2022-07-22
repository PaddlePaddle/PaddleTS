========
Analysis
========

The purpose of the Analysis module is to make it easy for users to analyze time series data. We provide a variety of `analyzers <../../api/paddlets.analysis.html>`_ 
to inspect data properties. Moreover, we provide `report <../../api/paddlets.analysis.analysis_report.html>`_  API to show aggregated results of `analyzers <../../api/paddlets.analysis.html>`_ .

1. Analyzer
====================================
Currently support analyzers:

- `Summary <../../api/paddlets.analysis.base_analyzers.html>`_ : Statistical indicators, currently support numbers, mean, variance, minimum, 25% median, 50% median, 75% median, maximum value, missing percentage, stationarity p value.
- `Max <../../api/paddlets.analysis.base_analyzers.html>`_ : Compute maximum values of given columns.
- `FFT <../../api/paddlets.analysis.frequency_domain.html>`_ : Frequency domain analysis of signal based on fast Fourier transform.
- `STFT <../../api/paddlets.analysis.frequency_domain.html>`_ : Time-frequency analysis of signal based on short-time Fourier transform.
- `CWT <../../api/paddlets.analysis.frequency_domain.html>`_ : Time-frequency analysis of signal based on continuous wavelet transform.
  
The following code snippet shows how to apply `analyzers <../../api/paddlets.analysis.html>`_ on a TSDataset object.
We use the ``UNI_WTH`` dataset as a sample, which is a univariate dataset containing weather from 2010 to 2014, where ``WetBulbCelsuis`` represents the wet bulb temperature.

.. code:: python

   from paddlets.datasets.repository import get_dataset
   from paddlets.analysis import Summary
   tsdataset = get_dataset('UNI_WTH')
   tsdataset.summary()
   sum = Summary()
   sum(tsdataset)
   
   #         WetBulbCelsius
   #missing        0.000000
   #count      35064.000000
   #mean           1.026081
   #std            6.898354
   #min          -26.400000
   #25%           -3.800000
   #50%            0.600000
   #75%            6.600000
   #max           16.300000


Note that `base analyzers <../../api/paddlets.analysis.base_analyzers.html>`_ can be invoked by ``TSdataset`` directly:

Currently support base analyzers: `Summary <../../api/paddlets.analysis.base_analyzers.html>`_ 
, `Max <../../api/paddlets.analysis.base_analyzers.html>`_ .

.. code:: python

   from paddlets.datasets.repository import get_dataset
   tsdataset = get_dataset('UNI_WTH')
   tsdataset.summary()




1. Analysis Report
====================================
`Analysis Report <../../api/paddlets.analysis.analysis_report.html>`_ is designed to show aggragated 
analysis results in the form of a report. Three examples to get a analysis report are demonstrated below :

2.1 Default Analysis Report
-------------------------------
.. code:: python

   from paddlets.analysis import AnalysisReport
   from paddlets.datasets.repository import get_dataset
   tsdataset = get_dataset('UNI_WTH')
   report = AnalysisReport(tsdataset)
   # export a file named "analysis_report.docx" to current path by default
   report.export_docx_report()

2.2 Customized Analyzers Report with Default Config
---------------------------------------------------------
.. code:: python

   from paddlets.analysis import AnalysisReport
   from paddlets.datasets.repository import get_dataset
   tsdataset = get_dataset('UNI_WTH')
   report = AnalysisReport(tsdataset, ["summary","fft"])
   # export a file named "analysis_report.docx" to current path by default
   report.export_docx_report()

2.3 Customized Analyzers Report with Customized Config
---------------------------------------------------------
.. code:: python

   from paddlets.analysis import AnalysisReport
   from paddlets.datasets.repository import get_dataset
   tsdataset = get_dataset('UNI_WTH')
   customized_config = {"fft":{
                    "norm":False,
                    "fs":1
                     }
            }
   report = AalysisReport(tsdataset, ["summary","fft"], customized_config)
   # export a file named "analysis_report.docx" to current path by default
   report.export_docx_report()
