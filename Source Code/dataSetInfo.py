

class dataSetInformaion:
   'Common base class for all employees'
   dataSetName = list()
   rowsName = None
   data = None
   classLabel = None

   def __init__(self, dataSetName, data, classLabel):
      self.dataSetName = dataSetName
      self.data = data
      self.classLabel = classLabel

   #def displayCount(self):
   #  print