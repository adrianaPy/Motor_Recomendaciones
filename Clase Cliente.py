import numpy as np
from sklearn.preprocessing import StandardScaler

class Client(object):

    def __init__(self,age=None,credit_amount=None,duration=None):

        if age:
            self.age = age            
        if credit_amount:
            self.mount = credit_amount
        if duration:
            self.duration = duration
        
                 
    def set_name(self,name):
      self.name = name

    def set_age(self,age):
      self.age = age

    def set_credit_amount(self,mount):
      self.mount = mount

    def set_duration(self,duration):
      self.duration = duration

    def set_n_age(self,nage):
      self.nage = nage

    def set_n_credit_amount(self,nmount):
      self.nmount = nmount

    def set_n_duration(self,nduration):
      self.nduration = nduration

    def set_key(self,key):
      self.key = key
    
    def set_num_key(self, num_key):
      self.num_key = num_key

    def get_name(self):
      return self.name

    def get_age(self):
      return self.age

    def get_credit_amount(self):
      return self.mount

    def get_duration(self):
      return self.duration

    def get_key(self):
      return self.key 
 
    def get_n_age(self):
      return self.nage

    def get_n_credit_amount(self):
      return self.nmount

    def get_n_duration(self):
      return self.nduration
    
    def get_num_key(self):
      return self.num_key


  

class Clients(object):
  def __init__(self,data):
    self.data = data
    self.nClients = len(data)
    self.clients = np.ndarray((self.nClients),dtype=np.object)
    self.y = np.ndarray(self.nClients,dtype=int)
    self.load_data()
    self.set_key()
    self.normalization()

  def load_data(self):
      for c in range(self.nClients):
          client = Client()
          client.set_age(self.data["Age"][c])
          client.set_credit_amount(self.data["Credit amount"][c])
          client.set_duration(self.data["Duration"][c])
          self.clients[c] = client
          #print(self.clients[c].get_age()) 

  def set_key(self):

      for c in range(self.nClients):

        #print(self.clients[c].get_age()) #OK: aparecen las edades
        #print(self.clients[c].get_credit_amount()) #OK: aparece el monto de cada cliente

        if self.clients[c].get_credit_amount() > 5000:
          if  self.clients[c].get_age() > 20  and self.clients[c].get_age() < 50:
            self.clients[c].set_key("key_A")
            self.clients[c].set_num_key(0)
          else:
            self.clients[c].set_key("key_B")
            self.clients[c].set_num_key(1)

        if self.clients[c].get_credit_amount() < 5000 and  self.clients[c].get_credit_amount() > 1000:
          if  self.clients[c].get_age() > 20  and self.clients[c].get_age() < 50:
            self.clients[c].set_key("key_B")
            self.clients[c].set_num_key(1)
          else:
            self.clients[c].set_key("key_C")
            self.clients[c].set_num_key(2)

        if self.clients[c].get_credit_amount() < 1000:
          self.clients[c].set_key("key_C")
          self.clients[c].set_num_key(2)

        #print(self.clients[c].get_key()) # OK: asignamos las llaves 

  def normalization(self):
      data_log = np.log(self.data)
      scaler = StandardScaler()
      self.data_log_scaler = scaler.fit_transform(data_log)
      #print(data_log_scaler)
      #print()

      for c in range(self.nClients):
        self.clients[c].set_n_age(self.data_log_scaler[c,0])
        self.clients[c].set_n_credit_amount(self.data_log_scaler[c,1])
        self.clients[c].set_n_duration(self.data_log_scaler[c,2])
        #print(self.clients[c].get_n_age(), self.clients[c].get_n_credit_amount(), self.clients[c].get_n_duration())

  def getX(self):
    return self.data_log_scaler
  
  def getY(self):
    for c in range(self.nClients):
      self.y[c] = self.clients[c].get_num_key();
    
    return self.y
