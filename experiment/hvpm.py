import Monsoon.HVPM as HVPM
import Monsoon.LVPM as LVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op
import Monsoon.pmapi as pmapi
import time
import sys
from threading import Event, Thread



class Monitor(object):
    def __init__(self, voltage = 5):
        self.HVMON = HVPM.Monsoon()
        self.HVMON.setup_usb()
        self.HVMON.fillStatusPacket()
        self.HVMON.setVout(voltage)
        
        self.HVengine = sampleEngine.SampleEngine(self.HVMON)
        self.HVengine.ConsoleOutput(False)
        print("initialization complete for HVPM serial number: " + repr(self.HVMON.getSerialNumber()))

    def start_sampling(self, csv_output):
        print("start sampling")
        self.HVengine.enableCSVOutput(csv_output) 
        self.HVengine.periodicStartSampling()
        self.condition = Event()
        
        t = Thread(target = self.sample)
        t.start()
        
    def sample(self):
        while not self.condition.is_set():
            self.HVengine.periodicCollectSamples(5, legacy_timestamp=True) 
            #print("collecting HVPM samples...")
            time.sleep(1) 
        
        print("exiting thread...")

    def stop_sampling(self, set_condition=False):   
        print("stop sampling")
        if set_condition:
            self.condition.set()

        time.sleep(2)
        self.HVengine.periodicStopSampling(closeCSV=True)
    
    def close(self):
        pass
        # avoid closing monitor by mistake, and interrupt voltage to rpi 
        # the rpi should be gracefully shutdown before interrupting the power supply 
        self.HVMON.setVout(0)
        #self.HVMON.closeDevice()

