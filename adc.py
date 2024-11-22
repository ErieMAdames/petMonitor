#!/usr/bin/env python3
from i2c import I2C

class ADC(I2C):
    ADDR=0x14
    def __init__(self, chn):
        super().__init__()
        if isinstance(chn, str):
            if chn.startswith("A"):
                chn = int(chn[1:])
            else:
                raise ValueError("ADC channel should be between [A0, A7], not {0}".format(chn))
        if chn < 0 or chn > 7:          
            self._error('Incorrect channel range')
        chn = 7 - chn
        self.chn = chn | 0x10          
        self.reg = 0x40 + self.chn
        
    def read(self):                     
        self.send([self.chn, 0, 0], self.ADDR)
        h = self.recv(1, self.ADDR)
        value_h = h[0]
        l = self.recv(1, self.ADDR)
        value_l = l[0]
        print('value_h : ' + str(value_h))
        print(value_h << 8)
        print(int(h))
        print('value_l : ' + str(value_l))
        print(int(l))
        value = (value_h << 8) + value_l
        return value

def test():
    import time
    adc = ADC(0)
    while True:
        print(adc.read())
        time.sleep(1)

if __name__ == '__main__':
    test()