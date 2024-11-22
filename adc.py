#!/usr/bin/env python3
from i2c import I2C

class ADC(I2C):
    ADDR = 0x14
    MAX_ADC_VALUE = 4095  # Assuming a 12-bit ADC (adjust if necessary)
    MIN_ADC_VALUE = 0     # Minimum ADC value

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
        
    def read_raw(self):                     
        # Send the channel select command
        self.send([self.chn], self.ADDR)

        # Read 2 bytes (high and low) from the ADC
        data = self.recv(2, self.ADDR)

        # Combine the high and low bytes into a single 16-bit value
        value = (data[0] << 8) | data[1]

        # If the ADC resolution is 12 bits, mask the lower 12 bits
        value &= 0x0FFF

        return value

    def read_normalized(self):
        """Normalize the ADC value to a range of 0-100."""
        raw_value = self.read_raw()
        normalized = int(
            (raw_value - self.MIN_ADC_VALUE) / 
            (self.MAX_ADC_VALUE - self.MIN_ADC_VALUE) * 100
        )
        # Clamp the value between 0 and 100
        return max(0, min(100, normalized))


def test():
    import time
    adc = ADC(0)  # Use ADC channel 0
    while True:
        raw_value = adc.read_raw()
        water_level = adc.read_normalized()
        print(f"Raw ADC Value: {raw_value}, Water Level: {water_level}%")
        time.sleep(1)

if __name__ == '__main__':
    test()
